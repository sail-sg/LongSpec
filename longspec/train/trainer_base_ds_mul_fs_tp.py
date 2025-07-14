import datetime
import glob
import logging
import os
import sys
from typing import Dict, Union
from collections import OrderedDict
import shutil

import deepspeed
import fairscale.nn.model_parallel.initialize as mpu
import hydra
import torch
import wandb
from deepspeed import comm as dist
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedModel

from general_util.dist_utils import get_pipeline_parallel_rank, get_pipeline_parallel_world_size, prepare_distributed_sampler
from general_util.evaluator import evaluate
from general_util.logger import setting_logger
from general_util.training_utils import batch_to_device, set_seed, note_best_checkpoint, load_and_cache_examples, set_seed_int, \
    organize_multiple_dataset, get_last_checkpoint

logger: logging.Logger

torch.backends.cuda.matmul.allow_tf32 = True

GLOBAL_SEED = 1
GLOBAL_WORKER_ID = None


def get_zero_stage(cfg: DictConfig):
    if hasattr(cfg, "zero_optimization"):
        return int(getattr(cfg.zero_optimization, "stage", 0))
    return 0


def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed_int(GLOBAL_SEED + worker_id)


def extract_and_rename(state_dict, prefix="draft_model."):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
    return new_state_dict


def save_model(model: Union[deepspeed.DeepSpeedEngine, deepspeed.PipelineEngine],
               cfg: DictConfig, output_dir: str, tokenizer: PreTrainedTokenizer = None, state_dict: Dict = None):
    unwrapped_model = model.module
    assert isinstance(unwrapped_model, PreTrainedModel)

    save_ds_state = getattr(cfg, "save_ds_state", True)
    zero_stage = get_zero_stage(cfg.ds_cfg)

    if not save_ds_state:
        if zero_stage == 3:
            logger.warning("Deepspeed ZeRO-3 has to save checkpoint states since the model is sharded.")
            saving_ds_state = True

    if not hasattr(save_model, "tag") or save_model.tag == 'last_2ds':
        save_model.tag = 'last_ds'
        save_model.hf_tag = 'checkpoint-last'
    else:
        save_model.tag = 'last_2ds'
        save_model.hf_tag = 'checkpoint-2last'

    if save_ds_state:
        os.makedirs(f"{cfg.output_dir}/{save_model.tag}", exist_ok=True)
        model.save_checkpoint(cfg.output_dir, tag=save_model.tag)

    if zero_stage == 3:
        state_dict = model._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = model.module.state_dict()

    if mpu.model_parallel_is_initialized():
        dp_rank = mpu.get_data_parallel_rank()
    else:
        if dist.is_initialized():
            dp_rank = dist.get_rank()
        else:
            dp_rank = -1

    if dist.is_initialized() and dp_rank != 0:
        dist.barrier()

    if dp_rank in [-1, 0]:
        output_dir = f"{cfg.output_dir}/{save_model.hf_tag}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        draft_model_weights = extract_and_rename(state_dict, prefix="draft_model.")
        torch.save(draft_model_weights, os.path.join(output_dir, "draft_model_weights.pth"))

        # if cfg.local_rank in [-1, 0]:
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

        OmegaConf.save(cfg, os.path.join(output_dir, "training_config.yaml"))
        logger.info("Saving model checkpoint to %s", output_dir)

        if dist.is_initialized():
            dist.barrier()


def forward_step(model, inputs: Dict[str, torch.Tensor]):
    outputs = model(**inputs)
    if isinstance(outputs, tuple):
        loss = outputs[0]
    else:
        loss = outputs["loss"]
    model.backward(loss)
    model.step()

    return loss.item(), outputs


def train(cfg, model, tokenizer, continue_from_global_step=0):
    """ Train the model """
    if cfg.local_rank in [-1, 0]:
        tb_helper = hydra.utils.instantiate(cfg.summary_helper) if "summary_helper" in cfg and cfg.summary_helper else None
    else:
        tb_helper = None

    cfg.train_batch_size = cfg.per_gpu_train_batch_size

    files, total_dataset_len = organize_multiple_dataset(cfg, tokenizer, _split="train")

    logger.warning(f"Rank No. {dist.get_rank()} has {total_dataset_len} samples.")

    if getattr(cfg, "do_preprocess", False):
        return

    if "extended_vocab" in cfg and cfg.extended_vocab:
        logger.info(f"Extended extra vocab size: {cfg.extended_vocab}")
        model.resize_token_embeddings(model.config.vocab_size + cfg.extended_vocab)

    dp_degree = cfg.dp_size
    _actual_train_batch_size = cfg.train_batch_size * cfg.gradient_accumulation_steps * dp_degree
    if cfg.max_steps > 0:
        t_total = cfg.max_steps
        cfg.num_train_epochs = cfg.max_steps // (total_dataset_len // _actual_train_batch_size) + 1
    else:
        t_total = total_dataset_len // _actual_train_batch_size * cfg.num_train_epochs

    num_warmup_steps = int(t_total * cfg.warmup_proportion) if cfg.warmup_proportion else cfg.warmup_steps

    ds_config = cfg.ds_cfg
    if "total_num_steps" in ds_config.scheduler.params:
        ds_config.scheduler.params.total_num_steps = t_total
    ds_config.scheduler.params.warmup_num_steps = num_warmup_steps
    ds_config = OmegaConf.to_container(ds_config, resolve=True)

    optimizer = hydra.utils.instantiate(cfg.optimizer, model) if getattr(cfg, "optimizer", None) else None

    if torch.__version__ >= "2" and (getattr(os.environ, "TORCH_COMPILE", False) or getattr(cfg, "compile", False)):
        model = torch.compile(model, mode="max-autotune")
    model, optimizer, _, scheduler = deepspeed.initialize(model=model,
                                                          model_parameters=[p for p in model.parameters() if p.requires_grad],
                                                          config=ds_config,
                                                          mpu=mpu if mpu.model_parallel_is_initialized() else None,
                                                          optimizer=optimizer)
    logger.info(optimizer.optimizer)

    unwrapped_model = model.module
    assert isinstance(unwrapped_model, PreTrainedModel)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", total_dataset_len)
    logger.info("  Num Epochs = %d", cfg.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", cfg.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", _actual_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", cfg.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup steps = %d", num_warmup_steps)

    save_tag = 'last_ds'
    save_tag_buffer = 'last_2ds'
    if continue_from_global_step > 0:
        logger.info("Fast forwarding to global step %d to resume training from latest checkpoint...", continue_from_global_step)
        # resume = os.path.dirname(cfg.resume)
        model.load_checkpoint(cfg.output_dir)

    elif os.path.exists(os.path.join(cfg.output_dir, save_tag)) or os.path.exists(os.path.join(cfg.output_dir, save_tag_buffer)):
        try:
            model.load_checkpoint(os.path.join(cfg.output_dir, save_tag))
        except RuntimeError as e:
            model.load_checkpoint(os.path.join(cfg.output_dir, save_tag_buffer))

    if cfg.local_rank in [-1, 0]:
        wandb.init(
            project=getattr(cfg, "wandb_project", "wandb_project"),
            name=cfg.exp_name,
            notes=cfg.exp_notes,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.define_metric(cfg.prediction_cfg.metric, summary=("max" if cfg.prediction_cfg.measure > 0 else "min"))

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(int(cfg.num_train_epochs), desc="Epoch", disable=cfg.local_rank not in [-1, 0])
    set_seed(cfg)  # Added here for reproducibility (even between python 2 and 3)

    for epoch in train_iterator:
        for _file in files:
            sub_train_dataset = load_and_cache_examples(cfg, tokenizer, _split="train", _file=_file)
            if cfg.local_rank == -1:
                if getattr(cfg, "shuffle_dataset", True):
                    sub_train_sampler = RandomSampler(sub_train_dataset)
                else:
                    sub_train_sampler = SequentialSampler(sub_train_dataset)
            else:
                sub_train_sampler = prepare_distributed_sampler(sub_train_dataset, cfg.seed, getattr(cfg, "shuffle_dataset", True))
            sub_train_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None
            sub_train_dataloader = DataLoader(dataset=sub_train_dataset,
                                              sampler=sub_train_sampler,
                                              batch_size=cfg.train_batch_size,
                                              collate_fn=sub_train_collator,
                                              num_workers=cfg.num_workers,
                                              pin_memory=True,
                                              drop_last=True,
                                              prefetch_factor=cfg.prefetch_factor,
                                              # worker_init_fn=worker_init_fn)
                                              )

            epoch_iterator = tqdm(sub_train_dataloader, desc="Iteration", disable=cfg.local_rank not in [-1, 0], dynamic_ncols=True)
            if cfg.local_rank != -1:
                sub_train_dataloader.sampler.set_epoch(epoch)

            if dist.is_initialized():
                dist.barrier()

            for step, batch in enumerate(epoch_iterator):
                # If training is continued from a checkpoint, fast forward
                # to the state of that checkpoint.
                if global_step < continue_from_global_step:
                    if (step + 1) % cfg.gradient_accumulation_steps == 0:
                        # scheduler.step()  # Update learning rate schedule  # Done by `load_checkpoint` of DS.
                        global_step += 1
                    continue
                batch = batch_to_device(batch, cfg.device)

                model.train()

                bsz = batch['input_ids'].size(1)
                # for loop here to avoid oom
                loss, outputs = forward_step(model, batch)
                loss /= cfg.gradient_accumulation_steps

                if tb_helper is not None:
                    tb_helper.update(last_batch=batch, last_outputs=outputs)

                tr_loss += loss
                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    global_step += 1

                    # Log metrics
                    log_metrics = {}
                    if cfg.local_rank in [-1, 0]:
                        log_metrics['lr'] = scheduler.get_lr()[0]
                        log_metrics['loss'] = tr_loss - logging_loss
                        logging_loss = tr_loss

                        if tb_helper is not None:
                            log_metrics.update(tb_helper(clear=True))

                    # Save model checkpoint
                    if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                        output_dir = os.path.join(cfg.output_dir, 'checkpoint-{}'.format(global_step))
                        if cfg.local_rank in [-1, 0] and not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        save_model(model, cfg, output_dir, tokenizer)

                    # Evaluation
                    if cfg.evaluate_during_training and cfg.eval_steps > 0 and global_step % cfg.eval_steps == 0:
                        # state_dict = get_state_dict(model, cfg)

                        if cfg.ddp_eval or cfg.local_rank in [-1, 0]:
                            results = evaluate(cfg, model, tokenizer, prefix=str(global_step), _split="dev")

                            if cfg.local_rank in [-1, 0]:
                                for key, value in results.items():
                                    log_metrics[f"eval/{key}"] = value

                            sub_path = os.path.join(cfg.output_dir, 'checkpoint-{}'.format(global_step))
                            flag = note_best_checkpoint(cfg, results, sub_path)
                            if cfg.save_best and flag:
                                save_model(model, cfg, cfg.output_dir, tokenizer)

                    if len(log_metrics) > 0 and cfg.local_rank in [-1, 0]:
                        wandb.log(log_metrics, step=global_step)
                        if global_step % cfg.logging_steps == 0:
                            logger.info(log_metrics)

                    del batch
                    del log_metrics

                if 0 < cfg.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < cfg.max_steps < global_step:
                train_iterator.close()
                break

        if 0 < cfg.max_steps < global_step:
            break
    
    # Delete the stored deepspeed engine to save memory
    if mpu.model_parallel_is_initialized():
        dp_rank = mpu.get_data_parallel_rank()
    else:
        if dist.is_initialized():
            dp_rank = dist.get_rank()
        else:
            dp_rank = -1
    if dp_rank in [-1, 0]:
        if os.path.exists(os.path.join(cfg.output_dir, save_tag)):
            shutil.rmtree(os.path.join(cfg.output_dir, save_tag))
        if os.path.exists(os.path.join(cfg.output_dir, save_tag_buffer)):
            shutil.rmtree(os.path.join(cfg.output_dir, save_tag_buffer))

    return global_step, tr_loss / global_step


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    if "LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] != -1:
        cfg.local_rank = int(os.environ["LOCAL_RANK"])
    # if "WORLD_SIZE" in os.environ and os.environ["WORLD_SIZE"]:
    #     cfg.world_size = int(os.environ["WORLD_SIZE"])
    # if "WORLD_RANK" in os.environ and os.environ["WORLD_RANK"]:
    #     cfg.world_rank = int(os.environ["WORLD_RANK"])

    if cfg.local_rank == -1 or cfg.no_cuda:
        device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
        cfg.n_gpu = torch.cuda.device_count()
        cfg.dp_size = 1
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device("cuda", cfg.local_rank))
        deepspeed.init_distributed(dist_backend="nccl", timeout=datetime.timedelta(seconds=9600))
        cfg.n_gpu = 1
        cfg.world_size = dist.get_world_size()
        cfg.dp_size = dist.get_world_size()
        if cfg.tp_size > 1:
            initialize_model_parallel(cfg.tp_size)
            cfg.dp_size = mpu.get_data_parallel_world_size()

    cfg.device = device

    global logger
    logger = setting_logger(cfg.output_dir, local_rank=cfg.local_rank)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   cfg.local_rank, cfg.device, cfg.n_gpu, bool(cfg.local_rank != -1), cfg.fp16)
    logger.warning(f"CPU cores: {os.cpu_count()}")
    logger.warning(f"Global rank: {dist.get_rank() if dist.is_initialized() else -1}")

    if mpu.model_parallel_is_initialized():
        dp_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
        mp_size = mpu.get_model_parallel_world_size()
        mp_rank = mpu.get_model_parallel_rank()
        pp_size = get_pipeline_parallel_world_size()
        pp_rank = get_pipeline_parallel_rank()
        logger.warning(f"Local Rank: {cfg.local_rank}, "
                       f"Global Rank: {dist.get_rank()}, "
                       f"Data Parallel: {dp_rank}/{dp_size}, "
                       f"Model Parallel: {mp_rank}/{mp_size}, "
                       f"Pipeline Parallel: {pp_rank}/{pp_size}")

    # Set seed
    set_seed(cfg)

    # Training
    if cfg.do_train:
        use_barrier = not os.path.exists(cfg.model_name_or_path)
        # Load pre-trained model and tokenizer
        if use_barrier and cfg.local_rank not in [-1, 0]:
            dist.barrier()  # Make sure only the first process in distributed training will download model & vocab

        if cfg.pretrain:  # TODO: How to load pretrain state dict and then split it to different GPUs.
            pretrain_state_dict = torch.load(cfg.pretrain, map_location='cpu')
        else:
            pretrain_state_dict = None

        if getattr(cfg, "tokenizer_init", None):
            tokenizer = hydra.utils.call(cfg.tokenizer_init)
        else:
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)

            from general_util.tokenization_utils import expand_special_tokenizer

            expand_special_tokenizer(tokenizer)

        try:
            model = hydra.utils.call(cfg.model, cfg.model_name_or_path, state_dict=pretrain_state_dict)
            if cfg.draft_model_name_or_path is not None:                model.draft_model.load_state_dict(torch.load(cfg.draft_model_name_or_path))
        except Exception as e:
            logger.warning(e)
            model = hydra.utils.call(cfg.model)
        # print('\n' * 4)
        # print(model.config)
        # print('\n' * 4)

        if use_barrier and cfg.local_rank == 0:
            dist.barrier()  # Make sure only the first process in distributed training will download model & vocab

        if dist.is_initialized():
            dist.barrier()

        # logger.info("Training/evaluation parameters %s", OmegaConf.to_yaml(cfg))
        if (cfg.local_rank == -1 or dist.get_rank() == 0) and cfg.do_train:
            if not os.path.exists(cfg.output_dir):
                os.makedirs(cfg.output_dir)
            OmegaConf.save(cfg, os.path.join(cfg.output_dir, "training_config.yaml"))

        continue_from_global_step = 0  # If set to 0, start training from the beginning
        if os.path.exists(cfg.output_dir) and getattr(cfg, "resume", None):
            if cfg.resume == "latest":
                checkpoint = get_last_checkpoint(cfg.output_dir)
            else:
                checkpoint = cfg.resume
            if checkpoint:
                logger.info("Resuming training from the latest checkpoint: %s", checkpoint)
                continue_from_global_step = int(checkpoint.split('-')[-1])

        # Catch keyboard interrupts
        try:
            global_step, tr_loss = train(cfg, model, tokenizer, continue_from_global_step)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt, normally exiting...")
            exit()
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["WANDB__SERVICE_WAIT"] = "1200"
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--"):])
        else:
            hydra_formatted_args.append(arg)
    sys.argv = hydra_formatted_args
    print(sys.argv)
    main()
