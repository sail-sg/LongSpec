import os
from logging import Logger
from typing import Optional, Union, Callable

import hydra
import omegaconf
import torch
from fairscale.nn.model_parallel import initialize as mpu
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    TaskType,
    get_peft_model,
)
from peft.tuners.lora import LoraLayer
from transformers import PreTrainedModel

from general_util.logger import get_child_logger
from general_util.training_utils import get_rank

logger: Logger = get_child_logger(__name__)

REFERENCE_MODEL: PreTrainedModel


def return_single_device_map():
    return {"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))}


def return_reference_model():
    return REFERENCE_MODEL


class PreTrainedModelPeftMixin(PreTrainedModel):
    _ref_model: PreTrainedModel = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        gradient_checkpointing = kwargs.pop("gradient_checkpointing", False)
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if gradient_checkpointing:
            model.config.use_cache = False
            model.gradient_checkpointing_enable()

        return model

    @classmethod
    def from_pretrained_with_ref_model(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], ref_model: PreTrainedModel,
                                       register_ref_model: bool = False,
                                       *model_args, **kwargs):
        global REFERENCE_MODEL
        REFERENCE_MODEL = ref_model
        REFERENCE_MODEL.eval()

        model = cls.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        if register_ref_model:
            model._ref_model = ref_model

        return model

    @classmethod
    def from_pretrained_with_ref_model_lora(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        lora_config = kwargs.pop("lora_config", None)
        assert lora_config is not None, "lora_config must be provided to enable lora training."

        base_model = cls.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        global REFERENCE_MODEL
        REFERENCE_MODEL = base_model

        enable_quantization = "quantization_config" in kwargs

        if lora_config is None:
            lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32,
                                     lora_dropout=0.1)

        logger.warning(lora_config)
        logger.info(lora_config.target_modules.__class__)
        if isinstance(lora_config.target_modules, omegaconf.listconfig.ListConfig):
            lora_config.target_modules = list(lora_config.target_modules)
        elif isinstance(lora_config.target_modules, omegaconf.DictConfig):
            lora_config.target_modules = hydra.utils.instantiate(lora_config.target_modules, model=base_model)
        else:
            raise ValueError(f"Unsupported type of target modules: {lora_config.target_modules.__class__}")

        if isinstance(lora_config.modules_to_save, omegaconf.listconfig.ListConfig):
            lora_config.modules_to_save = list(lora_config.modules_to_save)

        logger.info(lora_config.target_modules.__class__)
        logger.warning(lora_config.target_modules)

        gradient_checkpointing = base_model.model.gradient_checkpointing
        if enable_quantization:
            logger.warning(f"Rank {get_rank()} is being loaded with quantization.")
            logger.info(f"Quantization config: {kwargs['quantization_config']}")
            base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=gradient_checkpointing)

        model = get_peft_model(base_model, lora_config)

        logger.info(f"Reference model type: {REFERENCE_MODEL.__class__.__name__}")
        logger.info(f"Actor model type: {model.__class__.__name__}")

        compute_dtype = kwargs["torch_dtype"]
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if compute_dtype == torch.bfloat16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if compute_dtype and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

        model.print_trainable_parameters()

        logger.info(f"Config pad token id after loading pre-trained weights: {model.config.pad_token_id}")
        logger.info(model.lm_head.__class__.__name__)

        return model


class PretrainedModelParallelPreSplitMixin(PreTrainedModelPeftMixin):
    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            *model_args,
            **kwargs,
    ):
        if mpu.model_parallel_is_initialized():
            mp_rank = mpu.get_model_parallel_rank()
            pretrained_model_name_or_path = os.path.join(pretrained_model_name_or_path, f"mp_{mp_rank}-of-{mpu.get_model_parallel_world_size()}")
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            *args,
            **kwargs,
    ):
        if mpu.model_parallel_is_initialized():
            mp_rank = mpu.get_model_parallel_rank()
            save_directory = os.path.join(save_directory, f"mp_{mp_rank}-of-{mpu.get_model_parallel_world_size()}")
        super().save_pretrained(save_directory, *args, **kwargs)

    @classmethod
    def from_pretrained_with_ref_model(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], ref_model: PreTrainedModel,
                                       *model_args, **kwargs):
        global REFERENCE_MODEL
        REFERENCE_MODEL = ref_model
        REFERENCE_MODEL.eval()
        REFERENCE_MODEL.to(device=torch.cuda.current_device())

        model = cls.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model
