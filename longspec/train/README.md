#  LongSpec: Long-Context Lossless Speculative Decoding with Efficient Drafting and Verification

This repository contains the code for training our model submitted to NeurIPS 2025. Please follow the steps below to set up the environment and run the training script. Our Triton kernel can be found in `./models/triton_tree_attn.py`.

## 1. Environment Setup

Make sure you have the following installed:

* python >= 3.12
* pytorch >= 2.6.0
* deepspeed
* wandb
* flash_attn
* Any additional dependencies listed in `requirements.txt`


## 2. Prepare Training Data

Prepare the training dataset in JSON format. Each entry should follow the expected input structure required by the model.

Example format:

```json
[
  {
    "source": "...",
    "target": "..."
  },
  ...
]
```

Save this file to a location accessible during training (e.g., `./data/train.json`).

## 3. Configure YAML

Edit the corresponding YAML configuration file in `./conf/exp/` to point to your training data. For example, in `conf/exp/qwq_glide_8gpu_slim6b.yaml`, modify the `data_path` or equivalent field:

```yaml
data:
  train_path: ./data/train.json
  ...
```

Make sure all paths in the config file are correctly set relative to the project root.

## 4. Login to Weights & Biases

Authenticate with Weights & Biases before launching training. Run the following command in your terminal:

```bash
wandb login ******
```

## 5. Launch Training

Use DeepSpeed to launch training with 8 GPUs on a single machine:

```bash
deepspeed --include localhost:0,1,2,3,4,5,6,7 ./trainer_base_ds_mul_fs_tp.py -cp conf/exp/ -cn qwq_glide_8gpu_slim6b
```

Replace the config name and paths as needed based on your setup.

## Notes

* The script `trainer_base_ds_mul_fs_tp.py` supports multi-GPU and mixed precision training via DeepSpeed.
* Make sure the GPUs listed in `--include` are available and not in use by other processes.
* Training logs and checkpoints will be saved as specified in the YAML config.
