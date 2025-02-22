# LongSpec: Long-Context Speculative Decoding with Efficient Drafting and Verification

<p align="center">
  <a href="https://phyang.top/"><strong>Penghui Yang<sup>*</sup></strong></a>
  ·
  <a href="#"><strong>Cunxiao Du<sup>*</sup></strong></a>
  ·
  <a href="#"><strong>Fengzhuo Zhang</strong></a>
  ·
  <a href="#"><strong>Haonan Wang</strong></a>
  ·
  <a href="#"><strong>Tianyu Pang</strong></a>
  ·
  <a href="#"><strong>Chao Du</strong></a>
  ·
  <a href="#"><strong>Bo An</strong></a>
</p>

<div align=center><img src='./static/images/1.png' width=600></div>

## Introduction

Speculative decoding has emerged as a promising technique to mitigate the high inference latency inherent in autoregressive decoding for large language models (LLMs). However, its effective application in long-context scenarios faces three key challenges:
- **Memory Overhead:** The draft model requires a linearly growing Key-Value cache as the sequence length increases.
- **Distribution Shift:** Training on short-context data leads to a mismatch when performing long-context inference.
- **Inefficient Attention:** Existing implementations struggle with latency due to suboptimal attention mechanisms.

In LongSpec, we address these challenges by:
- **Memory-Efficient Drafting:** Introducing a draft model that maintains a constant-sized Key-Value cache regardless of context length.
- **Seamless Adaptation:** Proposing novel position indices for short-training data to bridge the gap with long-context inference.
- **Hybrid Attention Aggregation:** Developing an innovative attention aggregation method that combines fast prefix computation with standard attention for effective tree mask handling.

Our approach demonstrates significant improvements in latency reduction across a range of long-context tasks, including repository-level code completion, long-context summarization, and long CoT reasoning tasks.


## Installation

```bash
git clone https://github.com/sail-sg/LongSpec.git
cd longspec
pip install -r requirements.txt
```

## LongSpec Weights

We provide the weights for LongSpec models. Please download them from the following links:

| Base Model | LongSpec on Hugging Face |
|------------|--------------------------|
| [lmsys/Vicuna-7B-v1.5-16k](https://huggingface.co/lmsys/vicuna-7b-v1.5-16k) | [sail/longspec-vicuna-7b-v1.5-16k](https://huggingface.co/sail/longspec-vicuna-7b-v1.5-16k) |
| [lmsys/Vicuna-13B-v1.5-16k](https://huggingface.co/lmsys/vicuna-13b-v1.5-16k) | [sail/longspec-vicuna-13b-v1.5-16k](https://huggingface.co/sail/longspec-vicuna-13b-v1.5-16k) |
| [lmsys/LongChat-7B-v1.5-32k](https://huggingface.co/lmsys/longchat-7b-v1.5-32k) | [sail/longspec-longchat-7b-v1.5-32k](https://huggingface.co/sail/longspec-longchat-7b-v1.5-32k) |
| [lmsys/LongChat-13B-16k](https://huggingface.co/lmsys/longchat-13b-16k) | [sail/longspec-longchat-13b-16k](https://huggingface.co/sail/longspec-longchat-13b-16k) |
| [gradientai/Llama-3-8B-Instruct-262k](https://huggingface.co/gradientai/Llama-3-8B-Instruct-262k) | [sail/longspec-Llama-3-8B-Instruct-262k](https://huggingface.co/sail/longspec-Llama-3-8B-Instruct-262k) |
| [Qwen/QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview) | [sail/longspec-QwQ-32B-Preview](https://huggingface.co/sail/longspec-QwQ-32B-Preview) |

## Evaluation

We provide the whole inference speed test code in the folder `./longspec`. For example, you can use the following command line the test the performance of `longspec-Llama-3-8B-Instruct-262k` on the `GovReport` dataset:

```bash
python inference_long-bench.py\
    --model_name llama8b\
    --method tree\
    --task gov_report\
    --data_path_prefix [your data folder of longbench]\
    --test_length 0\
    --temperature 0\
    --tree_shape 4 16 16 16 16
```

You need to preprocess the longbench data into `.jsonl` before using the command above. `test_length` is set to 0 when you want to test on the whole dataset.

If you want to test QwQ on AIME24, use the command below:

```bash
python inference_qwq.py\
    --model_name qwq\
    --method tree\
    --id_min 60\
    --id_max 89\
    --max_gen_len 32000\
    --temperature 0\
    --tree_shape 4 16 16 16 16
```

The questions with id from 60 to 89 are AIME24 questions in the dataset `AI-MO/aimo-validation-aime`.
