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
