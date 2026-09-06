# Build Your First LLM from Scratch

This repository contains a from-scratch PyTorch implementation of the Llama
3.2 architecture for the Manning liveProject.

## Milestone 1: Coding the architecture

`llama3.py` implements:

- gated SwiGLU feedforward layers;
- Llama 3-scaled rotary position embeddings (RoPE);
- causal grouped-query attention;
- pre-normalized transformer blocks with RMSNorm;
- the complete decoder-only `Llama3Model`.

The production `LLAMA32_CONFIG` matches the course's Llama 3.2 1B settings.
Tests use a deliberately small configuration so they run quickly without a
GPU or downloading model weights.

## Run the tests

Create a Python 3.11 virtual environment, install PyTorch, and run:

```bash
python -m unittest -v
```

No Hugging Face token or pretrained weights are required for this milestone.

## Milestone 2: Initialize with GPU support

`milestone_2.py` assembles the complete model, reduces its context length,
selects CUDA, Apple Metal, or CPU in that order, and validates the model with a
dummy forward pass. It uses BF16 where supported, FP16 on a T4 GPU or Apple
Metal, and FP32 for the CPU fallback.

The full-size course model should be run in Google Colab with GPU acceleration:

[Open Milestone 2 in Google Colab](https://colab.research.google.com/github/manning-lp/debug-diary-1-build-your-first-llm-from-scratch-lp/blob/main/milestone_2_colab.ipynb)

Choose **Runtime → Change runtime type → T4 GPU**, then run all cells. The local
unit test uses the same complete architecture with small dimensions so it can
verify the full execution path without requiring several gigabytes of memory.
