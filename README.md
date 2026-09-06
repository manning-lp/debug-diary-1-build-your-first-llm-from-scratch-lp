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
