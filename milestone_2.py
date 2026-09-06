"""Milestone 2: assemble Llama 3.2 and run it on the best available device."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from llama3 import LLAMA32_CONFIG, Llama3Model


@dataclass(frozen=True)
class ModelRun:
    """Artifacts produced by a successful dummy forward pass."""

    model: Llama3Model
    device: torch.device
    inputs: Tensor
    logits: Tensor


def best_available_device() -> torch.device:
    """Prefer CUDA, then Apple Metal, and retain a CPU fallback for tests."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def best_dtype(device: torch.device) -> torch.dtype:
    """Choose a memory-efficient dtype supported by the selected hardware."""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def prepare_model_config(
    base_config: Mapping[str, Any] = LLAMA32_CONFIG,
    *,
    context_length: int = 4096,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Copy the course config and reduce it to a hardware-friendly context."""
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if context_length > int(base_config["context_length"]):
        raise ValueError("context_length must not exceed the base configuration")

    selected_device = device or best_available_device()
    config = deepcopy(dict(base_config))
    config["context_length"] = context_length
    config["dtype"] = best_dtype(selected_device)
    return config


def run_dummy_forward(
    base_config: Mapping[str, Any] = LLAMA32_CONFIG,
    *,
    context_length: int = 4096,
    sequence_length: int = 8,
    batch_size: int = 1,
    device: torch.device | None = None,
) -> ModelRun:
    """Build the complete model and validate it with token-shaped dummy input."""
    selected_device = device or best_available_device()
    config = prepare_model_config(
        base_config,
        context_length=context_length,
        device=selected_device,
    )
    if sequence_length <= 0 or sequence_length > context_length:
        raise ValueError("sequence_length must be between 1 and context_length")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model = Llama3Model(config).to(selected_device)
    model.eval()
    inputs = torch.randint(
        0,
        int(config["vocab_size"]),
        (batch_size, sequence_length),
        device=selected_device,
    )
    with torch.inference_mode():
        logits = model(inputs)
    return ModelRun(model=model, device=selected_device, inputs=inputs, logits=logits)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    run = run_dummy_forward()
    print(f"Device: {run.device}")
    print(f"Dtype: {next(run.model.parameters()).dtype}")
    print(f"Parameters: {parameter_count(run.model):,}")
    print(f"Input shape: {tuple(run.inputs.shape)}")
    print(f"Logits shape: {tuple(run.logits.shape)}")


if __name__ == "__main__":
    main()

