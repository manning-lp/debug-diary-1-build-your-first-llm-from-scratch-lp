"""Milestone 4: autoregressive generation, toy training, and weight loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from llama3 import Llama3Model


PRETRAINED_REPOSITORY = "meta-llama/Llama-3.2-1B"
PRETRAINED_FILENAME = "model.safetensors"


class TextTokenizer(Protocol):
    """The tokenizer operations needed to prepare causal-language-model data."""

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]: ...


def generate(
    model: nn.Module,
    input_ids: Tensor,
    *,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Generate tokens autoregressively with greedy or top-k sampling."""
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have shape (batch, sequence) with a non-empty sequence")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must not be negative")
    if context_size <= 0:
        raise ValueError("context_size must be positive")
    if temperature < 0:
        raise ValueError("temperature must not be negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")

    generated = input_ids
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits = model(generated[:, -context_size:])[:, -1, :]

                if top_k is not None:
                    effective_k = min(top_k, logits.shape[-1])
                    threshold = torch.topk(logits, effective_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))

                if temperature > 0:
                    probabilities = torch.softmax(logits / temperature, dim=-1)
                    next_ids = torch.multinomial(
                        probabilities,
                        num_samples=1,
                        generator=generator,
                    )
                else:
                    next_ids = torch.argmax(logits, dim=-1, keepdim=True)

                generated = torch.cat((generated, next_ids), dim=1)
                if eos_id is not None and torch.all(next_ids.squeeze(-1) == eos_id):
                    break
    finally:
        model.train(was_training)

    return generated


def prepare_toy_sequences(
    tokenizer: TextTokenizer,
    texts: Iterable[str],
    *,
    device: torch.device | str = "cpu",
) -> list[Tensor]:
    """Tokenize strings into batched sequences for next-token training."""
    sequences: list[Tensor] = []
    for text in texts:
        token_ids = tokenizer.encode(text, bos=True, eos=True)
        if len(token_ids) < 2:
            raise ValueError("each training sequence must contain at least two tokens")
        sequences.append(torch.tensor([token_ids], dtype=torch.long, device=device))
    if not sequences:
        raise ValueError("at least one training sequence is required")
    return sequences


def train_on_toy_data(
    model: nn.Module,
    sequences: Sequence[Tensor],
    *,
    epochs: int = 5,
    learning_rate: float = 1e-4,
) -> list[float]:
    """Train a causal language model and return the average loss per epoch."""
    if not sequences:
        raise ValueError("at least one training sequence is required")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for input_ids in sequences:
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            prediction_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            targets = input_ids[:, 1:].reshape(-1)
            loss = F.cross_entropy(prediction_logits.float(), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(sequences)
        losses.append(average_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f}")

    return losses


def download_pretrained_weights(
    *,
    token: str | None = None,
    local_dir: str | Path = ".models/llama32",
    repository: str = PRETRAINED_REPOSITORY,
    filename: str = PRETRAINED_FILENAME,
) -> dict[str, Tensor]:
    """Download the gated safetensors checkpoint without storing credentials."""
    access_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not access_token:
        raise RuntimeError(
            "A Hugging Face token is required. Set HF_TOKEN in the environment "
            "or add it to Colab Secrets."
        )

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    weights_path = hf_hub_download(
        repo_id=repository,
        filename=filename,
        local_dir=str(local_dir),
        token=access_token,
    )
    return load_file(weights_path)


def _assign_parameter(target: nn.Parameter, source: Tensor, name: str) -> nn.Parameter:
    if target.shape != source.shape:
        raise ValueError(
            f"Shape mismatch for {name}: expected {tuple(target.shape)}, "
            f"received {tuple(source.shape)}"
        )
    value = source.detach().to(device=target.device, dtype=target.dtype).clone()
    return nn.Parameter(value, requires_grad=target.requires_grad)


def load_pretrained_weights(
    model: Llama3Model,
    config: Mapping[str, Any],
    weights: Mapping[str, Tensor],
) -> bool:
    """Map Hugging Face Llama weights into this project's module structure.

    Returns ``True`` when the checkpoint omits ``lm_head.weight`` and the
    output projection is tied to the token embedding, as in the base model.
    """
    model.token_embedding.weight = _assign_parameter(
        model.token_embedding.weight,
        weights["model.embed_tokens.weight"],
        "model.embed_tokens.weight",
    )

    for layer_index in range(int(config["n_layers"])):
        block = model.blocks[layer_index]
        prefix = f"model.layers.{layer_index}"
        assignments = (
            (block.attention.query, f"{prefix}.self_attn.q_proj.weight"),
            (block.attention.key, f"{prefix}.self_attn.k_proj.weight"),
            (block.attention.value, f"{prefix}.self_attn.v_proj.weight"),
            (block.attention.output, f"{prefix}.self_attn.o_proj.weight"),
            (block.feedforward.fc1, f"{prefix}.mlp.gate_proj.weight"),
            (block.feedforward.fc2, f"{prefix}.mlp.up_proj.weight"),
            (block.feedforward.fc3, f"{prefix}.mlp.down_proj.weight"),
        )
        for module, weight_name in assignments:
            module.weight = _assign_parameter(module.weight, weights[weight_name], weight_name)

        block.attention_norm.weight = _assign_parameter(
            block.attention_norm.weight,
            weights[f"{prefix}.input_layernorm.weight"],
            f"{prefix}.input_layernorm.weight",
        )
        block.feedforward_norm.weight = _assign_parameter(
            block.feedforward_norm.weight,
            weights[f"{prefix}.post_attention_layernorm.weight"],
            f"{prefix}.post_attention_layernorm.weight",
        )

    model.final_norm.weight = _assign_parameter(
        model.final_norm.weight,
        weights["model.norm.weight"],
        "model.norm.weight",
    )

    tied = "lm_head.weight" not in weights
    if tied:
        model.output.weight = model.token_embedding.weight
    else:
        model.output.weight = _assign_parameter(
            model.output.weight,
            weights["lm_head.weight"],
            "lm_head.weight",
        )
    return tied
