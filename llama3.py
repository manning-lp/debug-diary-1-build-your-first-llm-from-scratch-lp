"""A compact, from-scratch implementation of the Llama 3.2 architecture."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


LLAMA32_CONFIG: dict[str, Any] = {
    "vocab_size": 128_256,
    "context_length": 131_072,
    "emb_dim": 2048,
    "n_heads": 32,
    "n_layers": 16,
    "hidden_dim": 8192,
    "n_kv_groups": 8,
    "rope_base": 500_000.0,
    "dtype": torch.bfloat16,
    "rope_freq": {
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
}


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by Llama."""

    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        normalized = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(input_dtype)


class FeedForward(nn.Module):
    """Llama's gated SwiGLU feedforward network."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        emb_dim = int(cfg["emb_dim"])
        hidden_dim = int(cfg["hidden_dim"])
        dtype = cfg.get("dtype", torch.float32)

        self.fc1 = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.fc2 = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, bias=False, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


def _scaled_rope_frequencies(
    inv_freq: Tensor, rope_freq: Mapping[str, float] | None
) -> Tensor:
    """Apply Llama 3's smooth frequency scaling for long contexts."""
    if rope_freq is None:
        return inv_freq

    factor = float(rope_freq["factor"])
    low_factor = float(rope_freq["low_freq_factor"])
    high_factor = float(rope_freq["high_freq_factor"])
    old_context = float(rope_freq["original_context_length"])

    wavelength = 2 * math.pi / inv_freq
    low_wavelength = old_context / low_factor
    high_wavelength = old_context / high_factor

    scaled = torch.where(wavelength > low_wavelength, inv_freq / factor, inv_freq)
    smooth = (old_context / wavelength - low_factor) / (high_factor - low_factor)
    smoothed = (1 - smooth) * (inv_freq / factor) + smooth * inv_freq
    medium = (wavelength <= low_wavelength) & (wavelength >= high_wavelength)
    return torch.where(medium, smoothed, scaled)


def precompute_rope_parameters(
    head_dim: int,
    context_length: int,
    rope_base: float = 10_000.0,
    rope_freq: Mapping[str, float] | None = None,
) -> tuple[Tensor, Tensor]:
    """Precompute cosine and sine matrices for rotary position embeddings."""
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even attention head dimension")

    dimensions = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (rope_base ** (dimensions / head_dim))
    inv_freq = _scaled_rope_frequencies(inv_freq, rope_freq)

    positions = torch.arange(context_length, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    angles = torch.cat((angles, angles), dim=-1)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate query or key vectors according to their token positions."""
    sequence_length = x.shape[-2]
    cos = cos[:sequence_length].unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
    sin = sin[:sequence_length].unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
    first_half, second_half = x.chunk(2, dim=-1)
    rotated = torch.cat((-second_half, first_half), dim=-1)
    return x * cos + rotated * sin


class GroupedQueryAttention(nn.Module):
    """Causal self-attention with fewer key/value heads than query heads."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        emb_dim = int(cfg["emb_dim"])
        self.n_heads = int(cfg["n_heads"])
        self.n_kv_groups = int(cfg["n_kv_groups"])
        context_length = int(cfg["context_length"])
        dtype = cfg.get("dtype", torch.float32)

        if emb_dim % self.n_heads != 0:
            raise ValueError("emb_dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_groups != 0:
            raise ValueError("n_heads must be divisible by n_kv_groups")

        self.head_dim = emb_dim // self.n_heads
        self.group_size = self.n_heads // self.n_kv_groups
        kv_dim = self.n_kv_groups * self.head_dim

        self.query = nn.Linear(emb_dim, emb_dim, bias=False, dtype=dtype)
        self.key = nn.Linear(emb_dim, kv_dim, bias=False, dtype=dtype)
        self.value = nn.Linear(emb_dim, kv_dim, bias=False, dtype=dtype)
        self.output = nn.Linear(emb_dim, emb_dim, bias=False, dtype=dtype)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch_size, sequence_length, emb_dim = x.shape

        queries = self.query(x).view(
            batch_size, sequence_length, self.n_heads, self.head_dim
        )
        keys = self.key(x).view(
            batch_size, sequence_length, self.n_kv_groups, self.head_dim
        )
        values = self.value(x).view(
            batch_size, sequence_length, self.n_kv_groups, self.head_dim
        )

        queries = apply_rope(queries.transpose(1, 2), cos, sin)
        keys = apply_rope(keys.transpose(1, 2), cos, sin)
        values = values.transpose(1, 2)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        scores = queries @ keys.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        mask = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores.float(), dim=-1).to(queries.dtype)

        context = weights @ values
        context = context.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, emb_dim
        )
        return self.output(context)


class TransformerBlock(nn.Module):
    """A pre-normalized Llama transformer block."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        emb_dim = int(cfg["emb_dim"])
        self.attention_norm = RMSNorm(emb_dim)
        self.attention = GroupedQueryAttention(cfg)
        self.feedforward_norm = RMSNorm(emb_dim)
        self.feedforward = FeedForward(cfg)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x), cos, sin)
        return x + self.feedforward(self.feedforward_norm(x))


class Llama3Model(nn.Module):
    """Decoder-only Llama 3.2 language model returning token logits."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.context_length = int(cfg["context_length"])
        emb_dim = int(cfg["emb_dim"])
        dtype = cfg.get("dtype", torch.float32)

        self.token_embedding = nn.Embedding(int(cfg["vocab_size"]), emb_dim, dtype=dtype)
        self.blocks = nn.ModuleList(
            TransformerBlock(cfg) for _ in range(int(cfg["n_layers"]))
        )
        self.final_norm = RMSNorm(emb_dim)
        self.output = nn.Linear(emb_dim, int(cfg["vocab_size"]), bias=False, dtype=dtype)

        head_dim = emb_dim // int(cfg["n_heads"])
        cos, sin = precompute_rope_parameters(
            head_dim=head_dim,
            context_length=self.context_length,
            rope_base=float(cfg["rope_base"]),
            rope_freq=cfg.get("rope_freq"),
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if token_ids.shape[1] > self.context_length:
            raise ValueError("input sequence exceeds configured context length")

        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        return self.output(self.final_norm(x))

