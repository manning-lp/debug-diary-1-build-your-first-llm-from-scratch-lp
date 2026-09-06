"""Llama 3.2 tokenizer with safe Hugging Face model retrieval."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import AbstractSet, Collection, Literal, Sequence

import tiktoken


MODEL_REPOSITORY = "meta-llama/Llama-3.2-1B"
MODEL_FILENAME = "original/tokenizer.model"
NUM_RESERVED_SPECIAL_TOKENS = 256

TOKEN_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def download_tokenizer_model(
    *,
    token: str | None = None,
    local_dir: str | Path = ".models/llama32",
    repository: str = MODEL_REPOSITORY,
    filename: str = MODEL_FILENAME,
) -> Path:
    """Download the gated tokenizer without ever storing the access token."""
    access_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not access_token:
        raise RuntimeError(
            "A Hugging Face token is required. Set HF_TOKEN in the environment "
            "or add it to Colab Secrets."
        )

    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id=repository,
        filename=filename,
        local_dir=str(local_dir),
        token=access_token,
    )
    return Path(model_path)


def _special_token_names() -> list[str]:
    named_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|finetune_right_pad_id|>",
        "<|step_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eom_id|>",
        "<|eot_id|>",
        "<|python_tag|>",
    ]
    remaining = NUM_RESERVED_SPECIAL_TOKENS - len(named_tokens)
    named_tokens.extend(
        f"<|reserved_special_token_{index}|>" for index in range(2, 2 + remaining)
    )
    return named_tokens


def _load_mergeable_ranks(model_path: Path) -> dict[bytes, int]:
    """Load tiktoken's base64-token/rank format without optional blob storage."""
    ranks: dict[bytes, int] = {}
    for line_number, line in enumerate(model_path.read_bytes().splitlines(), start=1):
        try:
            encoded_token, rank = line.split()
            ranks[base64.b64decode(encoded_token)] = int(rank)
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"Invalid tokenizer model entry on line {line_number}"
            ) from error
    if not ranks:
        raise ValueError("Tokenizer model contains no mergeable ranks")
    return ranks


class Tokenizer:
    """Encode and decode text with Llama 3.2's tiktoken vocabulary."""

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Tokenizer model not found: {path}")

        mergeable_ranks = _load_mergeable_ranks(path)
        base_token_count = len(mergeable_ranks)
        self.special_tokens = {
            name: base_token_count + index
            for index, name in enumerate(_special_token_names())
        }
        self.model = tiktoken.Encoding(
            name=path.name,
            pat_str=TOKEN_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.bos_id = self.special_tokens["<|begin_of_text|>"]
        self.eos_id = self.special_tokens["<|end_of_text|>"]
        self.pad_id = self.special_tokens["<|finetune_right_pad_id|>"]
        self.stop_tokens = {
            self.eos_id,
            self.special_tokens["<|eot_id|>"],
        }

    def encode(
        self,
        text: str,
        *,
        bos: bool = False,
        eos: bool = False,
        allowed_special: Literal["all"] | AbstractSet[str] = frozenset(),
        disallowed_special: Literal["all"] | Collection[str] = (),
    ) -> list[int]:
        """Convert text to IDs, optionally adding beginning/end markers."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        normalized_allowed = (
            allowed_special if allowed_special == "all" else set(allowed_special)
        )
        tokens = self.model.encode(
            text,
            allowed_special=normalized_allowed,
            disallowed_special=disallowed_special,
        )
        if bos:
            tokens.insert(0, self.bos_id)
        if eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, tokens: Sequence[int]) -> str:
        """Convert token IDs back to text, including named special tokens."""
        return self.model.decode(list(tokens))


def main() -> None:
    model_path = download_tokenizer_model()
    tokenizer = Tokenizer(model_path)
    sample = "Building a Llama tokenizer from scratch."
    encoded = tokenizer.encode(sample, bos=True, eos=True)
    decoded = tokenizer.decode(encoded)
    empty_tokens = tokenizer.encode("")

    print(f"Tokenizer model: {model_path}")
    print(f"Original: {sample}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Empty input: {empty_tokens}")


if __name__ == "__main__":
    main()
