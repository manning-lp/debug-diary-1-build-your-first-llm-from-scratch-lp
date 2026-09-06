"""Credential-free tests for generation, toy training, and weight mapping."""

import math
import unittest

import torch
from torch import nn

from llama3 import Llama3Model
from milestone_4 import generate, load_pretrained_weights, train_on_toy_data


TINY_CONFIG = {
    "vocab_size": 12,
    "context_length": 16,
    "emb_dim": 8,
    "n_heads": 2,
    "n_layers": 1,
    "hidden_dim": 16,
    "n_kv_groups": 1,
    "rope_base": 10_000.0,
    "dtype": torch.float32,
    "rope_freq": None,
}


class IncrementModel(nn.Module):
    def __init__(self, vocab_size: int, eos_id: int | None = None) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_id = eos_id

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        next_ids = (token_ids + 1) % self.vocab_size
        if self.eos_id is not None and token_ids.shape[1] >= 2:
            next_ids[:, -1] = self.eos_id
        return torch.nn.functional.one_hot(next_ids, self.vocab_size).float() * 10


class GenerationTest(unittest.TestCase):
    def test_greedy_generation_appends_multiple_tokens(self) -> None:
        model = IncrementModel(vocab_size=8)

        result = generate(
            model,
            torch.tensor([[1, 2]]),
            max_new_tokens=3,
            context_size=4,
        )

        self.assertEqual(result.tolist(), [[1, 2, 3, 4, 5]])

    def test_generation_stops_after_eos(self) -> None:
        model = IncrementModel(vocab_size=8, eos_id=7)

        result = generate(
            model,
            torch.tensor([[1, 2]]),
            max_new_tokens=5,
            context_size=4,
            eos_id=7,
        )

        self.assertEqual(result.tolist(), [[1, 2, 7]])


class TinyLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(6, 8)
        self.output = nn.Linear(8, 6)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.output(self.embedding(token_ids))


class TrainingTest(unittest.TestCase):
    def test_toy_training_logs_finite_decreasing_loss(self) -> None:
        torch.manual_seed(7)
        model = TinyLanguageModel()
        data = [torch.tensor([[0, 1, 2, 3, 4, 5]])]

        losses = train_on_toy_data(
            model,
            data,
            epochs=20,
            learning_rate=0.05,
        )

        self.assertEqual(len(losses), 20)
        self.assertTrue(all(math.isfinite(loss) for loss in losses))
        self.assertLess(losses[-1], losses[0])


class WeightLoadingTest(unittest.TestCase):
    def test_maps_hugging_face_names_and_ties_output_head(self) -> None:
        model = Llama3Model(TINY_CONFIG)
        block = model.blocks[0]
        weights = {
            "model.embed_tokens.weight": torch.full_like(model.token_embedding.weight, 1),
            "model.layers.0.self_attn.q_proj.weight": torch.full_like(block.attention.query.weight, 2),
            "model.layers.0.self_attn.k_proj.weight": torch.full_like(block.attention.key.weight, 3),
            "model.layers.0.self_attn.v_proj.weight": torch.full_like(block.attention.value.weight, 4),
            "model.layers.0.self_attn.o_proj.weight": torch.full_like(block.attention.output.weight, 5),
            "model.layers.0.mlp.gate_proj.weight": torch.full_like(block.feedforward.fc1.weight, 6),
            "model.layers.0.mlp.up_proj.weight": torch.full_like(block.feedforward.fc2.weight, 7),
            "model.layers.0.mlp.down_proj.weight": torch.full_like(block.feedforward.fc3.weight, 8),
            "model.layers.0.input_layernorm.weight": torch.full_like(block.attention_norm.weight, 9),
            "model.layers.0.post_attention_layernorm.weight": torch.full_like(block.feedforward_norm.weight, 10),
            "model.norm.weight": torch.full_like(model.final_norm.weight, 11),
        }

        tied = load_pretrained_weights(model, TINY_CONFIG, weights)

        self.assertTrue(tied)
        self.assertTrue(torch.equal(block.attention.query.weight, weights["model.layers.0.self_attn.q_proj.weight"]))
        self.assertIs(model.output.weight, model.token_embedding.weight)


if __name__ == "__main__":
    unittest.main()
