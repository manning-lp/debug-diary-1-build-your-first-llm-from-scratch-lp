"""Tests for the first Build Your First LLM from Scratch milestone."""

import unittest

import torch

from llama3 import FeedForward, GroupedQueryAttention, Llama3Model, precompute_rope_parameters


TINY_CONFIG = {
    "vocab_size": 64,
    "context_length": 16,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 2,
    "hidden_dim": 80,
    "n_kv_groups": 2,
    "rope_base": 500_000.0,
    "dtype": torch.float32,
    "rope_freq": None,
}


class FeedForwardTest(unittest.TestCase):
    def test_preserves_shape_and_supports_backpropagation(self) -> None:
        layer = FeedForward(TINY_CONFIG)
        inputs = torch.randn(2, 5, TINY_CONFIG["emb_dim"], requires_grad=True)

        output = layer(inputs)

        self.assertEqual(output.shape, inputs.shape)
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()
        self.assertIsNotNone(inputs.grad)


class ArchitectureTest(unittest.TestCase):
    def test_grouped_query_attention_is_causal(self) -> None:
        torch.manual_seed(7)
        attention = GroupedQueryAttention(TINY_CONFIG)
        cos, sin = precompute_rope_parameters(
            head_dim=TINY_CONFIG["emb_dim"] // TINY_CONFIG["n_heads"],
            context_length=TINY_CONFIG["context_length"],
            rope_base=TINY_CONFIG["rope_base"],
        )
        original = torch.randn(1, 5, TINY_CONFIG["emb_dim"])
        changed = original.clone()
        changed[:, -1] = torch.randn_like(changed[:, -1])

        original_output = attention(original, cos, sin)
        changed_output = attention(changed, cos, sin)

        torch.testing.assert_close(original_output[:, :-1], changed_output[:, :-1])

    def test_llama_model_returns_logits_for_each_token(self) -> None:
        model = Llama3Model(TINY_CONFIG)
        token_ids = torch.randint(0, TINY_CONFIG["vocab_size"], (2, 6))

        logits = model(token_ids)

        self.assertEqual(logits.shape, (2, 6, TINY_CONFIG["vocab_size"]))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()

