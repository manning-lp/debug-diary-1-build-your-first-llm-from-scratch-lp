"""Tests for hardware selection and full-model assembly in milestone 2."""

import unittest

import torch

from milestone_2 import best_dtype, prepare_model_config, run_dummy_forward


TINY_CONFIG = {
    "vocab_size": 96,
    "context_length": 32,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 2,
    "hidden_dim": 80,
    "n_kv_groups": 2,
    "rope_base": 500_000.0,
    "dtype": torch.bfloat16,
    "rope_freq": None,
}


class ModelSetupTest(unittest.TestCase):
    def test_cpu_config_reduces_context_and_uses_float32(self) -> None:
        config = prepare_model_config(
            TINY_CONFIG,
            context_length=12,
            device=torch.device("cpu"),
        )

        self.assertEqual(config["context_length"], 12)
        self.assertEqual(config["dtype"], torch.float32)
        self.assertEqual(TINY_CONFIG["context_length"], 32)

    def test_cpu_is_assigned_a_supported_dtype(self) -> None:
        self.assertEqual(best_dtype(torch.device("cpu")), torch.float32)

    def test_assembled_model_runs_dummy_input(self) -> None:
        run = run_dummy_forward(
            TINY_CONFIG,
            context_length=12,
            sequence_length=5,
            batch_size=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(run.device.type, "cpu")
        self.assertEqual(run.inputs.shape, (2, 5))
        self.assertEqual(run.logits.shape, (2, 5, TINY_CONFIG["vocab_size"]))
        self.assertTrue(torch.isfinite(run.logits).all())


if __name__ == "__main__":
    unittest.main()

