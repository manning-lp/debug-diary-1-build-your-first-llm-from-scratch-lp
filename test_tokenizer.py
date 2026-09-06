"""Credential-free validation for the Llama 3.2 Tokenizer class."""

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tokenizer import Tokenizer, download_tokenizer_model


class TokenizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        model_path = Path(self.temporary_directory.name) / "tokenizer.model"
        lines = [
            f"{base64.b64encode(bytes([byte])).decode('ascii')} {byte}"
            for byte in range(256)
        ]
        model_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.tokenizer = Tokenizer(model_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_encode_decode_round_trip(self) -> None:
        sample = "Hello, Llama 3.2! 🦙"

        encoded = self.tokenizer.encode(sample)

        self.assertEqual(self.tokenizer.decode(encoded), sample)
        self.assertTrue(all(isinstance(token, int) for token in encoded))

    def test_bos_and_eos_special_tokens(self) -> None:
        encoded = self.tokenizer.encode("test", bos=True, eos=True)

        self.assertEqual(encoded[0], self.tokenizer.bos_id)
        self.assertEqual(encoded[-1], self.tokenizer.eos_id)
        self.assertEqual(
            self.tokenizer.decode(encoded),
            "<|begin_of_text|>test<|end_of_text|>",
        )

    def test_empty_input_is_robust(self) -> None:
        self.assertEqual(self.tokenizer.encode(""), [])
        self.assertEqual(self.tokenizer.decode([]), "")

    def test_all_256_special_tokens_have_unique_ids(self) -> None:
        special_ids = list(self.tokenizer.special_tokens.values())

        self.assertEqual(len(special_ids), 256)
        self.assertEqual(len(set(special_ids)), 256)
        self.assertIn("<|start_header_id|>", self.tokenizer.special_tokens)
        self.assertIn("<|eot_id|>", self.tokenizer.special_tokens)

    def test_download_requires_a_hugging_face_token(self) -> None:
        with patch.dict(
            "os.environ",
            {"HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": ""},
        ):
            with self.assertRaisesRegex(RuntimeError, "Hugging Face token"):
                download_tokenizer_model()


if __name__ == "__main__":
    unittest.main()
