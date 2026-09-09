"""Focused offline tests for the Qwen3.8 DFlash2 CPU RTN converter.

The quantizer tests use small matrices.  The checkpoint test patches only the
strict contract constants while constructing a factored 81-tensor fixture; the
production converter still validates the real 81-tensor/1,924,404,480-parameter
contract unchanged.  Real-model conversion and runtime validation belong to the
lead's pinned CPU/XPU campaign.
"""

from __future__ import annotations

import copy
import math
import importlib.util
import sys
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "quantize-qwen38-dflash2-rtn.py"
spec = importlib.util.spec_from_file_location("qwen38_dflash2_rtn", SCRIPT)
converter = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = converter
spec.loader.exec_module(converter)
torch = converter.torch

try:
    from safetensors.torch import load_file, save_file
except ImportError:  # pragma: no cover - exercised by the local no-dependency checkout
    load_file = save_file = None


TORCH_REQUIRED = unittest.skipIf(torch is None, "PyTorch is unavailable")
CHECKPOINT_REQUIRED = unittest.skipIf(
    torch is None or save_file is None or load_file is None,
    "PyTorch and safetensors are required",
)


def _unpack_words(words):
    """Independent low-to-high nibble unpacking oracle for int32 words."""

    values = words.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, dtype=torch.int64) * 4
    return (values.unsqueeze(-1) >> shifts) & 0xF


def _bf16_scalar(value: float) -> float:
    return float(torch.tensor(value, dtype=torch.float32).to(torch.bfloat16).item())


def _oracle(weight):
    """Independent Python-loop RTN and GPTQ-v1 packing oracle."""

    n, k = map(int, weight.shape)
    groups = k // 128
    values = weight.to(dtype=torch.float32)
    codes = [[0] * k for _ in range(n)]
    scales = [[0.0] * n for _ in range(groups)]
    for row in range(n):
        for group in range(groups):
            group_values = [float(values[row, group * 128 + col]) for col in range(128)]
            absmax = max(abs(value) for value in group_values)
            scale = 1.0 if absmax == 0 else _bf16_scalar(2.0 * absmax / 15.0)
            if scale <= 0:
                scale = float(torch.finfo(torch.bfloat16).tiny)
            scales[group][row] = scale
            for col, value in enumerate(group_values):
                code = max(0, min(15, round(value / scale) + 8))
                codes[row][group * 128 + col] = code

    qweight_words = []
    for input_block in range(k // 8):
        for row in range(n):
            qweight_words.append(
                sum(codes[row][input_block * 8 + nibble] << (4 * nibble) for nibble in range(8))
            )
    qzeros_words = [[0x77777777] * (n // 8) for _ in range(groups)]
    return codes, scales, qweight_words, qzeros_words


def _fixture_config():
    return {
        "model_type": "qwen3",
        "architectures": ["DFlash2DraftModel"],
        "hidden_size": 5120,
        "num_hidden_layers": 5,
        "vocab_size": 248320,
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
        "max_position_embeddings": 262144,
        "native_fixture_field": {"kept": True},
        "dflash_config": {
            "target_layer_ids": [5, 19, 33, 47, 61],
            "block_size": 8,
            "selector_rank": 256,
            "selector_top_k": 16,
            "conv_kernel_size": 2,
            "conv_group_size": 16,
        },
    }


@TORCH_REQUIRED
class QuantizeWeightTests(unittest.TestCase):
    def test_independent_gptq_v1_packing_and_dequantization_oracle(self):
        weight = torch.zeros((8, 256), dtype=torch.float32)
        # absmax=7.5 makes the rounded scale exactly one.  These exercise the
        # signed high nibble, clamped endpoints, and ties-to-even at +/-x.5.
        weight[0, 0] = -7.5
        weight[0, 1] = 7.5
        weight[0, 7] = 7.5
        weight[1, 0] = 0.5
        weight[1, 1] = 1.5
        weight[1, 2] = -0.5
        weight[1, 3] = -1.5
        weight[1, 10] = 7.5
        # Row two is an all-zero group and must receive a positive safe scale.
        weight[2, 128:] = 0
        weight[3, :128] = torch.linspace(-3.0, 3.0, 128)
        weight[3, 128:] = torch.linspace(-2.0, 2.0, 128)

        result = converter.quantize_weight(weight)
        codes, scales, qweight_words, qzeros_words = _oracle(weight)

        self.assertEqual(result["qweight"].shape, (32, 8))
        self.assertEqual(result["scales"].shape, (2, 8))
        self.assertEqual(result["qzeros"].shape, (2, 1))
        self.assertEqual(result["g_idx"].shape, (256,))
        self.assertEqual(result["qweight"].dtype, torch.int32)
        self.assertEqual(result["scales"].dtype, torch.bfloat16)
        self.assertEqual(result["qzeros"].dtype, torch.int32)
        self.assertEqual(result["g_idx"].dtype, torch.int32)

        expected_qweight = torch.tensor(qweight_words, dtype=torch.int64).reshape(32, 8).to(torch.int32)
        expected_qzeros = torch.tensor(qzeros_words, dtype=torch.int64).to(torch.int32)
        expected_scales = torch.tensor(scales, dtype=torch.bfloat16)
        self.assertTrue(torch.equal(result["qweight"], expected_qweight))
        self.assertTrue(torch.equal(result["qzeros"], expected_qzeros))
        self.assertTrue(torch.equal(result["scales"], expected_scales))
        self.assertTrue(torch.equal(result["g_idx"], torch.arange(256, dtype=torch.int32) // 128))

        unpacked_codes = _unpack_words(result["qweight"]).permute(1, 0, 2).reshape(8, 256)
        # The word containing nibble seven has its sign bit set; unpacking must
        # still recover the original unsigned nibbles.
        self.assertLess(int(result["qweight"][0, 0]), 0)
        self.assertEqual(int(unpacked_codes[0, 0]), 0)
        self.assertEqual(int(unpacked_codes[0, 1]), 15)
        self.assertEqual(int(unpacked_codes[1, 0]), 8)   # 8.5 rounds to even 8
        self.assertEqual(int(unpacked_codes[1, 1]), 10)  # 9.5 rounds to even 10
        self.assertEqual(int(unpacked_codes[1, 2]), 8)   # 7.5 rounds to even 8
        self.assertEqual(int(unpacked_codes[1, 3]), 6)   # 6.5 rounds to even 6
        self.assertTrue(torch.all(unpacked_codes[2] == 8))

        unpacked_zero_minus_one = _unpack_words(result["qzeros"]).reshape(2, 8) + 1
        self.assertTrue(torch.all(unpacked_zero_minus_one == 8))
        actual_scales = result["scales"].transpose(0, 1)
        actual_zero = unpacked_zero_minus_one.transpose(0, 1).repeat_interleave(128, dim=1)
        dequantized = (unpacked_codes - actual_zero) * actual_scales.repeat_interleave(128, dim=1)
        expected_codes = torch.tensor(codes, dtype=torch.int64)
        oracle_dequantized = (expected_codes - 8) * expected_scales.transpose(0, 1).repeat_interleave(128, dim=1)
        self.assertTrue(torch.equal(unpacked_codes, expected_codes))
        torch.testing.assert_close(dequantized, oracle_dequantized.to(dtype=dequantized.dtype))

    def test_output_row_chunk_boundary_is_deterministic(self):
        weight = torch.arange(16 * 128, dtype=torch.float32).reshape(16, 128) / 100
        with patch.object(converter, "ROW_CHUNK_SIZE", 3):
            first = converter.quantize_weight(weight)
            second = converter.quantize_weight(weight)
        for name in first:
            self.assertTrue(torch.equal(first[name], second[name]), name)

    def test_finite_and_shape_rejection(self):
        finite = torch.zeros((8, 128), dtype=torch.float32)
        bad_nan = finite.clone()
        bad_nan[0, 0] = float("nan")
        bad_inf = finite.clone()
        bad_inf[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            converter.quantize_weight(bad_nan)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            converter.quantize_weight(bad_inf)
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            converter.quantize_weight(torch.zeros((8, 128, 1), dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "divisible by eight"):
            converter.quantize_weight(torch.zeros((7, 128), dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "group_size 128"):
            converter.quantize_weight(torch.zeros((8, 120), dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "floating-point"):
            converter.quantize_weight(torch.zeros((8, 128), dtype=torch.int32))


class ContractAndPathTests(unittest.TestCase):
    def test_exact_twenty_allowed_names_and_rejected_neighbors(self):
        self.assertEqual(len(converter.QUANTIZED_TENSOR_NAMES), 20)
        for layer in range(5):
            for suffix in (
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ):
                self.assertTrue(converter._is_quantized_name(f"layers.{layer}.{suffix}"))
        for name in (
            "candidate_selector.hidden_projection.weight",
            "candidate_selector.predecessor_codebook",
            "candidate_selector.successor_codebook",
            "fc.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.v_proj.weight",
            "layers.0.mlp.gate_up_proj.weight",
            "layers.5.mlp.down_proj.weight",
        ):
            self.assertFalse(converter._is_quantized_name(name), name)
        self.assertEqual(len(converter.EXPECTED_SHAPES), 81)
        self.assertEqual(sum(math.prod(shape) for shape in converter.EXPECTED_SHAPES.values()), 1_924_404_480)
        self.assertEqual(converter.EXPECTED_TENSOR_COUNT, 81)
        self.assertEqual(converter.EXPECTED_PARAMETER_COUNT, 1_924_404_480)
    def test_config_only_adds_the_standard_gptq_contract(self):
        source = _fixture_config()
        converter._validate_source_config(source)
        original = copy.deepcopy(source)
        output = converter._output_config(source)
        self.assertEqual(source, original)
        self.assertEqual(output["native_fixture_field"], {"kept": True})
        self.assertEqual(output["quantization_config"], converter.QUANTIZATION_CONFIG)
        self.assertNotIn("calibration", output)
        self.assertNotIn("rtn_receipt", output)

    def test_source_quantization_and_head_embed_metadata_is_rejected(self):
        for key in ("quantization_config", "vocab_remap", "lm_head"):  # all are forbidden source metadata
            config = _fixture_config()
            config[key] = {} if key != "lm_head" else "present"
            with self.assertRaisesRegex(ValueError, "quantization/remap/head/embed"):
                converter._validate_source_config(config)
        self.assertTrue(converter._forbidden_tensor_name("lm_head.weight"))
        self.assertTrue(converter._forbidden_tensor_name("model.embed_tokens.weight"))
        self.assertTrue(converter._forbidden_tensor_name("layers.0.self_attn.o_proj.qweight"))

    def test_existing_output_and_related_paths_are_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            existing = root / "existing"
            existing.mkdir()
            sentinel = existing / "do-not-delete"
            sentinel.write_text("owned", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                converter.quantize_checkpoint(source, existing)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "owned")
            with self.assertRaises(FileExistsError):
                converter.quantize_checkpoint(source, source)
            child = source / "new-output"
            with self.assertRaisesRegex(ValueError, "unrelated"):
                converter.quantize_checkpoint(source, child)
            self.assertFalse(child.exists())
            # An existing ancestor is refused before any source/output mutation.
            with self.assertRaises(FileExistsError):
                converter.quantize_checkpoint(source, root)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "owned")


@CHECKPOINT_REQUIRED
class SyntheticCheckpointTests(unittest.TestCase):
    def test_factored_checkpoint_preserves_native_tensors_and_rewrites_only_targets(self):
        fixture_shapes = {
            name: ((8, 128) if converter._is_quantized_name(name) else (2, 2))
            for name in converter.EXPECTED_SHAPES
        }
        fixture_parameter_count = sum(math.prod(shape) for shape in fixture_shapes.values())
        source_config = _fixture_config()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "rtn-output"
            source.mkdir()
            (source / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
            tensors = {}
            for index, (name, shape) in enumerate(fixture_shapes.items()):
                value = (index % 13 - 6) / 4
                tensors[name] = torch.full(shape, value, dtype=torch.bfloat16)
            save_file(tensors, str(source / "model.safetensors"), metadata={"format": "pt"})

            with patch.object(converter, "EXPECTED_SHAPES", fixture_shapes), patch.object(
                converter, "EXPECTED_TENSOR_COUNT", len(fixture_shapes)
            ), patch.object(converter, "EXPECTED_PARAMETER_COUNT", fixture_parameter_count):
                summary = converter.quantize_checkpoint(source, output)

            self.assertEqual(summary["method"], "RTN")
            self.assertEqual(summary["quantized_matrices"], 20)
            self.assertEqual(summary["retained_native_tensors"], len(fixture_shapes) - 20)
            self.assertEqual(summary["output_tensors"], len(fixture_shapes) + 60)
            output_tensors = load_file(str(output / "model.safetensors"), device="cpu")
            for name, tensor in tensors.items():
                if converter._is_quantized_name(name):
                    self.assertNotIn(name, output_tensors)
                    prefix = name[: -len(".weight")]
                    self.assertIn(prefix + ".qweight", output_tensors)
                    self.assertIn(prefix + ".scales", output_tensors)
                    self.assertIn(prefix + ".qzeros", output_tensors)
                    self.assertIn(prefix + ".g_idx", output_tensors)
                else:
                    self.assertIn(name, output_tensors)
                    self.assertTrue(torch.equal(output_tensors[name], tensor), name)
                    self.assertEqual(output_tensors[name].dtype, torch.bfloat16)
            saved_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            expected_config = copy.deepcopy(source_config)
            expected_config["quantization_config"] = converter.QUANTIZATION_CONFIG
            self.assertEqual(saved_config, expected_config)
            self.assertFalse(any("head" in name or "embed" in name for name in output_tensors))


if __name__ == "__main__":
    unittest.main()
