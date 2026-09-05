"""Regression checks for the pinned DFlash overlay's XPU RMSNorm fallback.

Numerical correctness is additionally exercised on the real B70 kernel; these
CPU-only checks cover patch replay and platform/layer dispatch.
"""
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

PATCH = Path(__file__).resolve().parents[1] / "scripts/patch-vllm-dflash-gptq-context-kv.py"
# Base fallback is already present in this minimal fixture. Test only the new
# normalization overlay without importing vLLM or requiring an XPU on CI.
SOURCE = '''# DFLASH_GPTQ_CONTEXT_KV_FALLBACK
class Draft:
    def _normalize_context_k(self, all_k):
        all_k_normed = torch.empty_like(all_k)
        ops.rms_norm(
            all_k_normed,
            all_k,
            self._k_norm_weights,
            self._rms_norm_eps,
        )
        return all_k_normed
'''


class Tensor:
    def __init__(self, shape, device="xpu"):
        self.shape = shape
        self.ndim = len(shape)
        self.device = SimpleNamespace(type=device)
        self.rows = {}

    def __getitem__(self, index):
        if index not in self.rows:
            self.rows[index] = Tensor(self.shape[1:], self.device.type)
        return self.rows[index]


class ContextNormTests(unittest.TestCase):
    def apply(self, path):
        return subprocess.run(
            [sys.executable, str(PATCH), str(path)],
            env={**os.environ, "DFLASH_KV_MODE": "none"},
            text=True, capture_output=True,
        )

    def patched_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qwen3_dflash.py"
            path.write_text(SOURCE)
            result = self.apply(path)
            self.assertEqual(result.returncode, 0, result.stderr)
            return path.read_text()

    def run_norm(self, device="xpu", weight_shape=(5, 128)):
        ops = SimpleNamespace(rms_norm=Mock())
        torch = SimpleNamespace(empty_like=lambda x: Tensor(x.shape, x.device.type))
        scope = {"ops": ops, "torch": torch}
        exec(compile(self.patched_source(), "patched_fixture", "exec"), scope)
        draft = scope["Draft"]()
        draft._k_norm_weights = Tensor(weight_shape, device)
        draft._rms_norm_eps = 1e-6
        x = Tensor((5, 4, 8, 128), device)
        y = draft._normalize_context_k(x)
        return ops, draft, x, y

    def test_xpu_uses_each_layers_weight(self):
        ops, draft, x, y = self.run_norm()
        self.assertEqual(ops.rms_norm.call_count, 5)
        for layer, call in enumerate(ops.rms_norm.call_args_list):
            self.assertEqual(call.args, (y[layer], x[layer], draft._k_norm_weights[layer], 1e-6))

    def test_non_xpu_keeps_grouped_call(self):
        ops, draft, x, y = self.run_norm(device="cuda")
        ops.rms_norm.assert_called_once_with(y, x, draft._k_norm_weights, 1e-6)

    def test_one_dimensional_weights_keep_original_call(self):
        ops, draft, x, y = self.run_norm(weight_shape=(128,))
        ops.rms_norm.assert_called_once_with(y, x, draft._k_norm_weights, 1e-6)

    def test_mismatched_layer_or_head_shape_rejected(self):
        for shape in ((4, 128), (5, 64)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.run_norm(weight_shape=shape)

    def test_patch_replay_is_identical_and_valid_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qwen3_dflash.py"
            path.write_text(SOURCE)
            for _ in range(2):
                result = self.apply(path)
                self.assertEqual(result.returncode, 0, result.stderr)
                current = path.read_text()
                ast.parse(current)
                self.assertEqual(current.count("# DFLASH_XPU_LAYERWISE_CONTEXT_K_NORM"), 1)
                if _ == 0:
                    first = current
                else:
                    self.assertEqual(first, current)
            self.assertEqual(path.with_suffix(".py.orig").read_text(), SOURCE)

    def test_unrecognized_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qwen3_dflash.py"
            changed = SOURCE.replace("all_k_normed =", "different_output =")
            path.write_text(changed)
            result = self.apply(path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("context K norm anchor missing or ambiguous", result.stderr)
            self.assertEqual(path.read_text(), changed)


if __name__ == "__main__":
    unittest.main()
