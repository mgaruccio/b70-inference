"""Focused overlay tests; CPU torch cases also run in the pinned serving image."""
import importlib.util
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / "patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b"
CASCADE = runpy.run_path(str(PATCHES / "patch_cascade_acceptance.py"))
HEAD = runpy.run_path(str(PATCHES / "patch_lossy_lmhead.py"))
HAS_TORCH = importlib.util.find_spec("torch") is not None


class PatchTests(unittest.TestCase):
    def test_head_patch_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "model_executor/models/qwen3_5.py"
            path.parent.mkdir(parents=True)
            path.write_text("import torch\nclass Target:\n" + HEAD["OLD"])
            HEAD["patch"](root)
            once = path.read_text()
            HEAD["patch"](root)
            self.assertEqual(once, path.read_text())
            compile(HEAD["HELPER"], "head-helper", "exec")

    def test_cascade_patch_idempotent_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "v1/sample/rejection_sampler.py"
            path.parent.mkdir(parents=True)
            path.write_text("import torch\ndef rejection_sample():\n    if True:\n" + CASCADE["ANCHOR"])
            CASCADE["patch"](root)
            once = path.read_text()
            CASCADE["patch"](root)
            self.assertEqual(once, path.read_text())
            self.assertIn("sampling_metadata.all_greedy and not synthetic_mode", once)
            path.write_text("import torch\n")
            with self.assertRaises(RuntimeError):
                CASCADE["patch"](root)


    def test_long_probe_uses_nonthinking_template_and_short_format_gate(self):
        probe = runpy.run_path(str(ROOT / "scripts/experiments/qwen38_lossy_probe.py"))
        cell = probe["Cell"].__new__(probe["Cell"])
        cell.summary = {}
        lengths = []
        template = [248045, 1, 198, 248046, 198, 248045, 2, 198, 248058, 271, 248059, 271]
        def fake_post(path, payload, timeout=180):
            if path == "/tokenize":
                self.assertFalse(payload["add_special_tokens"])
                if "messages" in payload:
                    self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
                    return {"tokens": template}
                return {"tokens": list(payload["prompt"].encode())}
            self.assertEqual(path, "/v1/completions")
            self.assertEqual(payload["max_tokens"], 32)
            self.assertEqual(payload["prompt"][:3], template[:3])
            self.assertEqual(payload["prompt"][-9:], template[-9:])
            lengths.append(len(payload["prompt"]))
            return {"usage": {"prompt_tokens": lengths[-1]}, "choices": [
                {"text": "684219", "finish_reason": "stop", "logprobs": {"token_logprobs": [0.0]}}]}
        with tempfile.TemporaryDirectory() as d:
            cell.out = Path(d)
            with patch.dict(cell.long_context.__globals__, {"post": fake_post}):
                cell.long_context()
        self.assertEqual(len(lengths), 2)
        self.assertLess(lengths[0], 200000)
        self.assertEqual(lengths[1], 200000)
        self.assertTrue(cell.summary["long_context"]["pass"])
        self.assertEqual(cell.summary["long_context"]["protocol"], "nonthinking-chat-v2")


@unittest.skipUnless(HAS_TORCH, "torch is available in the pinned XPU image, not the lead environment")
class TensorTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.helper = {}
        with patch.dict(os.environ, {"B70_CASCADE_ALPHA": "0"}):
            exec(CASCADE["HELPER"], self.helper)
        self.relax = self.helper["relax_argmax"]

    def logits(self, rows):
        t = self.torch.full((len(rows), 248320), -100.0)
        for i, row in enumerate(rows):
            for token, score in row.items():
                t[i, token] = score
        return t

    def test_probability_rank_padding_special_and_unchanged_logits(self):
        t = self.torch
        logits = self.logits([
            {1: 1, 2: .95}, {1: 1, 2: .7}, {1: 1, 2: .99, 3: .98},
            {1: 1, 2: .99}, {1: 1, 248046: .99}, {248046: 1, 2: .99},
        ])
        original = logits.clone()
        draft = t.tensor([2, 2, 3, -1, 248046, 2])
        argmax = logits.argmax(-1)
        self.assertIs(self.relax(logits, draft, argmax, 0), argmax)
        result = self.relax(logits, draft, argmax, .9)
        self.assertEqual(result.tolist(), [2, 1, 1, 1, 1, 248046])
        self.assertTrue(t.equal(logits, original))

    def test_prefix_stops_at_first_real_reject(self):
        t = self.torch
        logits = self.logits([{1: 1, 2: .99}, {3: 1, 4: .2}, {5: 1, 6: .99}])
        draft = t.tensor([2, 4, 6])
        effective = self.relax(logits, draft, logits.argmax(-1), .9).tolist()
        emitted = []
        for d, target in zip(draft.tolist(), effective):
            emitted.append(target)
            if d != target:
                break
        self.assertEqual(emitted, [2, 3])

    def test_nonfinite_does_not_relax(self):
        t = self.torch
        logits = self.logits([{1: float("inf"), 2: 1}, {1: 1, 2: float("nan")}])
        original = logits.argmax(-1)
        result = self.relax(logits, t.tensor([2, 1]), original, .9)
        self.assertTrue(t.equal(result, original))

    def test_invalid_config_rejected(self):
        for value in ("nan", "-1", "1.1"):
            with patch.dict(os.environ, {"B70_CASCADE_ALPHA": value}), self.assertRaises(ValueError):
                exec(CASCADE["HELPER"], {})

    def test_capture_requires_explicit_phase_and_keeps_splits_separate(self):
        import json
        from types import SimpleNamespace
        helper = {}
        exec(HEAD["HELPER"], helper)
        model = SimpleNamespace()
        hidden = self.torch.ones((5, 5120), dtype=self.torch.float16)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            helper["capture"](model, hidden, d)
            self.assertEqual(list(root.iterdir()), [])
            for split in ("calibration", "heldout"):
                (root / "capture-request.json").write_text(json.dumps({"split": split, "name": split + "-01"}))
                helper["capture"](model, hidden, d)
                files = list((root / split).glob("*.pt"))
                self.assertEqual(len(files), 1)
                result = self.torch.load(files[0], weights_only=True)
                self.assertTrue(self.torch.equal(result["hidden_states"], hidden))
            self.assertEqual(model._b70_capture_counts, {"calibration": 5, "heldout": 5})


    def test_loaded_head_stride_and_noncontiguous_hidden_shape(self):
        t = self.torch
        helper = {}
        exec(HEAD["HELPER"], helper)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "head.pt"
            t.save({"qweight": t.zeros((16, 32), dtype=t.int32),
                    "scales": t.ones((1, 32), dtype=t.float16),
                    "qzeros": t.tensor([8], dtype=t.int8), "group_size": 128}, path)
            method = helper["PackedHeadMethod"](path, t.empty((32, 128), dtype=t.float16))
            self.assertEqual(method.qweight.stride(), (1, 16))
            x = t.ones((2, 3, 128), dtype=t.float16).transpose(0, 1)
            def fake_op(flat, q, bias, scales, zeros, group, extra):
                self.assertEqual(tuple(flat.shape), (6, 128))
                self.assertTrue(flat.is_contiguous())
                return t.zeros((6, 32), dtype=t.float16)
            with patch.object(t.ops._xpu_C, "int4_gemm_w4a16", fake_op, create=True):
                self.assertEqual(tuple(method.apply(None, x).shape), (3, 2, 32))


if __name__ == "__main__":
    unittest.main()
