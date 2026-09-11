#!/usr/bin/env python3
"""Focused stdlib checks; no model import, dependency installation, or GPU."""
import copy
import os
from pathlib import Path
import runpy
import types
import unittest
from unittest.mock import patch

import parity_common as common
import parity_capture as capture

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
patcher = runpy.run_path(str(ROOT / "patch-capture.py"))


class Array:
    def __init__(self, value):
        self.value = value
    def __getitem__(self, key):
        result = self.value[key]
        return Array(result) if isinstance(key, slice) else result
    def detach(self):
        return self
    def cpu(self):
        return self
    def tolist(self):
        return self.value


def batch():
    return types.SimpleNamespace(num_reqs=1, num_tokens=3, num_draft_tokens=0,
                                 num_computed_tokens_np=[0], num_computed_prefill_tokens_np=[0],
                                 prefill_len_np=[3], input_ids=Array([10, 20, 30]))


def layout():
    n = 3
    return {"context_positions": Array(list(range(n))), "query_positions": Array(list(range(n, n + 7))),
            "sample_positions": Array(list(range(n + 1, n + 8))), "sample_indices": Array(list(range(7))),
            "query_ids": Array([80] + [248070] * 6), "draft_seq_lens": Array([n + 7]),
            "context_slots": [[16, 17, 18]], "query_slots": [list(range(19, 26))]}


class Assets(unittest.TestCase):
    def test_all_assets_compile(self):
        for path in ROOT.glob("*.py"):
            compile(path.read_text(), str(path), "exec")

    def test_prompt_stays_existing_code_prompt(self):
        source = REPO / "results/20260910-qwen38-dspark-v2-feasibility/qwen38_lossy_probe.py"
        self.assertEqual(runpy.run_path(str(source))["CODE"], common.CODE)

    def test_canonical_overlay_identity(self):
        self.assertEqual(common.sha(REPO / "scripts/patch-vllm-qwen38-dspark-bf16.py"), common.OVERLAY_SHA)

    def test_real_first_request_only(self):
        b = batch()
        armed = {"prompt_token_ids": [10, 20, 30]}
        self.assertTrue(common.first_request_matches(b, armed))
        self.assertFalse(common.first_request_matches(b, armed, dummy=True))
        self.assertFalse(common.first_request_matches(b, armed, profile=True))
        self.assertFalse(common.first_request_matches(b, {"prompt_token_ids": [10, 20, 99]}))
        for name, value in (("num_reqs", 2), ("num_tokens", 2), ("num_draft_tokens", 7),
                            ("num_computed_tokens_np", [3]), ("num_computed_prefill_tokens_np", [2])):
            wrong = copy.copy(b)
            setattr(wrong, name, value)
            self.assertFalse(common.first_request_matches(wrong, armed), name)

    def test_cache_visibility(self):
        common.check_layout(layout(), 3)
        for name, bad in (("query_positions", Array(list(range(2, 9)))),
                          ("context_positions", Array([1, 2, 3])),
                          ("sample_indices", Array(list(range(1, 8)))),
                          ("draft_seq_lens", Array([11])),
                          ("context_slots", [[16, -1, 18]]),
                          ("query_slots", [[16, 20, 21, 22, 23, 24, 25]])):
            with self.subTest(name=name):
                value = layout()
                value[name] = bad
                with self.assertRaises(ValueError):
                    common.check_layout(value, 3)

    def test_fused_weight_mapping(self):
        config = {"num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
                  "intermediate_size": 17408}
        self.assertEqual(common.weight_parts("layers.0.self_attn.qkv_proj.weight", [6144, 5120], config), [
            ("layers.0.self_attn.q_proj.weight", 0, 4096), ("layers.0.self_attn.k_proj.weight", 4096, 5120),
            ("layers.0.self_attn.v_proj.weight", 5120, 6144)])
        self.assertEqual(common.weight_parts("layers.0.mlp.gate_up_proj.weight", [34816, 5120], config)[1],
                         ("layers.0.mlp.up_proj.weight", 17408, 34816))
        with self.assertRaises(ValueError):
            common.weight_parts("layers.0.self_attn.qkv_proj.weight", [4096, 5120], config)

    def test_noop_wrapper_preserves_identity_and_restores(self):
        sentinel = object()
        class M:
            def f(self, value):
                self.value = value
                return sentinel
        obj = M()
        observer = object.__new__(capture.Capture)
        observer.restores, observer.handles, observer.errors = [], [], []
        def fail(*args):
            raise ValueError("observation failed")
        observer.wrap(obj, "f", before=fail, after=fail)
        self.assertIs(obj.f(sentinel), sentinel)
        self.assertIs(obj.value, sentinel)
        self.assertEqual(len(observer.errors), 2)
        observer.detach()
        self.assertNotIn("f", obj.__dict__)
        self.assertIs(obj.f(sentinel), sentinel)

    def test_direct_backbone_forward_is_observed(self):
        sentinel = object()
        class Backbone:
            def forward(self):
                return sentinel
        obj = Backbone()
        observer = object.__new__(capture.Capture)
        observer.restores, observer.handles, observer.errors = [], [], []
        seen = []
        observer.wrap(obj, "forward", after=seen.append)
        self.assertIs(obj.forward(), sentinel)
        self.assertEqual(seen, [sentinel])
        observer.detach()
        self.assertNotIn("forward", obj.__dict__)
        source = (ROOT / "parity_capture.py").read_text()
        self.assertIn('self.wrap(b, "forward", after=', source)
        self.assertNotIn('self.post(b, "final_hidden")', source)

    def test_hook_returns_none_even_if_put_returns_object(self):
        callbacks = []
        class Module:
            def register_forward_hook(self, callback):
                callbacks.append(callback)
                return object()
        observer = object.__new__(capture.Capture)
        observer.handles, observer.errors = [], []
        observer.put = lambda *args: object()
        observer.post(Module(), "example")
        self.assertIsNone(callbacks[0](None, (), object()))

    def test_distinct_attention_inputs_label_shared_rope(self):
        class Attention:
            def register_forward_pre_hook(self, callback):
                self.callback = callback
                return object()
        observer = object.__new__(capture.Capture)
        observer.handles, observer.errors = [], []
        recorded = {}
        observer.put = lambda name, value: recorded.setdefault(name, value)
        layers = [Attention(), Attention()]
        q, k = object(), object()  # outputs of one shared RoPE module
        for i, layer in enumerate(layers):
            observer.observe_qk(layer, f"layers.{i}.")
            self.assertIsNone(layer.callback(None, (q, k, object())))
        self.assertEqual(set(recorded), {f"layers.{i}.query_{part}_rope" for i in range(2) for part in ("q", "k")})
        self.assertIs(recorded["layers.1.query_q_rope"], q)
        self.assertFalse(observer.errors)

    def test_unarmed_propose_returns_original_object(self):
        sentinel = object()
        class Spec:
            def load_draft_model(self, target_model):
                return sentinel
            def propose(self, input_batch, dummy_run=False, is_profile=False):
                return sentinel
        with patch.dict(os.environ, {}, clear=True):
            capture.install(Spec)
            self.assertIs(Spec().propose(batch()), sentinel)

    def test_unrelated_failed_proposal_is_not_retried(self):
        calls = []
        class Spec:
            def load_draft_model(self, target_model):
                return None
            def propose(self, input_batch, dummy_run=False, is_profile=False):
                calls.append(1)
                raise RuntimeError("original failure")
        with patch.dict(os.environ, DSPARK_PARITY_OUTPUT="/unused"), patch.object(capture, "_DONE", False), patch.object(Path, "is_file", return_value=True), patch.object(Path, "read_text", return_value='{"prompt_token_ids": [1, 2, 3]}'):
            capture.install(Spec)
            with self.assertRaisesRegex(RuntimeError, "original failure"):
                Spec().propose(batch())
        self.assertEqual(len(calls), 1)

    def test_selector_failure_cannot_replace_model_output(self):
        callbacks = []
        class Module:
            def register_forward_hook(self, callback):
                callbacks.append(callback)
                return object()
        observer = object.__new__(capture.Capture)
        observer.handles, observer.errors = [], []
        observer.put = lambda *args: None
        observer.post(Module(), "bad", lambda x: x[123])
        self.assertIsNone(callbacks[0](None, (), ()))
        self.assertEqual(len(observer.errors), 1)

    def test_installed_pinned_source_transform(self):
        source_root = Path(os.environ.get("PINNED_VLLM_SOURCE", "/tmp/vllm-pinned-73029d424"))
        if not source_root.is_dir():
            self.skipTest("read-only installed-source export absent; set PINNED_VLLM_SOURCE")
        canonical = runpy.run_path(str(REPO / "scripts/patch-vllm-qwen38-dspark-bf16.py"))
        original = {p: (source_root / p).read_bytes().decode() for p in canonical["PINNED_SHA256"]}
        base = canonical["prepare"](original)
        hooked, restored = patcher["prepare"](base, canonical)
        self.assertEqual(restored, base)
        self.assertEqual(patcher["prepare"](hooked, canonical)[0], hooked)
        self.assertEqual(patcher["transform"](hooked, reverse=True), base)
        with self.assertRaises(RuntimeError):
            patcher["prepare"](original, canonical)  # canonical correction must not be bypassed
        tampered = dict(hooked)
        tampered[patcher["MODEL"]] += "\n# unapproved source drift\n"
        with self.assertRaises(RuntimeError):
            patcher["prepare"](tampered, canonical)
        damaged = dict(hooked)
        damaged[patcher["MODEL"]] = damaged[patcher["MODEL"]].replace('"context_norm"', '"other"')
        with self.assertRaises(RuntimeError):
            patcher["prepare"](damaged, canonical)

    def test_launch_reuses_native_corrected_configuration(self):
        driver = runpy.run_path(str(REPO / "results/20260911-qwen38-dspark-acceptance-diagnostics/run-acceptance-diagnostics.py"))
        runner = runpy.run_path(str(ROOT / "run-capture.py"))
        # Real helper with read-only host-device metadata mocked; no command execution.
        with patch.object(Path, "exists", return_value=True), patch.object(Path, "stat", return_value=types.SimpleNamespace(st_gid=109)), patch.dict(runner["launch"].__globals__, sha=lambda p: common.OVERLAY_SHA):
            argv, metadata = runner["launch"](driver, Path("/new/capture"), Path("/immutable"), Path("/canonical.py"))
        self.assertIn("--pull=never", argv)
        self.assertIn("--enforce-eager", metadata["serve"])
        self.assertEqual(metadata["speculative_config"]["draft_sample_method"], "greedy")
        self.assertIn("/canonical.py:/experiment/patch_dspark_bf16.py:ro", argv)
        self.assertIn("/parity/patch-capture.py", argv[-1])
        self.assertNotIn("run_matrix", argv[-1])


if __name__ == "__main__":
    unittest.main()
