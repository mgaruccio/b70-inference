#!/usr/bin/env python3
"""Stdlib-only checks; VLLM_SOURCE_ROOT enables exact pinned CPU transforms."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import b70_quant_observe as obs

COMPOSER = runpy.run_path(str(ROOT / "patch-quant-control.py"))
RUNNER = runpy.run_path(str(ROOT / "run-quant-control.py"))
CANONICAL = Path(os.environ.get("CANONICAL_OVERLAY", ROOT.parents[1] / "scripts/patch-vllm-qwen38-dspark-bf16.py"))
CANONICAL_SHA = obs.digest(CANONICAL)
DRIVER = ROOT.parent / "20260911-qwen38-dspark-acceptance-diagnostics/run-acceptance-diagnostics.py"


def overlay():
    return COMPOSER["build_overlay"](CANONICAL, CANONICAL_SHA)


def json_write(path, data):
    path.write_text(json.dumps(data))


class CompositionTests(unittest.TestCase):
    def test_canonical_hash_required(self):
        with self.assertRaisesRegex(RuntimeError, "SHA256"):
            COMPOSER["build_overlay"](CANONICAL, "0" * 64)

    def test_guard_only_changes_requested_quant_branch(self):
        original = runpy.run_path(str(CANONICAL))["CONFIG_HELPERS"]
        candidate = overlay()["CONFIG_HELPERS"]
        self.assertIn('target.quantization not in ("gptq", "auto_gptq", "fp8")', candidate)
        self.assertNotRegex(candidate, r'target\.quantization\s*=(?!=)')
        # Remaining dtype/vocab/cache/rejection/draft validation is byte-identical.
        self.assertEqual(candidate[candidate.index('    if (spec.quantization'):],
                         original[original.index('    if (spec.quantization'):])
        for literal in ('target.dtype != torch.float16', 'target.get_vocab_size() != 248320',
                        'target.hf_text_config.num_hidden_layers != 64'):
            self.assertIn(literal, candidate)

    def test_unknown_canonical_shape_fails(self):
        with self.assertRaises(RuntimeError):
            COMPOSER["replace_once"]("changed guard", "old guard", "new guard")

    def test_pristine_exact_replay_and_negative_source_pins(self):
        source_root = os.environ.get("VLLM_SOURCE_ROOT")
        if not source_root:
            self.skipTest("set VLLM_SOURCE_ROOT to the exported pinned image package")
        ns = overlay()
        sources = {k: (Path(source_root) / k).read_bytes().decode() for k in ns["PINNED_SHA256"]}
        modified = ns["prepare"](sources)
        self.assertEqual(ns["prepare"](modified), modified)
        self.assertEqual(len(sources), 38)
        self.assertEqual({k for k in sources if sources[k] != modified[k]} - set(runpy.run_path(str(CANONICAL))["transformations"]()),
                         {"model_executor/offloader/uva.py", "model_executor/model_loader/base_loader.py",
                          "model_executor/model_loader/utils.py", "v1/worker/gpu/model_runner.py"})
        # Unchanged native kernels and cache/rejection sources are still pinned.
        for name in ("model_executor/kernels/linear/scaled_mm/xpu.py", "v1/worker/gpu/spec_decode/rejection_sampler.py"):
            damaged = dict(modified)
            damaged[name] += "\n# incompatible\n"
            with self.assertRaises(RuntimeError):
                ns["prepare"](damaged)
        mixed = dict(modified)
        mixed["model_executor/offloader/uva.py"] = sources["model_executor/offloader/uva.py"]
        with self.assertRaisesRegex(RuntimeError, "mixed"):
            ns["prepare"](mixed)
        # Run the real canonical apply API in a disposable exported tree.
        with tempfile.TemporaryDirectory(prefix="quant-control-source-") as tmp:
            root = Path(tmp)
            for name, source in sources.items():
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text(source)
            with patch.dict(os.environ, B70_DSPARK_BF16="1"):
                self.assertTrue(ns["apply"](root))
                self.assertEqual(ns["apply"](root), [])
            self.assertEqual({k: (root / k).read_text() for k in sources}, modified)
        print("Pinned CPU source transform/replay: 38 sources; no vLLM/torch import or GPU execution")


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.quant = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
                      "weight_block_size": [128, 128], "modules_to_not_convert": ["example"]}
        self.target = NS(model="/model", quantization="fp8", dtype="float16", head_dtype=None,
                         get_hidden_size=lambda: 5120, get_vocab_size=lambda: 248320,
                         hf_text_config=NS(num_hidden_layers=64),
                         hf_config=NS(architectures=["Qwen3_5ForConditionalGeneration"], model_type="qwen3_5",
                                      quantization_config=self.quant))
        self.spec = NS(method="dspark", model="/draft", target_model_config=self.target, quantization=None,
                       num_speculative_tokens=7, draft_sample_method="greedy", rejection_sample_method="standard",
                       enable_adaptive_verification=False, dspark_draft_topk=None, use_heterogeneous_vocab=False,
                       use_local_argmax_reduction=False, kv_cache_dtype="bfloat16", revision=obs.REVISION)
        self.spec.revision = "b9a5dbdf03bc999c6c73c426b19c2d9041cea393"
        self.ctx = patch.dict(os.environ, B70_DSPARK_BF16="1", VLLM_USE_V2_MODEL_RUNNER="1", B70_QUANT_CONTROL_ARM="fp8")
        self.ctx.start()
        self.addCleanup(self.ctx.stop)
        modules = {"torch": NS(float16="float16", bfloat16="bfloat16"),
                   "vllm.platforms": NS(current_platform=NS(is_xpu=lambda: True)),
                   "vllm._b70_quant_observe": obs}
        self.mods = patch.dict(sys.modules, modules)
        self.mods.start()
        self.addCleanup(self.mods.stop)
        qhash = hashlib.sha256(json.dumps(self.quant, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.hash_patch = patch.object(obs, "FP8_QUANT_SHA", qhash)
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)
        self.digest_patch = patch.object(obs, "digest", side_effect=lambda p:
            obs.FP8_CONFIG_SHA if str(p).endswith("/config.json") else obs.FP8_INDEX_SHA)
        self.digest_patch.start()
        self.addCleanup(self.digest_patch.stop)
        self.ns = {}
        exec(overlay()["CONFIG_HELPERS"], self.ns)

    def test_fp8_allowed_without_spoofing(self):
        self.assertTrue(self.ns["_b70_dspark_requested"](self.spec))
        self.assertEqual(self.target.quantization, "fp8")

    def test_fp8_pinned_identity_and_explicit_enable(self):
        for key, value in (("weight_block_size", None), ("fmt", "e5m2"), ("modules_to_not_convert", [])):
            with self.subTest(key=key):
                old = self.quant[key]
                self.quant[key] = value
                with self.assertRaisesRegex(RuntimeError, "config changed"):
                    self.ns["_b70_dspark_requested"](self.spec)
                self.quant[key] = old
        with patch.dict(os.environ, B70_QUANT_CONTROL_ARM="gptq"):
            with self.assertRaisesRegex(RuntimeError, "not explicitly enabled"):
                obs.validate_fp8_target(self.target)
        with patch.object(obs, "digest", return_value="0" * 64):
            with self.assertRaisesRegex(RuntimeError, "wrong pinned"):
                obs.validate_fp8_target(self.target)

    def test_original_target_and_spec_guards_survive(self):
        for obj, key, value in ((self.target, "dtype", "bfloat16"), (self.target, "head_dtype", "bfloat16"),
                               (self.target.hf_config, "architectures", ["Wrong"]),
                               (self.target.hf_text_config, "num_hidden_layers", 63),
                               (self.spec, "num_speculative_tokens", 8), (self.spec, "quantization", "fp8"),
                               (self.spec, "rejection_sample_method", "synthetic"),
                               (self.spec, "enable_adaptive_verification", True),
                               (self.spec, "kv_cache_dtype", "fp8"), (self.spec, "use_heterogeneous_vocab", True)):
            with self.subTest(key=key):
                old = getattr(obj, key)
                setattr(obj, key, value)
                with self.assertRaises((RuntimeError, ValueError)):
                    self.ns["_b70_dspark_requested"](self.spec)
                setattr(obj, key, old)

    def test_original_gptq_quant_guards_survive(self):
        self.target.quantization = "gptq"
        self.target.hf_config.quantization_config = {"quant_method": "gptq", "bits": 4, "group_size": 128,
                                                    "sym": True, "desc_act": False, "lm_head": False}
        self.assertTrue(self.ns["_b70_dspark_requested"](self.spec))
        self.target.hf_config.quantization_config["bits"] = 8
        with self.assertRaises(ValueError):
            self.ns["_b70_dspark_requested"](self.spec)


class RunnerTests(unittest.TestCase):
    def test_original_matrix_is_reused_unchanged(self):
        self.assertEqual(obs.digest(DRIVER), RUNNER["DRIVER_SHA"])
        ns = runpy.run_path(str(DRIVER))
        requests = []
        class Client:
            def chat(self, label, prompt, count, forced, **kwargs):
                requests.append((label, prompt, count, forced, kwargs))
                return {"label": label, "sampling": kwargs, "usage": {}, "transport_pass": True,
                        "draftsteps_delta": 1, "proposals_delta": 7, "accepts_delta": 1,
                        "emitted_per_step": 2, "output_sha256": "unused", "per_position_accepted_counters": {"0": 1}}
        result = ns["run_matrix"](Client(), NS(CODE="code", PROSE="prose"))
        self.assertEqual(result["completed_requests"], 36)
        self.assertEqual(len({r[0] for r in requests}), 36)
        self.assertTrue(all(r[2:4] == (512, True) for r in requests))
        self.assertEqual({r[4]["seed"] for r in requests}, {42, 43, 44})
        self.assertEqual({r[4]["temperature"] for r in requests}, {0.0, 1.0})

    def test_pair_inputs_not_target_output_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            a.mkdir(); b.mkdir()
            row = {"label": "experiment-test", "prompt": "same", "sampling": {"seed": 42},
                   "cache_salt": "same", "rendered_prompt_token_ids": [1, 2], "output_token_ids": [10],
                   "proposals_delta": 7, "accepts_delta": 1, "emitted_per_step": 2,
                   "per_position_accepted_counters": {"0": 1}}
            for path in (a, b):
                json_write(path / "experiment-test-result.json", row)
                json_write(path / "experiment-test-request.json", {"messages": ["same"]})
            row["output_token_ids"] = [20]
            json_write(b / "experiment-test-result.json", row)
            self.assertTrue(RUNNER["compare_request"](a, b, "experiment-test")["inputs_equal"])
            row["rendered_prompt_token_ids"] = [1, 3]
            json_write(b / "experiment-test-result.json", row)
            with self.assertRaisesRegex(RuntimeError, "paired input mismatch"):
                RUNNER["compare_request"](a, b, "experiment-test")

    def test_real_driver_launch_composition_and_pre_matrix_native_gate(self):
        with tempfile.TemporaryDirectory(prefix="quant-control-launch-") as tmp:
            tmp = Path(tmp)
            target, baseline, out = (tmp / name for name in ("target", "baseline", "out"))
            for path in (target, baseline, out):
                path.mkdir()
            json_write(target / "config.json", {"quantization_config": {"quant_method": "gptq", "bits": 4,
                "group_size": 128, "sym": True, "desc_act": False, "lm_head": False}})
            json_write(target / "model.safetensors.index.json", {"weight_map": {}})
            ns = runpy.run_path(str(DRIVER))["run"].__globals__
            ns["TARGET"] = str(target)
            # Only the /dev/dri metadata lookup is faked; launch argv construction
            # and all configuration values come from the pinned real driver.
            ns["Path"] = lambda p: NS(exists=lambda: True, stat=lambda: NS(st_gid=109)) if str(p).startswith("/dev/dri") else Path(p)
            json_write(baseline / "summary.json", {"status": "passed", "target": str(target),
                "image": ns["IMAGE"], "context": 8192, "kv_cache_dtype": "fp8", "runner": "v2-eager-C1",
                "speculative_config": ns["SPEC_CONFIG"]})
            json_write(baseline / "launch-metadata.json", {"serve": ["vllm", "serve", "/model"]})
            json_write(baseline / "effective-dspark-source.json", {})
            source_checks = []
            ns["source_check"] = lambda *args: source_checks.append(args)
            options = NS(canonical_overlay=CANONICAL, canonical_sha256=CANONICAL_SHA,
                         previous_campaign=tmp, target=None, arm="gptq", target_manifest=None,
                         no_offload_reference=baseline, paired_with=None, driver=DRIVER)
            control = RUNNER["configure"](ns, options)
            argv, metadata = ns["dependency_mounts"](out, tmp / "draft", "dspark", "fp8")
            serve = metadata["serve"]
            for flag, value in (("--quantization", "gptq"), ("--dtype", "float16"), ("--max-model-len", "8192"),
                                ("--cpu-offload-gb", "8"), ("--kv-cache-dtype", "fp8"), ("--max-num-seqs", "1")):
                self.assertEqual(serve[serve.index(flag) + 1], value)
            self.assertIn("--enforce-eager", serve)
            self.assertIn("--no-enable-prefix-caching", serve)
            self.assertIn("B70_QUANT_CONTROL_ARM=gptq", argv)
            self.assertIn("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=0", argv)
            self.assertIn("--cpu-offload-gb 8", argv[-1])
            self.assertEqual(obs.digest(out / "canonical.py"), CANONICAL_SHA)
            self.assertEqual(metadata["patch_order"], ["prefill", "draft", "boundary"])
            self.assertFalse(control["throughput_comparison_valid"])
            with self.assertRaises(FileNotFoundError):
                ns["source_check"](out, metadata["container_name"], "dspark")
            json_write(out / "target-native.json", {"passed": False, "arm": "gptq"})
            json_write(out / "memory-after-load.json", {"arm": "gptq"})
            with self.assertRaisesRegex(RuntimeError, "native kernel/offload"):
                ns["source_check"](out, metadata["container_name"], "dspark")
            self.assertEqual(len(source_checks), 2)
            # Same composited path selects FP8 honestly, never labels it GPTQ.
            options.arm = "fp8"
            fp8_out = tmp / "fp8-out"
            fp8_out.mkdir()
            options.target_manifest = baseline / "summary.json"
            argv, metadata = ns["dependency_mounts"](fp8_out, tmp / "draft", "dspark", "fp8")
            self.assertEqual(metadata["serve"][metadata["serve"].index("--quantization") + 1], "fp8")
            self.assertIn("B70_QUANT_CONTROL_ARM=fp8", argv)

    def test_offload_record_does_not_retain_tensor(self):
        obs._OFFLOADED.clear()
        fake = NS(device="xpu:0", untyped_storage=lambda: NS(data_ptr=lambda: 123, nbytes=lambda: 32),
                  numel=lambda: 32, element_size=lambda: 1)
        obs.record_offload(NS(uva_offloading=True, pin_memory=True), fake)
        self.assertEqual(obs._OFFLOADED, {("xpu:0", 123, 32): 32})
        obs._REOFFLOADED.clear()
        obs.record_reoffload(fake)
        self.assertEqual(obs._REOFFLOADED, {("xpu:0", 123, 32): 32})
        with self.assertRaisesRegex(RuntimeError, "no fallback"):
            obs.record_offload(NS(uva_offloading=False, pin_memory=True), fake)

    def test_unexpected_native_kernel_fails_without_forcing(self):
        with patch.dict(sys.modules, {"torch": NS()}):
            method = NS(fp8_linear=object())
            with patch.object(obs, "qualified", side_effect=[obs.FP8_METHOD, "wrong.NativeKernel", "wrong.NativeKernel"]):
                with self.assertRaisesRegex(RuntimeError, "unexpected FP8 native kernel"):
                    obs.check_linear(method, NS(), "fp8")


if __name__ == "__main__":
    unittest.main(verbosity=2)
