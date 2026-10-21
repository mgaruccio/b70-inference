# SPDX-License-Identifier: Apache-2.0
"""Exact-source installer tests plus CPU tests of the installed engine call sites.

Run the installed contracts ONLY in the frozen throwaway runtime, without devices:
  EAGLE4_CPU_CONTRACTS=1 EAGLE4_FROZEN_PACKAGE=/tmp/e4-base/vllm \
  EAGLE4_ENGINE_OVERLAY=/check/overlay EAGLE4_OFFICIAL_ROOT=/check/official \
  /opt/venv/bin/python /check/tests/test_eagle4_full_engine_overlay.py -v

These are engine protocol/lifecycle tests, NOT evidence of native tree kernels,
model parity, measured cost decisions, or the complete served EAGLE4 algorithm.
"""
from __future__ import annotations

from contextlib import nullcontext
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

OVERLAY = Path(os.environ.get("EAGLE4_ENGINE_OVERLAY", Path(__file__).resolve().parents[1] / "patches/eagle4-vllm-ac7509e2b"))
BASE = Path(os.environ.get("EAGLE4_FROZEN_PACKAGE", "/home/mike/code/vllm-b70-ac750-python/vllm"))
OFFICIAL = Path(os.environ.get("EAGLE4_OFFICIAL_ROOT", "/home/mike/code/EAGLE-4-b70"))
CPU = os.environ.get("EAGLE4_CPU_CONTRACTS") == "1"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


installer = load_file("eagle4_engine_overlay_installer", OVERLAY / "install.py")
MANIFEST = json.loads((OVERLAY / "manifest.json").read_text())


@unittest.skipUnless(BASE.is_dir(), "set EAGLE4_FROZEN_PACKAGE to the immutable frozen export")
class OverlayReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "vllm"
        for entry in MANIFEST["files"]:
            dst = self.root / entry["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if entry["before_sha256"] is not None:
                shutil.copy2(BASE / entry["path"], dst)

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}

    def test_all_files_checked_before_writes_and_idempotent_replay(self):
        original = self.snapshot()
        check = installer.apply(self.root, check=True)
        self.assertGreater(check["pending"], 10)
        self.assertEqual(original, self.snapshot())
        first = installer.apply(self.root)
        self.assertEqual(first["written"], check["pending"])
        installed = self.snapshot()
        self.assertEqual(installer.apply(self.root)["written"], 0)
        self.assertEqual(installer.apply(self.root, check=True)["pending"], 0)
        self.assertEqual(installed, self.snapshot())
        self.assertEqual(original["version.py"], installed["version.py"])
        for entry in MANIFEST["files"]:
            self.assertEqual(installer.sha((self.root / entry["path"]).read_bytes()), entry["after_sha256"])

    def test_late_source_mismatch_writes_nothing(self):
        last = next(e for e in reversed(MANIFEST["files"]) if e["before_sha256"])
        path = self.root / last["path"]
        path.write_bytes(path.read_bytes() + b"\n# drift\n")
        original = self.snapshot()
        with self.assertRaisesRegex(installer.OverlayError, "Unsupported source"):
            installer.apply(self.root)
        self.assertEqual(original, self.snapshot())

    def test_conflicting_new_file_rejected_before_writes(self):
        entry = next(e for e in MANIFEST["files"] if e["before_sha256"] is None)
        (self.root / entry["path"]).write_text("# other overlay\n")
        original = self.snapshot()
        with self.assertRaises(installer.OverlayError):
            installer.apply(self.root)
        self.assertEqual(original, self.snapshot())

    def test_symlink_target_rejected(self):
        path = self.root / "platforms/xpu.py"
        path.unlink()
        path.symlink_to(BASE / "platforms/xpu.py")
        with self.assertRaisesRegex(installer.OverlayError, "Symlink"):
            installer.apply(self.root)

    def test_modified_patch_rejected(self):
        bad = Path(self.tmp.name) / "bad"
        bad.mkdir()
        shutil.copy2(OVERLAY / "manifest.json", bad)
        (bad / "engine.patch").write_bytes((OVERLAY / "engine.patch").read_bytes() + b"# drift\n")
        with self.assertRaisesRegex(installer.OverlayError, "patch digest"):
            installer.apply(self.root, bad)

    def test_only_scoped_changes_and_release_lifecycle(self):
        installer.apply(self.root)
        source = (self.root / "v1/worker/gpu/model_runner.py").read_text()
        self.assertNotIn("defer_finalize", source)
        self.assertNotIn("finish_forward(", source)  # frozen post_forward owns finalization
        self.assertEqual(source.count("self.kv_connector.post_forward(finished_req_ids)"), 3)
        self.assertIn("self.speculator.prepare_verify_inputs(", source)
        self.assertIn("self.speculator.post_accept(", source)
        scheduler = (self.root / "v1/core/sched/scheduler.py").read_text()
        self.assertIn("self.num_lookahead_tokens = vllm_config.num_lookahead_tokens", scheduler)
        self.assertIn("num_lookahead_tokens=self.num_lookahead_tokens", scheduler)
        warmup = (self.root / "v1/worker/gpu/warmup.py").read_text()
        self.assertIn("model_runner.vllm_config.num_lookahead_tokens", warmup)
        xpu = (self.root / "v1/worker/xpu_model_runner.py").read_text()
        self.assertIn("GPUModelRunnerV2", xpu)
        self.assertIn("_torch_cuda_wrapper", xpu)


@unittest.skipUnless(CPU, "installed engine tests require the isolated frozen CPU container")
class InstalledEngineContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import vllm
        import vllm.v1.worker.gpu.model_runner as runner
        from vllm.config import SpeculativeConfig, VllmConfig
        from vllm.v1.spec_decode import registry
        from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
        cls.torch, cls.runner = torch, runner
        cls.SpeculativeConfig, cls.VllmConfig = SpeculativeConfig, VllmConfig
        cls.registry, cls.BaseSpeculator = registry, BaseSpeculator
        assert vllm.__version__ == MANIFEST["base_version"]
        assert "/opt/venv/lib/python3.12/site-packages/vllm/" in vllm.__file__, vllm.__file__
        assert torch.xpu.device_count() == 0
        sys.path.insert(0, str(OFFICIAL))
        cls.addClassCleanup(lambda: sys.path.remove(str(OFFICIAL)))

    def tearDown(self):
        self.registry.unregister_speculative_method("engine_contract")

    def test_official_hook_map_real_imports(self):
        from eagle_harness.vllm.adapter import missing_hooks
        self.assertEqual(missing_hooks(), [])

    def test_opaque_config_real_official_lookahead_and_v2_factory(self):
        from eagle_harness.vllm.config import configure_speculative_config as configure, num_lookahead_tokens
        from vllm.v1.worker.gpu.spec_decode import init_speculator
        class Plugin:
            def __init__(self, config, device):
                self.config, self.device = config, device
        with mock.patch.object(self.registry, "resolve_obj_by_qualname", return_value=Plugin):
            self.registry.register_speculative_method("engine_contract", self.registry.SpeculativeMethodSpec(
                speculator_cls="plugin:Speculator", configure=configure,
                num_lookahead_tokens=num_lookahead_tokens, uses_aux_hidden_states=True))
            cfg = self.SpeculativeConfig(method="engine_contract", method_config={
                "max_verify_width": 16, "node_top_k": 4, "tree_depth": 14, "depth_stop_gain": 0.0})
            self.assertEqual(cfg.method, "engine_contract")
            self.assertEqual(cfg.num_speculative_tokens, 15)
            self.assertEqual(cfg.method_config["node_top_k"], 4)
            self.assertTrue(cfg.use_eagle())
            vc = NS(speculative_config=cfg, num_speculative_tokens=15)
            self.assertEqual(self.VllmConfig.num_lookahead_tokens.fget(vc), 56)
            self.assertIs(init_speculator(vc, "cpu").config, vc)
            old_hash = cfg.compute_hash()
            cfg.method_config["tree_depth"] = 13
            self.assertNotEqual(old_hash, cfg.compute_hash())
            with mock.patch("vllm.envs.VLLM_USE_V2_MODEL_RUNNER", None):
                self.assertTrue(self.VllmConfig.use_v2_model_runner.fget(vc))
            with mock.patch("vllm.envs.VLLM_USE_V2_MODEL_RUNNER", False):
                with self.assertRaisesRegex(ValueError, "requires Model Runner V2"):
                    self.VllmConfig.use_v2_model_runner.fget(vc)
        with self.assertRaises(ValueError):
            self.SpeculativeConfig(method="not_registered", num_speculative_tokens=2)
        with self.assertRaisesRegex(ValueError, "built-in"):
            self.registry.register_speculative_method("mtp", self.registry.SpeculativeMethodSpec("x:X"))
        self.assertEqual(self.VllmConfig.num_lookahead_tokens.fget(NS(speculative_config=None)), 0)
        native = NS(use_dflash=lambda: False, registered_method=lambda: None, use_eagle=lambda: True)
        self.assertEqual(self.VllmConfig.num_lookahead_tokens.fget(NS(speculative_config=native, num_speculative_tokens=4)), 4)

    def test_custom_backend_selected_through_actual_xpu_platform(self):
        from vllm.platforms.xpu import XPUPlatform
        from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
        import vllm.v1.attention.backends.registry as attention_registry
        class ContractTreeBackend(FlashAttentionBackend):
            @staticmethod
            def get_name():
                return "ENGINE_CONTRACT_TREE"
        path = f"{__name__}.ContractTreeBackend"
        # A resolvable backend class, not an invented/unimportable string. The
        # device tree implementation is supplied/tested by the adapter worker.
        with mock.patch.object(sys.modules[__name__], "ContractTreeBackend", ContractTreeBackend, create=True), \
             mock.patch.dict(attention_registry._ATTN_OVERRIDES, clear=False):
            register_backend(AttentionBackendEnum.CUSTOM, path)
            cfg = NS(kv_cache_dtype="auto", dtype=self.torch.float32, use_sparse=False,
                     use_mla=False, use_mm_prefix=False)
            selected = XPUPlatform.get_attn_backend_cls(AttentionBackendEnum.CUSTOM, cfg)
            self.assertEqual(selected, path)
            self.assertIs(AttentionBackendEnum.CUSTOM.get_class(), ContractTreeBackend)
            self.assertEqual(XPUPlatform.get_attn_backend_cls(AttentionBackendEnum.TRITON_ATTN, cfg),
                             AttentionBackendEnum.TRITON_ATTN.get_path())
            AttentionBackendEnum.CUSTOM.clear_override()
            with self.assertRaises(ValueError):
                XPUPlatform.get_attn_backend_cls(AttentionBackendEnum.CUSTOM, cfg)

    def test_sampler_factory_default_noops_and_varlen_validation(self):
        native = object()
        runner = NS(sampler=object(), speculative_config=object(), device="cpu", speculator=None)
        with mock.patch.object(self.runner, "RejectionSampler", return_value=native) as chain:
            self.assertIs(self.runner.GPUModelRunner._build_rejection_sampler(runner), native)
            chain.assert_called_once()
            custom = object()
            runner.speculator = NS(make_rejection_sampler=lambda sampler: custom)
            self.assertIs(self.runner.GPUModelRunner._build_rejection_sampler(runner), custom)
            self.assertEqual(chain.call_count, 1)
        self.assertIsNone(self.BaseSpeculator.make_rejection_sampler(None, runner.sampler))
        self.assertIsNone(self.BaseSpeculator.prepare_verify_inputs(None, object(), {}, dummy_run=True))
        self.assertIsNone(self.BaseSpeculator.post_accept(None, None, None, None, None))
        self.assertFalse(self.BaseSpeculator.uses_varlen_verify_graphs)
        self.assertEqual(self.BaseSpeculator.num_extra_query_per_req, 0)
        from vllm.v1.attention.backend import AttentionCGSupport as CG
        from vllm.v1.worker.gpu.attn_utils import get_varlen_cudagraph_unsupported_backend
        class Builder:
            get_cudagraph_support = staticmethod(lambda vc, spec: CG.UNIFORM_BATCH)
            get_varlen_cudagraph_max_query_len = staticmethod(lambda vc, spec: 16)
        group = NS(layer_names=["target"], get_metadata_builder=lambda _: Builder(),
                   kv_cache_spec=None, backend=Builder)
        self.assertIsNone(get_varlen_cudagraph_unsupported_backend([[group]], None, 16))
        self.assertEqual(get_varlen_cudagraph_unsupported_backend([[group]], None, 17), ("Builder", 16))
        runner = NS(adaptive_verification=None, speculator=NS(uses_varlen_verify_graphs=True),
                    kv_cache_config=NS(kv_cache_groups=[NS(layer_names=["target"])]),
                    attn_groups=[[group]], vllm_config=None, decode_query_len=16,
                    model_state=NS(get_additional_cg_support=lambda: (CG.ALWAYS, None)))
        self.assertTrue(self.runner.GPUModelRunner._use_varlen_verify_graphs(runner, None))
        runner.decode_query_len = 17
        with self.assertRaises(ValueError):
            self.runner.GPUModelRunner._use_varlen_verify_graphs(runner, None)
        runner.speculator = None
        self.assertFalse(self.runner.GPUModelRunner._use_varlen_verify_graphs(runner, None))

    def test_rejection_transport_preserves_native_ids_and_masks_only_verifier(self):
        import numpy as np
        import vllm.v1.worker.gpu.spec_decode.rejection_sampler as rejection
        from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
        t = self.torch
        batch = NS(input_ids=t.tensor([5, 6, 7, 8]), positions=t.arange(10, 14),
            logits_indices=t.arange(4), idx_mapping=t.tensor([0]), idx_mapping_np=np.array([0]),
            expanded_idx_mapping=t.zeros(4, dtype=t.int64), expanded_local_pos=t.arange(4),
            cu_num_logits=t.tensor([0, 4]), cu_num_logits_np=np.array([0, 4]), num_reqs=1,
            seq_lens=t.tensor([14]), seq_lens_cpu_upper_bound=t.tensor([14]))
        seen = []
        def process(logits, expanded, idx, idx_np, pos, ids, local):
            seen.append(("process", ids.clone()))
            return logits
        sampler = NS(compute_nans=False, logprobs_mode="raw_logprobs", use_fp64_gumbel=False,
            sampling_states=NS(max_num_logprobs=lambda _: NO_LOGPROBS,
                               temperature=NS(gpu=t.ones(1)), seeds=NS(gpu=t.zeros(1))),
            req_states=NS(prefill_len=NS(gpu=t.zeros(1))), apply_sampling_params=process)
        r = rejection.RejectionSampler(sampler, NS(num_speculative_tokens=3, rejection_sample_method="standard"), "cpu")
        def accept(logits, draft_logits, ids, *args, **kwargs):
            seen.append(("verify", ids.clone()))
            return t.tensor([[21, 22, -1, -1]]), t.tensor([2])
        counts = lambda n, seq, cu, idx, prefill: (n, cu.diff() - n)
        for invalid in (None, t.tensor([False, True, False, True])):
            with mock.patch.object(rejection, "rejection_sample", side_effect=accept), \
                 mock.patch.object(rejection, "get_num_sampled_and_rejected", side_effect=counts):
                output = r(t.zeros(4, 32), batch, None, invalid)
            self.assertEqual(seen[-2][1].tolist(), [5, 6, 7, 8])
            self.assertEqual(seen[-1][1].tolist(), [5, 6, 7, 8] if invalid is None else [5, -1, 7, -1])
            self.assertEqual(output.num_sampled.tolist(), [2])
            self.assertEqual(output.num_rejected.tolist(), [2])
            self.assertEqual(batch.input_ids.tolist(), [5, 6, 7, 8])

    def test_official_tree_rejection_subclass_receives_release_metadata(self):
        import numpy as np
        import eagle_harness.vllm.tree_accept as tree
        import vllm.v1.worker.gpu.spec_decode.rejection_sampler as rejection
        from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
        t, seen = self.torch, []
        batch = NS(input_ids=t.tensor([5, 6, 7, 8]), positions=t.arange(10, 14),
            logits_indices=t.arange(4), idx_mapping=t.tensor([0]), idx_mapping_np=np.array([0]),
            expanded_idx_mapping=t.zeros(4, dtype=t.int64), expanded_local_pos=t.arange(4),
            cu_num_logits=t.tensor([0, 4]), cu_num_logits_np=np.array([0, 4]), num_reqs=1,
            seq_lens=t.tensor([14]), seq_lens_cpu_upper_bound=t.tensor([14]),
            num_tokens=4, has_structured_output_reqs=True)
        def process(logits, expanded, idx, idx_np, pos, ids, local, bounds):
            self.assertEqual(pos.tolist(), [10, 11, 11, 12])
            self.assertEqual(ids.tolist(), [5, 6, 7, 8])
            self.assertEqual(bounds.tolist(), [14])
            seen.append("process")
            return logits
        sampler = NS(compute_nans=False, logprobs_mode="raw_logprobs", use_fp64_gumbel=False,
            return_sampling_mask=False,
            sampling_states=NS(max_num_logprobs=lambda _: NO_LOGPROBS,
                               temperature=NS(gpu=t.ones(1)), seeds=NS(gpu=t.zeros(1))),
            req_states=NS(prefill_len=NS(gpu=t.zeros(1))), apply_sampling_params=process)
        stage = NS(restore_positions=lambda *a: seen.append("restore_positions"),
                   slot_depths=t.tensor([[0, 1, 1, 2]]), slot_parents=t.tensor([[-1, 0, 0, 2]]))
        store = object()
        r = tree.TreeRejectionSampler(sampler, NS(num_speculative_tokens=3, rejection_sample_method="standard"),
                                      "cpu", store, stage)
        def accept(target, ids, cu, idx, parents, actual_store):
            self.assertEqual(ids.tolist(), [5, -1, -1, -1])  # structured-output root-only invalid mask
            self.assertIs(actual_store, store)
            seen.append("tree_accept")
            return t.tensor([[21, -1, -1, -1]]), t.tensor([1])
        with mock.patch.object(tree, "gumbel_sample", return_value=t.tensor([21, 22, 23, 24])), \
             mock.patch.object(tree, "accept_tree", side_effect=accept), \
             mock.patch.object(rejection, "get_num_sampled_and_rejected", side_effect=lambda n, seq, cu, idx, p: (n, cu.diff() - n)):
            output = r(t.zeros(4, 32), batch)
        self.assertEqual(seen, ["restore_positions", "process", "tree_accept"])
        self.assertEqual(output.num_sampled.tolist(), [1])
        self.assertEqual(output.num_rejected.tolist(), [3])

    def test_dummy_profile_reserves_extra_query_slots_without_changing_native(self):
        for extra, expected in ((2, 56), (0, 64)):
            seen = []
            runner = NS(is_encoder_only=False, max_num_reqs=4, max_num_tokens=64,
                speculator=NS(num_extra_query_per_req=extra, num_query_per_req=3),
                is_first_pp_rank=True, is_last_pp_rank=False, lora_config=None,
                kv_connector=NS(set_disabled=lambda disabled: None),
                maybe_dummy_run_with_lora=lambda *a, **kw: nullcontext(),
                execute_model=lambda schedule, **kw: seen.append((schedule, kw)))
            inspect.unwrap(self.runner.GPUModelRunner._dummy_run)(runner, 64, is_profile=True)
            schedule, flags = seen[0]
            self.assertEqual(schedule.total_num_scheduled_tokens, expected)
            self.assertEqual(sum(schedule.num_scheduled_tokens.values()), expected)
            self.assertEqual(len(schedule.num_scheduled_tokens), 4)
            self.assertTrue(flags["dummy_run"])

    def _dummy_state_tables(self, mode="none", width=16, num_blocks=128):
        import vllm.v1.worker.gpu.buffer_utils as buffers
        from vllm.v1.worker.gpu.block_table import BlockTables
        from vllm.v1.kv_cache_interface import MambaSpec, FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
        t = self.torch
        # Only the accelerator-backed UVA allocation is doubled. The actual
        # BlockTables constructor, persistent storage and dummy methods run.
        def cpu_uva(size, dtype):
            cpu = t.zeros(size, dtype=dtype)
            return NS(cpu=cpu, np=cpu.numpy(), uva=cpu)
        with mock.patch.object(buffers, "UvaBuffer", side_effect=cpu_uva):
            tables = BlockTables([16, 16], 4, 64, [32, 32 if mode == "align" else width],
                                 t.device("cpu"), [16, 16])
        state_spec = MambaSpec(block_size=16, shapes=((1,), (1,)), dtypes=(t.float32, t.float32),
                               num_speculative_blocks=width - 1, mamba_cache_mode=mode)
        attention_spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=t.float32)
        cache = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["attention"], kv_cache_spec=attention_spec),
            KVCacheGroupSpec(layer_names=["gdn"], kv_cache_spec=state_spec)])
        return tables, cache

    def test_context_rewrite_invalidates_upstream_first_column_state_slots(self):
        from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers, set_dummy_context
        t = self.torch
        tables, cache = self._dummy_state_tables()
        batch = InputBatch.make_dummy(2, 32, InputBuffers(4, 64, t.device("cpu")), max_query_len=16)
        views = tables.get_dummy_block_tables(2)
        for table in views:
            table[:, 0].copy_(t.tensor([1, 2]))  # exact upstream first-column assignment
        set_dummy_context(batch, tables, 64, cache.num_blocks, 128)
        self.assertEqual(views[1][:, 0].tolist(), [0, 5])  # first request now NULL again
        self.assertEqual(views[1][:, 5:].count_nonzero().item(), 0)  # verify slots also NULL
        self.assertEqual(batch.seq_lens.tolist(), [80, 80])
        self.assertEqual(batch.num_computed_tokens_np.tolist(), [64, 64])
        print("UPSTREAM OVERWRITE: C2 q16 context64 -> roots [0, 5], remaining verify slots include NULL")

    def _calibration_metadata_journey(self, requests, query, flag, mode="none", full=False,
                                      profile=False, method="engine_contract"):
        from vllm.v1.worker.gpu.input_batch import InputBuffers
        from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
        import vllm.v1.attention.backends.gdn_attn as gdn
        t, mod = self.torch, self.runner
        width = 16 if method else 1
        tables, cache = self._dummy_state_tables(mode, width)
        # Inactive persistent rows must not be overwritten by the opt-in.
        for table in tables.input_block_tables:
            table[requests:].fill_(77)
        pointers = [table.data_ptr() for table in tables.input_block_tables]
        spec = NS(method=method, num_speculative_tokens=width - 1, parallel_drafting=False) if method else None
        config = NS(speculative_config=spec, num_speculative_tokens=width - 1,
            cache_config=NS(mamba_cache_mode=mode),
            compilation_config=NS(cudagraph_mode=mod.CUDAGraphMode.FULL if full else mod.CUDAGraphMode.NONE,
                                  max_cudagraph_capture_size=None),
            scheduler_config=NS(max_num_seqs=4), parallel_config=NS(decode_context_parallel_size=1))
        extension = object()
        with mock.patch("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn._resolve_gdn_prefill_backend",
                        return_value=("triton", "triton")), \
             mock.patch("vllm.model_executor.layers.mamba.gdn.verify_extension.resolve_obj_by_qualname",
                        return_value=lambda _: extension):
            builder = gdn.GDNAttentionMetadataBuilder(cache.kv_cache_groups[1].kv_cache_spec,
                                                     ["gdn"], config, t.device("cpu"))
        group = NS(layer_names=["gdn"], get_metadata_builder=lambda _: builder)
        model_state = MambaHybridModelState.__new__(MambaHybridModelState)
        model_state.vllm_config, model_state.max_model_len = config, 128
        model_state.num_accepted_tokens_gpu = t.ones(4, dtype=t.int32)
        model_state.prepare_inputs = lambda *a: {}
        prepare_attn = model_state.prepare_attn
        observed = {}
        def metadata(*args, **kwargs):
            result = prepare_attn(*args, **kwargs)  # actual hybrid -> common -> GDN builders
            observed.update(batch=args[0], metadata=result["gdn"])
            return result
        model_state.prepare_attn = metadata
        plugin = NS(num_extra_query_per_req=0, num_query_per_req=width,
                    prepare_verify_inputs=lambda *a, **kw: None, supports_mm_inputs=False,
                    propose=lambda *a, **kw: None) if method else None
        runner = NS(device=t.device("cpu"), max_num_reqs=4, max_num_tokens=64, max_model_len=128,
            decode_query_len=width, speculator=plugin, speculative_config=spec,
            is_first_pp_rank=True, is_last_pp_rank=True, is_encoder_only=False,
            is_encoder_decoder=False, supports_mm_inputs=False, use_aux_hidden_state_outputs=False,
            lora_config=None, pcp_manager=None, pp_handler=None, dp_size=1, dp_rank=0,
            vllm_config=config, cache_config=config.cache_config, model_config=None,
            model_state=model_state, block_tables=tables, kv_cache_config=cache,
            input_buffers=InputBuffers(4, 64, t.device("cpu")), attn_groups=[[], [group]],
            req_states=NS(last_sampled_tokens=None, next_prefill_tokens=None),
            sampler=NS(sampling_states=NS(temperature=NS(gpu=None), seeds=NS(gpu=None))),
            eplb=NS(prepare_forward=lambda *a: None, step=lambda **kw: None),
            step_timing=NS(record_batch=lambda *a: None, forward_start=lambda: None,
                           forward_end=lambda: None, drafter_start=lambda: None, drafter_end=lambda: None),
            kv_connector=NS(set_disabled=lambda _: None, pre_forward=lambda _: None),
            maybe_dummy_run_with_lora=lambda *a, **kw: nullcontext(),
            gather_batch_req_state=lambda *a: (None, query))
        runner.prepare_dummy_attn = lambda *a, **kw: mod.GPUModelRunner.prepare_dummy_attn(runner, *a, **kw)
        runner.execute_model = lambda *a, **kw: mod.GPUModelRunner.execute_model(runner, *a, **kw)
        runner.model = lambda **kw: t.zeros(requests * query, 2)
        desc = NS(num_tokens=requests * query, num_reqs=requests, max_query_len=query,
                  num_active_loras=0, cg_mode=mod.CUDAGraphMode.FULL if full else mod.CUDAGraphMode.NONE)
        runner.cudagraph_manager = NS(run_fullgraph=lambda _: runner.model())
        flags = {} if flag is None else {"valid_dummy_state_slots": flag}
        with mock.patch.object(mod, "dispatch_cg_and_sync_dp", return_value=(desc, None)), \
             mock.patch.object(mod, "set_forward_context", return_value=nullcontext()), \
             mock.patch("vllm.utils.torch_utils.PIN_MEMORY", False), \
             mock.patch("vllm.v1.attention.backends.utils.PIN_MEMORY", False), \
             mock.patch.object(gdn, "async_tensor_h2d", side_effect=lambda value, **kw: value):
            mod.GPUModelRunner._dummy_run(runner, requests * query, uniform_decode=query > 1,
                                          context_len=64, is_profile=profile, **flags)
        batch, metadata = observed["batch"], observed["metadata"]
        self.assertEqual(batch.seq_lens.tolist(), [64 + query] * requests)
        self.assertEqual(batch.seq_lens_cpu_upper_bound.tolist(), [64 + query] * requests)
        self.assertEqual(batch.num_computed_tokens_np.tolist(), [64] * requests)
        self.assertEqual(batch.positions.tolist(), list(range(64, 64 + query)) * requests)
        self.assertEqual([table.data_ptr() for table in tables.input_block_tables], pointers)
        for table in tables.input_block_tables:
            self.assertTrue((table[requests:] == 77).all())
        self.assertTrue((tables.slot_mappings == -1).all())
        self.assertTrue((runner.input_buffers.seq_lens[requests:] == 0).all())
        self.assertTrue((runner.input_buffers.query_start_loc[requests + 1:] == requests * query).all())
        states = metadata.spec_state_indices_tensor if metadata.num_spec_decodes else metadata.non_spec_state_indices_tensor
        snapshot = dict(states=states.tolist(), spec_decodes=metadata.num_spec_decodes,
                        prefills=metadata.num_prefills, decodes=metadata.num_decodes,
                        attention=tables.input_block_tables[0][:requests].tolist(),
                        state_table=tables.input_block_tables[1][:requests].tolist(),
                        drafts=None if batch.num_draft_tokens_per_req is None else batch.num_draft_tokens_per_req.tolist())
        if flag:
            self.assertTrue((states > 0).all())
            self.assertTrue((states < cache.num_blocks).all())
            self.assertEqual(states.unique().numel(), states.numel())
            self.assertEqual(metadata.num_spec_decodes, requests if query > 1 else 0)
            self.assertEqual(metadata.num_prefills, 0)
            if query > 1:
                self.assertEqual(tuple(states.shape), (requests, width))
                self.assertIs(metadata.verify_extension, extension)
                self.assertEqual(metadata.num_spec_decode_tokens, requests * query)
            # Exercise actual graph-builder padding with an extra zero-length row.
            from vllm.v1.attention.backend import CommonAttentionMetadata
            padded = CommonAttentionMetadata(query_start_loc=t.cat((batch.query_start_loc, batch.query_start_loc[-1:])),
                query_start_loc_cpu=t.cat((batch.query_start_loc.cpu(), batch.query_start_loc[-1:].cpu())),
                seq_lens=t.cat((batch.seq_lens, t.zeros(1, dtype=t.int32))), num_reqs=requests + 1,
                num_actual_tokens=requests * query, max_query_len=query, max_seq_len=64 + query,
                block_table_tensor=t.cat((tables.input_block_tables[1][:requests], t.zeros_like(tables.input_block_tables[1][:1]))),
                slot_mapping=tables.slot_mappings[1, :requests * query], causal=True)
            with mock.patch.object(gdn, "async_tensor_h2d", side_effect=lambda value, **kw: value):
                # Use persistent padded metadata buffers even in the eager case.
                builder.use_full_cuda_graph = True
                padded_meta = builder.build_for_cudagraph_capture(padded)
            padded_states = padded_meta.spec_state_indices_tensor if query > 1 else padded_meta.non_spec_state_indices_tensor
            self.assertEqual(padded_states[-1].count_nonzero().item(), 0)
            self.assertEqual(padded_states[:requests].tolist(), states.tolist())
        return snapshot

    def test_valid_calibration_state_metadata_c1_c2_q1_q16(self):
        self.registry.register_speculative_method("engine_contract", self.registry.SpeculativeMethodSpec(
            "plugin:Spec", linear_attn_verify_extension="plugin:make_extension"))
        for mode in ("none", "all", "align"):
            for requests in (1, 2):
                for query in (1, 16):
                    for full, profile in ((False, False), (False, True), (True, False)):
                        with self.subTest(mode=mode, C=requests, q=query, full=full, profile=profile):
                            default = self._calibration_metadata_journey(requests, query, None, mode, full, profile)
                            disabled = self._calibration_metadata_journey(requests, query, False, mode, full, profile)
                            enabled = self._calibration_metadata_journey(requests, query, True, mode, full, profile)
                            self.assertEqual(default, disabled)
                            self.assertEqual(enabled["attention"], default["attention"])
                            print(f"CALIBRATION C{requests} q{query} {mode=} {full=} {profile=}: "
                                  f"state_slots={enabled['states']} spec_rows={enabled['spec_decodes']} context=64")

    def test_dummy_state_flag_defaults_preserve_native_no_spec_and_mtp(self):
        for method, query in ((None, 1), ("mtp", 16)):
            for full in (False, True):
                with self.subTest(method=method, full=full):
                    default = self._calibration_metadata_journey(2, query, None, full=full, method=method)
                    disabled = self._calibration_metadata_journey(2, query, False, full=full, method=method)
                    self.assertEqual(default, disabled)
                    self.assertIsNone(default["drafts"])

    def test_valid_dummy_state_slots_refuse_pool_aliasing(self):
        from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
        tables, cache = self._dummy_state_tables(num_blocks=32)  # needs 32 non-NULL + NULL
        batch = InputBatch.make_dummy(2, 32, InputBuffers(4, 64, self.torch.device("cpu")), max_query_len=16)
        runner = NS(block_tables=tables, kv_cache_config=cache, pcp_manager=None,
                    max_model_len=128, cache_config=NS(mamba_cache_mode="none"), speculative_config=None)
        with self.assertRaisesRegex(ValueError, "Insufficient distinct Mamba state slots"):
            self.runner.GPUModelRunner.prepare_dummy_attn(runner, batch, True, context_len=64)

    def test_real_runner_verify_commit_propose_finalize_order(self):
        self._runner_journey(plugin=True)
        self._runner_journey(plugin=False)
        self._runner_journey(plugin=True, dummy=True, full_graph=True)
        self._runner_journey(plugin=True, dummy=True, full_graph=False)

    def _runner_journey(self, plugin, dummy=False, full_graph=False):
        t, mod = self.torch, self.runner
        events = []
        def mark(name, result=None):
            def call(*args, **kwargs):
                events.append(name)
                return result
            return call
        batch = NS(num_tokens=5, num_tokens_after_padding=5, num_reqs=1, req_ids=["r"],
                   input_ids=t.arange(5), positions=t.arange(10, 15), is_padding=False,
                   idx_mapping=t.tensor([0]), query_start_loc=t.tensor([0, 5]))
        kv, gdn = t.arange(100, 105), t.arange(200, 205)
        output = NS(sampled_token_ids=t.tensor([[7, 8, 9, -1, -1]]))
        accepted, rejected = t.tensor([3]), t.tensor([2])
        state = NS(num_computed_tokens=NS(gpu=t.tensor([10])), all_token_ids=NS(gpu=None),
                   prompt_len=NS(np=None), last_sampled_tokens=None, next_prefill_tokens=None,
                   draft_tokens=t.zeros((1, 4), dtype=t.int64))
        def prepare_verify(b, inputs, *, dummy_run):
            events.append("verify")
            self.assertIs(b, batch)
            self.assertEqual(dummy_run, dummy)
            self.assertEqual(events[-3:-1], ["metadata", "model_inputs"])
            inputs["positions"].copy_(t.tensor([10, 11, 11, 12, 12]))
        def target(**inputs):
            events.append("forward")
            self.assertEqual(inputs["positions"].tolist(), [10, 11, 11, 12, 12] if plugin else list(range(10, 15)))
            return t.ones((5, 2))
        def post_accept(b, sampler_output, num_sampled, num_rejected):
            events.append("post_accept")
            self.assertIs(sampler_output, output)
            self.assertIs(num_sampled, accepted)
            self.assertIs(num_rejected, rejected)
            self.assertEqual(state.num_computed_tokens.gpu.item(), 10)
            # Non-prefix accepted parent path; copy via scratch to avoid overlaps.
            path = t.tensor([0, 2, 4])
            kv[:3] = kv[path].clone()
            gdn[:3] = gdn[path].clone()
        def advance(*args):
            events.append("advance")
            if plugin:
                self.assertEqual(kv[:3].tolist(), [100, 102, 104])
                self.assertEqual(gdn[:3].tolist(), [200, 202, 204])
            state.num_computed_tokens.gpu += 5 - rejected
        def propose(*args, **kwargs):
            events.append("propose")
            self.assertEqual(state.num_computed_tokens.gpu.item(), 13)
            self.assertEqual(gdn[2].item(), 204)
            return t.tensor([[1, 2, 3, 4]])
        sp = NS(prepare_verify_inputs=prepare_verify, post_accept=post_accept, propose=propose,
                supports_mm_inputs=False) if plugin else None
        runner = NS(speculator=sp, is_first_pp_rank=True, is_last_pp_rank=True,
            is_encoder_only=False, is_encoder_decoder=False, supports_mm_inputs=False,
            use_aux_hidden_state_outputs=False, lora_config=None, pcp_manager=None,
            pp_handler=None, dp_size=1, dp_rank=0, vllm_config=None, model_config=None,
            model=target, req_states=state, adaptive_verification=None, kv_cache_config=None,
            input_buffers=None, attn_groups=None, num_speculative_steps=4 if plugin else 0,
            main_stream=None, output_copy_stream=None, check_ep_fault=None,
            routed_experts_capturer=None, block_tables=NS(apply_staged_writes=lambda: None),
            update_pp_decode_requests=lambda: None, finish_requests=lambda _: None,
            free_states=lambda _: None, add_requests=lambda _: None, update_requests=lambda _: None,
            gather_batch_req_state=lambda *a: (None if dummy else NS(num_tokens=5), 5),
            prepare_inputs=lambda *a: batch, prepare_attn=mark("slots", (object(), object())),
            prepare_dummy_attn=mark("slots", (object(), object())),
            model_state=NS(preprocess_state=lambda *a: None,
                           prepare_attn=mark("metadata", {}), prepare_inputs=mark("model_inputs", {})),
            eplb=NS(prepare_forward=lambda *a: None),
            step_timing=NS(record_batch=lambda *a: None, forward_start=lambda: None),
            kv_connector=NS(pre_forward=mark("pre_forward"), post_forward=mark("post_forward", "kv_output")),
            sample=mark("sample", (output, accepted, rejected)),
            sampler=NS(sampling_states=NS(temperature=NS(gpu=None), seeds=NS(gpu=None))),
            prompt_logprobs_worker=NS(compute_prompt_logprobs=lambda *a: {}),
            postprocess_sampled=advance, draft_tokens_handler=NS(set_draft_tokens=lambda *a: None))
        target.compute_logits = None
        desc = NS(num_tokens=5, num_reqs=1, max_query_len=5, num_active_loras=0,
                  cg_mode=mod.CUDAGraphMode.FULL if full_graph else mod.CUDAGraphMode.NONE)
        runner.cudagraph_manager = NS(run_fullgraph=lambda desc: target(positions=batch.positions))
        schedule = NS(num_scheduled_tokens={"r": 5}, total_num_scheduled_tokens=5, finished_req_ids=set())
        with mock.patch.multiple(mod, dispatch_cg_and_sync_dp=lambda *a, **kw: (desc, None),
                                 build_slot_mappings_by_layer=lambda *a: {},
                                 set_forward_context=lambda *a, **kw: nullcontext(),
                                 AsyncOutput=lambda **kw: NS(**kw)), \
             mock.patch.object(mod.pcp, "maybe_restore_pcp_for_sampling", side_effect=lambda p, h, b: (h, b)), \
             mock.patch.object(mod.InputBatch, "make_dummy", return_value=batch):
            inspect.unwrap(mod.GPUModelRunner.execute_model)(runner, schedule, dummy_run=dummy)
            if dummy:
                self.assertEqual(events, ["slots", "metadata", "model_inputs", "verify", "pre_forward", "forward"])
                self.assertIs(runner.execute_model_state.input_batch, batch)
                return
            result = inspect.unwrap(mod.GPUModelRunner.sample_tokens)(runner, None)
        self.assertEqual(result.model_runner_output.kv_connector_output, "kv_output")
        self.assertEqual(events.count("post_forward"), 1)
        expected = ["slots", "metadata", "model_inputs"]
        if plugin:
            expected += ["verify"]
        expected += ["pre_forward", "forward", "sample"]
        if plugin:
            expected += ["post_accept"]
        expected += ["advance"]
        if plugin:
            expected += ["propose"]
        expected += ["post_forward"]
        self.assertEqual(events, expected)

    def test_actual_gdn_builder_carries_extension(self):
        from vllm.model_executor.layers.mamba.gdn.verify_extension import create_linear_attn_verify_extension
        import vllm.v1.attention.backends.gdn_attn as gdn
        t = self.torch
        extension = object()
        self.registry.register_speculative_method("engine_contract", self.registry.SpeculativeMethodSpec(
            "plugin:Spec", linear_attn_verify_extension="plugin:make_extension"))
        vc = NS(speculative_config=NS(method="engine_contract", num_speculative_tokens=3, parallel_drafting=False),
                cache_config=NS(mamba_cache_mode="all"),
                compilation_config=NS(cudagraph_mode=self.runner.CUDAGraphMode.NONE, max_cudagraph_capture_size=None),
                scheduler_config=NS(max_num_seqs=1), parallel_config=NS(decode_context_parallel_size=1))
        with mock.patch("vllm.model_executor.layers.mamba.gdn.verify_extension.resolve_obj_by_qualname",
                        return_value=lambda config: extension), \
             mock.patch("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn._resolve_gdn_prefill_backend",
                        return_value=("triton", "triton")):
            self.assertIs(create_linear_attn_verify_extension(vc), extension)
            b = gdn.GDNAttentionMetadataBuilder(None, ["layer"], vc, t.device("cpu"))
            self.assertIs(b.verify_extension, extension)
        self.assertIsNone(create_linear_attn_verify_extension(NS(speculative_config=None)))
        q = t.tensor([0, 4], dtype=t.int32)
        m = NS(query_start_loc=q, query_start_loc_cpu=q, block_table_tensor=t.tensor([[0, 1, 2, 3]]),
               seq_lens=t.tensor([12]), num_reqs=1, num_actual_tokens=4)
        with mock.patch.object(gdn, "mamba_get_block_table_tensor", side_effect=lambda table, *a: table), \
             mock.patch.object(gdn, "async_tensor_h2d", side_effect=lambda value, **kw: value):
            metadata = b.build(0, m, t.tensor([2]), t.tensor([3]))
        self.assertIs(metadata.verify_extension, extension)
        self.assertEqual(metadata.num_accepted_tokens.tolist(), [2])
        self.assertEqual(metadata.spec_state_indices_tensor.tolist(), [[0, 1, 2, 3]])

    def test_actual_xpu_gdn_extension_and_native_paths(self):
        self._gdn_journey(mixed=False, extension=True)
        self._gdn_journey(mixed=True, extension=True)
        self._gdn_journey(mixed=False, extension=False)
        self._gdn_journey(mixed=False, extension=True, common=True)

    def _gdn_journey(self, mixed, extension, common=False):
        import vllm._xpu_ops as xpu
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
        from vllm.model_executor.layers.mamba.gdn.verify_extension import LinearAttnVerifyExtension
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
        t, calls = self.torch, []
        class Verify(LinearAttnVerifyExtension):
            def forward_spec_rows(_, rows):
                calls.append(("extension", rows))
                # Deliberately branched CPU recurrence: siblings read the same
                # parent, not the previous row. Device kernels are tested elsewhere.
                initial = rows.ssm_state[0, 0, 0, 0].clone()
                parents = [-1, 0, 0, 2]
                states = []
                for i, parent in enumerate(parents):
                    state = (initial if parent < 0 else states[parent]) + rows.mixed_qkv[i, 0]
                    states.append(state)
                rows.out[:, 0, 0] = t.stack(states)
                rows.ssm_state[:4, 0, 0, 0] = t.stack(states)
                rows.conv_state[:4, 0, 0] = t.stack(states)
        count = 6 if mixed else 4
        projected = t.arange(count * 4, dtype=t.float32).reshape(count, 4)
        ba = t.ones(count, 2)
        out, z = t.full((count, 1, 1), -99.0), t.full((count, 1, 1), -99.0)
        layer = NS(prefix="layer", gqa_interleaved_layout=False, key_dim=1, value_dim=1,
                   tp_size=1, num_k_heads=1, num_v_heads=1, head_k_dim=1, head_v_dim=1,
                   kv_cache=(t.zeros(8, 3, 2), t.zeros(8, 1, 1, 1)),
                   conv1d=NS(weight=t.ones(3, 1, 3), bias=None), activation="silu",
                   A_log=t.zeros(1), dt_bias=t.zeros(1))
        layer._forward_spec_rows = lambda *a: QwenGatedDeltaNetAttention._forward_spec_rows(layer, *a)
        meta = GDNAttentionMetadata(num_prefills=int(mixed), num_prefill_tokens=2 if mixed else 0,
            num_decodes=0, num_decode_tokens=0, num_spec_decodes=1, num_spec_decode_tokens=4,
            num_actual_tokens=count, verify_extension=Verify() if extension else None,
            spec_sequence_masks=t.tensor([True, False] if mixed else [True]),
            spec_query_start_loc=t.tensor([0, 4]), spec_token_indx=t.tensor([0, 2, 3, 5] if mixed else [0, 1, 2, 3]),
            non_spec_token_indx=t.tensor([1, 4]) if mixed else None,
            non_spec_query_start_loc=t.tensor([0, 2]) if mixed else None,
            non_spec_state_indices_tensor=t.tensor([7]) if mixed else None,
            has_initial_state=t.tensor([True]) if mixed else None,
            spec_state_indices_tensor=t.tensor([[0, 1, 2, 3]]), num_accepted_tokens=t.tensor([1]))
        def native(o, zz, qkvz, gates, *args, **kwargs):
            calls.append(("native", kwargs))
            if extension:
                self.assertTrue(mixed)
                self.assertEqual(kwargs["num_spec_decodes"], 0)
                self.assertIsNone(kwargs["spec_state_indices_tensor"])
                self.assertEqual(qkvz.tolist(), projected[[1, 4]].tolist())
            o.fill_(777)
            zz.copy_(qkvz[:, 3:].reshape_as(zz))
        context = NS(no_compile_layers={"layer": layer}, attn_metadata={"layer": meta})
        with mock.patch("vllm.forward_context.get_forward_context", return_value=context), \
             mock.patch("vllm.model_executor.layers.mamba.gdn.verify_extension_xpu.is_conv_state_dim_first", return_value=True), \
             mock.patch.object(t.ops._xpu_C, "gdn_attention", side_effect=native, create=True):
            if common:
                import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as core
                layer.enable_packed_recurrent_decode = False
                layer.rearrange_mixed_qkv = lambda qkv: (None, None, None) if qkv is None else tuple(qkv.chunk(3, dim=-1))
                z.copy_(projected[:, 3:].reshape_as(z))
                with mock.patch.object(core, "get_forward_context", return_value=context), \
                     mock.patch.object(core, "is_conv_state_dim_first", return_value=True), \
                     mock.patch.object(core, "causal_conv1d_update", side_effect=AssertionError("chain conv called")), \
                     mock.patch.object(core, "fused_sigmoid_gating_delta_rule_update", side_effect=AssertionError("chain GDN called")):
                    core.QwenGatedDeltaNetAttention._forward_core(layer, projected[:, :3], ba[:, :1], ba[:, 1:], out)
            else:
                xpu._gdn_attention_core_xpu_impl(out, z, projected, ba, "layer")
        self.assertEqual(z.flatten().tolist(), projected[:, 3].tolist())
        if extension:
            rows = calls[0][1]
            self.assertIs(rows.state_indices, meta.spec_state_indices_tensor)
            self.assertIs(rows.num_accepted_tokens, meta.num_accepted_tokens)
            self.assertIs(rows.query_start_loc, meta.spec_query_start_loc)
            self.assertEqual([name for name, _ in calls], ["extension", "native"] if mixed else ["extension"])
            expected = [0., 777., 8., 12., 777., 32.] if mixed else [0., 4., 8., 20.]
            self.assertEqual(out.flatten().tolist(), expected)
        else:
            self.assertEqual([name for name, _ in calls], ["native"])
            self.assertEqual(calls[0][1]["num_spec_decodes"], 1)
            self.assertEqual(out.flatten().tolist(), [777.] * count)


if __name__ == "__main__":
    unittest.main()
