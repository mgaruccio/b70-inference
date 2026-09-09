"""Focused offline tests for the pinned research overlay (no vLLM/XPU launch).

B70_DFLASH2_SOURCE_ROOT must point at a pristine exported vllm package, or an
already-patched one. The default is the lead's actual installed-image export.
No vendored vLLM copy or network downloads. Source-dependent tests explicitly
skip if the export is absent; tensor tests additionally require CPU PyTorch.
Run: python -m unittest discover -s tests -p test_qwen38_dflash2_bf16.py -v
The lead separately owns real-kernel XPU and public serving validation.
"""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass, replace
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:
    torch = nn = F = None

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "scripts/patch-vllm-qwen38-dflash2-bf16.py"
spec = importlib.util.spec_from_file_location("dflash2_overlay", PATCH)
overlay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay)
SOURCE_ROOT = Path(os.environ.get("B70_DFLASH2_SOURCE_ROOT", "/tmp/qwen38-dflash2-runtime-source"))


def exported_sources():
    if not all((SOURCE_ROOT / name).is_file() for name in overlay.PINNED_SHA256):
        raise unittest.SkipTest("export pinned image sources and set B70_DFLASH2_SOURCE_ROOT")
    return {name: (SOURCE_ROOT / name).read_text() for name in overlay.PINNED_SHA256}


def execute(nodes, scope):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])),
                 "actual_pinned_source", "exec"), scope)


def actual_class(source, name, methods, scope, base=None):
    node = deepcopy(next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name))
    node.decorator_list = []
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    for method in node.body:
        method.decorator_list = []
    if base:
        node.bases = [ast.parse(base, mode="eval").body]
    execute([node], scope)
    return scope[name]


def actual_function(source, name, scope):
    node = deepcopy(next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    execute([node], scope)
    return scope[name]


class OverlaySourceTests(unittest.TestCase):
    def test_explicit_flag_required_without_writes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            sentinel = Path(tmp) / "sentinel"
            sentinel.write_text("unchanged")
            env = {k: v for k, v in os.environ.items() if k != "B70_DFLASH2_BF16"}
            result = subprocess.run([sys.executable, str(PATCH), "--root", tmp],
                                    env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("explicit B70_DFLASH2_BF16=1", result.stderr)
            self.assertEqual(list(Path(tmp).iterdir()), [sentinel])
            self.assertEqual(sentinel.read_text(), "unchanged")

    def test_actual_source_apply_and_replay(self):
        sources = exported_sources()
        expected = overlay.prepare(sources)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            for name, source in sources.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source)
            command = [sys.executable, str(PATCH), "--root", tmp]
            env = {**os.environ, "B70_DFLASH2_BF16": "1"}
            for _ in range(2):
                result = subprocess.run(command, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                for name, source in expected.items():
                    self.assertEqual((root / name).read_text(), source)
                    ast.parse(source)
            self.assertIn("already applied", result.stdout)
        self.assertEqual(overlay.prepare(expected), expected)
        self.assertNotIn("DFLASH_GPTQ_CONTEXT_KV_FALLBACK", expected["model_executor/models/qwen3_dflash.py"])

    def test_unknown_anchor_and_damaged_replay_do_not_write_any_file(self):
        sources = exported_sources()
        for initial in (sources, overlay.prepare(sources)):
            with self.subTest(patched=initial != sources), tempfile.TemporaryDirectory(dir=ROOT) as tmp:
                root = Path(tmp)
                changed = dict(initial)
                # Last file checked: failure must not partially patch earlier files.
                changed["model_executor/models/registry.py"] += "\n# drift\n"
                for name, source in changed.items():
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(source)
                with self.assertRaises(RuntimeError):
                    overlay.apply(root)
                self.assertEqual({n: (root / n).read_text() for n in changed}, changed)
        broken = overlay.prepare(sources)
        name = "model_executor/models/qwen3_dflash2.py"
        broken[name] = broken[name].replace('"candidate.logits"', '"different.logits"')
        with self.assertRaisesRegex(RuntimeError, "incompatible source"):
            overlay.prepare(broken)

    def test_partial_application_rejected(self):
        pristine = exported_sources()
        patched = overlay.prepare(pristine)
        if pristine == patched:
            self.skipTest("partial-install test needs pristine export")
        mixed = dict(pristine)
        mixed["config/speculative.py"] = patched["config/speculative.py"]
        with self.assertRaisesRegex(RuntimeError, "partially applied"):
            overlay.prepare(mixed)

    def test_no_argument_root_resolution_uses_importlib(self):
        fake_spec = NS(submodule_search_locations=["/installed/vllm"])
        with patch.object(overlay.importlib.util, "find_spec", return_value=fake_spec), \
                patch.object(overlay, "apply") as apply, \
                patch.dict(os.environ, {"B70_DFLASH2_BF16": "1"}), \
                patch.object(sys, "argv", ["/overlay.py"]):
            overlay.main()
        apply.assert_called_once_with(Path("/installed/vllm"))


@unittest.skipIf(torch is None, "CPU PyTorch required; lead can run in pinned image")
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = overlay.prepare(exported_sources())

    def setUp(self):
        torch.manual_seed(12)
        self.scope = {"torch": torch, "nn": nn, "F": F, "_B70_DFLASH2_AUDIT": False}
        actual_function(self.sources["model_executor/models/qwen3_dflash.py"], "_b70_audit_tensor", self.scope)

    def config(self, enabled=True, audit=False):
        scope = {}
        exec(overlay.CONFIG_HELPERS, scope)
        scope.update(_B70_DFLASH2_BF16=enabled, _B70_DFLASH2_AUDIT=audit)
        hf = NS(architectures=["DFlash2DraftModel"], dtype="bfloat16", num_hidden_layers=5,
                hidden_size=5120, vocab_size=248320, tie_word_embeddings=False,
                dflash_config=dict(block_size=8, selector_rank=256, selector_top_k=16,
                                   conv_kernel_size=2, conv_group_size=16,
                                   target_layer_ids=[5, 19, 33, 47, 61]))
        target = NS(dtype=torch.float16, quantization="gptq", head_dtype=None,
                    enforce_eager=True, get_vocab_size=lambda: 248320, get_hidden_size=lambda: 5120)
        spec = NS(target_model_config=target, draft_model_config=NS(hf_config=hf, dtype=torch.bfloat16,
                  quantization=None), method="dflash", quantization=None, num_speculative_tokens=7,
                  draft_sample_method="greedy", rejection_sample_method="standard", use_heterogeneous_vocab=False,
                  use_local_argmax_reduction=False, kv_cache_dtype="auto", enforce_eager=None)
        return scope, spec

    def test_config_guards_and_dtype_do_not_modify_target(self):
        scope, spec = self.config()
        platform = NS(current_platform=NS(is_xpu=lambda: True))
        with patch.dict(sys.modules, {"vllm.platforms": platform}), patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "0"}):
            scope["_b70_dflash2_validate"](spec)
            vc = NS(speculative_config=spec, model_config=spec.target_model_config)
            self.assertEqual(scope["_b70_dflash2_dtype"](vc), torch.bfloat16)
            self.assertEqual(vc.model_config.dtype, torch.float16)
            for attr, value in (("method", "eagle3"), ("quantization", "gptq"),
                                ("num_speculative_tokens", 8), ("draft_sample_method", "probabilistic"),
                                ("rejection_sample_method", "synthetic"), ("kv_cache_dtype", None),
                                ("use_heterogeneous_vocab", True), ("use_local_argmax_reduction", True)):
                with self.subTest(attr=attr):
                    invalid = deepcopy(spec)
                    setattr(invalid, attr, value)
                    with self.assertRaises(ValueError):
                        scope["_b70_dflash2_validate"](invalid)
            invalid = deepcopy(spec)
            invalid.draft_model_config.hf_config.architectures = ["DFlashDraftModel"]
            with self.assertRaises(ValueError):
                scope["_b70_dflash2_validate"](invalid)
            platform.current_platform.is_xpu = lambda: False
            with self.assertRaises(ValueError):
                scope["_b70_dflash2_validate"](spec)
        scope["_B70_DFLASH2_BF16"] = False
        self.assertEqual(scope["_b70_dflash2_dtype"](vc), torch.float16)
        scope["_b70_dflash2_validate"](spec)  # Disabled overlay does not validate/change ordinary runs.

    def test_normalized_xpu_gptq_alias_and_explicit_runner_selection(self):
        scope, spec = self.config()
        spec.target_model_config.quantization = "auto_gptq"
        platform = NS(current_platform=NS(is_xpu=lambda: True))
        with patch.dict(sys.modules, {"vllm.platforms": platform}), \
                patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "0"}):
            scope["_b70_dflash2_validate"](spec)
            for value in (None, "1"):
                if value is None:
                    os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
                else:
                    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = value
                with self.assertRaisesRegex(ValueError, "legacy"):
                    scope["_b70_dflash2_validate"](spec)
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
            spec.target_model_config.quantization = "awq"
            with self.assertRaisesRegex(ValueError, "GPTQ"):
                scope["_b70_dflash2_validate"](spec)


    def test_legacy_support_gate_requires_opt_in_selector_overlay(self):
        scope, spec = self.config()
        source = ast.parse(self.sources["config/vllm.py"])
        gate = next(n for n in ast.walk(source) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "self._is_dflash2_draft()")
        module = NS(_b70_dflash2_enabled=scope["_b70_dflash2_enabled"])
        for enabled, expected in ((False, ["dflash2 drafts"]), (True, [])):
            scope["_B70_DFLASH2_BF16"] = enabled
            runtime = {"self": NS(_is_dflash2_draft=lambda: True, speculative_config=spec),
                       "unsupported": []}
            with patch.dict(sys.modules, {"vllm.config.speculative": module}):
                execute([deepcopy(gate)], runtime)
            self.assertEqual(runtime["unsupported"], expected)

    def test_audit_requires_eager_and_finite_checks_are_off_by_default(self):
        scope, spec = self.config(audit=True)
        with patch.dict(sys.modules, {"vllm.platforms": NS(current_platform=NS(is_xpu=lambda: True))}), \
                patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "0"}):
            for target_eager, draft_eager in ((False, None), (True, False)):
                spec.target_model_config.enforce_eager = target_eager
                spec.enforce_eager = draft_eager
                with self.assertRaisesRegex(ValueError, "eager"):
                    scope["_b70_dflash2_validate"](spec)
        audit = self.scope["_b70_audit_tensor"]
        with patch.object(torch, "isfinite", side_effect=AssertionError("synchronous check")), redirect_stdout(io.StringIO()) as output:
            audit("disabled", torch.tensor([float("inf")]))
        self.assertEqual(output.getvalue(), "")
        self.scope["_B70_DFLASH2_AUDIT"] = True
        with redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                audit("bad", torch.tensor([float("inf")]))
        self.assertIn("finite=False", output.getvalue())
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "torch.compile"):
                audit("compiled", torch.ones(1))

    def models(self):
        base_source = self.sources["model_executor/models/qwen3_dflash.py"]
        draft_source = self.sources["model_executor/models/qwen3_dflash2.py"]
        actual_class(base_source, "DFlashQwen3Model", {"embed_input_ids"}, self.scope, "nn.Module")
        model_cls = actual_class(draft_source, "DFlash2Qwen3Model", {"embed_input_ids"}, self.scope)
        actual_class(base_source, "DFlashQwen3ForCausalLM", {"combine_hidden_states", "compute_logits"}, self.scope, "nn.Module")
        draft_cls = actual_class(draft_source, "DFlash2Qwen3ForCausalLM",
                                {"combine_hidden_states", "compute_logits", "compute_candidates", "_b70_head_input", "_b70_validate_shared"}, self.scope)
        model = model_cls()
        model._b70_dflash2_bf16 = True
        model.embed_tokens = nn.Embedding(16, 4, dtype=torch.float16)
        model.has_separate_mask_embedding = True
        model.mask_token_id = 15
        model.mask_embedding = nn.Parameter(torch.full((4,), 1e6, dtype=torch.bfloat16))
        model.input_embedding_scale = 1.25
        model.use_aux_hidden_state = True
        model.fc = nn.Linear(6, 4, bias=False, dtype=torch.bfloat16)
        model.fc.input_size = 6
        model.candidate_selector = NS(top_k=2)
        draft = draft_cls()
        draft.model = model
        draft.lm_head = nn.Linear(4, 16, bias=False, dtype=torch.float16)
        draft.draft_id_to_target_id = None
        draft._b70_shared_checked = True

        class Processor:
            head_dtype = None
            def __init__(self, scale):
                self.scale = scale
                self.calls = []
            def __call__(self, head, hidden):
                self.calls.append((head, hidden.dtype))
                return F.linear(hidden, head.weight).float() * self.scale
            def get_top_k_tokens(self, head, hidden, k):
                values, ids = self(head, hidden).topk(k)
                return ids, values
        draft.logits_processor = Processor(2.0)
        draft.candidate_logits_processor = Processor(3.0)
        return draft

    def test_real_embedding_aux_logits_candidates_boundaries_and_target_identity(self):
        draft = self.models()
        head, embedding = draft.lm_head, draft.model.embed_tokens
        weights = [p.detach().clone() for p in (head.weight, embedding.weight)]
        pointers = [p.data_ptr() for p in (head.weight, embedding.weight)]
        x = torch.randn(3, 6, dtype=torch.float16)
        x_before = x.clone()
        hidden = draft.combine_hidden_states(x)
        self.assertEqual(hidden.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(x, x_before))
        self.assertTrue(torch.equal(hidden, draft.model.fc(x.bfloat16())))
        embeds = draft.model.embed_input_ids(torch.tensor([0, 15]))
        self.assertEqual(embeds.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(embeds).all())  # Mask never makes an FP16 round trip.
        logits = draft.compute_logits(hidden)
        ids, values = draft.compute_candidates(hidden)
        expected = F.linear(hidden.half(), head.weight).float()
        self.assertTrue(torch.equal(logits, expected * 2))
        self.assertTrue(torch.equal(values, (expected * 3).topk(2).values))
        self.assertEqual(draft.logits_processor.calls, [(head, torch.float16)])
        self.assertEqual(draft.candidate_logits_processor.calls, [(head, torch.float16)])
        self.assertIs(draft.lm_head, head)
        self.assertIs(draft.model.embed_tokens, embedding)
        for i, param in enumerate((head.weight, embedding.weight)):
            self.assertEqual(param.data_ptr(), pointers[i])
            self.assertEqual(param.dtype, torch.float16)
            self.assertTrue(torch.equal(param, weights[i]))
        draft.model._b70_dflash2_bf16 = False
        self.assertIs(draft._b70_head_input(hidden), hidden)

    def test_actual_sharing_methods_preserve_target_objects(self):
        draft = self.models()
        target = nn.Module()
        target.model = nn.Module()
        target.model.embed_tokens = nn.Embedding(16, 4, dtype=torch.float16)
        target.lm_head = nn.Linear(4, 16, bias=False, dtype=torch.float16)
        saved = [p.detach().clone() for p in target.parameters()]
        scope = {"torch": torch, "nn": nn, "logger": Mock(), "get_pp_group": lambda: NS(world_size=1)}
        proposer_cls = actual_class(self.sources["v1/spec_decode/llm_base_proposer.py"], "SpecDecodeBaseProposer",
                                   {"_maybe_share_embeddings", "_maybe_share_lm_head"}, scope, "object")
        proposer = proposer_cls()
        proposer.model = draft
        proposer.vllm_config = NS(speculative_config=None)
        proposer.use_local_argmax_reduction = False
        proposer._maybe_share_embeddings(target)
        proposer._maybe_share_lm_head(target)
        self.assertIs(draft.lm_head, target.lm_head)
        self.assertIs(draft.model.embed_tokens, target.model.embed_tokens)
        for before, after in zip(saved, target.parameters()):
            self.assertTrue(torch.equal(before, after))
            self.assertEqual(after.dtype, torch.float16)

    def test_loaded_parameter_audit_rejects_wrong_dtype_without_target_conversion(self):
        draft = self.models()
        class Unquantized:
            pass
        modules = {
            "vllm.model_executor.layers.linear": NS(UnquantizedLinearMethod=Unquantized),
            "vllm.model_executor.layers.vocab_parallel_embedding": NS(UnquantizedEmbeddingMethod=Unquantized),
        }
        draft.lm_head.quant_method = Unquantized()
        draft.model.embed_tokens.quant_method = Unquantized()
        target = NS(lm_head=draft.lm_head, model=NS(embed_tokens=draft.model.embed_tokens))
        attn = NS(dtype=torch.bfloat16, kv_cache_torch_dtype=torch.bfloat16,
                  kv_cache_dtype="bfloat16", impl=NS(kv_cache_dtype="bfloat16"))
        draft.model.layers = [NS(self_attn=NS(attn=attn))]
        for name in ("_fused_kv_weight", "_k_norm_weights", "_hidden_norm_weight"):
            setattr(draft.model, name, torch.ones(2, 4, dtype=torch.bfloat16))
        self.scope["_B70_DFLASH2_AUDIT"] = True
        with patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()) as output:
            draft._b70_validate_shared(target)
            self.assertTrue(draft._b70_shared_checked)
            self.assertEqual(attn.kv_cache_torch_dtype, torch.bfloat16)
            self.assertEqual(attn.kv_cache_dtype, "auto")
            self.assertEqual(attn.impl.kv_cache_dtype, "auto")
            attn.impl.kv_cache_dtype = "fp8"
            with self.assertRaisesRegex(RuntimeError, "backend KV dispatch"):
                draft._b70_validate_shared(target)
            attn.impl.kv_cache_dtype = "auto"
            draft.model.fc.weight = nn.Parameter(draft.model.fc.weight.half())
            with self.assertRaisesRegex(RuntimeError, "loaded parameter model.fc.weight"):
                draft._b70_validate_shared(target)
            self.assertEqual(target.lm_head.weight.dtype, torch.float16)
            self.assertIs(draft.lm_head, target.lm_head)
            draft.model.fc.weight = nn.Parameter(draft.model.fc.weight.bfloat16())
            draft.candidate_logits_processor.head_dtype = torch.bfloat16
            with self.assertRaisesRegex(RuntimeError, "would copy/cast"):
                draft._b70_validate_shared(target)
        self.assertIn("shared=True dtype=torch.float16", output.getvalue())
        self.assertIn("parameter.model.fc.weight: dtype=torch.bfloat16", output.getvalue())

    def test_native_checkpoint_load_rejects_fp16_and_private_vocab_tensors(self):
        class Base:
            def load_weights(self, weights):
                return list(weights)
        scope = {"torch": torch, "DFlashQwen3ForCausalLM": Base}
        cls = actual_class(self.sources["model_executor/models/qwen3_dflash2.py"],
                           "DFlash2Qwen3ForCausalLM", {"load_weights"}, scope)
        draft = cls()
        draft.model = NS(_b70_dflash2_bf16=True)
        tensor = torch.ones(2, 2, dtype=torch.bfloat16)
        self.assertEqual(draft.load_weights([("fc.weight", tensor)])[0][0], "fc.weight")
        for name, weight in (("fc.weight", tensor.half()), ("embed_tokens.weight", tensor),
                             ("lm_head.weight", tensor), ("d2t", tensor)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                draft.load_weights([(name, weight)])

    def test_actual_grouped_conv_intermediates_and_selector_projection_are_bf16(self):
        source = self.sources["model_executor/models/qwen3_dflash2.py"]
        grouped_conv = actual_function(source, "_grouped_conv", self.scope)
        actual_function(source, "_score_edges", self.scope)
        actual_class(source, "CandidateSelector", {"forward"}, self.scope)
        selector = self.scope["CandidateSelector"]()
        selector.top_k = 2
        selector.hidden_projection = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
        selector.predecessor_codebook = nn.Parameter(torch.randn(16, 3, dtype=torch.bfloat16))
        selector.successor_codebook = nn.Parameter(torch.randn(16, 3, dtype=torch.bfloat16))
        self.scope["_B70_DFLASH2_AUDIT"] = True
        with redirect_stdout(io.StringIO()) as output:
            x = torch.randn(16, 4, dtype=torch.bfloat16)
            delta = torch.randn(16, 2, 2, dtype=torch.bfloat16)
            base = torch.randn(2, 4, dtype=torch.bfloat16)
            result = grouped_conv(x, delta, base, 8, 2, 2, 2)
            self.assertEqual(result.dtype, torch.bfloat16)
            coefficients = base.reshape(1, 2, 2, 2) + delta.unsqueeze(-1)
            blocks = x.reshape(16, 2, 2)
            expected = coefficients[:, 0] * blocks
            expected[1:] += coefficients[1:, 1] * blocks[:-1] * (torch.arange(1, 16) % 8 != 0).reshape(-1, 1, 1)
            self.assertTrue(torch.equal(result, expected.flatten(-2)))
            scores = selector(torch.randint(0, 16, (2, 7, 2)), torch.ones(2, 7, 2),
                              torch.randn(2, 7, 4, dtype=torch.bfloat16), torch.tensor([1, 3]))
            self.assertEqual(scores.shape, (2, 7, 2, 2))
            self.assertTrue(torch.isfinite(scores).all())
        for name in ("conv.hidden", "conv.delta", "conv.base", "conv.coefficients", "conv.output", "selector.projection", "selector.scores"):
            self.assertIn(name, output.getvalue())

    def test_actual_greedy_selector_walk_is_predecessor_conditioned(self):
        scope = {"torch": torch, "_b70_dflash2_enabled": lambda _: True, "SpecDecodeBaseProposer": object}
        cls = actual_class(self.sources["v1/spec_decode/dflash.py"], "DFlashProposer", {"_sample_draft_tokens"}, scope)
        proposer = cls()
        proposer.speculative_config = NS(draft_sample_method="greedy", rejection_sample_method="standard")
        proposer.num_speculative_tokens = 7
        proposer.input_ids = torch.tensor([3, 15, 15, 15, 15, 15, 15, 15, 5, 15, 15, 15, 15, 15, 15, 15])
        ids = torch.tensor([[[6, 7]] * 7, [[8, 9]] * 7])
        scores = torch.zeros(2, 7, 2, 2)
        scores[:, :, 0, 1] = 1
        scores[:, :, 1, 0] = 1
        class Selector:
            top_k = 2
            def __call__(self, candidates, unary, hidden, anchors):
                self.seen_anchors = anchors.clone()
                return scores
        selector = Selector()
        proposer.model = NS(model=NS(candidate_selector=selector),
                            compute_candidates=lambda hidden: (ids.flatten(0, 1), torch.zeros(14, 2)))
        tokens, probs = proposer._sample_draft_tokens(torch.zeros(14, 4, dtype=torch.bfloat16), NS(all_greedy=False))
        self.assertIsNone(probs)  # Existing standard deterministic-proposal rejection unchanged.
        self.assertEqual(tokens.reshape(2, 7).tolist(), [[7, 6, 7, 6, 7, 6, 7], [9, 8, 9, 8, 9, 8, 9]])
        self.assertEqual(selector.seen_anchors.tolist(), [3, 5])
        proposer.speculative_config.draft_sample_method = "probabilistic"
        with self.assertRaises(ValueError):
            proposer._sample_draft_tokens(torch.zeros(14, 4), NS())

    def test_actual_norm_method_uses_per_layer_weights_only_for_enabled_xpu(self):
        class Tensor:
            def __init__(self, data, device="xpu"):
                self.data, self.device = data, NS(type=device)
                self.shape = data.shape
            def __getitem__(self, index):
                return Tensor(self.data[index], self.device.type)
        def rms(out, x, weight, eps):
            out.data.copy_((x.data.float() * torch.rsqrt(x.data.float().square().mean(-1, keepdim=True) + eps) * weight.data.float()).to(x.data.dtype))
        ops = NS(rms_norm=Mock(side_effect=rms))
        scope = {"torch": NS(empty_like=lambda x: Tensor(torch.empty_like(x.data), x.device.type)), "ops": ops}
        cls = actual_class(self.sources["model_executor/models/qwen3_dflash.py"], "DFlashQwen3Model", {"_normalize_context_k"}, scope, "object")
        draft = cls()
        draft._b70_dflash2_bf16 = True
        draft._rms_norm_eps = 1e-6
        draft._k_norm_weights = Tensor(torch.arange(1, 6).reshape(5, 1).expand(5, 4).bfloat16())
        x = Tensor(torch.randn(5, 3, 2, 4, dtype=torch.bfloat16))
        y = draft._normalize_context_k(x)
        self.assertEqual(ops.rms_norm.call_count, 5)
        for layer, call in enumerate(ops.rms_norm.call_args_list):
            self.assertEqual(call.args[2].data.data_ptr(), draft._k_norm_weights.data[layer].data_ptr())
        expected = x.data.float() * torch.rsqrt(x.data.float().square().mean(-1, keepdim=True) + 1e-6)
        expected *= draft._k_norm_weights.data[:, None, None, :].float()
        torch.testing.assert_close(y.data, expected.bfloat16())
        draft._k_norm_weights = Tensor(torch.ones(4, 4))
        with self.assertRaises(ValueError):
            draft._normalize_context_k(x)
        # Non-XPU and unflagged DFlash retain the exact original single dispatch.
        ops.rms_norm.side_effect = None
        for enabled, device in ((True, "cuda"), (False, "xpu")):
            ops.rms_norm.reset_mock()
            draft._b70_dflash2_bf16 = enabled
            x.device.type = device
            draft._normalize_context_k(x)
            self.assertEqual(ops.rms_norm.call_count, 1)

    def test_proposer_buffer_allocations_follow_draft_dtype_and_cache_copy_is_bf16(self):
        scope, spec = self.config()
        vc = NS(speculative_config=spec, model_config=spec.target_model_config)
        instance = NS(max_num_tokens=3, hidden_size=4, inputs_embeds_size=4)
        tree = ast.parse(self.sources["v1/spec_decode/llm_base_proposer.py"])
        fields = {"self.dtype", "self.hidden_states", "self.inputs_embeds"}
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and len(node.targets) == 1 and ast.unparse(node.targets[0]) in fields]
        assignments.sort(key=lambda node: node.lineno)
        execute(assignments, {**scope, "self": instance, "torch": torch, "vllm_config": vc, "device": torch.device("cpu")})
        self.assertEqual(instance.hidden_states.dtype, torch.bfloat16)
        self.assertEqual(instance.inputs_embeds.dtype, torch.bfloat16)
        self.assertEqual(vc.model_config.dtype, torch.float16)
        @dataclass
        class Cache:
            cache_dtype: str
        @dataclass
        class Config:
            cache_config: Cache
            model_config: object
            attention_config: object
        @dataclass
        class Attention:
            use_non_causal: bool = False
        class Base:
            def _create_draft_vllm_config(self):
                return self.vllm_config
        source = self.sources["v1/spec_decode/dflash.py"]
        cls = actual_class(source, "DFlashProposer", {"_create_draft_vllm_config"},
                           {"SpecDecodeBaseProposer": Base, "replace": replace, "_b70_dflash2_enabled": lambda _: True})
        proposer = cls()
        proposer.speculative_config = spec
        proposer.dflash_causal = False
        proposer.vllm_config = Config(Cache("fp8"), NS(model_arch_config=NS(is_mm_prefix_lm=False)), Attention())
        draft_config = proposer._create_draft_vllm_config()
        self.assertEqual(draft_config.cache_config.cache_dtype, "bfloat16")
        self.assertEqual(proposer.vllm_config.cache_config.cache_dtype, "fp8")
        self.assertIsNot(draft_config.cache_config, proposer.vllm_config.cache_config)


if __name__ == "__main__":
    unittest.main()
