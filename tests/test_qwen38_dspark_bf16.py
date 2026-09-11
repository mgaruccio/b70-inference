"""Offline regression tests; not a substitute for the lead's real XPU/API gate.

python -B -m unittest discover -s tests -p test_qwen38_dspark_bf16.py -v
B70_DSPARK_SOURCE_ROOT must contain the complete pristine installed-image export
(or this exact overlay). Missing exports/CPU PyTorch explicitly skip the relevant
checks. No downloads, package installs or accelerator operations. TensorTests
can run read-only in the pinned image, using its installed vllm as SOURCE_ROOT.
"""
import ast
from contextlib import contextmanager
from copy import copy, deepcopy
import hashlib
import importlib.util
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
PATCH = ROOT / "scripts/patch-vllm-qwen38-dspark-bf16.py"
# Allows a read-only CPU container to execute this same suite entirely in memory.
overlay = sys.modules.get("qwen38_dspark_overlay")
if overlay is None:
    spec = importlib.util.spec_from_file_location("qwen38_dspark_overlay", PATCH)
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)
SOURCE_ROOT = Path(os.environ.get("B70_DSPARK_SOURCE_ROOT", "/tmp/qwen38-dspark-source-73029d424"))
DFLASH = "model_executor/models/qwen3_dflash.py"
DSPARK = "model_executor/models/qwen3_dspark.py"
BASE = "v1/worker/gpu/spec_decode/speculator.py"
FLASH_SPEC = "v1/worker/gpu/spec_decode/dflash/speculator.py"
SPARK_SPEC = "v1/worker/gpu/spec_decode/dspark/speculator.py"
DTYPES = torch or NS(float16="float16", bfloat16="bfloat16", float32="float32")


def exported_sources():
    missing = [name for name in overlay.PINNED_SHA256 if not (SOURCE_ROOT / name).is_file()]
    if missing:
        raise unittest.SkipTest("complete pinned export required: B70_DSPARK_SOURCE_ROOT; missing " + missing[0])
    return {name: (SOURCE_ROOT / name).read_bytes().decode() for name in overlay.PINNED_SHA256}


def execute(nodes, scope):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])),
                 "actual_pinned_source", "exec"), scope)


def actual_class(source, name, methods, scope, base="nn.Module"):
    node = deepcopy(next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name))
    node.decorator_list = []
    node.bases = [ast.parse(base, mode="eval").body]
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    for method in node.body:
        method.decorator_list = [d for d in method.decorator_list if isinstance(d, ast.Name) and d.id == "property"]
    execute([node], scope)
    return scope[name]


def actual_function(source, name, scope):
    node = deepcopy(next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    execute([node], scope)
    return scope[name]


def config():
    qc = dict(quant_method="gptq", bits=4, group_size=128, sym=True, desc_act=False, lm_head=False)
    target = NS(dtype=DTYPES.float16, quantization="gptq", head_dtype=None,
                hf_config=NS(architectures=["Qwen3_5ForConditionalGeneration"], model_type="qwen3_5",
                             quantization_config=qc), hf_text_config=NS(num_hidden_layers=64),
                get_hidden_size=lambda: 5120, get_vocab_size=lambda: 248320,
                max_model_len=8192, use_fp64_gumbel=False)
    hf = NS(architectures=["DSparkDraftModel"], model_type="qwen3", dtype="bfloat16",
            hidden_size=5120, num_hidden_layers=5, vocab_size=248320, draft_vocab_size=248320,
            intermediate_size=17408, num_attention_heads=32, num_key_value_heads=8, head_dim=128,
            block_size=7, markov_rank=256, markov_head_type="vanilla", projector_type="dspark",
            attention_mode="gqa", mask_token_id=248070, target_layer_ids=[5, 19, 33, 47, 61],
            layer_types=["full_attention"] * 5, enable_confidence_head=True,
            confidence_head_with_markov=True,
            dflash_config=dict(target_layer_ids=[5, 19, 33, 47, 61], mask_token_id=248070,
                               markov_rank=256, attention_mode="gqa", projector_type="dspark"))
    draft = NS(hf_config=hf, dtype=DTYPES.bfloat16, quantization=None,
               get_hidden_size=lambda: 5120, get_vocab_size=lambda: 248320)
    pc = NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
            data_parallel_rank=0, decode_context_parallel_size=1, prefill_context_parallel_size=1)
    spec = NS(method="dspark", model="RadixArk/Qwen3.8-27B-DSpark",
              revision="b9a5dbdf03bc999c6c73c426b19c2d9041cea393", target_model_config=target,
              draft_model_config=draft, quantization=None, num_speculative_tokens=7,
              draft_sample_method="greedy", rejection_sample_method="standard",
              enable_adaptive_verification=False, dspark_draft_topk=None,
              use_heterogeneous_vocab=False, use_local_argmax_reduction=False,
              kv_cache_dtype="bfloat16", draft_parallel_config=copy(pc), attention_backend=None)
    return NS(speculative_config=spec, model_config=target, parallel_config=pc,
              use_v2_model_runner=True, cache_config=NS(cache_dtype="fp8", kv_cache_dtype_skip_layers=[]),
              attention_config=NS(backend=None, use_non_causal=False),
              scheduler_config=NS(max_num_seqs=2, max_num_batched_tokens=4))


def config_scope(flag="1"):
    scope = {}
    with patch.dict(os.environ, {"B70_DSPARK_BF16": flag}):
        exec(overlay.CONFIG_HELPERS, scope)
    return scope


@contextmanager
def guard_imports():
    with patch.dict(sys.modules, {"torch": DTYPES, "vllm.platforms": NS(current_platform=NS(is_xpu=lambda: True))}), \
            patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"}):
        yield


class GuardTests(unittest.TestCase):
    def test_off_and_absent_do_not_inspect_or_modify_spec(self):
        for flag in ("", "0", "true", "yes", "01"):
            scope = config_scope(flag)
            self.assertFalse(scope["_b70_dspark_requested"](object()))
            self.assertFalse(scope["_b70_dspark_enabled"](object()))
            scope["_b70_dspark_validate"](object())
            c = NS(speculative_config=None, model_config=NS(dtype=DTYPES.float16))
            scope["_b70_dspark_runtime"](c)
            self.assertEqual(scope["_b70_dspark_dtype"](c), DTYPES.float16)
        with patch.dict(os.environ, {}, clear=True):
            scope = {}
            exec(overlay.CONFIG_HELPERS, scope)
            self.assertFalse(scope["_B70_DSPARK_BF16"])

    def test_valid_raw_and_normalized_config_and_target_unchanged(self):
        with guard_imports():
            c, scope = config(), config_scope()
            before = repr(c)
            scope["_b70_dspark_runtime"](c)
            self.assertEqual(repr(c), before)
            self.assertEqual(scope["_b70_dspark_dtype"](c), DTYPES.bfloat16)
            c.speculative_config.draft_model_config.hf_config.architectures = ["Qwen3DSparkModel"]
            scope["_b70_dspark_runtime"](c)

    def test_invalid_request_target_and_quantization(self):
        cases = {
            "method": ["dflash", "eagle3", None], "model": [None],
            "revision": ["main", "wrong"], "quantization": ["gptq", "fp8"],
            "num_speculative_tokens": [0, 6, 8], "draft_sample_method": ["probabilistic"],
            "rejection_sample_method": ["synthetic", "block"], "enable_adaptive_verification": [True],
            "dspark_draft_topk": [16], "use_heterogeneous_vocab": [True],
            "use_local_argmax_reduction": [True], "kv_cache_dtype": [None, "auto", "float16", "fp8"],
        }
        scope = config_scope()
        with guard_imports():
            for key, values in cases.items():
                for value in values:
                    with self.subTest(key=key, value=value):
                        c = config()
                        setattr(c.speculative_config, key, value)
                        with self.assertRaises(ValueError):
                            scope["_b70_dspark_requested"](c.speculative_config)
            for key, value in (("dtype", DTYPES.bfloat16), ("quantization", None),
                               ("head_dtype", DTYPES.bfloat16)):
                c = config()
                setattr(c.model_config, key, value)
                with self.assertRaises(ValueError):
                    scope["_b70_dspark_requested"](c.speculative_config)
            for key, value in (("bits", 8), ("group_size", 64), ("sym", False),
                               ("desc_act", True), ("lm_head", True), ("quant_method", "awq")):
                c = config()
                c.model_config.hf_config.quantization_config[key] = value
                with self.assertRaises(ValueError):
                    scope["_b70_dspark_requested"](c.speculative_config)
            for key, value in (("architectures", ["Qwen3ForCausalLM"]), ("model_type", "qwen3")):
                c = config()
                setattr(c.model_config.hf_config, key, value)
                with self.assertRaises(ValueError):
                    scope["_b70_dspark_requested"](c.speculative_config)
            for runner in ("", "0"):
                with patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": runner}), self.assertRaises(ValueError):
                    scope["_b70_dspark_requested"](config().speculative_config)
            with patch.dict(sys.modules, {"vllm.platforms": NS(current_platform=NS(is_xpu=lambda: False))}), \
                    self.assertRaises(ValueError):
                scope["_b70_dspark_requested"](config().speculative_config)

    def test_invalid_draft_config_and_precision(self):
        scope = config_scope()
        cases = {"architectures": ["Gemma4DSparkModel"], "model_type": "deepseek_v4", "dtype": "float16",
                 "num_hidden_layers": 4, "hidden_size": 4096, "draft_vocab_size": 240000,
                 "target_layer_ids": [5, 19, 33, 47, 60], "layer_types": ["sliding_attention"] * 5,
                 "sliding_window": 4096, "use_sliding_window": True, "tie_word_embeddings": True,
                 "sample_from_anchor": False, "is_causal": True, "dspark_draft_topk": 16,
                 "quantization_config": {}, "eagle_config": {"use_aux_hidden_state": False},
                 "eagle_aux_hidden_state_layer_ids": [6, 20, 34], "target_hidden_size": 4096,
                 "markov_rank": 128, "block_size": 8, "mask_token_id": 0}
        with guard_imports():
            for key, value in cases.items():
                with self.subTest(key=key):
                    c = config()
                    setattr(c.speculative_config.draft_model_config.hf_config, key, value)
                    with self.assertRaises(ValueError):
                        scope["_b70_dspark_validate"](c.speculative_config)
            for key, value in (("dtype", DTYPES.float16), ("quantization", "gptq")):
                c = config()
                setattr(c.speculative_config.draft_model_config, key, value)
                with self.assertRaises(ValueError):
                    scope["_b70_dspark_validate"](c.speculative_config)
            for key, value in (("use_aux_hidden_state", False), ("causal", True), ("use_swa", True),
                               ("sample_from_anchor", True), ("target_layer_ids", [0]), ("markov_rank", 128)):
                c = config()
                c.speculative_config.draft_model_config.hf_config.dflash_config[key] = value
                with self.assertRaises(ValueError):
                    scope["_b70_dspark_validate"](c.speculative_config)

    def test_runtime_rejects_actual_runner_parallel_cache_and_lora(self):
        scope = config_scope()
        with guard_imports():
            for key in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                        "decode_context_parallel_size", "prefill_context_parallel_size"):
                c = config()
                setattr(c.parallel_config, key, 2)
                with self.assertRaises(ValueError):
                    scope["_b70_dspark_runtime"](c)
            for mutate in (
                lambda c: setattr(c, "use_v2_model_runner", False),
                lambda c: setattr(c.cache_config, "cache_dtype", "auto"),
                lambda c: setattr(c.cache_config, "kv_cache_dtype_skip_layers", ["0"]),
                lambda c: setattr(c, "lora_config", object()),
                lambda c: setattr(c.speculative_config, "attention_backend", NS(name="TRITON_ATTN")),
                lambda c: setattr(c.speculative_config.draft_parallel_config, "tensor_parallel_size", 2),
            ):
                c = config()
                mutate(c)
                with self.assertRaises(ValueError):
                    scope["_b70_dspark_runtime"](c)


class SourceTests(unittest.TestCase):
    def write_sources(self, root, sources):
        for name, source in sources.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source.encode())

    def test_cli_and_apply_require_explicit_flag_without_writes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            for flag in (None, "0", "true"):
                env = {k: v for k, v in os.environ.items() if k != "B70_DSPARK_BF16"}
                if flag is not None:
                    env["B70_DSPARK_BF16"] = flag
                result = subprocess.run([sys.executable, "-B", str(PATCH), "--root", tmp],
                                        env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("explicit B70_DSPARK_BF16=1", result.stderr)
                with patch.dict(os.environ, env, clear=True), self.assertRaises(RuntimeError):
                    overlay.apply(tmp)
                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_whole_source_apply_reverse_and_idempotent_cli_replay(self):
        sources = exported_sources()
        expected = overlay.prepare(sources)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            self.write_sources(root, sources)
            for _ in range(2):
                result = subprocess.run([sys.executable, "-B", str(PATCH), "--root", tmp],
                                        env={**os.environ, "B70_DSPARK_BF16": "1"},
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual({n: (root / n).read_bytes().decode() for n in sources}, expected)
            self.assertIn("already applied", result.stdout)
        self.assertEqual(overlay.prepare(expected), expected)
        for name, source in expected.items():
            compile(source, name, "exec")
            for old, new in reversed(overlay.transformations().get(name, [])):
                self.assertEqual(source.count(new), 1)
                source = source.replace(new, old, 1)
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), overlay.PINNED_SHA256[name])

    def test_every_source_and_helper_drift_refuses_all_writes(self):
        sources = exported_sources()
        for initial in (sources, overlay.prepare(sources)):
            with tempfile.TemporaryDirectory(dir=ROOT) as tmp, patch.dict(os.environ, {"B70_DSPARK_BF16": "1"}):
                root = Path(tmp)
                self.write_sources(root, initial)
                for name in initial:
                    with self.subTest(name=name, patched=initial != sources):
                        path = root / name
                        path.write_bytes((initial[name] + "\n# unrelated drift\n").encode())
                        with patch.object(Path, "write_bytes", side_effect=AssertionError("unexpected write")), \
                                self.assertRaises(RuntimeError):
                            overlay.apply(root)
                        path.write_bytes(initial[name].encode())

    def test_partial_and_damaged_helper_replay_rejected(self):
        sources = exported_sources()
        expected = overlay.prepare(sources)
        for name in overlay.transformations():
            mixed = {**sources, name: expected[name]}
            with self.assertRaisesRegex(RuntimeError, "partially applied"):
                overlay.prepare(mixed)
        broken = dict(expected)
        broken[DFLASH] = broken[DFLASH].replace("def _b70_dspark_loaded(", "def _b70_dspark_damaged(")
        with self.assertRaises(RuntimeError):
            overlay.prepare(broken)

    def test_compilation_failure_and_crlf_refuse_all_writes(self):
        sources = exported_sources()
        edits = overlay.transformations()
        name = "v1/worker/gpu/spec_decode/dspark/utils.py"
        old, new = edits[name][-1]
        edits[name][-1] = (old, new + "\nreturn\n")  # AST-valid, compile-invalid.
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp, patch.dict(os.environ, {"B70_DSPARK_BF16": "1"}):
            root = Path(tmp)
            self.write_sources(root, sources)
            with patch.object(overlay, "transformations", return_value=edits), \
                    patch.object(Path, "write_bytes", side_effect=AssertionError("unexpected write")), \
                    self.assertRaises(SyntaxError):
                overlay.apply(root)
            (root / name).write_bytes(sources[name].replace("\n", "\r\n").encode())
            with patch.object(Path, "write_bytes", side_effect=AssertionError("unexpected write")), \
                    self.assertRaises(RuntimeError):
                overlay.apply(root)

    def test_symlink_escape_and_importlib_package_resolution(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp, patch.dict(os.environ, {"B70_DSPARK_BF16": "1"}):
            root = Path(tmp) / "package"
            (root / "config").mkdir(parents=True)
            outside = Path(tmp) / "outside.py"
            outside.write_text("untouched\n")
            (root / "config/speculative.py").symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, "outside package root"):
                overlay.apply(root)
            self.assertEqual(outside.read_text(), "untouched\n")
        with patch.object(overlay.importlib.util, "find_spec", return_value=NS(submodule_search_locations=["/installed/vllm"])), \
                patch.object(overlay, "apply") as apply, patch.dict(os.environ, {"B70_DSPARK_BF16": "1"}), \
                patch.object(sys, "argv", ["overlay.py"]):
            overlay.main()
            apply.assert_called_once_with(Path("/installed/vllm"))

    def test_pinned_v2_routing_cache_and_default_dtype_contract(self):
        sources = exported_sources()
        patched = overlay.prepare(sources)
        for name in set(sources) - set(overlay.transformations()):
            self.assertEqual(patched[name], sources[name])
        self.assertIn('with set_default_torch_dtype(model_config.dtype):', sources["model_executor/model_loader/base_loader.py"])
        self.assertIn('model_config=draft_model_config', sources["v1/worker/gpu/spec_decode/dspark/utils.py"])
        self.assertIn('"bfloat16",', sources["v1/attention/backends/flash_attn.py"])
        self.assertIn('kv_cache_dtype, vllm_config.model_config', sources["model_executor/layers/attention/attention.py"])
        self.assertIn('Qwen3DSparkForCausalLM', sources["model_executor/models/registry.py"])
        self.assertIn('DSparkSpeculator', sources["v1/worker/gpu/spec_decode/__init__.py"])
        self.assertIn('"dspark"', sources["v1/worker/gpu/model_runner.py"])
        # The exact native sampler, graph entry and aux routing are not replaced.
        self.assertEqual(patched[SPARK_SPEC], sources[SPARK_SPEC])
        self.assertIn('self._generate_draft,', patched[FLASH_SPEC])
        self.assertIn('torch.cat(aux_hidden_states, dim=-1)', patched[FLASH_SPEC])
        self.assertIn('target_layer_ids', sources["v1/worker/gpu/spec_decode/eagle/eagle3_utils.py"])
        self.assertNotIn('.to(torch.bfloat16)', patched["v1/worker/gpu/spec_decode/dspark/utils.py"])
        scope = {}
        aux = actual_function(sources["v1/worker/gpu/spec_decode/eagle/eagle3_utils.py"],
                              "get_eagle3_aux_layers_from_config", scope)
        self.assertEqual(aux(config().speculative_config), (6, 20, 34, 48, 62))
        fc_input = actual_function(sources[DFLASH], "_get_dflash_fc_input_size", scope)
        self.assertEqual(fc_input(config()), 25600)
        runner = ast.parse(sources["v1/worker/gpu/model_runner.py"])
        routes = [node for node in ast.walk(runner) if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Compare)
                  and ast.unparse(node.test.left) == "self.speculative_config.method"
                  and 'dspark' in ast.unparse(node.test)]
        self.assertTrue(any(any(isinstance(n, ast.Assign)
                                and ast.unparse(n.targets[0]) == "self.use_aux_hidden_state_outputs"
                                and isinstance(n.value, ast.Constant) and n.value.value is True
                                for n in node.body) for node in routes))
        # Preserve the loader's target config for layer numbering, but replace the
        # two explicitly target-typed draft parameters before creating them.
        for old, new in overlay.transformations()[DFLASH]:
            if "dtype=vllm_config.model_config.dtype" in old:
                self.assertIn("_b70_dspark_dtype(vllm_config)", new)


@unittest.skipIf(torch is None, "CPU PyTorch required; run TensorTests read-only in the pinned image")
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = overlay.prepare(exported_sources())
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(12)
        self.scope = {"torch": torch, "nn": nn, "F": F}
        actual_function(self.sources[DFLASH], "_b70_dspark_loaded", self.scope)

    def model(self, enabled=True):
        inner_cls = actual_class(self.sources[DFLASH], "DFlashQwen3Model", ["embed_input_ids"], self.scope)
        inner = inner_cls()
        inner._b70_dspark_bf16 = enabled
        inner.embed_tokens = nn.Embedding(16, 4, dtype=torch.float16)
        inner.mask_embedding = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
        inner.has_separate_mask_embedding = False
        inner.use_aux_hidden_state = True
        inner.fc = nn.Linear(12, 4, bias=False, dtype=torch.bfloat16 if enabled else torch.float16)
        inner.fc.input_size = 12
        base = actual_class(self.sources[DFLASH], "DFlashQwen3ForCausalLM", ["combine_hidden_states"], self.scope)
        cls = actual_class(self.sources[DSPARK], "Qwen3DSparkForCausalLM", ["compute_draft_logits"], self.scope,
                           base="DFlashQwen3ForCausalLM")
        model = cls()
        model.model = inner
        model.lm_head = nn.Linear(4, 16, bias=False, dtype=torch.float16)
        model.logits_processor = lambda head, hidden: F.linear(hidden, head.weight).float()
        return model

    def test_real_embedding_fc_head_boundaries_and_shared_storage(self):
        for enabled in (False, True):
            model = self.model(enabled)
            embed, head = model.model.embed_tokens, model.lm_head
            snapshots = [(p, p.data_ptr(), p.detach().clone()) for m in (embed, head) for p in m.parameters()]
            ids = torch.tensor([1, 2, 3])
            embeds = model.model.embed_input_ids(ids)
            dtype = torch.bfloat16 if enabled else torch.float16
            self.assertEqual(embeds.dtype, dtype)
            aux = torch.randn(3, 12, dtype=torch.float16)
            aux_copy = aux.clone()
            hidden = model.combine_hidden_states(aux)
            self.assertEqual(hidden.dtype, dtype)
            torch.testing.assert_close(hidden, model.model.fc(aux.to(dtype)))
            torch.testing.assert_close(model.combine_hidden_states(aux[0]), hidden[0])
            torch.testing.assert_close(model.compute_draft_logits(hidden), F.linear(hidden.half(), head.weight).float())
            self.assertTrue(torch.equal(aux, aux_copy))
            self.assertIs(model.model.embed_tokens, embed)
            self.assertIs(model.lm_head, head)
            for param, ptr, original in snapshots:
                self.assertEqual(param.dtype, torch.float16)
                self.assertEqual(param.data_ptr(), ptr)
                self.assertTrue(torch.equal(param, original))
            with self.assertRaises(ValueError):
                model.combine_hidden_states(torch.ones(2, 11, dtype=torch.float16))

    def test_implicit_markov_and_explicit_confidence_precision(self):
        def lm_head(vocab, hidden, **kwargs):
            return nn.Linear(hidden, vocab, bias=False)

        def replicated_linear(inputs, outputs, *, bias, params_dtype, **kwargs):
            return nn.Linear(inputs, outputs, bias=bias, dtype=params_dtype)

        self.scope.update(ParallelLMHead=lm_head, ReplicatedLinear=replicated_linear,
                          maybe_prefix=lambda a, b: a + "." + b)
        markov_cls = actual_class(self.sources[DSPARK], "DSparkMarkovHead", ["__init__", "embed", "bias"], self.scope)
        confidence_cls = actual_class(self.sources[DSPARK], "DSparkConfidenceHead", ["__init__", "forward"], self.scope)
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)  # Actual BaseModelLoader contract, pinned above.
            markov = markov_cls(16, 16, 4, prefix="markov")
            confidence = confidence_cls(8, prefix="confidence", bias=True)
        finally:
            torch.set_default_dtype(original_dtype)
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in markov.parameters()))
        self.assertTrue(all(p.dtype == torch.float32 for p in confidence.parameters()))
        embed = markov.embed(torch.tensor([1, 2]))
        bias = markov.bias(embed, lambda head, x: head(x).float())
        self.assertEqual(embed.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(bias).all())
        self.assertEqual(confidence(torch.ones(2, 4, dtype=torch.bfloat16), embed).dtype, torch.float32)
        with torch.no_grad():
            markov.markov_w1.weight.fill_(70000)
        self.assertTrue(torch.isfinite(markov.embed(torch.tensor([1]))).all())
        self.assertFalse(torch.isfinite(markov.markov_w1.weight.half()).all())

    def test_v2_context_and_query_buffers_created_in_draft_dtype(self):
        scope = {**self.scope, **config_scope(), "_target_feeds_hc_residual": lambda c: False,
                 "InputBuffers": lambda **kw: NS(), "get_parallel_drafting_token_id": lambda hf: hf.mask_token_id}
        base = actual_class(self.sources[BASE], "DraftModelSpeculator", ["__init__"], scope, base="object")
        dflash = actual_class(self.sources[FLASH_SPEC], "DFlashSpeculator", ["__init__"], scope, base="DraftModelSpeculator")
        dspark = actual_class(self.sources[SPARK_SPEC], "DSparkSpeculator", ["__init__"], scope, base="DFlashSpeculator")
        fake_dflash = NS(dflash_has_any_non_causal=lambda hf: True)
        c = config()
        with guard_imports(), patch.dict(sys.modules, {"vllm.model_executor.models.qwen3_dflash": fake_dflash}):
            speculator = dspark(c, torch.device("cpu"))
        self.assertEqual(speculator.dtype, torch.bfloat16)
        self.assertEqual(speculator.hidden_states.dtype, torch.bfloat16)
        self.assertEqual(speculator.hidden_states.shape, (4, 5120))
        self.assertEqual(speculator.num_query_per_req, 7)
        self.assertEqual(speculator.context_positions.dtype, torch.int64)
        self.assertEqual(speculator.draft_tokens.dtype, torch.int64)
        self.assertEqual(speculator.draft_token_confidence_probs.dtype, torch.float32)
        self.assertIsNone(speculator.draft_logits)
        self.assertEqual(c.model_config.dtype, torch.float16)
        self.assertEqual(c.cache_config.cache_dtype, "fp8")

    def test_attn_metadata_config_isolated_and_allocated_cache_validated(self):
        scope = {**self.scope, **config_scope(), "copy": NS(copy=copy),
                 "replace": lambda obj, **kw: NS(**{**vars(obj), **kw})}
        class Base:
            @property
            def attn_vllm_config(self):
                return self.vllm_config
        scope["Base"] = Base
        cls = actual_class(self.sources[FLASH_SPEC], "DFlashSpeculator", ["attn_vllm_config"], scope, base="Base")
        obj = cls()
        obj.vllm_config = config()
        obj.speculative_config = obj.vllm_config.speculative_config
        obj.draft_model_config = obj.speculative_config.draft_model_config
        obj.requires_non_causal = True
        config_copy = obj.attn_vllm_config
        self.assertIs(config_copy.model_config, obj.draft_model_config)
        self.assertEqual(config_copy.cache_config.cache_dtype, "bfloat16")
        self.assertEqual(obj.vllm_config.cache_config.cache_dtype, "fp8")
        self.assertFalse(obj.vllm_config.attention_config.use_non_causal)
        self.assertTrue(config_copy.attention_config.use_non_causal)
        # Execute the exact inserted startup check (no GPU allocation).
        source = "def check(self, kv_cache_config):\n" + overlay.CACHE_CHECK
        exec(source, scope)
        cache = NS(dtype=torch.bfloat16)
        group = NS(layer_names=["draft.64"], kv_cache_spec=NS(dtype=torch.bfloat16))
        obj.model = NS(get_draft_kv_cache_layer_names=lambda: ["draft.64"],
                       model=NS(layers=[NS(self_attn=NS(attn=NS(kv_cache=cache)))]))
        obj.hidden_states = torch.zeros(4, 4, dtype=torch.bfloat16)
        scope["check"](obj, NS(kv_cache_groups=[group]))
        group.kv_cache_spec = NS(kv_cache_specs={"draft.64": NS(dtype=torch.bfloat16)})
        scope["check"](obj, NS(kv_cache_groups=[group]))
        group.kv_cache_spec.kv_cache_specs["draft.64"].dtype = torch.float16
        with self.assertRaises(ValueError):
            scope["check"](obj, NS(kv_cache_groups=[group]))

    def test_actual_loader_shares_target_modules_and_checks_loaded_precision(self):
        from contextlib import nullcontext
        model = self.model()
        model.draft_id_to_target_id = None
        model.model._fused_kv_weight = torch.ones(4, 4, dtype=torch.bfloat16)
        model.model._hidden_norm_weight = torch.ones(4, dtype=torch.bfloat16)
        model.model._k_norm_weights = torch.ones(2, 2, dtype=torch.bfloat16)
        attn = NS(dtype=torch.bfloat16, kv_cache_dtype="bfloat16",
                  kv_cache_torch_dtype=torch.bfloat16, backend=NS(name="FLASH_ATTN"))
        model.model.layers = [NS(self_attn=NS(attn=attn))]
        target_embed = nn.Embedding(16, 4, dtype=torch.float16)
        target_head = nn.Linear(4, 16, bias=False, dtype=torch.float16)
        target_inner = NS(model=NS(embed_tokens=target_embed), lm_head=target_head)
        target = NS(get_language_model=lambda: target_inner)
        c = config()
        scope = {**self.scope, "replace": lambda obj, **kw: NS(**{**vars(obj), **kw}),
                 "get_pp_group": lambda: NS(world_size=1)}
        helpers = {}
        for name in ("_should_share", "get_target_lm_head"):
            actual_function(self.sources["v1/worker/gpu/spec_decode/eagle/utils.py"], name, helpers)
        load = actual_function(self.sources["v1/worker/gpu/spec_decode/dspark/utils.py"], "load_dspark_model", scope)
        actual_function(self.sources["v1/worker/gpu/spec_decode/dspark/utils.py"], "_resolve_dspark_attention_backend", scope)
        getter = Mock(return_value=model)
        modules = {
            "vllm.compilation.backends": NS(set_model_tag=lambda tag: nullcontext()),
            "vllm.model_executor.model_loader": NS(get_model=getter),
            "vllm.model_executor.models.utils": NS(get_draft_quant_config=lambda cfg: None),
            "vllm.model_executor.models.qwen3_dflash": NS(dflash_has_any_non_causal=lambda hf: True,
                                                        _b70_dspark_loaded=self.scope["_b70_dspark_loaded"]),
            "vllm.v1.worker.gpu.spec_decode.eagle.utils": NS(**helpers),
            "vllm.config.speculative": NS(**config_scope()),
        }
        with guard_imports(), patch.dict(sys.modules, modules):
            loaded = load(target, c)
        self.assertIs(loaded.model.embed_tokens, target_embed)
        self.assertIs(loaded.lm_head, target_head)
        kwargs = getter.call_args.kwargs
        self.assertIs(kwargs["model_config"], c.speculative_config.draft_model_config)
        self.assertIs(kwargs["vllm_config"].model_config, c.model_config)  # Target layer numbering retained.
        self.assertEqual(kwargs["vllm_config"].cache_config.cache_dtype, "bfloat16")
        self.assertIsNone(kwargs["vllm_config"].quant_config)
        self.assertEqual(c.cache_config.cache_dtype, "fp8")
        validate = self.scope["_b70_dspark_loaded"]
        for mutate, restore in (
            (lambda: setattr(model, "has_own_lm_head", True), lambda: setattr(model, "has_own_lm_head", False)),
            (lambda: setattr(attn, "kv_cache_torch_dtype", torch.float16), lambda: setattr(attn, "kv_cache_torch_dtype", torch.bfloat16)),
            (lambda: setattr(model.model, "_fused_kv_weight", torch.ones(4, 4, dtype=torch.float16)),
             lambda: setattr(model.model, "_fused_kv_weight", torch.ones(4, 4, dtype=torch.bfloat16))),
        ):
            mutate()
            with self.assertRaises(ValueError):
                validate(model, target_embed, target_head)
            restore()
        with self.assertRaises(ValueError):
            validate(model, nn.Embedding(16, 4, dtype=torch.float16), target_head)
        model.model.fc.weight = nn.Parameter(model.model.fc.weight.float())
        with self.assertRaisesRegex(ValueError, "parameter dtype"):
            validate(model, target_embed, target_head)

    def test_actual_query_entry_checks_cache_before_eager_or_graph_forward(self):
        from contextlib import nullcontext
        scope = {**self.scope, **config_scope(), "CUDAGraphMode": NS(NONE="NONE"),
                 "BatchDescriptor": NS, "set_forward_context": lambda *a, **kw: nullcontext()}
        cls = actual_class(self.sources[FLASH_SPEC], "DFlashSpeculator", ["_run_model"], scope, base="object")
        obj = cls()
        obj.vllm_config = config()
        obj.speculative_config = obj.vllm_config.speculative_config
        attn = NS(kv_cache=torch.ones(2, 4, dtype=torch.bfloat16))
        forward = Mock(return_value=torch.ones(1, 4, dtype=torch.bfloat16))
        forward.model = NS(layers=[NS(self_attn=NS(attn=attn))])
        obj.model = forward
        obj.input_buffers = NS(input_ids=torch.tensor([1]), positions=torch.tensor([0]))
        for mode in ("NONE", "FULL_DECODE_ONLY"):
            result = obj._run_model(1, None, None, None, mode)
            self.assertEqual(result.dtype, torch.bfloat16)
            attn.kv_cache = attn.kv_cache.half()
            forward.reset_mock()
            with self.assertRaisesRegex(ValueError, "allocated query cache"):
                obj._run_model(1, None, None, None, mode)
            forward.assert_not_called()
            attn.kv_cache = attn.kv_cache.bfloat16()
        attn.kv_cache = torch.empty(0)  # Profiling before allocation remains supported.
        obj._run_model(1, None, None, None)

    def test_fused_context_parameters_and_cache_write_boundary(self):
        def rms_norm(out, x, weight, eps):
            self.assertEqual(x.dtype, torch.bfloat16)
            self.assertEqual(weight.dtype, torch.bfloat16)
            if weight.ndim == 2:
                weight = weight.view(weight.shape[0], 1, 1, weight.shape[1])
            normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
            out.copy_((normalized * weight).to(out.dtype))

        ops = NS(rms_norm=rms_norm, rotary_embedding=Mock())
        scope = {**self.scope, "ops": ops}
        cls = actual_class(self.sources[DFLASH], "DFlashQwen3Model", [
            "_build_context_kv_buffers", "_build_fused_kv_buffers", "_project_context_kv",
            "_normalize_context_k", "precompute_and_store_context_kv"], scope)
        obj = cls()
        obj._b70_dspark_bf16 = True
        obj.hidden_norm = nn.Linear(4, 1, bias=False, dtype=torch.bfloat16)
        obj.hidden_norm.weight = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
        attns = []
        for _ in range(2):
            attns.append(NS(qkv_proj=nn.Linear(4, 6, bias=False, dtype=torch.bfloat16),
                            q_size=2, kv_size=2, head_dim=2, num_kv_heads=1,
                            q_norm=NS(variance_epsilon=1e-6), k_norm=NS(weight=torch.ones(2, dtype=torch.bfloat16)),
                            rotary_emb=NS(head_size=2, cos_sin_cache=torch.ones(8, 2), is_neox_style=True),
                            attn=NS(kv_cache=torch.zeros(8, 2, dtype=torch.bfloat16),
                                    impl=NS(do_kv_cache_update=Mock()))))
        obj.layers = [NS(self_attn=a) for a in attns]
        obj._build_fused_kv_buffers()
        self.assertEqual(obj._fused_kv_weight.dtype, torch.bfloat16)
        self.assertEqual(obj._k_norm_weights.dtype, torch.bfloat16)
        context = torch.randn(3, 4, dtype=torch.bfloat16)
        positions = torch.arange(3)
        slots = torch.arange(3)
        obj.precompute_and_store_context_kv(context, positions, slots)
        for a in attns:
            _, key, value, cache, mapping = a.attn.impl.do_kv_cache_update.call_args.args
            self.assertEqual(key.dtype, torch.bfloat16)
            self.assertEqual(value.dtype, torch.bfloat16)
            self.assertIs(cache, a.attn.kv_cache)
            self.assertIs(mapping, slots)
        attns[0].attn.kv_cache = attns[0].attn.kv_cache.half()
        attns[0].attn.impl.do_kv_cache_update.reset_mock()
        with self.assertRaisesRegex(ValueError, "allocated context cache"):
            obj.precompute_and_store_context_kv(context, positions, slots)
        attns[0].attn.impl.do_kv_cache_update.assert_not_called()

    def test_checkpoint_dtype_and_ownership_rejected_before_weight_loader(self):
        scope = {**self.scope, "process_eagle_weight": lambda *args: None,
                 "AutoWeightsLoader": Mock(side_effect=AssertionError("loader must not run"))}
        cls = actual_class(self.sources[DSPARK], "Qwen3DSparkForCausalLM", ["load_weights"], scope)
        obj = cls()
        obj.model = NS(_b70_dspark_bf16=True)
        obj.config = NS(vocab_size=16, draft_vocab_size=16)
        obj.target_vocab_size = 16
        for weights in (
            [("fc.weight", torch.ones(1, dtype=torch.float16))],
            [("embed_tokens.weight", torch.ones(1, dtype=torch.bfloat16))],
            [("lm_head.weight", torch.ones(1, dtype=torch.bfloat16))],
            [("d2t", torch.ones(1, dtype=torch.bfloat16))],
            [("fc.weight", torch.ones(1, dtype=torch.bfloat16))] * 2,
            [("fc.weight", torch.ones(1, dtype=torch.bfloat16))],
        ):
            with self.assertRaises(ValueError):
                obj.load_weights(weights)
        scope["AutoWeightsLoader"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
