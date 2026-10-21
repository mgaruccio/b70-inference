"""CPU checks for EAGLE4 math, checkpoint routing and pinned-source installation.

Run with an external PyTorch runtime, not the interactive Pi environment:
  /home/mike/code/model-router/.venv/bin/python -m unittest discover -s tests \
      -p test_eagle4_xpu_model.py -v

The vLLM base is stubbed for unit tests; these are NOT an XPU/HTTP serving test.
The integration owner must load the actual checkpoint and compare greedy HTTP
outputs to the same target without drafting, checking accepted/proposed counters.
"""

import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = nn = None


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "scripts/eagle4_xpu"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Only the external vLLM imports are replaced. Numerical tests execute the
# production adapter's PyTorch operations, not a second implementation of them.
if torch is not None:
    base_stub = ModuleType("vllm.model_executor.models.llama_eagle3")
    base_stub.Eagle3LlamaForCausalLM = nn.Module
    utils_stub = ModuleType("vllm.model_executor.models.utils")
    utils_stub.AutoWeightsLoader = Mock()
    with patch.dict(sys.modules, {
        base_stub.__name__: base_stub, utils_stub.__name__: utils_stub,
    }):
        model = load_module("eagle4_model_under_test", ADAPTER / "model.py")
installer = load_module("eagle4_installer_under_test", ADAPTER / "install.py")


def release_config():
    return NS(
        architectures=["LlamaForCausalLMEagle4"], hidden_size=5120,
        target_hidden_size=5120, intermediate_size=16384, vocab_size=248320,
        draft_vocab_size=248320, num_hidden_layers=1, num_attention_heads=40,
        num_key_value_heads=40, head_dim=128, num_aux_hidden_states=3,
        hidden_act="silu", fc_norm=True, norm_output=True,
        layer_types=["sliding_attention"], sliding_window=4096, rms_norm_eps=1e-6,
        tie_word_embeddings=False, attention_bias=False, mlp_bias=False,
        efficient_lm_head_config={"type": "low_rank", "rank": 128, "num_candidates": 128},
        eagle_config={"eagle_aux_hidden_state_layer_ids": [3, 31, 59],
                      "use_aux_hidden_state": True},
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0},
    )


def runtime_config():
    draft = NS(dtype=torch.float16, enforce_eager=True, quantization=None)
    spec = NS(
        method="eagle3", parallel_drafting=False, target_model_config=copy.copy(draft),
        draft_model_config=draft, quantization=None, kv_cache_dtype=None,
        draft_sample_method="greedy", rejection_sample_method="standard",
        use_local_argmax_reduction=False,
    )
    return NS(speculative_config=spec, parallel_config=NS(tensor_parallel_size=1),
              scheduler_config=NS(max_num_seqs=1), cache_config=NS(cache_dtype="auto"))


def tiny_model():
    config = release_config()
    config.hidden_size = config.target_hidden_size = 4
    config.vocab_size = config.draft_vocab_size = 7
    config.intermediate_size = 8
    config.num_attention_heads = config.num_key_value_heads = 2
    config.head_dim = 2
    config.efficient_lm_head_config = {"rank": 2, "num_candidates": 3, "type": "low_rank"}
    draft = model.LlamaForCausalLMEagle4.__new__(model.LlamaForCausalLMEagle4)
    nn.Module.__init__(draft)
    draft.config = config
    draft.num_candidates = 3
    draft.lm_head = nn.Linear(4, 7, bias=False, dtype=torch.float16)
    draft.efficient_lm_head = nn.Linear(2, 7, bias=False, dtype=torch.float16)
    draft.efficient_lm_head_down_proj = nn.Linear(4, 2, bias=False, dtype=torch.float16)
    return draft


@unittest.skipIf(torch is None, "run with an external PyTorch runtime")
class CandidateHeadTests(unittest.TestCase):
    def test_selected_logits_match_fp32_dense_reference(self):
        torch.manual_seed(38)
        draft = tiny_model()
        hidden = torch.randn(4, 4, dtype=torch.float16)
        logits = draft.compute_logits(hidden)
        low = hidden @ draft.efficient_lm_head_down_proj.weight.T
        scores = low @ draft.efficient_lm_head.weight.T
        ids = scores.argsort(dim=-1, descending=True, stable=True)[:, :3]
        reference = hidden.float() @ draft.lm_head.weight.float().T
        expected = torch.full_like(reference, -torch.inf)
        expected.scatter_(1, ids, reference.gather(1, ids))
        self.assertEqual(logits.dtype, torch.float32)
        torch.testing.assert_close(logits, expected, atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.equal(torch.isfinite(logits).sum(-1), torch.full((4,), 3)))

    def test_cutoff_ties_have_upstream_order_on_every_backend(self):
        scores = torch.tensor([[2., 3., 2., 1., 4., 2.], [0., 0., 0., 0., 0., 0.]])
        # All strictly-above-cutoff IDs, then the lowest cutoff-equal IDs.
        expected = torch.tensor([[1, 4, 0, 2], [0, 1, 2, 3]])
        self.assertTrue(torch.equal(model.candidate_ids(scores, 4), expected))
        self.assertTrue(torch.equal(model.candidate_ids(scores, 4), expected))

    def test_ties_at_real_candidate_count(self):
        scores = torch.zeros((1, 248320), dtype=torch.float16)
        scores[0, 248319] = 1
        ids = model.candidate_ids(scores, 128)
        expected = torch.cat([torch.tensor([248319]), torch.arange(127)]).unsqueeze(0)
        self.assertTrue(torch.equal(ids, expected))

    def test_learned_head_not_full_vocab_argmax_and_excludes_padding(self):
        hidden = torch.tensor([[1., 2.]], dtype=torch.float16)
        down = torch.tensor([[1., 0.]], dtype=torch.float16)
        efficient = torch.tensor([[0.], [2.], [3.], [1000.]], dtype=torch.float16)
        full = torch.tensor([[100., 100.], [2., 1.], [3., 1.], [999., 999.]],
                            dtype=torch.float16)
        logits = model.candidate_logits(hidden, down, efficient, full, 2, 3)
        torch.testing.assert_close(logits, torch.tensor([[-torch.inf, 4., 5.]]))
        self.assertEqual(logits.argmax().item(), 2)
        self.assertEqual((hidden @ full[:3].T).argmax().item(), 0)

    def test_exact_products_accumulate_in_fp32(self):
        hidden = torch.tensor([[256., 256.]], dtype=torch.float16)
        down = torch.tensor([[0.001, 0.001]], dtype=torch.float16)
        efficient = torch.ones((1, 1), dtype=torch.float16)
        full = torch.tensor([[256., 256.]], dtype=torch.float16)
        logits = model.candidate_logits(hidden, down, efficient, full, 1, 1)
        self.assertEqual(logits.item(), 131072.)  # FP16 dot product would overflow.

    def test_empty_rows_and_noncontiguous_hidden(self):
        draft = tiny_model()
        self.assertEqual(tuple(draft.compute_logits(torch.empty(0, 4, dtype=torch.float16)).shape), (0, 7))
        h = torch.randn(3, 8, dtype=torch.float16)[:, ::2]
        torch.testing.assert_close(draft.compute_logits(h), draft.compute_logits(h.contiguous()))

    def test_invalid_candidate_count_rejected(self):
        for count in (0, 4):
            with self.assertRaises(ValueError):
                model.candidate_ids(torch.zeros(1, 3), count)


@unittest.skipIf(torch is None, "run with an external PyTorch runtime")
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.draft = tiny_model()
        self.weights = {name: torch.randn(shape, dtype=torch.bfloat16)
                        for name, shape in model.checkpoint_shapes(self.draft.config).items()}

    def test_pinned_header_dimensions_and_tensor_count(self):
        shapes = model.checkpoint_shapes(release_config())
        self.assertEqual(len(shapes), 19)
        self.assertEqual(shapes["efficient_lm_head_down_proj.weight"], (128, 5120))
        self.assertEqual(shapes["efficient_lm_head.weight"], (248320, 128))
        self.assertEqual(shapes["layers.0.self_attn.q_proj.weight"], (5120, 10240))
        self.assertEqual(shapes["fc.weight"], (5120, 15360))
        self.assertEqual(shapes["embed_tokens.weight"], shapes["lm_head.weight"])

    def test_heads_are_root_and_trunk_uses_inherited_stacking_loader(self):
        received = {}
        def load(items):
            received.update(items)
            # Head modules are normal torch parameters (the production vLLM
            # AutoWeightsLoader performs these copies, including BF16 -> FP16).
            for head in ("lm_head", "efficient_lm_head", "efficient_lm_head_down_proj"):
                with torch.no_grad():
                    getattr(self.draft, head).weight.copy_(received[head + ".weight"])
            return {"loaded_parameter_names"}
        loader = Mock()
        loader.load_weights.side_effect = load
        with patch.object(model, "AutoWeightsLoader", return_value=loader) as factory:
            loaded = self.draft.load_weights(iter(self.weights.items()))
        factory.assert_called_once_with(self.draft)
        self.assertEqual(loaded, {"loaded_parameter_names"})
        self.assertEqual(len(received), 19)
        for name, tensor in self.weights.items():
            expected_name = name if name.startswith(("lm_head.", "efficient_lm_head")) else "model." + name
            self.assertIs(received[expected_name], tensor)
        for head in ("lm_head", "efficient_lm_head", "efficient_lm_head_down_proj"):
            torch.testing.assert_close(getattr(self.draft, head).weight,
                                       self.weights[head + ".weight"].half())
        self.assertTrue(self.draft.has_own_lm_head)
        self.assertTrue(self.draft.has_own_embed_tokens)

    def test_prefixed_trunk_is_not_double_prefixed(self):
        weights = [("model." + name, t) for name, t in self.weights.items()]
        with patch.object(model, "AutoWeightsLoader") as factory:
            self.draft.load_weights(weights)
            actual = dict(factory.return_value.load_weights.call_args.args[0])
        self.assertIn("model.fc.weight", actual)
        self.assertIn("efficient_lm_head_down_proj.weight", actual)
        self.assertFalse(any(name.startswith("model.model.") for name in actual))

    def test_every_released_weight_is_required_before_any_copy(self):
        for missing in self.weights:
            with self.subTest(missing=missing), patch.object(model, "AutoWeightsLoader") as loader:
                with self.assertRaisesRegex(ValueError, "Missing EAGLE4"):
                    self.draft.load_weights((n, t) for n, t in self.weights.items() if n != missing)
                loader.assert_not_called()

    def test_unknown_duplicate_wrong_shape_and_integer_tensors_fail_closed(self):
        good = list(self.weights.items())
        cases = [
            good + [("down_proj.weight", torch.zeros(2, 4))],
            good + [("model.fc.weight", self.weights["fc.weight"])],
            [("lm_head.weight", torch.zeros(6, 4))],
            [("lm_head.weight", torch.zeros(7, 4, dtype=torch.int64))],
            good + [("d2t", torch.arange(7))],
        ]
        for items in cases:
            with self.subTest(name=items[-1][0]), patch.object(model, "AutoWeightsLoader") as loader:
                with self.assertRaises(ValueError):
                    self.draft.load_weights(items)
                loader.assert_not_called()


@unittest.skipIf(torch is None, "run with an external PyTorch runtime")
class ConfigTests(unittest.TestCase):
    def test_released_config_and_runtime(self):
        model.validate_config(release_config())
        model.validate_runtime(runtime_config())

    def test_graph_dispatch_is_owned_by_the_pinned_runner(self):
        for target_eager, draft_eager in ((False, False), (False, True), (True, False)):
            config = runtime_config()
            config.speculative_config.target_model_config.enforce_eager = target_eager
            config.speculative_config.draft_model_config.enforce_eager = draft_eager
            with self.subTest(target_eager=target_eager, draft_eager=draft_eager):
                model.validate_runtime(config)

    def test_checkpoint_output_layers_become_serving_input_layers_before_base_init(self):
        config = runtime_config()
        hf = config.speculative_config.draft_model_config.hf_config = release_config()
        # Stop at the real superclass boundary: no full-sized parameters are needed.
        class BaseReached(RuntimeError):
            pass
        for _ in range(2):
            with patch.object(model.Eagle3LlamaForCausalLM, "__init__", side_effect=BaseReached):
                with self.assertRaises(BaseReached):
                    model.LlamaForCausalLMEagle4(vllm_config=config)
            self.assertEqual(hf.eagle_aux_hidden_state_layer_ids, (4, 32, 60))
            self.assertEqual(hf.eagle_config["eagle_aux_hidden_state_layer_ids"], [3, 31, 59])

    def test_unsupported_model_math_rejected(self):
        for name, bad in (("fc_norm", False), ("norm_output", False), ("attn_norm", True),
                          ("num_hidden_layers", 2), ("draft_vocab_size", 100),
                          ("norm_before_residual", True), ("use_rope", False)):
            config = release_config()
            setattr(config, name, bad)
            with self.subTest(name=name), self.assertRaises(ValueError):
                model.validate_config(config)
        config = release_config()
        config.eagle_config["eagle_aux_hidden_state_layer_ids"] = [1, 2, 3]
        with self.assertRaises(ValueError):
            model.validate_config(config)

    def test_unsupported_runtime_rejected(self):
        for component, name, bad in (
            ("parallel_config", "tensor_parallel_size", 2),
            ("scheduler_config", "max_num_seqs", 2),
            ("cache_config", "cache_dtype", "fp8"),
            ("speculative_config", "method", "eagle"),
            ("speculative_config", "parallel_drafting", True),
            ("speculative_config", "quantization", "gptq"),
            ("speculative_config", "kv_cache_dtype", "bfloat16"),
            ("speculative_config", "use_local_argmax_reduction", True),
        ):
            config = runtime_config()
            setattr(getattr(config, component), name, bad)
            with self.subTest(name=name), self.assertRaises(ValueError):
                model.validate_runtime(config)
        config = runtime_config()
        config.speculative_config.draft_model_config.dtype = torch.bfloat16
        with self.assertRaises(ValueError):
            model.validate_runtime(config)


class OverlayTests(unittest.TestCase):
    def setUp(self):
        # Never create fixtures in a runtime checkout or another worker's files.
        self.temp = tempfile.TemporaryDirectory(prefix=".test-", dir=ADAPTER)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        base = "# synthetic EAGLE3 unit-test fixture\n"
        self.write(installer.BASE_PATH, base)
        self.write("_version.py", f"__version__ = version = {installer.PINNED_VERSION!r}\n")
        self.write("model_executor/models/registry.py", "_SPECULATIVE_DECODING_MODELS = {\n}\n")
        self.write("transformers_utils/configs/eagle.py", '''def wrap(architectures):
    return [
                arch
                if arch.startswith("Eagle3") or arch.endswith("Eagle3")
                else f"Eagle3{arch}"
                for arch in architectures
    ]
''')
        self.write("config/speculative.py", '''def detect(self):
    if True:
        if True:
            if True:
                # Automatically detect the method
                return self.method
''')
        pin = patch.object(installer, "BASE_SHA256", hashlib.sha256(base.encode()).hexdigest())
        pin.start()
        self.addCleanup(pin.stop)

    def write(self, path, content):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def contents(self):
        return {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*.py")}

    def test_install_check_replay_and_architecture_dispatch(self):
        before = self.contents()
        self.assertEqual(len(installer.install(self.root, check=True)), 4)
        self.assertEqual(before, self.contents())
        self.assertEqual(len(installer.install(self.root)), 4)
        installed = self.contents()
        self.assertEqual(installer.install(self.root), [])
        self.assertEqual(installed, self.contents())
        for name, source in installed.items():
            ast.parse(source, name)
        scope = {}
        exec(installed["transformers_utils/configs/eagle.py"], scope)
        arches = ["LlamaForCausalLMEagle4", "LlamaForCausalLM", "LlamaForCausalLMEagle3", "Eagle3LlamaForCausalLM"]
        self.assertEqual(scope["wrap"](arches), [arches[0], "Eagle3LlamaForCausalLM", *arches[2:]])
        exec(installed["model_executor/models/registry.py"], scope)
        self.assertEqual(scope["_SPECULATIVE_DECODING_MODELS"][arches[0]],
                         ("eagle4_xpu", "LlamaForCausalLMEagle4"))
        exec(installed["config/speculative.py"], scope)
        for method in (None, "draft_model", "eagle3"):
            spec = NS(method=method, draft_model_config=NS(architectures=[arches[0]]))
            self.assertEqual(scope["detect"](spec), "eagle3")
        with self.assertRaises(ValueError):
            scope["detect"](NS(method="eagle", draft_model_config=NS(architectures=[arches[0]])))
        self.assertEqual(scope["detect"](NS(method="eagle", draft_model_config=NS(architectures=["LlamaForCausalLM"]))), "eagle")

    def test_wrong_version_or_base_has_no_partial_writes(self):
        for filename, content in (("_version.py", "__version__ = '0.28.0'\n"),
                                  (installer.BASE_PATH, "# different base\n")):
            old = (self.root / filename).read_text()
            self.write(filename, content)
            before = self.contents()
            with self.assertRaises(ValueError):
                installer.install(self.root)
            self.assertEqual(before, self.contents())
            self.write(filename, old)

    def test_missing_ambiguous_or_conflicting_anchor_has_no_partial_writes(self):
        path = "config/speculative.py"
        source = (self.root / path).read_text()
        for changed in (source.replace("Automatically", "Differently"), source + source,
                        source + "# LlamaForCausalLMEagle4 already customized\n"):
            self.write(path, changed)
            before = self.contents()
            with self.assertRaises(ValueError):
                installer.install(self.root)
            self.assertEqual(before, self.contents())

    def test_different_installed_model_is_not_overwritten(self):
        installer.install(self.root)
        self.write(installer.MODEL_PATH, "# existing unrelated adapter\n")
        before = self.contents()
        with self.assertRaises(ValueError):
            installer.install(self.root)
        self.assertEqual(before, self.contents())


if __name__ == "__main__":
    unittest.main()
