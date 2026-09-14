"""Focused CPU checks; live XPU replay/stock identity/ABBA remain the lead E2E.

Run inside the pinned CPU-only disposable image with B70_MTP_TEST_VLLM_ROOT
pointing to its vllm source to include actual loader/packing and seam checks.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


ROOT = Path(__file__).parents[1]
PATCHES = ROOT / "patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b"


def load_file(name):
    spec = importlib.util.spec_from_file_location(name, PATCHES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(monkeypatch):
    pytest.importorskip("torch")
    for key in ("B70_MTP_WEIGHTS", "B70_MTP_CAPTURE_DIR", "B70_MTP_CAPTURE_MAX_TOKENS"):
        monkeypatch.delenv(key, raising=False)
    return load_file("b70_mtp_training")


@pytest.fixture
def small_weights(runtime, monkeypatch):
    torch = runtime.torch
    shapes = {
        "mtp.fc.weight": (4, 8),
        "mtp.layers.0.input_layernorm.weight": (4,),
        "mtp.layers.0.mlp.down_proj.weight": (4, 6),
        "mtp.layers.0.mlp.gate_proj.weight": (6, 4),
        "mtp.layers.0.mlp.up_proj.weight": (6, 4),
        "mtp.layers.0.post_attention_layernorm.weight": (4,),
        "mtp.layers.0.self_attn.k_norm.weight": (2,),
        "mtp.layers.0.self_attn.k_proj.weight": (2, 4),
        "mtp.layers.0.self_attn.o_proj.weight": (4, 4),
        "mtp.layers.0.self_attn.q_norm.weight": (2,),
        "mtp.layers.0.self_attn.q_proj.weight": (8, 4),
        "mtp.layers.0.self_attn.v_proj.weight": (2, 4),
        "mtp.norm.weight": (4,),
        "mtp.pre_fc_norm_embedding.weight": (4,),
        "mtp.pre_fc_norm_hidden.weight": (4,),
    }
    assert shapes.keys() == runtime.MTP_SHAPES.keys()
    monkeypatch.setattr(runtime, "MTP_SHAPES", shapes)
    return {key: torch.full(shape, i + 1, dtype=torch.bfloat16)
            for i, (key, shape) in enumerate(shapes.items())}


def save_overlay(weights, tmp_path, monkeypatch):
    safetensors = pytest.importorskip("safetensors.torch")
    path = tmp_path / "mtp.safetensors"
    safetensors.save_file(weights, str(path))
    monkeypatch.setenv("B70_MTP_WEIGHTS", str(path))
    return path


def test_pinned_schema_count(runtime):
    assert len(runtime.MTP_SHAPES) == 15
    assert sum(math.prod(shape) for shape in runtime.MTP_SHAPES.values()) == 424699392
    assert all(name.startswith("mtp.") for name in runtime.MTP_SHAPES)


def test_overlay_unset_is_exact_nonconsuming_passthrough(runtime):
    class NeverRead:
        def __iter__(self):
            raise AssertionError("disabled overlay consumed weights")
    weights = NeverRead()
    assert runtime.overlay_weights(weights) is weights
    runtime.capture_replay_step(None, None, None)


def test_stock_overlay_identity_preserves_shared_tensors(runtime, small_weights, tmp_path, monkeypatch):
    save_overlay(small_weights, tmp_path, monkeypatch)
    shared = {name: runtime.torch.ones(2) for name in (
        "model.language_model.embed_tokens.weight", "lm_head.weight", "model.layers.0.qweight"
    )}
    original = {**shared, **small_weights}
    actual = dict(runtime.overlay_weights(iter(original.items())))
    assert actual.keys() == original.keys()
    for key, value in original.items():
        assert runtime.torch.equal(actual[key], value)
        if key in shared:
            assert actual[key] is value


@pytest.mark.parametrize("bad", ["missing", "shared", "shape", "float16", "float32", "nan", "inf"])
def test_overlay_validates_every_tensor_before_consuming_source(
    runtime, small_weights, tmp_path, monkeypatch, bad
):
    torch = runtime.torch
    invalid = {key: value.clone() for key, value in small_weights.items()}
    key = "mtp.pre_fc_norm_hidden.weight"  # Last tensor must also be checked eagerly.
    if bad == "missing":
        invalid.pop(key)
    elif bad == "shared":
        invalid["lm_head.weight"] = torch.zeros(1, dtype=torch.bfloat16)
    elif bad == "shape":
        invalid[key] = torch.zeros(5, dtype=torch.bfloat16)
    elif bad in ("float16", "float32"):
        invalid[key] = invalid[key].to(getattr(torch, bad))
    else:
        invalid[key][0] = float(bad)
    save_overlay(invalid, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="keys mismatch|shape/dtype|nonfinite"):
        runtime.overlay_weights(None)  # Would fail with TypeError if iteration began.


@pytest.mark.parametrize("bad", ["missing", "duplicate", "extra", "dtype", "shape"])
def test_overlay_rejects_incompatible_checkpoint_stream(runtime, small_weights, tmp_path, monkeypatch, bad):
    save_overlay(small_weights, tmp_path, monkeypatch)
    items = list(small_weights.items())
    if bad == "missing":
        items.pop()
    elif bad == "duplicate":
        items.append(items[0])
    elif bad == "extra":
        items.append(("mtp.embed_tokens.weight", runtime.torch.zeros(1)))
    elif bad == "dtype":
        items[0] = (items[0][0], items[0][1].float())
    else:
        items[0] = (items[0][0], items[0][1][:1])
    with pytest.raises(ValueError, match="checkpoint"):
        list(runtime.overlay_weights(iter(items)))


def replay_fixture(runtime, tmp_path, *, dtype=None, mrope=False):
    torch = runtime.torch
    dtype = dtype or torch.bfloat16
    ids = [101, 102, 103, 104, 105, 106]
    control = {"name": "example-0001", "input_ids": ids, "loss_mask": [False, False, True, True, True, True]}
    (tmp_path / "capture-request.json").write_text(json.dumps(control))
    request = NS(prompt_token_ids=ids, mm_features=[], prompt_embeds=None, lora_request=None,
                 prompt_is_token_ids=None, sampling_params=NS(max_tokens=1))
    model = type("Qwen3_5ForConditionalGeneration", (), {})()
    runner = NS(
        speculative_config=None,
        cache_config=NS(enable_prefix_caching=False, kv_sharing_fast_prefill=False),
        scheduler_config=NS(max_num_seqs=1, async_scheduling=False),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
                           use_ubatching=False),
        vllm_config=NS(kv_transfer_config=None, ec_transfer_config=None),
        model_config=NS(hf_text_config=NS(hidden_size=5120)),
        use_aux_hidden_state_outputs=False, is_pooling_model=False, get_model=lambda: model,
        input_batch=NS(req_ids=["replay-1"], num_reqs=1, num_computed_tokens_cpu=[0]),
        requests={"replay-1": request},
    )
    output = NS(scheduled_spec_decode_tokens={}, scheduled_cached_reqs=NS(resumed_req_ids=set()))
    # Nonuniform rows distinguish arbitrary/sampled-row capture and accidental norm application.
    hidden = (torch.arange(len(ids) * 5120, dtype=torch.float32).reshape(len(ids), 5120) / 1000).to(dtype)

    def chunk(start, end):
        n = end - start
        runner.input_batch.num_computed_tokens_cpu[0] = start
        runner.input_ids = NS(gpu=torch.tensor(ids[start:end] + [-1, -1], dtype=torch.int32))
        pos = torch.arange(start, end + 2, dtype=torch.int64)
        if mrope:
            pos = pos.unsqueeze(0).expand(3, -1).clone()
        runner._get_positions = lambda count: pos[..., :count]
        output.num_scheduled_tokens = {runner.input_batch.req_ids[0]: n}
        output.total_num_scheduled_tokens = n
        padded = torch.cat((hidden[start:end].clone(), torch.full((2, 5120), float("nan"), dtype=dtype)))
        return padded

    return runner, output, hidden, chunk, control


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
@pytest.mark.parametrize("mrope", [False, True])
def test_complete_chunked_capture_matches_proposer_source_and_ignores_padding(
    runtime, tmp_path, dtype, mrope
):
    torch = runtime.torch
    runner, output, expected, chunk, control = replay_fixture(
        runtime, tmp_path, dtype=getattr(torch, dtype), mrope=mrope
    )
    capture = runtime.ReplayCapture(tmp_path)
    first = chunk(0, 2)
    original = first.clone()
    capture.step(runner, output, first)
    assert torch.equal(first[:2], original[:2])
    assert not list(tmp_path.glob("*.pt"))
    # Model buffers may be reused immediately after the callback returns.
    first.fill_(99)
    runner.input_ids.gpu.fill_(-1)
    capture.step(runner, output, chunk(2, 6))
    assert capture.active is None and capture.total_tokens == 6 and capture.sequences == 1
    payload = torch.load(tmp_path / "example-0001.pt", weights_only=True)
    assert set(payload) == {"input_ids", "positions", "target_last_hidden_states", "loss_mask"}
    assert payload["input_ids"].dtype == payload["positions"].dtype == torch.int64
    assert payload["positions"].tolist() == list(range(6))
    assert payload["input_ids"].tolist() == control["input_ids"]
    assert payload["loss_mask"].dtype == torch.bool
    assert payload["loss_mask"].tolist() == control["loss_mask"]
    assert payload["target_last_hidden_states"].dtype == getattr(torch, dtype)
    assert torch.equal(payload["target_last_hidden_states"], expected)
    # Explicit native alignment: x[t+1], h[t] -> CE label x[t+2].
    assert payload["input_ids"][1:-1].tolist() == [102, 103, 104, 105]
    assert payload["input_ids"][2:].tolist() == [103, 104, 105, 106]
    assert payload["loss_mask"][2:].all()
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("path,value,match", [
    ("speculative_config", NS(method="mtp"), "target-only"),
    ("cache_config.enable_prefix_caching", True, "prefix caching"),
    ("cache_config.kv_sharing_fast_prefill", True, "fast prefill"),
    ("scheduler_config.max_num_seqs", 2, "max_num_seqs"),
    ("scheduler_config.async_scheduling", True, "async"),
    ("parallel_config.tensor_parallel_size", 2, "TP=PP=DP"),
    ("parallel_config.pipeline_parallel_size", 2, "TP=PP=DP"),
    ("parallel_config.data_parallel_size", 2, "TP=PP=DP"),
    ("parallel_config.use_ubatching", True, "microbatching"),
    ("vllm_config.kv_transfer_config", NS(), "KV transfer"),
    ("vllm_config.ec_transfer_config", NS(), "encoder transfer"),
    ("use_aux_hidden_state_outputs", True, "final hidden"),
    ("is_pooling_model", True, "final hidden"),
    ("model_config.hf_text_config.hidden_size", 12, "5120"),
    ("input_batch.num_reqs", 2, "one active request"),
])
def test_unsupported_runner_modes_fail(runtime, tmp_path, path, value, match):
    runner, output, _, chunk, _ = replay_fixture(runtime, tmp_path)
    obj = runner
    *parents, field = path.split(".")
    for parent in parents:
        obj = getattr(obj, parent)
    setattr(obj, field, value)
    with pytest.raises(ValueError, match=match):
        runtime.ReplayCapture(tmp_path).step(runner, output, chunk(0, 6))
    assert not list(tmp_path.glob("*.pt"))


@pytest.mark.parametrize("bad", ["multimodal", "embeds", "lora", "soft_token", "generation", "rejected", "resumed"])
def test_unsupported_request_modes_fail(runtime, tmp_path, bad):
    runner, output, _, chunk, _ = replay_fixture(runtime, tmp_path)
    request = runner.requests["replay-1"]
    if bad == "multimodal":
        request.mm_features = [object()]
    elif bad == "embeds":
        request.prompt_embeds = runtime.torch.ones(2)
    elif bad == "lora":
        request.lora_request = object()
    elif bad == "soft_token":
        request.prompt_is_token_ids = [True, False]
    elif bad == "generation":
        request.sampling_params.max_tokens = 2
    elif bad == "rejected":
        output.scheduled_spec_decode_tokens = {"replay-1": [3]}
    else:
        output.scheduled_cached_reqs.resumed_req_ids = {"replay-1"}
    with pytest.raises(ValueError, match="text token IDs|max_tokens|speculative/rejected|preempted"):
        runtime.ReplayCapture(tmp_path).step(runner, output, chunk(0, 6))


@pytest.mark.parametrize("bad", ["name", "token_bool", "mask_int", "empty_loss", "length", "unknown_key", "prompt"])
def test_invalid_controls_fail(runtime, tmp_path, bad):
    runner, output, _, chunk, control = replay_fixture(runtime, tmp_path)
    if bad == "name":
        control["name"] = "../escape"
    elif bad == "token_bool":
        control["input_ids"][0] = True
    elif bad == "mask_int":
        control["loss_mask"][2] = 1
    elif bad == "empty_loss":
        control["loss_mask"] = [True, True, False, False, False, False]
    elif bad == "length":
        control["loss_mask"].pop()
    elif bad == "unknown_key":
        control["extra"] = "bad"
    else:
        runner.requests["replay-1"].prompt_token_ids = [1, 2, 3]
    (tmp_path / "capture-request.json").write_text(json.dumps(control))
    with pytest.raises(ValueError):
        runtime.ReplayCapture(tmp_path).step(runner, output, chunk(0, 6))
    assert not list(tmp_path.glob("*.pt"))


@pytest.mark.parametrize("bad", ["ids", "positions", "mrope", "nonfinite", "width", "missing_prefix"])
def test_runtime_rows_fail_closed(runtime, tmp_path, bad):
    runner, output, _, chunk, _ = replay_fixture(runtime, tmp_path, mrope=(bad == "mrope"))
    hidden = chunk(0, 6)
    if bad == "ids":
        runner.input_ids.gpu[1] = 999
    elif bad == "positions":
        runner._get_positions = lambda n: runtime.torch.arange(1, n + 1)
    elif bad == "mrope":
        runner._get_positions(6)[1, 2] += 1
    elif bad == "nonfinite":
        hidden[1, 0] = float("nan")
    elif bad == "width":
        hidden = hidden[:, :-1]
    else:
        runner.input_batch.num_computed_tokens_cpu[0] = 2
    with pytest.raises(ValueError):
        runtime.ReplayCapture(tmp_path).step(runner, output, hidden)
    assert not list(tmp_path.glob("*.pt"))


@pytest.mark.parametrize("bad", ["gap", "request", "control", "dtype"])
def test_chunk_continuity_is_mandatory(runtime, tmp_path, bad):
    runner, output, _, chunk, control = replay_fixture(runtime, tmp_path)
    capture = runtime.ReplayCapture(tmp_path)
    capture.step(runner, output, chunk(0, 2))
    hidden = chunk(2, 6)
    if bad == "gap":
        runner.input_batch.num_computed_tokens_cpu[0] = 3
    elif bad == "request":
        runner.requests["replay-2"] = runner.requests["replay-1"]
        runner.input_batch.req_ids = ["replay-2"]
    elif bad == "control":
        control["name"] = "changed"
        (tmp_path / "capture-request.json").write_text(json.dumps(control))
    else:
        hidden = hidden.float()
    with pytest.raises(ValueError, match="interrupted|control changed|dtype changed"):
        capture.step(runner, output, hidden)
    assert not list(tmp_path.glob("*.pt"))


def test_capture_budget_no_overwrite_and_no_partial_publish(runtime, tmp_path, monkeypatch):
    runner, output, _, chunk, control = replay_fixture(runtime, tmp_path)
    monkeypatch.setenv("B70_MTP_CAPTURE_MAX_TOKENS", "5")
    with pytest.raises(ValueError, match="budget"):
        runtime.ReplayCapture(tmp_path).step(runner, output, chunk(0, 6))
    monkeypatch.setenv("B70_MTP_CAPTURE_MAX_TOKENS", "10")
    capture = runtime.ReplayCapture(tmp_path)
    capture.step(runner, output, chunk(0, 6))
    original = (tmp_path / "example-0001.pt").read_bytes()
    with pytest.raises(FileExistsError):
        runtime.ReplayCapture(tmp_path).step(runner, output, chunk(0, 6))
    assert (tmp_path / "example-0001.pt").read_bytes() == original
    control["name"] = "example-0002"
    (tmp_path / "capture-request.json").write_text(json.dumps(control))
    runner.requests["replay-2"] = runner.requests["replay-1"]
    runner.input_batch.req_ids = ["replay-2"]
    with pytest.raises(ValueError, match="budget"):
        capture.step(runner, output, chunk(0, 6))
    assert not (tmp_path / "example-0002.pt").exists()


def test_callback_rejects_compilation_or_forward_context(runtime, tmp_path, monkeypatch):
    monkeypatch.setenv("B70_MTP_CAPTURE_DIR", str(tmp_path))
    context = NS(is_forward_context_available=lambda: True)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", context)
    with pytest.raises(RuntimeError, match="outside compilation/XPU graph"):
        runtime.capture_replay_step(None, None, None)
    context.is_forward_context_available = lambda: False
    monkeypatch.setattr(runtime.torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(RuntimeError, match="outside compilation/XPU graph"):
        runtime.capture_replay_step(None, None, None)


@pytest.fixture
def pinned_source():
    root = os.environ.get("B70_MTP_TEST_VLLM_ROOT")
    if not root:
        pytest.skip("pinned-image CPU test: set B70_MTP_TEST_VLLM_ROOT")
    return Path(root)


def copy_sources(pinned_source, tmp_path):
    for relative in ("model_executor/models/qwen3_5_mtp.py", "v1/worker/gpu_model_runner.py"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pinned_source / relative, destination)
    return tmp_path


def test_patch_real_runner_seam_is_outside_forward_and_before_row_selection(pinned_source, tmp_path):
    patcher = load_file("patch_mtp_training")
    root = copy_sources(pinned_source, tmp_path)
    patcher.patch(root)
    once = {str(path): path.read_bytes() for path in root.rglob("*.py")}
    patcher.patch(root)
    assert once == {str(path): path.read_bytes() for path in root.rglob("*.py")}
    tree = ast.parse((root / "v1/worker/gpu_model_runner.py").read_text())
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GPUModelRunner")
    execute = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "execute_model")
    calls = [node for node in execute.body if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
             and node.value.func.id == "capture_replay_step"]
    assert len(calls) == 1  # Direct method body, not inside any forward context or graph.
    forward = next(node for node in execute.body if isinstance(node, ast.With)
                   and "self._model_forward(" in ast.unparse(node))
    postprocess = next(node for node in execute.body if isinstance(node, ast.With)
                      and "sample_hidden_states = hidden_states[logits_indices]" in ast.unparse(node))
    assert forward.end_lineno < calls[0].lineno < postprocess.lineno
    assert "target_hidden_states = hidden_states[:num_scheduled_tokens]" in ast.unparse(tree)
    qwen = (pinned_source / "model_executor/models/qwen3_5.py").read_text()
    assert "class Qwen3_5Model(Qwen3NextModel):" in qwen
    assert "get_mtp_target_hidden_states" not in qwen
    qwen_next = (pinned_source / "model_executor/models/qwen3_next.py").read_text()
    assert "hidden_states, _ = self.norm(hidden_states, residual)" in qwen_next


def test_patch_anchor_failure_is_atomic(pinned_source, tmp_path):
    root = copy_sources(pinned_source, tmp_path)
    patcher = load_file("patch_mtp_training")
    runner = root / "v1/worker/gpu_model_runner.py"
    runner.write_text(runner.read_text().replace("gpu_model_runner: postprocess", "changed"))
    before = (root / "model_executor/models/qwen3_5_mtp.py").read_bytes()
    with pytest.raises(RuntimeError, match="anchor mismatch"):
        patcher.patch(root)
    assert (root / "model_executor/models/qwen3_5_mtp.py").read_bytes() == before
    assert not (root / "model_executor/models/b70_mtp_training.py").exists()


@pytest.mark.parametrize("rtn_first", [True, False])
def test_real_native_loader_packs_overlay_before_existing_rtn_hooks(
    runtime, small_weights, pinned_source, tmp_path, monkeypatch, rtn_first
):
    """Real pinned loader + Qwen mapper + QKV/gate-up weight loaders, CPU tensors.

    No model construction/forward/GPU: tiny modules exercise the original loading
    method, and a recorder replaces ONLY the XPU RTN operation after loading.
    """
    torch = runtime.torch
    from vllm.model_executor.models.utils import AutoWeightsLoader
    from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
    from vllm.model_executor.layers.linear import QKVParallelLinear, MergedColumnParallelLinear
    # Parameter constructors still query TP even with disable_tp=True; CPU rank 0 only.
    monkeypatch.setattr("vllm.distributed.parallel_state._TP", NS(rank_in_group=0, world_size=1))

    root = copy_sources(pinned_source, tmp_path / "source")
    # Apply the actual RTN installer on either side of this patch.
    rtn_path = (ROOT / "results/20260909-qwen38-dflash2-rtn-standard/mtp4-clients/reference-source/patches/patch_draft_mtp_int4.py")
    spec = importlib.util.spec_from_file_location("rtn_patch", rtn_path)
    rtn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rtn)
    if rtn_first:
        rtn._patch_qwen3_5_mtp(str(root))
    load_file("patch_mtp_training").patch(root)
    if not rtn_first:
        rtn._patch_qwen3_5_mtp(str(root))
    source = (root / "model_executor/models/qwen3_5_mtp.py").read_text()
    cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5MTP")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "load_weights")
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    namespace = {"AutoWeightsLoader": AutoWeightsLoader, "os": os}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "pinned_load_weights", "exec"), namespace)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.b70_mtp_training", runtime)

    class Predictor(torch.nn.Module):
        def load_weights(self, weights):
            return AutoWeightsLoader(self).load_weights(weights, mapper=Qwen3_5Model.hf_to_vllm_mapper)

    def build_model():
        model = torch.nn.Module()
        model.model = Predictor()
        # All nonpacked HF parameters, including shared tensors, use stock default loader.
        for key, value in {**small_weights, "mtp.embed_tokens.weight": torch.ones(4, 4, dtype=torch.bfloat16)}.items():
            if any(part in key for part in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")):
                continue
            obj = model.model
            *parts, leaf = key.removeprefix("mtp.").split(".")
            for part in parts:
                if not hasattr(obj, part):
                    obj.add_module(part, torch.nn.Module())
                obj = getattr(obj, part)
            obj.register_parameter(leaf, torch.nn.Parameter(torch.zeros_like(value), requires_grad=False))
        layer = getattr(model.model.layers, "0")
        attn = layer.self_attn
        attn.qkv_proj = QKVParallelLinear(4, 2, 4, 1, bias=False, params_dtype=torch.bfloat16, disable_tp=True)
        layer.mlp.gate_up_proj = MergedColumnParallelLinear(4, [6, 6], bias=False, params_dtype=torch.bfloat16, disable_tp=True)
        model.lm_head = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        return model

    stock = {**small_weights,
             "model.embed_tokens.weight": torch.ones(4, 4, dtype=torch.bfloat16),
             "lm_head.weight": torch.full((4, 4), 2, dtype=torch.bfloat16)}
    observed = []
    def after_load(model):
        observed.append({key: value.detach().clone() for key, value in model.named_parameters()})
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.b70_draft_mtp_int4", NS(build_draft_mtp_int4=after_load))
    monkeypatch.setenv("B70_DRAFT_MTP_INT4", "1")
    baseline = build_model()
    namespace["load_weights"](baseline, iter(stock.items()))
    save_overlay(small_weights, tmp_path, monkeypatch)
    identity = build_model()
    namespace["load_weights"](identity, iter(stock.items()))
    assert observed[0].keys() == observed[1].keys()
    for key in observed[0]:
        assert torch.equal(observed[0][key], observed[1][key]), key
    tuned = {key: value + 1 for key, value in small_weights.items()}
    save_overlay(tuned, tmp_path, monkeypatch)
    candidate = build_model()
    namespace["load_weights"](candidate, iter(stock.items()))
    assert len(observed) == 3
    assert torch.equal(observed[2]["model.layers.0.self_attn.qkv_proj.weight"], torch.cat([
        tuned[f"mtp.layers.0.self_attn.{part}_proj.weight"] for part in ("q", "k", "v")
    ]))
    assert torch.equal(observed[2]["model.layers.0.mlp.gate_up_proj.weight"], torch.cat([
        tuned[f"mtp.layers.0.mlp.{part}_proj.weight"] for part in ("gate", "up")
    ]))
    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        assert torch.equal(observed[0][key], observed[2][key])


def test_opt_in_callback_writes_complete_replay(runtime, tmp_path, monkeypatch):
    runner, output, expected, chunk, _ = replay_fixture(runtime, tmp_path, mrope=True)
    monkeypatch.setenv("B70_MTP_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "vllm.forward_context", NS(is_forward_context_available=lambda: False))
    runtime.capture_replay_step(runner, output, chunk(0, 6))
    saved = runtime.torch.load(tmp_path / "example-0001.pt", weights_only=True)
    assert runtime.torch.equal(saved["target_last_hidden_states"], expected)


def test_full_stock_safetensors_export_overlay_identity(runtime, tmp_path, monkeypatch):
    """Optional real 424M-parameter identity check; source checkpoint is mounted read-only."""
    model = os.environ.get("B70_MTP_TEST_MODEL")
    if not model:
        pytest.skip("full stock identity: set B70_MTP_TEST_MODEL to the pinned read-only checkpoint")
    from safetensors import safe_open
    index = json.loads((Path(model) / "model.safetensors.index.json").read_text())["weight_map"]
    keys = {key for key in index if key.startswith("mtp.")}
    assert keys == runtime.MTP_SHAPES.keys()
    stock = {}
    for filename in sorted({index[key] for key in keys}):
        with safe_open(str(Path(model) / filename), framework="pt", device="cpu") as handle:
            for key in sorted(keys):
                if index[key] == filename:
                    stock[key] = handle.get_tensor(key)
    assert sum(value.numel() for value in stock.values()) == 424699392
    save_overlay(stock, tmp_path, monkeypatch)
    seen = set()
    for key, value in runtime.overlay_weights(iter(stock.items())):
        assert value.dtype == runtime.torch.bfloat16
        assert tuple(value.shape) == runtime.MTP_SHAPES[key]
        assert runtime.torch.equal(value, stock[key]), key
        seen.add(key)
    assert seen == keys
