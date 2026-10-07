"""Tiny CPU-only native-MTP contracts; real B70 reload/acceptance is lead-owned.

Run with the cached inference-host image (no devices/network/installations):
  python -m pytest -q tests/test_qwen38_train_mtp.py
The CLI test trains only a synthetic native head under 200K parameters, not the 27B.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/experiments/qwen38_train_mtp.py"
RTN_PATCH = ROOT / "results/20260909-qwen38-dflash2-rtn-standard/mtp4-clients/reference-source/patches/patch_draft_lmhead_int4.py"


@pytest.fixture(scope="module")
def trainer():
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("qwen38_train_mtp", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def raw_config():
    return {
        "model_type": "qwen3_5", "tie_word_embeddings": False,
        "quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": True},
        "text_config": {
            # All core input widths are g128-packable, as in the deployed model.
            "model_type": "qwen3_5_text", "hidden_size": 128, "intermediate_size": 128,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
            "num_hidden_layers": 4, "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "vocab_size": 32, "max_position_embeddings": 128,
            "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False,
            "attn_output_gate": True, "attention_bias": False, "attention_dropout": 0.0,
            "hidden_act": "silu", "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0,
                                "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0],
                                "mrope_interleaved": True},
        },
    }

@pytest.fixture
def native_raw_config(raw_config):
    # Copy: full-model fixtures mutate text/vision fields and must not strip GPTQ from raw_config.
    config = {**raw_config, "text_config": dict(raw_config["text_config"])}
    config.pop("quantization_config", None)
    return config

@pytest.fixture
def config(trainer, raw_config):
    return trainer.native_config(raw_config)


def state_for(trainer, config):
    torch = trainer.runtime().torch
    torch.manual_seed(7)
    return {key: (torch.zeros(shape) if len(shape) == 1 else torch.randn(shape) * 0.02).bfloat16()
            for key, shape in trainer.expected_mtp_shapes(config).items()}


@pytest.fixture
def record(trainer, config):
    torch = trainer.runtime().torch
    torch.manual_seed(23)
    return {
        "prompt_id": "train-1",
        "input_ids": torch.arange(8, dtype=torch.int64),
        "positions": torch.tensor([0, 1, 3, 4, 8, 9, 12, 13], dtype=torch.int64),
        "target_last_hidden_states": torch.randn(8, config.hidden_size).half(),
        "loss_mask": torch.tensor([False, False, False, False, True, True, True, True]),
    }


@pytest.fixture
def checkpoint(tmp_path, trainer, config, raw_config):
    torch = trainer.runtime().torch
    model = tmp_path / "stock-model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(raw_config))
    tensors = state_for(trainer, config)
    trainer.runtime().save_file(tensors, model / "mtp.safetensors")
    mapping = {key: "mtp.safetensors" for key in tensors}
    for key, name in [(trainer.Checkpoint.EMBEDDING_KEY, "embedding"),
                      (trainer.Checkpoint.LM_HEAD_KEY, "lm_head")]:
        weight = (torch.randn(config.vocab_size, config.hidden_size) * 0.02).bfloat16()
        trainer.runtime().save_file({key: weight}, model / f"{name}.safetensors")
        mapping[key] = f"{name}.safetensors"
    # The verifier shard deliberately DOES NOT EXIST: touching it must fail.
    mapping["model.language_model.layers.0.self_attn.q_proj.qweight"] = "DO_NOT_READ_VERIFIER.safetensors"
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    return trainer.Checkpoint(model)


@pytest.fixture
def native_checkpoint(tmp_path, trainer, native_raw_config):
    torch = trainer.runtime().torch
    model = tmp_path / "native-bf16-model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(native_raw_config))
    config = trainer.native_config(native_raw_config)
    tensors = state_for(trainer, config)
    trainer.runtime().save_file(tensors, model / "mtp.safetensors")
    mapping = {key: "mtp.safetensors" for key in tensors}
    for key, name in [(trainer.Checkpoint.EMBEDDING_KEY, "embedding"),
                      (trainer.Checkpoint.LM_HEAD_KEY, "lm_head")]:
        weight = (torch.randn(config.vocab_size, config.hidden_size) * 0.02).bfloat16()
        trainer.runtime().save_file({key: weight}, model / f"{name}.safetensors")
        mapping[key] = f"{name}.safetensors"
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    return trainer.Checkpoint(model)

def test_native_bf16_config_and_frozen_heads(trainer, native_raw_config, native_checkpoint):
    torch = trainer.runtime().torch
    assert trainer.precision_regime(native_raw_config) == "native_bf16"
    assert native_checkpoint.regime == native_checkpoint.precision_regime == "native_bf16"
    embedding, head = trainer.frozen_heads(native_checkpoint, "cpu")
    expected_embedding = native_checkpoint.tensor(trainer.Checkpoint.EMBEDDING_KEY)
    expected_head = native_checkpoint.tensor(trainer.Checkpoint.LM_HEAD_KEY)
    assert embedding.weight.dtype == torch.bfloat16
    assert head.weight.dtype == torch.bfloat16
    assert torch.equal(embedding.weight, expected_embedding)
    assert torch.equal(head.weight, expected_head)


def test_native_bf16_cpu_forward_and_actual_metric_stage(
    trainer, native_checkpoint, record, tmp_path, monkeypatch
):
    torch = trainer.runtime().torch
    model = trainer.build_native_mtp(native_checkpoint.config, native_checkpoint.mtp_state())
    embedding, head = trainer.frozen_heads(native_checkpoint, "cpu")
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    hidden, labels, mask = trainer.sequence_hidden(model, embedding, record, "cpu")
    assert hidden.dtype == torch.bfloat16
    assert torch.isfinite(hidden).all()
    losses = trainer.depth_losses([(hidden, labels, mask)], head, [1], chunk_tokens=2)
    assert losses[0]["tokens"] == int(mask.sum().item())

    output = tmp_path / "native-stock.safetensors"
    trainer.export_mtp(native_checkpoint.mtp_state(), native_checkpoint, output, {})
    capture = tmp_path / "dev.pt"
    torch.save(dict(record, prompt_id="native-dev"), capture)
    args = trainer.parser().parse_args(["--model", str(native_checkpoint.path), "--output", str(output),
                                       "--device", "cpu"])
    args.depth_weights = [1.0]
    monkeypatch.setattr(trainer, "rtn_effective_core",
                        lambda _: pytest.fail("RTN must not be evaluated for native BF16"))
    metrics = trainer.evaluate_export(output, native_checkpoint, embedding, head, [capture], args)
    assert metrics["BF16_export"] is not None
    assert metrics["RTN_effective_dense"] is None
    candidate = {"step": 0, "path": str(output), "metrics": metrics}
    selected = trainer.select_dev_checkpoint([candidate])
    assert selected["metric"] == "dev.BF16_export.objective"

def test_public_cli_native_bf16_cpu_train_report(trainer, native_checkpoint, record, tmp_path):
    torch = trainer.runtime().torch
    train, heldout = tmp_path / "train", tmp_path / "heldout"
    train.mkdir()
    heldout.mkdir()
    torch.save(record, train / "train.pt")
    torch.save(dict(record, prompt_id="native-heldout"), heldout / "heldout.pt")
    output = tmp_path / "native-cli.safetensors"
    command = [sys.executable, str(SCRIPT), "--model", str(native_checkpoint.path),
               "--train-dir", str(train), "--eval-dir", str(heldout), "--output", str(output),
               "--steps", "1", "--grad-accum", "1", "--logits-chunk", "2", "--lr", "0.001",
               "--max-length", "16", "--device", "cpu"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["precision"]["regime"] == "native_bf16"
    assert report["precision"]["lm_head"] == "frozen original BF16 checkpoint weight"
    assert report["eval_before"]["objective"] is not None
    assert all(candidate["metrics"]["RTN_effective_dense"] is None
               for candidate in report["checkpoints"])
    assert report["dev_selection"]["metric"] == "dev.BF16_export.objective"
    with trainer.runtime().safe_open(output, framework="pt") as handle:
        assert all(handle.get_tensor(key).dtype == torch.bfloat16 for key in handle.keys())


@pytest.mark.parametrize("checkpoint_fixture", ["native_checkpoint", "checkpoint"])
def test_checkpoint_validation_releases_training_gradients(
    trainer, record, tmp_path, monkeypatch, request, checkpoint_fixture
):
    checkpoint = request.getfixturevalue(checkpoint_fixture)
    torch = trainer.runtime().torch
    train, heldout = tmp_path / "train", tmp_path / "heldout"
    train.mkdir()
    heldout.mkdir()
    torch.save(record, train / "train.pt")
    torch.save(dict(record, prompt_id="gradient-heldout"), heldout / "heldout.pt")
    models, updates, evaluations = [], [], []
    original_build = trainer.build_native_mtp
    original_step = torch.optim.AdamW.step
    original_evaluate = trainer.evaluate_export

    def build(*args, **kwargs):
        model = original_build(*args, **kwargs)
        models.append(model)
        return model

    def step(optimizer, *args, **kwargs):
        assert any(p.grad is not None for p in models[0].parameters())
        updates.append(True)
        return original_step(optimizer, *args, **kwargs)

    def evaluate(*args, **kwargs):
        assert all(p.grad is None for p in models[0].parameters()), (
            "Training gradients must be released before allocating the export-validation copy"
        )
        evaluations.append(True)
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(trainer, "build_native_mtp", build)
    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    monkeypatch.setattr(trainer, "evaluate_export", evaluate)
    args = trainer.parser().parse_args([
        "--model", str(checkpoint.path), "--train-dir", str(train),
        "--eval-dir", str(heldout), "--output", str(tmp_path / "released.safetensors"),
        "--steps", "3", "--checkpoint-every", "1", "--max-length", "16",
        "--recursive-depth", "4", "--roots", "1", "--logits-chunk", "2",
        "--device", "cpu",
    ])
    report = trainer.run(args)
    assert report["optimizer_steps"] == len(updates) == 3
    assert len(evaluations) == len(report["checkpoints"]) == 4

@pytest.mark.parametrize("available,bf16,message", [
    (False, True, "not available"),
    (True, False, "BF16 CUDA support required"),
])
def test_cuda_capability_checks_fail_fast(
    trainer, native_checkpoint, tmp_path, monkeypatch, available, bf16, message
):
    torch = trainer.runtime().torch
    calls = {"available": 0, "bf16": 0}
    def cuda_available():
        calls["available"] += 1
        return available
    def cuda_bf16_supported():
        calls["bf16"] += 1
        return bf16
    monkeypatch.setattr(torch.cuda, "is_available", cuda_available)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", cuda_bf16_supported)
    args = trainer.parser().parse_args(["--model", str(native_checkpoint.path),
                                       "--train-dir", str(tmp_path / "missing-train"),
                                       "--output", str(tmp_path / "cuda.safetensors"),
                                       "--device", "cuda"])
    with pytest.raises(trainer.TrainingError, match=message):
        trainer.run(args)
    assert calls["available"] == 1
    assert calls["bf16"] == int(available)


def test_alignment_mask_and_actual_positions(trainer, config, record):
    torch = trainer.runtime().torch
    trainer.validate_record(record, config, 16)
    ids, hidden, positions, labels, mask = trainer.aligned_inputs(record)
    assert ids.tolist() == [1, 2, 3, 4, 5, 6]
    assert labels.tolist() == [2, 3, 4, 5, 6, 7]
    assert mask.tolist() == [False, False, True, True, True, True]
    assert positions.tolist() == [0, 1, 3, 4, 8, 9]
    torch.testing.assert_close(hidden, record["target_last_hidden_states"][:-2])


def test_alignment_matches_installed_vllm_first_pass(trainer, record):
    """Execute the real pinned input-setup method, without importing/starting vLLM."""
    torch = trainer.runtime().torch
    spec = importlib.util.find_spec("vllm")
    if spec is None:
        pytest.skip("Source-semantic check requires the pinned inference-host vLLM image")
    source = (Path(next(iter(spec.submodule_search_locations))) / "v1/spec_decode/llm_base_proposer.py").read_text()
    method = next(node for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.FunctionDef) and node.name == "set_inputs_first_pass")
    namespace = {"torch": torch}
    exec("from __future__ import annotations\n" + ast.get_source_segment(source, method), namespace)
    ids = record["input_ids"]
    fake = SimpleNamespace(
        use_heterogeneous_vocab=False, needs_extra_input_slots=False,
        uses_xdrope_dim=0, draft_uses_xdrope_dim=0,
        input_ids=torch.empty_like(ids[:-1]),
        hidden_states=torch.empty_like(record["target_last_hidden_states"][:-1]),
    )
    fake._set_positions = lambda n, p: setattr(fake, "positions", p.clone())
    namespace["set_inputs_first_pass"](
        fake, ids[:-1], ids[-1:], record["positions"][:-1],
        record["target_last_hidden_states"][:-1], None,
        SimpleNamespace(query_start_loc=torch.tensor([0, 7])), None,
    )
    aligned = trainer.aligned_inputs(record)
    torch.testing.assert_close(fake.input_ids[:-1], aligned[0])
    torch.testing.assert_close(fake.hidden_states[:-1], aligned[1])
    torch.testing.assert_close(fake.positions[:-1], aligned[2])


@pytest.mark.parametrize("mutation,match", [
    ("missing_prompt", "prompt_id"), ("blank_prompt", "prompt_id"),
    ("missing_hidden", "tensor fields"), ("ids_dtype", "int64"),
    ("positions_dtype", "positions"), ("positions_repeated", "increasing"),
    ("positions_decreasing", "increasing"), ("positions_negative", "nonnegative"),
    ("positions_context", "context"), ("bad_hidden_shape", "hidden_size"),
    ("hidden_double", "runtime float"), ("hidden_nan", "Nonfinite"),
    ("hidden_inf", "Nonfinite"), ("bad_mask_dtype", "loss_mask"),
    ("bad_mask_shape", "loss_mask"), ("no_shifted_labels", "x\\[t\\+2\\]"),
    ("outside_vocab", "vocabulary"), ("too_long", "no truncation"),
])
def test_strict_capture_validation(trainer, config, record, mutation, match):
    torch = trainer.runtime().torch
    length_limit = 16
    if mutation == "missing_prompt":
        del record["prompt_id"]
    elif mutation == "blank_prompt":
        record["prompt_id"] = "  "
    elif mutation == "missing_hidden":
        del record["target_last_hidden_states"]
    elif mutation == "ids_dtype":
        record["input_ids"] = record["input_ids"].int()
    elif mutation == "positions_dtype":
        record["positions"] = record["positions"].float()
    elif mutation == "positions_repeated":
        record["positions"][1] = 0
    elif mutation == "positions_decreasing":
        record["positions"][2] = 0
    elif mutation == "positions_negative":
        record["positions"][0] = -1
    elif mutation == "positions_context":
        record["positions"][-1] = config.max_position_embeddings
    elif mutation == "bad_hidden_shape":
        record["target_last_hidden_states"] = record["target_last_hidden_states"][:, :-1]
    elif mutation == "hidden_double":
        record["target_last_hidden_states"] = record["target_last_hidden_states"].double()
    elif mutation in ("hidden_nan", "hidden_inf"):
        record["target_last_hidden_states"][0, 0] = float("nan" if mutation == "hidden_nan" else "inf")
    elif mutation == "bad_mask_dtype":
        record["loss_mask"] = record["loss_mask"].float()
    elif mutation == "bad_mask_shape":
        record["loss_mask"] = record["loss_mask"][:-1]
    elif mutation == "no_shifted_labels":
        record["loss_mask"][:] = False
        record["loss_mask"][0] = True
    elif mutation == "outside_vocab":
        record["input_ids"][-1] = config.vocab_size
    elif mutation == "too_long":
        length_limit = 7
    with pytest.raises(trainer.TrainingError, match=match):
        trainer.validate_record(record, config, length_limit)


def test_splits_reject_prompt_overlap_and_never_skip_corrupt_file(trainer, config, record, tmp_path):
    torch = trainer.runtime().torch
    train, evaluation = tmp_path / "train", tmp_path / "heldout"
    train.mkdir()
    evaluation.mkdir()
    torch.save(record, train / "a.pt")
    torch.save(record, evaluation / "different-filename.pt")
    with pytest.raises(trainer.TrainingError, match="prompt_id overlap"):
        trainer.capture_sets(train, evaluation, config, 16)
    record["prompt_id"] = "heldout-1"
    torch.save(record, evaluation / "different-filename.pt")
    assert tuple(map(len, trainer.capture_sets(train, evaluation, config, 16))) == (1, 1)
    (train / "corrupt.pt").write_bytes(b"not a tensor capture")
    with pytest.raises(trainer.TrainingError, match="corrupt.pt"):
        trainer.capture_sets(train, evaluation, config, 16)


def test_native_gated_decoder_causality_prefix_and_norm_order(trainer, checkpoint, record):
    torch, rt = trainer.runtime().torch, trainer.runtime()
    model = trainer.build_native_mtp(checkpoint.config, checkpoint.mtp_state())
    embedding, _ = trainer.frozen_heads(checkpoint, "cpu")
    assert isinstance(model.layers[0], rt.Decoder)
    assert model.layers[0].block_type == "full_attention"
    assert model.layers[0].self_attn.q_proj.weight.shape == (4 * 32 * 2, 128)
    assert {"mtp." + k for k in model.state_dict()} == set(checkpoint.shapes)
    assert all(p.dtype == torch.float32 and p.requires_grad for p in model.parameters())
    seen = {}
    hook = model.fc.register_forward_pre_hook(lambda _, args: seen.update(fc=args[0].detach()))
    rope_hook = model.rotary_emb.register_forward_pre_hook(lambda _, args: seen.update(positions=args[1].clone()))
    ids, hidden, positions, _, _ = trainer.aligned_inputs(record)
    hidden = hidden.float()
    output = model(ids, hidden, positions, embedding)
    expected = torch.cat([model.pre_fc_norm_embedding(embedding(ids)), model.pre_fc_norm_hidden(hidden)], -1)
    torch.testing.assert_close(seen["fc"], expected)
    torch.testing.assert_close(seen["positions"], positions[None])
    hook.remove()
    rope_hook.remove()
    future_hidden = hidden.clone()
    future_hidden[4:] += 10
    future_ids = ids.clone()
    future_ids[4:] += 5
    future_output = model(future_ids, future_hidden, positions, embedding)
    torch.testing.assert_close(output[:4], future_output[:4], atol=1e-6, rtol=1e-6)
    # Prompt rows have no loss but must condition the supervised suffix.
    prefix_hidden = hidden.clone()
    prefix_hidden[:2] *= -1
    assert not torch.allclose(output[-1], model(ids, prefix_hidden, positions, embedding)[-1])
    norm = rt.Norm(128)
    norm.weight.data.fill_(0.25)
    sample = torch.randn(2, 128)
    expected_norm = sample * torch.rsqrt(sample.square().mean(-1, keepdim=True) + norm.eps) * 1.25
    torch.testing.assert_close(norm(sample), expected_norm)


def test_effective_head_matches_deployed_rtn_patch(trainer):
    """Execute the repo's actual packer, then unpack its nibbles/scales on CPU."""
    torch = trainer.runtime().torch
    source = ast.parse(RTN_PATCH.read_text())
    helper = next(ast.literal_eval(node.value) for node in source.body
                  if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HELPER_SOURCE" for t in node.targets))
    function = next(node for node in ast.parse(helper).body
                    if isinstance(node, ast.FunctionDef) and node.name == "quantize_lmhead_to_int4")
    namespace = {"torch": torch}
    exec(ast.get_source_segment(helper, function), namespace)
    torch.manual_seed(91)
    weight = torch.randn(5, 256).bfloat16()
    packed, scales, zeros, group = namespace["quantize_lmhead_to_int4"](weight.half())
    shifts = torch.arange(8, dtype=torch.int32) * 4
    codes = ((packed.t().contiguous()[..., None] >> shifts) & 15).reshape(5, 256) - 8
    expected = (codes.float().reshape(5, 2, 128) * scales.t().float()[..., None]).reshape(5, 256).half()
    effective = trainer.rtn_effective_lm_head(weight, row_chunk=2)
    assert torch.equal(effective, expected)
    assert not torch.equal(effective, weight.half())
    assert zeros.item() == 8 and group == 128
    assert torch.equal(trainer.rtn_effective_lm_head(torch.zeros(1, 128)), torch.zeros(1, 128).half())


def test_chunked_loss_gradients_and_accumulation_match_dense(trainer):
    torch = trainer.runtime().torch
    torch.manual_seed(11)
    head = torch.nn.Linear(128, 32, bias=False).requires_grad_(False)
    hidden = torch.randn(7, 128, requires_grad=True)
    labels = torch.tensor([2, 5, 1, 11, 8, 12, 3])
    mask = torch.tensor([False, True, True, False, True, True, True])
    dense_hidden = hidden.detach().clone().requires_grad_(True)
    dense_loss = torch.nn.functional.cross_entropy(head(dense_hidden)[mask], labels[mask])
    dense_loss.backward()
    # Two sequences share one denominator, just as gradient accumulation does.
    total = 0.0
    for start, end in [(0, 3), (3, 7)]:
        loss, _ = trainer.chunked_ce(hidden[start:end], head, labels[start:end], mask[start:end],
                                     chunk_tokens=1, backward=True, normalizer=mask.sum().item())
        total += loss
    assert total / mask.sum().item() == pytest.approx(dense_loss.item(), rel=1e-6)
    torch.testing.assert_close(hidden.grad, dense_hidden.grad, atol=1e-7, rtol=1e-5)
    assert head.weight.grad is None


def test_native_gradients_frozen_heads_and_strict_export(trainer, checkpoint, record, tmp_path):
    torch = trainer.runtime().torch
    model = trainer.build_native_mtp(checkpoint.config, checkpoint.mtp_state())
    embedding, head = trainer.frozen_heads(checkpoint, "cpu")
    before_embedding, before_head = embedding.weight.clone(), head.weight.clone()
    before_fc = model.fc.weight.clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    hidden, labels, mask = trainer.sequence_hidden(model, embedding, record, "cpu")
    trainer.chunked_ce(hidden, head, labels, mask, chunk_tokens=2, backward=True)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert embedding.weight.grad is None and head.weight.grad is None
    optimizer.step()
    assert not torch.equal(before_fc, model.fc.weight)
    assert torch.equal(before_embedding, embedding.weight) and torch.equal(before_head, head.weight)
    state = {"mtp." + k: v for k, v in model.state_dict().items()}
    output = tmp_path / "trained.safetensors"
    trainer.export_mtp(state, checkpoint, output, {"optimizer_steps": 1})
    with trainer.runtime().safe_open(output, framework="pt") as handle:
        assert set(handle.keys()) == set(checkpoint.shapes)
        loaded = {k: handle.get_tensor(k) for k in handle.keys()}
    assert all(v.dtype == torch.bfloat16 for v in loaded.values())
    reloaded = trainer.build_native_mtp(checkpoint.config, loaded)
    assert set(reloaded.state_dict()) == set(model.state_dict())
    invalid = dict(state, **{"lm_head.weight": head.weight})
    with pytest.raises(trainer.TrainingError, match="keys mismatch"):
        trainer.export_mtp(invalid, checkpoint, tmp_path / "bad.safetensors", {})
    wrong_shape = dict(state, **{"mtp.fc.weight": torch.zeros(1)})
    with pytest.raises(trainer.TrainingError, match="shape mismatch"):
        trainer.validate_mtp_state(wrong_shape, checkpoint.shapes)
    with pytest.raises(trainer.TrainingError, match="overwrite"):
        trainer.export_mtp(state, checkpoint, output, {})
    with pytest.raises(trainer.TrainingError, match="outside the stock"):
        trainer.export_mtp(state, checkpoint, checkpoint.path / "overlay.safetensors", {})
    with pytest.raises(trainer.TrainingError, match="verifier tensor"):
        checkpoint.tensor("model.language_model.layers.0.self_attn.q_proj.qweight")


def test_stock_export_requires_no_optimizer_or_embedding_reads(trainer, checkpoint, tmp_path, monkeypatch):
    torch = trainer.runtime().torch
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("optimizer in stock export"))
    (checkpoint.path / "embedding.safetensors").unlink()
    (checkpoint.path / "lm_head.safetensors").unlink()
    output = tmp_path / "stock.safetensors"
    args = trainer.parser().parse_args(["--model", str(checkpoint.path), "--output", str(output), "--export-stock"])
    report = trainer.run(args)
    assert report["optimizer_steps"] == 0 and report["train_steps"] == []
    with trainer.runtime().safe_open(output, framework="pt") as handle:
        for key, value in checkpoint.mtp_state().items():
            assert torch.equal(handle.get_tensor(key), value)


@pytest.mark.parametrize("key,value", [("mtp_num_hidden_layers", 2), ("attn_output_gate", False),
                                      ("model_type", "qwen3_5_moe_text"), ("attention_bias", True)])
def test_unsupported_native_models_fail(trainer, raw_config, key, value):
    raw_config["text_config"][key] = value
    with pytest.raises(trainer.TrainingError, match="Only dense"):
        trainer.native_config(raw_config)


def test_public_cli_tiny_export_train_reload(trainer, checkpoint, record, tmp_path):
    torch = trainer.runtime().torch
    train, heldout = tmp_path / "train", tmp_path / "heldout"
    train.mkdir()
    heldout.mkdir()
    torch.save(record, train / "train.pt")
    record["prompt_id"] = "heldout-1"
    torch.save(record, heldout / "eval.pt")
    output = tmp_path / "cli-tuned.safetensors"
    command = [sys.executable, str(SCRIPT), "--model", str(checkpoint.path),
               "--train-dir", str(train), "--eval-dir", str(heldout), "--output", str(output),
               "--steps", "2", "--grad-accum", "2", "--logits-chunk", "2", "--lr", "0.001",
               "--max-length", "16", "--device", "cpu"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["optimizer_steps"] == 2
    assert len(report["train_steps"]) == 2
    assert all(step["tokens"] == 8 and step["sequences"] == 2 for step in report["train_steps"])
    assert report["eval_before"]["tokens"] == report["eval_after"]["tokens"] == 4
    assert all(step["ce"] > 0 for step in report["train_steps"])
    with trainer.runtime().safe_open(output, framework="pt") as handle:
        state = {key: handle.get_tensor(key) for key in handle.keys()}
    reloaded = trainer.build_native_mtp(checkpoint.config, state)
    assert set(state) == set(checkpoint.shapes)
    assert sum(p.numel() for p in reloaded.parameters()) < 200_000
    assert any(not torch.equal(state[key], value) for key, value in checkpoint.mtp_state().items())


def test_stock_sized_cpu_training_fails_before_reading_weights(trainer, raw_config, tmp_path):
    raw_config["text_config"].update(hidden_size=5120, intermediate_size=17408,
                                     num_attention_heads=24, num_key_value_heads=4, head_dim=256)
    config = trainer.native_config(raw_config)
    shapes = trainer.expected_mtp_shapes(config)
    assert sum(__import__("math").prod(shape) for shape in shapes.values()) == 424699392
    model = tmp_path / "large-model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(raw_config))
    mapping = dict.fromkeys(shapes, "must-not-read.safetensors")
    mapping.update({trainer.Checkpoint.EMBEDDING_KEY: "must-not-read.safetensors",
                    trainer.Checkpoint.LM_HEAD_KEY: "must-not-read.safetensors"})
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    args = trainer.parser().parse_args(["--model", str(model), "--train-dir", str(tmp_path),
                                       "--output", str(tmp_path / "bad.safetensors"), "--device", "cpu"])
    with pytest.raises(trainer.TrainingError, match="Stock-size CPU training is forbidden"):
        trainer.run(args)


@pytest.fixture
def native_record(trainer, config):
    torch = trainer.runtime().torch
    torch.manual_seed(42)
    return {
        "prompt_id": "native-train",
        "input_ids": torch.arange(12, dtype=torch.int64),
        "positions": torch.arange(11, dtype=torch.int64),
        "target_last_hidden_states": torch.randn(11, config.hidden_size),
        "loss_mask": torch.tensor([False, False] + [True] * 10),
    }


@pytest.mark.parametrize("observed", [10, 11, 12])
def test_observed_prefix_alignment_never_invents_hidden(trainer, checkpoint, native_record, observed):
    torch = trainer.runtime().torch
    record = native_record
    full = torch.randn(12, checkpoint.config.hidden_size)
    record["positions"] = torch.arange(observed)
    record["target_last_hidden_states"] = full[:observed]
    trainer.validate_record(record, checkpoint.config, 16)
    ids, hidden, positions, labels, mask = trainer.aligned_inputs(record)
    assert ids.tolist() == list(range(1, 11))
    assert labels.tolist() == list(range(2, 12))
    assert positions.tolist() == list(range(10)) and mask.all()
    assert torch.equal(hidden, full[:10])
    model = trainer.build_native_mtp(checkpoint.config, checkpoint.mtp_state())
    embedding, _ = trainer.frozen_heads(checkpoint, "cpu")
    actual = trainer.sequence_hidden(model, embedding, record, "cpu")[0]
    expected = model(ids, full[:10], positions, embedding)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("bad", ["too_short", "too_long", "position_length", "hole", "offset"])
def test_observed_prefix_rejects_missing_or_nonprefix_rows(trainer, config, native_record, bad):
    torch = trainer.runtime().torch
    if bad in ("too_short", "too_long"):
        n = 9 if bad == "too_short" else 13
        native_record["target_last_hidden_states"] = torch.zeros(n, config.hidden_size)
        native_record["positions"] = torch.arange(n)
    elif bad == "position_length":
        native_record["positions"] = torch.arange(12)
    elif bad == "hole":
        native_record["positions"][-1] += 1
    else:
        native_record["positions"] += 1
    with pytest.raises(trainer.TrainingError, match="observed|prefix|T-2 <= H <= T"):
        trainer.validate_record(native_record, config, 16)


def test_root_sampling_is_seeded_and_all_labels_are_valid(trainer, native_record):
    import random
    expected = sorted(random.Random(5).sample(list(range(7)), 3))
    assert trainer.sample_roots(native_record, 4, 3, random.Random(5)) == expected
    assert trainer.sample_roots(native_record, 4, 3, random.Random(5)) == expected
    assert trainer.sample_roots(native_record, 4, 3, random.Random(6)) != expected
    native_record["loss_mask"][4] = False
    assert trainer.sample_roots(native_record, 4, 32, random.Random(0)) == [3, 4, 5, 6]
    native_record["loss_mask"][6:] = False
    assert trainer.sample_roots(native_record, 4, 32, random.Random(0)) == []
    assert trainer.sample_roots(native_record, 1, 32, random.Random(0)) == []


@pytest.mark.parametrize("legacy_positions", [False, True])
def test_real_hf_sequential_cache_matches_sliced_branches_at_every_depth(
    trainer, checkpoint, native_record, legacy_positions
):
    torch = trainer.runtime().torch
    if legacy_positions:
        native_record["positions"] = torch.arange(12) * 3
        native_record["target_last_hidden_states"] = torch.cat([native_record["target_last_hidden_states"],
                                                              torch.zeros(1, checkpoint.config.hidden_size)])
    trainer.validate_record(native_record, checkpoint.config, 16)
    model = trainer.build_native_mtp(checkpoint.config, checkpoint.mtp_state())
    embedding, _ = trainer.frozen_heads(checkpoint, "cpu")
    roots = [0, 2, 6]
    actual = trainer.sequence_depths(model, embedding, native_record, "cpu", 4, roots)
    ids, hidden, positions, labels, _ = trainer.aligned_inputs(native_record)
    tokens = native_record["input_ids"]
    for row, root in enumerate(roots):
        # Reference never slices/reuses a precomputed base: genuine row-by-row
        # HF cache decode of the accepted prefix followed by its recursive tail.
        cache = trainer.runtime().Cache()
        for j in range(root + 1):
            previous = model(ids[j:j + 1], hidden[j:j + 1], positions[j:j + 1],
                             embedding, past_key_values=cache)
            torch.testing.assert_close(previous[0], actual[0][0][j], atol=2e-6, rtol=2e-5)
        for d in range(2, 5):
            previous = model(tokens[root + d:root + d + 1], previous, positions[root:root + 1] + d - 1,
                             embedding, past_key_values=cache)
            torch.testing.assert_close(previous[0], actual[d - 1][0][row], atol=2e-6, rtol=2e-5)
            assert actual[d - 1][1][row] == tokens[root + d + 1]
            assert cache.get_seq_length() == root + d
    assert torch.equal(actual[0][1], labels)


def test_branches_exclude_future_base_hidden_and_siblings(trainer, checkpoint, native_record):
    torch = trainer.runtime().torch
    model = trainer.build_native_mtp(checkpoint.config, checkpoint.mtp_state())
    embedding, _ = trainer.frozen_heads(checkpoint, "cpu")
    roots = [1, 4, 6]
    baseline = trainer.sequence_depths(model, embedding, native_record, "cpu", 4, roots)
    reversed_roots = trainer.sequence_depths(model, embedding, native_record, "cpu", 4, roots[::-1])
    solo = trainer.sequence_depths(model, embedding, native_record, "cpu", 4, [1])
    changed = dict(native_record)
    changed["target_last_hidden_states"] = native_record["target_last_hidden_states"].clone()
    changed["target_last_hidden_states"][2:] = torch.randn_like(changed["target_last_hidden_states"][2:]) * 50
    future = trainer.sequence_depths(model, embedding, changed, "cpu", 4, [1])
    for d in range(1, 4):
        torch.testing.assert_close(baseline[d][0], reversed_roots[d][0].flip(0), atol=0, rtol=0)
        torch.testing.assert_close(baseline[d][0][:1], solo[d][0], atol=0, rtol=0)
        torch.testing.assert_close(baseline[d][0][:1], future[d][0], atol=0, rtol=0)
    changed["target_last_hidden_states"][0] *= -1
    prefix_changed = trainer.sequence_depths(model, embedding, changed, "cpu", 4, [1])
    assert not torch.allclose(solo[3][0], prefix_changed[3][0])


def test_depth_four_loss_reaches_own_hidden_chain_and_base_kv(trainer, checkpoint, native_record, monkeypatch):
    torch = trainer.runtime().torch
    model = trainer.build_native_mtp(checkpoint.config, checkpoint.mtp_state())
    embedding, head = trainer.frozen_heads(checkpoint, "cpu")
    native_record["target_last_hidden_states"].requires_grad_()
    seen, cache_tensors = [], []
    def retain_output(_, args, output):
        output.retain_grad()
        seen.append(output)
    hook = model.register_forward_hook(retain_output)
    original = trainer.prefix_cache
    def retain_cache(cache, length):
        for tensor in (cache.layers[0].keys, cache.layers[0].values):
            tensor.retain_grad()
            cache_tensors.append(tensor)
        return original(cache, length)
    monkeypatch.setattr(trainer, "prefix_cache", retain_cache)
    outputs = trainer.sequence_depths(model, embedding, native_record, "cpu", 4, [2])
    trainer.depth_losses(outputs, head, [0, 0, 0, 1], chunk_tokens=1, backward=True)
    hook.remove()
    assert len(seen) == 4 and all(t.grad is not None and t.grad.abs().sum() > 0 for t in seen)
    assert len(cache_tensors) == 2
    for tensor in cache_tensors:
        assert tensor.grad[..., :3, :].abs().sum() > 0
        assert tensor.grad[..., 3:, :].count_nonzero() == 0
    target_grad = native_record["target_last_hidden_states"].grad
    assert target_grad[:3].abs().sum() > 0 and target_grad[3:].count_nonzero() == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in model.parameters())
    assert embedding.weight.grad is None and head.weight.grad is None


def test_recursive_chunked_gradients_match_weighted_dense_accumulation(trainer, checkpoint, native_record):
    torch = trainer.runtime().torch
    model = trainer.build_native_mtp(checkpoint.config, checkpoint.mtp_state())
    reference = trainer.build_native_mtp(checkpoint.config, checkpoint.mtp_state())
    embedding, head = trainer.frozen_heads(checkpoint, "cpu")
    weights, denominators = [1, 0.5, 0.25, 0.125], [20, 4, 4, 4]
    dense_loss = 0
    for roots in ([0, 2, 6], [1]):
        outputs = trainer.sequence_depths(model, embedding, native_record, "cpu", 4, roots)
        trainer.depth_losses(outputs, head, weights, chunk_tokens=2, backward=True, normalizers=denominators)
        dense = trainer.sequence_depths(reference, embedding, native_record, "cpu", 4, roots)
        for (hidden, labels, mask), weight, denominator in zip(dense, weights, denominators):
            dense_loss += weight * torch.nn.functional.cross_entropy(head(hidden)[mask], labels[mask],
                                                                     reduction="sum") / denominator
    dense_loss.backward()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=3e-5)


def test_empty_roots_keep_all_depth_one_labels(trainer, checkpoint, native_record):
    torch = trainer.runtime().torch
    native_record["loss_mask"][3::2] = False
    model = trainer.build_native_mtp(checkpoint.config, checkpoint.mtp_state())
    embedding, head = trainer.frozen_heads(checkpoint, "cpu")
    outputs = trainer.sequence_depths(model, embedding, native_record, "cpu", 4, [])
    losses = trainer.depth_losses(outputs, head, [1] * 4, chunk_tokens=2, backward=True)
    metrics = trainer.loss_metrics(losses, [1] * 4)
    assert [row["tokens"] for row in metrics["depths"]] == [5, 0, 0, 0]
    assert metrics["depths"][1]["ce"] is None
    assert metrics["objective"] == metrics["ce"]
    for bad_roots in ([0], [7], [-1], [2, 2]):
        with pytest.raises(trainer.TrainingError, match="roots"):
            trainer.sequence_depths(model, embedding, native_record, "cpu", 4, bad_roots)


def test_effective_core_matches_deployment_fused_g128_packer(trainer, checkpoint):
    torch = trainer.runtime().torch
    patch = RTN_PATCH.with_name("patch_draft_mtp_int4.py")
    source = ast.parse(patch.read_text())
    helper = next(ast.literal_eval(node.value) for node in source.body
                  if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HELPER_SOURCE" for t in node.targets))
    function = next(node for node in ast.parse(helper).body
                    if isinstance(node, ast.FunctionDef) and node.name == "quantize_to_int4")
    namespace = {"torch": torch}
    exec(ast.get_source_segment(helper, function), namespace)
    state = checkpoint.mtp_state()
    effective = trainer.rtn_effective_core(state)
    groups = [
        ["mtp.fc.weight"],
        [f"mtp.layers.0.self_attn.{name}_proj.weight" for name in ("q", "k", "v")],
        ["mtp.layers.0.self_attn.o_proj.weight"],
        [f"mtp.layers.0.mlp.{name}_proj.weight" for name in ("gate", "up")],
        ["mtp.layers.0.mlp.down_proj.weight"],
    ]
    for keys in groups:
        weight = torch.cat([state[key] for key in keys]).half()
        packed, scales, zeros, group_size = namespace["quantize_to_int4"](weight)
        shifts = torch.arange(8, dtype=torch.int32) * 4
        codes = ((packed.t()[..., None] >> shifts) & 15).reshape_as(weight) - 8
        expected = (codes.float().reshape(weight.shape[0], -1, 128) * scales.t()[..., None].float()).reshape_as(weight).half()
        assert torch.equal(torch.cat([effective[key] for key in keys]), expected.float())
        assert group_size == 128 and zeros.item() == 8
    assert all(torch.equal(effective[key], value.float()) for key, value in state.items() if value.ndim == 1)


def test_checkpoint_metrics_use_loaded_bf16_and_effective_core_not_masters(
    trainer, checkpoint, native_record, tmp_path, monkeypatch
):
    torch = trainer.runtime().torch
    model = trainer.build_native_mtp(checkpoint.config, checkpoint.mtp_state())
    embedding, head = trainer.frozen_heads(checkpoint, "cpu")
    with torch.no_grad():
        model.fc.weight.add_(0.00013)
    masters = {"mtp." + key: value.detach().clone() for key, value in model.state_dict().items()}
    output = tmp_path / "rounded.safetensors"
    trainer.export_mtp(masters, checkpoint, output, {})
    with trainer.runtime().safe_open(output, framework="pt") as handle:
        loaded = {key: handle.get_tensor(key) for key in handle.keys()}
    assert not torch.equal(masters["mtp.fc.weight"], loaded["mtp.fc.weight"].float())
    capture = tmp_path / "dev.pt"
    torch.save(dict(native_record, prompt_id="dev-1"), capture)
    args = trainer.parser().parse_args(["--model", str(checkpoint.path), "--output", str(output),
                                       "--device", "cpu", "--recursive-depth", "4", "--roots", "2"])
    args.depth_weights = [1, 0.5, 0.25, 0.125]
    original, seen = trainer.evaluate, []
    def observe(actual_model, *rest):
        seen.append({"mtp." + key: value.detach().clone() for key, value in actual_model.state_dict().items()})
        return original(actual_model, *rest)
    monkeypatch.setattr(trainer, "evaluate", observe)
    metrics = trainer.evaluate_export(output, checkpoint, embedding, head, [capture], args)
    effective = trainer.rtn_effective_core(loaded)
    for stage, expected in zip(seen, (loaded, effective)):
        for key in stage:
            assert torch.equal(stage[key], expected[key].float())
    assert not torch.equal(seen[0]["mtp.fc.weight"], seen[1]["mtp.fc.weight"])
    assert all(torch.equal(value, masters["mtp." + key]) for key, value in model.state_dict().items())
    for stage in metrics.values():
        assert [row["tokens"] for row in stage["depths"]] == [10, 2, 2, 2]
        assert stage["objective"] == pytest.approx(sum(row["weight"] * row["ce"] for row in stage["depths"]))
        assert all(0 <= row["argmax_agreement"] <= 1 for row in stage["depths"])


def test_stock_remains_eligible_and_selection_uses_only_dev_rtn_objective(trainer):
    def candidate(step, dense_ce, rtn_ce):
        return {"step": step, "path": str(step), "metrics": {
            "BF16_export": {"objective": dense_ce}, "RTN_effective_dense": {"objective": rtn_ce}}}
    stock, tuned = candidate(0, 3, 2), candidate(1, 1, 3)
    assert trainer.select_dev_checkpoint([stock, tuned])["step"] == 0
    tied = candidate(2, 0, 2)
    assert trainer.select_dev_checkpoint([stock, tied])["step"] == 0
    better = candidate(3, 4, 1)
    selected = trainer.select_dev_checkpoint([stock, better])
    assert selected["step"] == 3 and selected["metric"] == "dev.RTN_effective_dense.objective"
    assert selected["promotion"] is False
    assert trainer.select_dev_checkpoint([{"metrics": {"RTN_effective_dense": None}}]) is None


@pytest.mark.parametrize("options", [
    ["--roots", "0"], ["--roots", "33"], ["--epochs", "0"], ["--epochs", "101"],
    ["--checkpoint-every", "0"], ["--depth-weights", "nan"], ["--depth-weights", "0"],
    ["--depth-weights", "-1"], ["--recursive-depth", "4", "--depth-weights", "1"],
    ["--steps", "0"], ["--steps", "10001"], ["--kl-weight", "-1"],
    ["--kl-weight", "nan"], ["--kl-weight", "inf"], ["--kl-temperature", "0"],
    ["--kl-temperature", "nan"], ["--kl-temperature", "inf"],
])
def test_recursive_cli_bounds_fail_before_checkpoint_reads(trainer, tmp_path, options):
    args = trainer.parser().parse_args(["--model", str(tmp_path / "must-not-read"),
                                       "--output", str(tmp_path / "out.safetensors"), *options])
    with pytest.raises(trainer.TrainingError, match="bounds|weight"):
        trainer.run(args)


def test_public_cli_supports_more_than_pilot_update_budget(checkpoint, native_record, tmp_path):
    import torch
    train, dev = tmp_path / "train", tmp_path / "dev"
    train.mkdir()
    dev.mkdir()
    torch.save(dict(native_record, prompt_id="train"), train / "train.pt")
    torch.save(dict(native_record, prompt_id="dev"), dev / "dev.pt")
    output = tmp_path / "longer.safetensors"
    command = [sys.executable, str(SCRIPT), "--model", str(checkpoint.path),
               "--train-dir", str(train), "--eval-dir", str(dev), "--output", str(output),
               "--steps", "1001", "--grad-accum", "1", "--checkpoint-every", "1000", "--max-length", "16",
               "--recursive-depth", "4", "--roots", "2", "--logits-chunk", "2",
               "--lr", "0.00001", "--seed", "19", "--device", "cpu"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["optimizer_steps"] == len(report["train_steps"]) == 1001
    assert report["counts"]["sequences_seen"] == 1001
    assert [row["step"] for row in report["checkpoints"]] == [0, 1000, 1001]
    assert output.exists()


@pytest.mark.parametrize("zero_head", [False, True])
def test_public_recursive_cli_epochs_export_reload_and_dev_selection(
    trainer, checkpoint, native_record, tmp_path, zero_head
):
    torch = trainer.runtime().torch
    if zero_head:
        # Real CLI tie case: frozen zero head gives equal CE at every stage, so
        # even after actual optimizer steps stock must remain the eligible winner.
        key = checkpoint.LM_HEAD_KEY
        trainer.runtime().save_file({key: torch.zeros_like(checkpoint.tensor(key))},
                                    checkpoint.path / "lm_head.safetensors")
    train, dev, fresh_test = (tmp_path / name for name in ("train", "dev", "fresh-test"))
    for directory in (train, dev, fresh_test):
        directory.mkdir()
    for i in range(3):
        torch.save(dict(native_record, prompt_id=f"train-{i}"), train / f"{i}.pt")
    torch.save(dict(native_record, prompt_id="dev"), dev / "dev.pt")
    # There is no test-set CLI input. Traversing sibling directories would fail.
    (fresh_test / "unread.pt").write_bytes(b"NOT A CAPTURE; MUST NEVER READ")
    output = tmp_path / "recursive.safetensors"
    command = [sys.executable, str(SCRIPT), "--model", str(checkpoint.path),
               "--train-dir", str(train), "--eval-dir", str(dev), "--output", str(output),
               "--epochs", "1", "--grad-accum", "2", "--logits-chunk", "2", "--lr", "0.00001",
               "--recursive-depth", "4", "--roots", "2", "--depth-weights", "1", "0.5", "0.25", "0.125",
               "--checkpoint-every", "1", "--max-length", "16", "--seed", "19", "--device", "cpu"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["mode"] == "recursive_teacher_forcing" and report["optimizer_steps"] == 2
    assert [step["sequences"] for step in report["train_steps"]] == [2, 1]
    assert report["counts"] == {"train_sequences": 3, "dev_sequences": 1, "sequences_seen": 3,
                                "input_tokens_seen": 36, "observed_hidden_rows_seen": 33,
                                "useful_positions_seen": 30, "loss_tokens_by_depth": [30, 6, 6, 6]}
    assert report["memory_peak"]["process_max_rss_kib"] > 0
    assert [row["step"] for row in report["checkpoints"]] == [0, 1, 2]
    for candidate in report["checkpoints"]:
        with trainer.runtime().safe_open(candidate["path"], framework="pt") as handle:
            assert set(handle.keys()) == set(checkpoint.shapes)
            assert all(handle.get_tensor(key).dtype == torch.bfloat16 for key in handle.keys())
            if candidate["step"] == 0:
                assert all(torch.equal(handle.get_tensor(key), value) for key, value in checkpoint.mtp_state().items())
        for metrics in candidate["metrics"].values():
            assert [row["tokens"] for row in metrics["depths"]] == [10, 2, 2, 2]
    assert report["eval_before"] == report["checkpoints"][0]["metrics"]["BF16_export"]
    assert report["eval_after"] == report["checkpoints"][-1]["metrics"]["BF16_export"]
    assert report["dev_selection"] == trainer.select_dev_checkpoint(report["checkpoints"])
    if zero_head:
        assert report["dev_selection"]["step"] == 0
    else:
        with trainer.runtime().safe_open(output, framework="pt") as handle:
            assert any(not torch.equal(handle.get_tensor(key), value) for key, value in checkpoint.mtp_state().items())
    # Keep exact public-boundary commands and stage/count evidence in pytest -s output.
    print(json.dumps({"command": command, "report": report}, allow_nan=False))


def test_kl_direction_temperature_and_extreme_logits(trainer):
    torch = trainer.runtime().torch
    student = torch.tensor([[2.0, -1.0, 0.5], [-0.3, 1.5, 0.2]], requires_grad=True)
    teacher = torch.tensor([[-1.0, 2.0, 0.0], [1.2, -0.4, 0.1]], requires_grad=True)
    temperature = 2.0
    value = trainer.kl_divergence(student, teacher, temperature=temperature)
    expected = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student.detach() / temperature, dim=-1),
        torch.nn.functional.softmax(teacher.detach() / temperature, dim=-1),
        reduction="sum") * temperature ** 2
    torch.testing.assert_close(value.detach(), expected)
    reverse = trainer.kl_divergence(teacher.detach(), student.detach(), temperature=temperature)
    assert not torch.allclose(value.detach(), reverse.detach())
    value.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher.grad is None
    extreme = trainer.kl_divergence(
        torch.tensor([[10000.0, -10000.0]], requires_grad=True),
        torch.tensor([[-10000.0, 10000.0]]),
        temperature=3.0)
    assert torch.isfinite(extreme) and extreme.item() > 0


def test_kl_teacher_rows_use_future_indices_and_masks(trainer, config, native_record):
    torch = trainer.runtime().torch
    record = dict(native_record)
    record["positions"] = torch.arange(10, dtype=torch.int64)
    record["target_last_hidden_states"] = native_record["target_last_hidden_states"][:10].clone()
    record["loss_mask"] = native_record["loss_mask"].clone()
    record["loss_mask"][5] = False
    record["target_last_hidden_states"][4, 0] = float("nan")
    roots = [0, 2, 6]
    outputs = [(
        torch.zeros(10, config.hidden_size),
        torch.zeros(10, dtype=torch.int64),
        record["loss_mask"][2:],
    )]
    outputs.extend((
        torch.zeros(3, config.hidden_size),
        torch.zeros(3, dtype=torch.int64),
        torch.ones(3, dtype=torch.bool),
    ) for _ in range(3))
    pairs = trainer.kl_teacher_pairs(record, outputs, 4, roots, "cpu")
    assert [int(mask.sum()) for _, mask in pairs] == [8, 2, 3, 1]
    assert trainer.kl_pair_counts(record, 4, roots) == [8, 2, 3, 1]
    assert torch.equal(pairs[0][0][0], record["target_last_hidden_states"][1])
    assert torch.equal(pairs[1][0][0], record["target_last_hidden_states"][2])
    assert all(not teacher.requires_grad for teacher, _ in pairs)


def test_kl_zero_weight_equivalence_and_frozen_teacher(trainer):
    torch = trainer.runtime().torch
    torch.manual_seed(101)
    head = torch.nn.Linear(4, 5, bias=False).bfloat16()
    head.weight.requires_grad_(False)
    seed = torch.randn(4, 4)
    labels = torch.tensor([0, 2, 1, 4], dtype=torch.int64)
    mask = torch.ones(4, dtype=torch.bool)
    teacher = torch.randn(4, 4)
    def calculate(weight, teacher_state):
        hidden = seed.clone().requires_grad_()
        losses = trainer.depth_losses(
            [(hidden, labels, mask)], head, [1], chunk_tokens=2, backward=True,
            teachers=[(teacher_state, mask)], kl_weight=weight, kl_temperature=2.0)
        return hidden, trainer.loss_metrics(losses, [1])
    baseline_hidden, baseline = calculate(0.0, teacher.clone().requires_grad_())
    zero_hidden, zero = calculate(0.0, teacher.clone().requires_grad_())
    torch.testing.assert_close(zero_hidden.grad, baseline_hidden.grad)
    assert zero == baseline
    assert "kl" not in zero and "kl_pairs" not in zero
    enabled_teacher = teacher.clone().requires_grad_()
    enabled_hidden, enabled = calculate(1.0, enabled_teacher)
    assert enabled["kl_pairs"] == 4 and enabled["kl"] > 0
    assert enabled["depths"][0]["kl_pairs"] == 4
    assert enabled_hidden.grad is not None and torch.isfinite(enabled_hidden.grad).all()
    assert enabled_teacher.grad is None
    assert head.weight.grad is None


def test_recursive_kl_backward_reaches_base_and_branch_states(
    trainer, checkpoint, native_record
):
    torch = trainer.runtime().torch
    model = trainer.build_native_mtp(checkpoint.config, checkpoint.mtp_state())
    embedding, head = trainer.frozen_heads(checkpoint, "cpu")
    roots = [2]
    seen = []
    def retain_output(_, args, output):
        output.retain_grad()
        seen.append(output)
    hook = model.register_forward_hook(retain_output)
    outputs = trainer.sequence_depths(model, embedding, native_record, "cpu", 4, roots)
    teachers = trainer.kl_teacher_pairs(native_record, outputs, 4, roots, "cpu")
    normalizers = trainer.kl_pair_counts(native_record, 4, roots)
    losses = trainer.depth_losses(
        outputs, head, [0, 0, 0, 1], chunk_tokens=1, backward=True,
        teachers=teachers, kl_weight=1.0, kl_temperature=1.0,
        kl_normalizers=normalizers)
    hook.remove()
    assert [loss["kl_pairs"] for loss in losses] == [10, 1, 1, 1]
    assert seen[0][2].abs().sum() > 0
    assert len(seen) == 4 and all(output.grad.abs().sum() > 0 for output in seen[1:])
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.parameters())


def test_kl_cli_defaults_are_off_and_temperature_is_one(trainer):
    args = trainer.parser().parse_args(["--model", "unused", "--output", "unused.safetensors"])
    assert args.kl_weight == 0.0
    assert args.kl_temperature == 1.0
    assert args.greedy_kl is False and args.validate_only is False


@pytest.fixture
def full_checkpoint(tmp_path, trainer, native_raw_config):
    """A real, tiny original multimodal architecture, not a CausalLM key proxy."""
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    torch = trainer.runtime().torch
    native_raw_config["text_config"].update(
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_conv_kernel_dim=4)
    native_raw_config["vision_config"] = {
        "model_type": "qwen3_5_vision", "hidden_size": 32, "intermediate_size": 64,
        "num_heads": 4, "depth": 1, "out_hidden_size": 128, "patch_size": 2,
        "temporal_patch_size": 1, "spatial_merge_size": 2, "num_position_embeddings": 16,
    }
    torch.manual_seed(137)
    original = Qwen3_5ForConditionalGeneration(Qwen3_5Config.from_dict(native_raw_config)).bfloat16().eval()
    model = tmp_path / "full-original"
    original.save_pretrained(model, max_shard_size="100KB")
    config = trainer.native_config(json.loads((model / "config.json").read_text()))
    mtp = state_for(trainer, config)
    trainer.runtime().save_file(mtp, model / "mtp.safetensors")
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].update({key: "mtp.safetensors" for key in mtp})
    index_path.write_text(json.dumps(index))
    return trainer.Checkpoint(model)


@pytest.fixture
def full_target(trainer, full_checkpoint):
    return trainer.FrozenTarget(full_checkpoint, "cpu")


@pytest.fixture
def replay_record(trainer, full_target, native_record):
    torch = trainer.runtime().torch
    with torch.no_grad():
        hidden = full_target.text(input_ids=native_record["input_ids"][None], use_cache=False).last_hidden_state[0]
    return dict(native_record, positions=torch.arange(12), target_last_hidden_states=hidden)


def greedy_args(trainer, checkpoint, tmp_path, **overrides):
    args = trainer.parser().parse_args([
        "--model", str(checkpoint.path), "--output", str(tmp_path / "validation.safetensors"),
        "--device", "cpu", "--recursive-depth", "4", "--roots", "8",
        "--depth-weights", "1", "1", ".8", ".8", "--kl-weight", "1",
        "--greedy-kl", "--validate-only", "--grad-accum", "1", "--logits-chunk", "2",
    ])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_greedy_teacher_rows_come_from_the_same_request_capture(trainer, replay_record):
    torch = trainer.runtime().torch
    tokens = replay_record["input_ids"]
    root = 2
    draft = tokens[root + 2:root + 6].clone()
    rows, valid = trainer.native_greedy_teacher_rows(replay_record, [root], draft.view(1, 4))
    assert rows.shape == (1, 4, replay_record["target_last_hidden_states"].shape[1])
    assert valid.tolist() == [[True, True, True, True]]
    for depth in range(4):
        assert torch.equal(rows[0, depth], replay_record["target_last_hidden_states"][root + depth + 1])
    changed = draft.clone()
    changed[1] += 1
    _, masked = trainer.native_greedy_teacher_rows(replay_record, [root], changed.view(1, 4))
    assert masked.tolist() == [[True, True, False, False]]


def test_full_target_original_structure_shared_weights_and_fresh_prefix_rows(
    trainer, full_checkpoint, full_target, replay_record
 ):
    torch = trainer.runtime().torch
    assert type(full_target.model).__name__ == "Qwen3_5ForConditionalGeneration"
    assert hasattr(full_target.model.model, "visual")
    assert full_target.embedding is full_target.text.embed_tokens
    assert full_target.head is full_target.model.lm_head
    assert torch.equal(full_target.head.weight.cpu(), full_checkpoint.tensor("lm_head.weight"))
    assert torch.equal(full_target.embedding.weight.cpu(), full_checkpoint.tensor(full_checkpoint.EMBEDDING_KEY))
    assert all(not p.requires_grad and p.grad is None for p in full_target.model.parameters())
    assert not full_target.model.training
    proposals = torch.tensor([[25, 24, 23, 22], [21, 20, 19, 18]])
    roots, calls = [1, 5], []
    def capture(_, args, kwargs):
        calls.append(kwargs)
    hook = full_target.text.register_forward_pre_hook(capture, with_kwargs=True)
    rows = full_target.replay(replay_record, roots, proposals)
    hook.remove()
    assert len(calls) == 2 and rows.shape == (2, 4, 128) and not rows.requires_grad
    for index, (root, call) in enumerate(zip(roots, calls)):
        expected = torch.cat([replay_record["input_ids"][:root + 2], proposals[index, :3]])
        assert torch.equal(call["input_ids"][0], expected)
        assert call["past_key_values"] is None and call["use_cache"] is False
        assert call["position_ids"].tolist() == [list(range(root + 5))]
        with torch.no_grad():
            for d in range(4):
                prefix = torch.cat([replay_record["input_ids"][:root + 2], proposals[index, :d]])
                sequential = full_target.text(input_ids=prefix[None], use_cache=False).last_hidden_state[0, -1]
                torch.testing.assert_close(rows[index, d], sequential, atol=0.016, rtol=0.016)
    changed = dict(replay_record, target_last_hidden_states=torch.full_like(replay_record["target_last_hidden_states"], 999))
    assert torch.equal(rows, full_target.replay(changed, roots, proposals))


@pytest.mark.parametrize("divergence", [None, 1, 2, 3])
def test_greedy_own_argmax_offsets_and_divergent_teacher_history(
    trainer, full_checkpoint, full_target, replay_record, monkeypatch, divergence
 ):
    torch = trainer.runtime().torch
    model = trainer.build_native_mtp(full_checkpoint.config, full_checkpoint.mtp_state())
    root = 2
    expected = replay_record["input_ids"][root + 2:root + 6].clone()
    if divergence is not None:
        expected[divergence - 1] = 31
    calls, chosen = [], []
    original = full_target.head.forward
    def forced_argmax(hidden):
        logits = original(hidden)
        logits[:, expected[len(chosen)]] += 100
        chosen.append(logits.argmax(-1))
        return logits
    monkeypatch.setattr(full_target.head, "forward", forced_argmax)
    def capture(_, args, kwargs):
        calls.append((args[0].clone(), args[2].clone(), kwargs["past_key_values"].get_seq_length()))
    hook = model.register_forward_pre_hook(capture, with_kwargs=True)
    states, draft = trainer.greedy_depths(model, full_target.embedding, full_target.head, replay_record, "cpu", [root])
    hook.remove()
    monkeypatch.setattr(full_target.head, "forward", original)
    assert torch.equal(draft[0], expected) and not draft.requires_grad
    assert calls[0][0].tolist() == list(range(1, 11))
    for d, (tokens, positions, cache_length) in enumerate(calls[1:], start=1):
        assert tokens.tolist() == [expected[d - 1].item()]
        assert positions.tolist() == [root + d]
        assert cache_length == root + d
    replay_calls = []
    hook = full_target.text.register_forward_pre_hook(
        lambda _, args, kwargs: replay_calls.append(kwargs["input_ids"].clone()), with_kwargs=True)
    teacher_rows = full_target.replay(replay_record, [root], draft)
    hook.remove()
    assert replay_calls[0][0].tolist() == list(range(root + 2)) + expected[:3].tolist()
    if divergence is None:
        for d in range(4):
            # CPU BF16 hybrid prefix-vs-full is not bitwise; a wrong row is far outside this band.
            torch.testing.assert_close(teacher_rows[0, d], replay_record["target_last_hidden_states"][root + d + 1],
                                       atol=0.05, rtol=0.25)
    else:
        # Later matching tokens do not repair an earlier changed prefix.
        assert not torch.equal(teacher_rows[0, divergence], replay_record["target_last_hidden_states"][root + divergence + 1])
    assert all(state.requires_grad for state in states)


def test_greedy_prefix_cache_isolation_append_and_no_future_capture_leak(
    trainer, native_checkpoint, native_record, monkeypatch
 ):
    torch = trainer.runtime().torch
    model = trainer.build_native_mtp(native_checkpoint.config, native_checkpoint.mtp_state())
    embedding, head = trainer.frozen_heads(native_checkpoint, "cpu")
    original, bases, branches = trainer.prefix_cache, [], []
    def track(cache, length):
        bases.append((cache, cache.layers[0].keys.clone(), cache.layers[0].values.clone()))
        result = original(cache, length)
        branches.append((result, length))
        return result
    monkeypatch.setattr(trainer, "prefix_cache", track)
    roots = [1, 4, 6]
    states, draft = trainer.greedy_depths(model, embedding, head, native_record, "cpu", roots)
    assert len({id(cache) for cache, _ in branches}) == 3
    for cache, length in branches:
        assert cache.get_seq_length() == length + 3
    for cache, keys, values in bases:
        assert torch.equal(cache.layers[0].keys, keys) and torch.equal(cache.layers[0].values, values)
    reversed_states, reversed_draft = trainer.greedy_depths(model, embedding, head, native_record, "cpu", roots[::-1])
    assert torch.equal(draft, reversed_draft.flip(0))
    for actual, reverse in zip(states, reversed_states):
        assert torch.equal(actual, reverse.flip(0))
    changed = dict(native_record)
    changed["target_last_hidden_states"] = native_record["target_last_hidden_states"].clone()
    changed["target_last_hidden_states"][2:] *= -100
    changed["input_ids"] = native_record["input_ids"].clone()
    changed["input_ids"][3:] = 30
    for record in (native_record, changed, native_record):
        solo, tokens = trainer.greedy_depths(model, embedding, head, record, "cpu", [1])
        assert torch.equal(tokens[0], draft[0])
        for actual, expected in zip(solo, states):
            assert torch.equal(actual, expected[:1])


def test_greedy_kl_backprop_through_bf16_cast_state_and_base_kv_not_teacher(
    trainer, full_checkpoint, full_target, replay_record, monkeypatch
 ):
    torch = trainer.runtime().torch
    model = trainer.build_native_mtp(full_checkpoint.config, full_checkpoint.mtp_state())
    replay_record["target_last_hidden_states"].requires_grad_()
    seen, kv = [], []
    def retain(_, args, output):
        output.retain_grad()
        seen.append(output)
    hook = model.register_forward_hook(retain)
    original = trainer.prefix_cache
    def retain_cache(cache, length):
        for tensor in (cache.layers[0].keys, cache.layers[0].values):
            tensor.retain_grad()
            kv.append(tensor)
        return original(cache, length)
    monkeypatch.setattr(trainer, "prefix_cache", retain_cache)
    states, tokens = trainer.greedy_depths(model, full_target.embedding, full_target.head, replay_record, "cpu", [2])
    target = full_target.replay(replay_record, [2], tokens)
    with trainer.autocast("cpu", dtype=torch.bfloat16):
        loss = trainer.kl_divergence(full_target.head(states[3]), full_target.head(target[:, 3]))
    loss.backward()
    hook.remove()
    assert len(seen) == 4 and all(s.grad is not None and s.grad.abs().sum() > 0 for s in seen)
    for tensor in kv:
        assert tensor.grad[..., :3, :].abs().sum() > 0
        assert tensor.grad[..., 3:, :].count_nonzero() == 0
    assert all(p.dtype == torch.float32 and p.grad is not None and torch.isfinite(p.grad).all()
               and p.grad.abs().sum() > 0 for p in model.parameters())
    assert replay_record["target_last_hidden_states"].grad is None
    assert all(p.grad is None and not p.requires_grad for p in full_target.model.parameters())


def test_greedy_ce_anchor_masks_missing_rows_and_chunked_dense_gradients(
    trainer, full_checkpoint, full_target, replay_record, tmp_path
 ):
    torch = trainer.runtime().torch
    args = greedy_args(trainer, full_checkpoint, tmp_path)
    state = full_checkpoint.mtp_state()
    model, reference = (trainer.build_native_mtp(full_checkpoint.config, state) for _ in range(2))
    replay_record["positions"] = replay_record["positions"][:10]
    replay_record["target_last_hidden_states"] = replay_record["target_last_hidden_states"][:10]
    normalizers, kl_normalizers = [20, 3, 3, 3], [18, 3, 3, 3]
    dense_loss = 0
    for roots in ([0, 2], [6]):
        outputs, teachers, students = trainer.objective_inputs(
            model, full_target.embedding, full_target.head, replay_record, args, roots, full_target)
        control = trainer.sequence_depths(model, full_target.embedding, replay_record, "cpu", 4, roots)
        for actual, expected in zip(outputs, control):
            assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        losses = trainer.depth_losses(
            outputs, full_target.head, args.depth_weights, chunk_tokens=1, backward=True,
            teachers=teachers, kl_students=students, kl_weight=1, kl_temperature=2,
            normalizers=normalizers, kl_normalizers=kl_normalizers)
        assert [loss["tokens"] for loss in losses] == [10] + [len(roots)] * 3
        assert [loss["kl_pairs"] for loss in losses] == [9, 0, 0, 0]
        assert trainer.objective_pair_counts(replay_record, args, roots) == [9] + [len(roots)] * 3
        with torch.no_grad():
            ce_only = trainer.depth_losses(control, full_target.head, args.depth_weights, chunk_tokens=1)
        assert all(a["loss_sum"] == b["loss_sum"] and a["correct"] == b["correct"] for a, b in zip(losses, ce_only))
        dense, target, branch = trainer.objective_inputs(
            reference, full_target.embedding, full_target.head, replay_record, args, roots, full_target)
        for d, ((hidden, labels, mask), (teacher_rows, valid)) in enumerate(zip(dense, target)):
            with trainer.autocast("cpu", dtype=torch.bfloat16):
                ce = torch.nn.functional.cross_entropy(full_target.head(hidden)[mask].float(), labels[mask], reduction="sum")
                kl = trainer.kl_divergence(
                    full_target.head(hidden if d == 0 else branch[d])[valid],
                    full_target.head(teacher_rows)[valid], temperature=2)
            dense_loss = dense_loss + args.depth_weights[d] * (ce / normalizers[d] + kl / kl_normalizers[d])
    dense_loss.backward()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        # BF16 accumulation order differs between token chunks and dense GEMMs.
        torch.testing.assert_close(actual.grad, expected.grad, atol=0.003, rtol=0.03)
    assert full_target.head.weight.grad is None and full_target.embedding.weight.grad is None


def test_greedy_empty_roots_and_nonfinite_teacher_replay(
    trainer, full_checkpoint, full_target, replay_record, tmp_path, monkeypatch
 ):
    torch = trainer.runtime().torch
    args = greedy_args(trainer, full_checkpoint, tmp_path)
    model = trainer.build_native_mtp(full_checkpoint.config, full_checkpoint.mtp_state())
    outputs, teachers, students = trainer.objective_inputs(
        model, full_target.embedding, full_target.head, replay_record, args, [], full_target)
    losses = trainer.depth_losses(outputs, full_target.head, args.depth_weights, chunk_tokens=2, backward=True,
                                  teachers=teachers, kl_students=students, kl_weight=1)
    assert [loss["tokens"] for loss in losses] == [10, 0, 0, 0]
    assert [loss["kl_pairs"] for loss in losses] == [10, 0, 0, 0]
    monkeypatch.setattr(full_target.text, "forward", lambda **kw: SimpleNamespace(
        last_hidden_state=torch.full((1, kw["input_ids"].shape[1], 128), float("nan"))))
    with pytest.raises(trainer.TrainingError, match="Nonfinite") :
        full_target.replay(replay_record, [0], torch.tensor([[10, 11, 12, 13]]))


def test_bf16_current_parameter_view_equals_real_export_reload(
    trainer, native_checkpoint, native_record, tmp_path
 ):
    torch = trainer.runtime().torch
    model = trainer.build_native_mtp(native_checkpoint.config, native_checkpoint.mtp_state())
    embedding, head = trainer.frozen_heads(native_checkpoint, "cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.003)
    path = tmp_path / "current.safetensors"
    trainer.export_mtp({"mtp." + key: value for key, value in model.state_dict().items()}, native_checkpoint, path, {})
    with trainer.runtime().safe_open(path, framework="pt") as handle:
        state = {key: handle.get_tensor(key) for key in handle.keys()}
    reloaded = trainer.build_native_mtp(native_checkpoint.config, state).bfloat16()
    # RoPE's nonpersistent inv_freq is FP32 in the actual serving/export view.
    reloaded.rotary_emb = trainer.runtime().Rotary(native_checkpoint.config)
    states, proposals = trainer.greedy_depths(model, embedding, head, native_record, "cpu", [1, 4])
    with torch.no_grad(), trainer.autocast("cpu", dtype=torch.bfloat16):
        ids, hidden, positions, _, _ = trainer.aligned_inputs(native_record)
        cache = trainer.runtime().Cache()
        base = reloaded(ids, hidden.bfloat16(), positions, embedding, past_key_values=cache)
        for index, root in enumerate([1, 4]):
            branch, previous = trainer.prefix_cache(cache, root + 1), base[root:root + 1]
            for d in range(4):
                if d:
                    previous = reloaded(token, previous, positions[root:root + 1] + d, embedding, past_key_values=branch)
                assert torch.equal(previous, states[d][index:index + 1])
                token = head(previous).argmax(-1)
                assert token.item() == proposals[index, d].item()


@pytest.mark.parametrize("extra,match", [
    (["--greedy-kl"], "depth 4"),
    (["--greedy-kl", "--recursive-depth", "4"], "positive KL"),
    (["--greedy-kl", "--recursive-depth", "4", "--kl-weight", "1", "--export-stock"], "export-stock"),
    (["--validate-only"], "depth 4"),
])
def test_greedy_incompatible_flags_fail_before_weight_reads(trainer, extra, match):
    args = trainer.parser().parse_args(["--model", "absent", "--output", "unused.safetensors"] + extra)
    with pytest.raises(trainer.TrainingError, match=match):
        trainer.run(args)


def test_greedy_architecture_precision_revision_and_positions_guards(
    trainer, full_checkpoint, checkpoint, native_checkpoint, replay_record, tmp_path, monkeypatch
 ):
    torch = trainer.runtime().torch
    with pytest.raises(trainer.TrainingError, match="unquantized"):
        trainer.validate_greedy_checkpoint(checkpoint, "cpu")
    with pytest.raises(trainer.TrainingError, match="original.*architecture"):
        trainer.validate_greedy_checkpoint(native_checkpoint, "cpu")
    with pytest.raises(trainer.TrainingError, match="exact BF16 model revision"):
        trainer.validate_greedy_checkpoint(full_checkpoint, "cuda")
    full_checkpoint.raw["_commit_hash"] = trainer.PINNED_BF16_REVISION
    trainer.validate_greedy_checkpoint(full_checkpoint, "cuda")
    for positions in (torch.arange(12) * 2, torch.arange(12) + 1):
        record = dict(replay_record, positions=positions)
        trainer.validate_record(record, full_checkpoint.config, 16)  # Deliberately valid legacy input.
        with pytest.raises(trainer.TrainingError, match="contiguous native positions starting at zero"):
            trainer.validate_greedy_positions(record)
    train = tmp_path / "gapped"
    train.mkdir()
    torch.save(record, train / "record.pt")
    args = greedy_args(trainer, full_checkpoint, tmp_path, train_dir=str(train))
    monkeypatch.setattr(trainer, "FrozenTarget", lambda *a: pytest.fail("invalid input loaded teacher"))
    with pytest.raises(trainer.TrainingError, match="contiguous native"):
        trainer.run(args)


@pytest.mark.parametrize("greedy", [False, True])
def test_public_cli_original_multimodal_validate_only(
    trainer, full_checkpoint, replay_record, tmp_path, greedy
 ):
    torch = trainer.runtime().torch
    train, dev = tmp_path / "train", tmp_path / "dev"
    train.mkdir()
    dev.mkdir()
    torch.save(replay_record, train / "train.pt")
    torch.save(dict(replay_record, prompt_id="dev-diagnostic"), dev / "dev.pt")
    output = tmp_path / "validated.safetensors"
    command = [sys.executable, str(SCRIPT), "--model", str(full_checkpoint.path),
               "--train-dir", str(train), "--eval-dir", str(dev), "--output", str(output),
               "--validate-only", "--recursive-depth", "4", "--roots", "8",
               "--depth-weights", "1", "1", ".8", ".8", "--device", "cpu", "--seed", "42",
               "--lr", "1e-6", "--max-length", "2048", "--grad-accum", "1", "--logits-chunk", "64",
               "--kl-weight", "1", "--kl-temperature", "1"] + (["--greedy-kl"] if greedy else [])
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["status"] == "validation-only" and report["optimizer_steps"] == 0
    assert report["train_steps"] == [] and report["checkpoints"] == []
    assert report["counts"]["loss_tokens_by_depth"] == [20, 14, 14, 14]
    assert report["counts"]["kl_pairs_by_depth"] == ([20, 0, 0, 0] if greedy else [20, 14, 14, 14])
    assert report["counts"]["sampled_roots"] == 14
    assert report["recurrence"]["teacher_replays"] == 0
    assert report["recurrence"].get("native_teacher_rows", 0) == (14 if greedy else 0)
    assert report["validation"]["ce_anchor_identical"]
    assert all(report["validation"]["gradient_checks"].values())
    export = report["validation"]["export_reload"]
    assert export["greedy_proposals_equal"] and export["proposal_tokens"] == 56
    assert export["stock_mtp_sha256"] == export["export_mtp_sha256"]
    with trainer.runtime().safe_open(output, framework="pt") as handle:
        assert all(torch.equal(handle.get_tensor(key), tensor) for key, tensor in full_checkpoint.mtp_state().items())
    print(json.dumps({"command": command, "report": report}, allow_nan=False))


def test_validate_only_constructs_no_optimizer_and_off_path_no_teacher(
    trainer, native_checkpoint, native_record, tmp_path, monkeypatch
 ):
    torch = trainer.runtime().torch
    train = tmp_path / "train"
    train.mkdir()
    torch.save(native_record, train / "train.pt")
    args = greedy_args(trainer, native_checkpoint, tmp_path, train_dir=str(train), greedy_kl=False)
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **kw: pytest.fail("validation constructed an optimizer"))
    monkeypatch.setattr(trainer, "FrozenTarget", lambda *a: pytest.fail("off path loaded a teacher"))
    report = trainer.run(args)
    assert report["optimizer_steps"] == 0 and report["recurrence"]["teacher_replays"] == 0


def test_greedy_off_objective_outputs_gradients_and_report_unchanged(
    trainer, native_checkpoint, native_record, tmp_path, monkeypatch
 ):
    torch = trainer.runtime().torch
    args = greedy_args(trainer, native_checkpoint, tmp_path, greedy_kl=False, validate_only=False)
    model, reference = (trainer.build_native_mtp(native_checkpoint.config, native_checkpoint.mtp_state()) for _ in range(2))
    embedding, head = trainer.frozen_heads(native_checkpoint, "cpu")
    monkeypatch.setattr(trainer, "FrozenTarget", lambda *a: pytest.fail("off path loaded a teacher"))
    outputs, teachers, students = trainer.objective_inputs(model, embedding, head, native_record, args, [1, 4])
    legacy = trainer.sequence_depths(reference, embedding, native_record, "cpu", 4, [1, 4])
    captured = trainer.kl_teacher_pairs(native_record, legacy, 4, [1, 4], "cpu")
    assert students is None
    actual = trainer.depth_losses(outputs, head, args.depth_weights, chunk_tokens=2, backward=True,
                                   teachers=teachers, kl_students=students, kl_weight=1)
    expected = trainer.depth_losses(legacy, head, args.depth_weights, chunk_tokens=2, backward=True,
                                     teachers=captured, kl_weight=1)
    assert actual == expected
    for a, b in zip(model.parameters(), reference.parameters()):
        assert torch.equal(a.grad, b.grad)
    args.export_stock = True
    report = trainer.run(args)
    assert "greedy_kl" not in report["config"] and "validate_only" not in report["config"]
    assert "recurrence" not in report and "validation" not in report and "status" not in report
