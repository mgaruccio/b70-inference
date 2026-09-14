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
            "model_type": "qwen3_5_text", "hidden_size": 128, "intermediate_size": 192,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
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
    assert model.layers[0].self_attn.q_proj.weight.shape == (4 * 8 * 2, 128)
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
