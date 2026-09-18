"""Focused contracts for the evaluation-only Qwen MTP acceptance scorer."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/experiments/qwen38_mtp_acceptance.py"


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture(scope="module")
def trainer(torch):
    pytest.importorskip("safetensors")
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("qwen38_train_mtp_acceptance_tests", ROOT / "scripts/experiments/qwen38_train_mtp.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def acceptance():
    spec = importlib.util.spec_from_file_location("qwen38_mtp_acceptance_tests", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def raw_config():
    return {
        "model_type": "qwen3_5", "tie_word_embeddings": False,
        "quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": True},
        "text_config": {
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
def config(trainer, raw_config):
    return trainer.native_config(raw_config)


def state_for(trainer, config):
    torch = trainer.runtime().torch
    torch.manual_seed(7)
    return {
        key: (torch.zeros(shape) if len(shape) == 1 else torch.randn(shape) * 0.02).bfloat16()
        for key, shape in trainer.expected_mtp_shapes(config).items()
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
    mapping["model.language_model.layers.0.self_attn.q_proj.qweight"] = "NO_VERIFIER_READ.safetensors"
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    return trainer.Checkpoint(model)


def record_for(torch, hidden_size, *, length=12, observed=None, mask=None, prompt_id="dev-1"):
    observed = length - 2 if observed is None else observed
    if mask is None:
        mask = [False, False] + [True] * (length - 2)
    return {
        "prompt_id": prompt_id,
        "input_ids": torch.arange(length, dtype=torch.int64),
        "positions": torch.arange(observed, dtype=torch.int64),
        "target_last_hidden_states": torch.randn(observed, hidden_size),
        "loss_mask": torch.tensor(mask, dtype=torch.bool),
    }


@pytest.mark.parametrize(
    "correct,expected",
    [
        ([False, True, True, True], 0),
        ([True, False, True, True], 1),
        ([True, True, False, True], 2),
        ([True, True, True, False], 3),
        ([True, True, True, True], 4),
    ],
)
def test_first_mismatch_and_full_acceptance(acceptance, correct, expected):
    scores = [
        {"loss_sum": 1.0, "tokens": 1, "correct": int(value), "correct_by_root": [value]}
        for value in correct
    ]
    result = acceptance.summarize_scores(scores, eligible_roots=1)
    assert result["length_histogram"] == [int(expected == index) for index in range(5)]
    assert result["mean_sum_j"] == expected
    assert [row["count"] for row in result["survival"]] == [
        int(all(correct[:index + 1])) for index in range(4)
    ]


def test_later_matches_after_mismatch_are_masked(acceptance):
    correct = [True, False, True, True]
    scores = [
        {"loss_sum": 0.0, "tokens": 1, "correct": int(value), "correct_by_root": [value]}
        for value in correct
    ]
    result = acceptance.summarize_scores(scores, eligible_roots=1)
    assert [row["count"] for row in result["survival"]] == [1, 0, 0, 0]
    assert result["length_histogram"] == [0, 1, 0, 0, 0]


def test_boundary_terminal_and_hidden_coverage_conventions(acceptance, trainer, config, torch):
    # t=0 has only x[t+2] supervised, so it is the explicitly excluded prompt-2 root.
    record = record_for(torch, config.hidden_size, length=12, observed=11)
    eligibility = acceptance.root_eligibility(record)
    assert eligibility["eligible"] == [1, 2, 3, 4, 5, 6]
    assert eligibility["counts"] == {
        "candidate_roots": 10,
        "hidden_excluded": 0,
        "terminal_excluded": 3,
        "boundary_excluded": 1,
        "masked_excluded": 0,
        "eligible_roots": 6,
    }
    # H=T-2 and H=T-1 both use the same aligned base rows; no extra row is invented.
    short_hidden = record_for(torch, config.hidden_size, length=12, observed=10)
    assert acceptance.eligible_roots(short_hidden) == eligibility["eligible"]
    trainer.validate_record(short_hidden, config, 16)
    trainer.validate_record(record, config, 16)

    terminal_mask = [False, False] + [True] * 6 + [False] * 4
    terminal = record_for(torch, config.hidden_size, length=12, mask=terminal_mask)
    result = acceptance.root_eligibility(terminal)
    assert result["eligible"] == [1, 2]
    assert result["counts"]["terminal_excluded"] == 7


def test_fixed_root_order_and_label_digest(acceptance, torch):
    record = record_for(torch, 8, length=20, prompt_id="stable")
    first = acceptance.make_sequence_plan(Path("capture.pt"), record, family="family", roots=4)
    second = acceptance.make_sequence_plan(Path("capture.pt"), record, family="family", roots=4)
    assert first["root_order"] == sorted(first["root_order"])
    assert len(first["root_order"]) == len(set(first["root_order"])) == 4
    assert first["root_order"] == second["root_order"]
    assert first["plan_sha256"] == second["plan_sha256"]
    assert len(first["labels_sha256_by_depth"]) == 4
    assert all(len(row["label_token_ids_sha256"]) == 64 for row in first["root_identities"])


def test_sequence_depths_preserves_future_and_sibling_isolation(trainer, checkpoint, config, torch):
    record = record_for(torch, config.hidden_size, length=12, observed=11, prompt_id="native")
    model = trainer.build_native_mtp(config, checkpoint.mtp_state())
    embedding, _ = trainer.frozen_heads(checkpoint, "cpu")
    roots = [1, 4, 6]
    baseline = trainer.sequence_depths(model, embedding, record, "cpu", 4, roots)
    reversed_roots = trainer.sequence_depths(model, embedding, record, "cpu", 4, roots[::-1])
    solo = trainer.sequence_depths(model, embedding, record, "cpu", 4, [1])
    changed = dict(record)
    changed["target_last_hidden_states"] = record["target_last_hidden_states"].clone()
    changed["target_last_hidden_states"][2:] = torch.randn_like(changed["target_last_hidden_states"][2:]) * 50
    future = trainer.sequence_depths(model, embedding, changed, "cpu", 4, [1])
    for depth in range(1, 4):
        torch.testing.assert_close(baseline[depth][0], reversed_roots[depth][0].flip(0), atol=0, rtol=0)
        torch.testing.assert_close(baseline[depth][0][:1], solo[depth][0], atol=0, rtol=0)
        torch.testing.assert_close(baseline[depth][0][:1], future[depth][0], atol=0, rtol=0)


def test_stage_routing_and_checkpoint_are_read_only(acceptance, trainer, checkpoint, tmp_path, monkeypatch, torch):
    candidate = tmp_path / "candidate.safetensors"
    original = checkpoint.mtp_state()
    trainer.runtime().save_file(original, candidate)
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in checkpoint.path.rglob("*") if path.is_file()
    }
    seen = []

    def build(config, state, device):
        seen.append({key: value.detach().clone() for key, value in state.items()})
        return object()

    monkeypatch.setattr(acceptance.trainer, "build_native_mtp", build)
    monkeypatch.setattr(acceptance, "_evaluate_model", lambda *args, **kwargs: {"aggregate": {}, "families": {}, "sequences": []})
    embedding, head = object(), object()
    acceptance._candidate_report("stock", None, checkpoint, embedding, head, [], device="cpu", max_length=16, logits_chunk=2)
    assert len(seen) == 2
    assert all(value.dtype == torch.bfloat16 for value in seen[0].values())
    assert any(value.dtype == torch.float32 for value in seen[1].values() if value.ndim == 2)
    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in checkpoint.path.rglob("*") if path.is_file()
    }
    assert before == after
    assert all(torch.equal(original[key], value) for key, value in checkpoint.mtp_state().items())


def test_public_tiny_cli_roundtrip_and_exclusive_output(acceptance, trainer, checkpoint, config, tmp_path, monkeypatch, torch):
    captures = tmp_path / "captures"
    captures.mkdir()
    record = record_for(torch, config.hidden_size, length=12, observed=10, prompt_id="cli-1")
    record["metadata"] = {}
    torch.save(record, captures / "one.pt")
    prompts = tmp_path / "dev.jsonl"
    prompts.write_text(json.dumps({"id": "cli-1", "source_group": "tiny-family"}) + "\n")
    candidate_path = tmp_path / "tuned.safetensors"
    trainer.runtime().save_file(checkpoint.mtp_state(), candidate_path)
    output = tmp_path / "report.json"
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in checkpoint.path.rglob("*") if path.is_file()
    }
    original_adamw = torch.optim.AdamW
    monkeypatch.setattr(torch.optim, "AdamW", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("optimizer used")))
    assert acceptance.main([
        "--model", str(checkpoint.path), "--captures", str(captures), "--prompts", str(prompts),
        "--candidate", f"tuned={candidate_path}", "--output", str(output), "--device", "cpu",
        "--max-length", "16", "--roots", "1", "--logits-chunk", "2",
    ]) == 0
    monkeypatch.setattr(torch.optim, "AdamW", original_adamw)
    report = json.loads(output.read_text())
    assert report["mode"] == "evaluation_only_joint_prefix"
    assert [row["name"] for row in report["candidates"]] == ["stock", "tuned"]
    assert report["counts"]["selected_roots"] == 1
    assert report["root_plan"][0]["source_group"] == "tiny-family"
    for candidate in report["candidates"]:
        assert set(candidate["stages"]) == {"BF16_export", "RTN_effective_dense"}
        aggregate = candidate["stages"]["RTN_effective_dense"]["aggregate"]
        assert aggregate["evaluated_roots"] == 1
        assert sum(aggregate["length_histogram"]) == 1
        assert aggregate["root_denominator"] == 1
    assert before == {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in checkpoint.path.rglob("*") if path.is_file()
    }
    with pytest.raises(SystemExit, match="overwrite"):
        acceptance.main([
            "--model", str(checkpoint.path), "--captures", str(captures),
            "--output", str(output), "--device", "cpu", "--roots", "1",
        ])
