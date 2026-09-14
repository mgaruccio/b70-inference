"""Small protocol checks; the actual qualification uses the serving API on B70."""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "experiments"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("mtp_tune_probe_test", SCRIPTS / "qwen38_mtp_tune_probe.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_per_position_counter_is_retained():
    text = '\n'.join([
        'vllm:spec_decode_num_drafts_total{model_name="qwen38"} 10',
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="qwen38",position="0"} 8',
        'vllm:unrelated_total 99',
    ])
    with patch.object(module.probe, "get", return_value=text):
        counters = module.probe.counters()
    assert len(counters) == 2
    assert sum(counters.values()) == 18


def test_acceptance_uses_explicit_denominators():
    metrics = {
        'vllm:spec_decode_num_drafts_total{model_name="qwen38"}': 10,
        'vllm:spec_decode_num_draft_tokens_total{model_name="qwen38"}': 40,
        'vllm:spec_decode_num_accepted_tokens_total{model_name="qwen38"}': 20,
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"}': 8,
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="1"}': 6,
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="2"}': 4,
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="3"}': 2,
    }
    result = module.aggregate([{"metric_deltas": metrics, "decode_tps": 60, "elapsed_s": 5}])
    assert result["accepted_per_draft_pass"] == 2
    assert result["draft_token_acceptance"] == 0.5
    assert result["position_acceptance_per_draft_pass"] == {"0": .8, "1": .6, "2": .4, "3": .2}


def test_missing_acceptance_is_not_reported_as_zero_success():
    try:
        module.aggregate([{"metric_deltas": {}, "decode_tps": 60, "elapsed_s": 5}])
    except RuntimeError as error:
        assert "missing live" in str(error)
    else:
        raise AssertionError("missing counters accepted")


def test_functional_check_rejects_forced_decode(tmp_path):
    (tmp_path / "summary.json").write_text('{"status": "completed", "generate": false}')
    try:
        module.check_outputs(tmp_path)
    except ValueError as error:
        assert "natural-EOS" in str(error)
    else:
        raise AssertionError("forced decode accepted as natural output")


def test_replay_preserves_exact_ids_and_masks(tmp_path):
    import json
    import qwen38_mtp_replay as replay
    (tmp_path / "summary.json").write_text(json.dumps({"status": "completed", "generate": True}))
    (tmp_path / "prompts.jsonl").write_text(json.dumps({"id": "mbpp-601", "split": "train"}) + "\n")
    (tmp_path / "mbpp-601-result.json").write_text(json.dumps({
        "transport_pass": True, "prompt_token_ids": [1, 2, 3], "token_ids": [4, 5]}))
    assert list(replay.sequences([tmp_path])) == [{"name": "mbpp-601", "split": "train",
        "input_ids": [1, 2, 3, 4, 5], "loss_mask": [False, False, False, True, True]}]
    try:
        list(replay.sequences([tmp_path, tmp_path]))
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate prompts admitted")


def test_replay_parity_failure_is_explicit(tmp_path):
    import json
    import qwen38_mtp_replay as replay
    left, right = tmp_path / "a", tmp_path / "b"
    left.mkdir()
    right.mkdir()
    (left / "replay-outputs.json").write_text(json.dumps({"a": {"token_ids": [1], "text": "x"}}))
    (right / "replay-outputs.json").write_text(json.dumps({"a": {"token_ids": [2], "text": "y"}}))
    try:
        replay.compare_replays(left, right)
    except RuntimeError as error:
        assert "stop before training" in str(error)
    else:
        raise AssertionError("mismatched replay accepted")
    assert json.loads((right / "capture-parity.json").read_text())["mismatches"] == ["a"]
