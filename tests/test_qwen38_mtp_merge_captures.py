"""Synthetic validation of train-only trajectory deduplication."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/experiments/qwen38_mtp_merge_captures.py"


@pytest.fixture
def merger(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("test_mtp_merger", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_data(merger, tmp_path):
    rt = merger.native_runtime()
    source = rt.private_directory(tmp_path / "source", create=True)
    record = dict(prompt_id="train-prompt", input_ids=rt.torch.tensor([1, 2, 3, 4, 5]),
                  positions=rt.torch.arange(4), target_last_hidden_states=rt.torch.ones(4, 4),
                  loss_mask=rt.torch.tensor([False, False, True, True, True]),
                  metadata={"source_group": "train-family"})
    config = SimpleNamespace(hidden_size=4, vocab_size=64, max_position_embeddings=64)
    return rt, source, record, config


def test_merge_deduplicates_and_preserves_originals(merger, tmp_path):
    rt, source, record, config = fixture_data(merger, tmp_path)
    rt.save_tensor(source / "a.pt", record)
    rt.save_tensor(source / "b.pt", record)
    output = tmp_path / "merged"
    allowed = {"train-prompt": "train-family"}
    counts = merger.merge([source], output, allowed, config)
    assert counts["sequences"] == 1 and counts["useful_positions"] == 3
    assert counts["duplicate_records"] == 1 and counts["added_sequences"] == 1
    repeated = merger.merge([source], output, allowed, config)
    assert repeated["added_sequences"] == 0 and repeated["useful_positions"] == 3
    record["input_ids"][-1] = 6
    rt.save_tensor(source / "c.pt", record)
    expanded = merger.merge([source], output, allowed, config)
    assert expanded["sequences"] == 2 and expanded["useful_positions"] == 6
    assert len(list(source.glob("*.pt"))) == 3
    for path in output.glob("*.pt"):
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("kind", ["prompt", "family", "nonfinite"])
def test_merge_rejects_heldout_or_invalid_records(merger, tmp_path, kind):
    rt, source, record, config = fixture_data(merger, tmp_path)
    if kind == "prompt":
        record["prompt_id"] = "heldout-prompt"
    elif kind == "family":
        record["metadata"]["source_group"] = "heldout-family"
    else:
        record["target_last_hidden_states"][0, 0] = float("nan")
    rt.save_tensor(source / "bad.pt", record)
    with pytest.raises((ValueError, merger.trainer.TrainingError)):
        merger.merge([source], tmp_path / "merged", {"train-prompt": "train-family"}, config)
    assert not list((tmp_path / "merged").glob("*.pt"))


def test_merge_cli_uses_real_serialized_tensor_path(merger, tmp_path):
    rt, source, record, config = fixture_data(merger, tmp_path)
    rt.save_tensor(source / "a.pt", record)
    rt.save_json(tmp_path / "allowed.json", {"train-prompt": "train-family"})
    rt.save_json(tmp_path / "config.json", {"text_config": vars(config)})
    result = subprocess.run([sys.executable, str(SCRIPT), "--source", str(source),
                             "--output", str(tmp_path / "merged"), "--allowlist", str(tmp_path / "allowed.json"),
                             "--model-config", str(tmp_path / "config.json")],
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["useful_positions"] == 3
    assert "train-prompt" not in result.stdout
