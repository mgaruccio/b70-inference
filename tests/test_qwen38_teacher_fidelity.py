"""CPU contracts for the teacher-fidelity diagnostic. Not GPU or live-parity proof."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load():
    path = ROOT / "scripts/experiments/qwen38_teacher_fidelity.py"
    spec = importlib.util.spec_from_file_location("teacher_fidelity_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fid = load()


def entry(ids, name="a"):
    n = len(ids)
    return {"name": name, "input_ids": ids, "scored_teacher_positions": list(range(n - 4, n))}


def write_case(tmp_path, ids, hidden, hf_rows, positions=None, hf_prefix=None, native_ids=None, name="a"):
    hf_dir, native_dir = tmp_path / "hf", tmp_path / "native"
    hf_dir.mkdir(parents=True)
    native_dir.mkdir(parents=True)
    torch.save({"teacher_rows": hf_rows, "prefix": torch.tensor(ids if hf_prefix is None else hf_prefix)},
               hf_dir / f"{name}.pt")
    torch.save({"input_ids": torch.tensor(ids if native_ids is None else native_ids),
                "positions": torch.arange(len(ids)) if positions is None else positions,
                "target_last_hidden_states": hidden}, native_dir / f"{name}.pt")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([entry(ids, name)]))
    return manifest, hf_dir, native_dir


def test_offline_detects_swapped_row_without_wrap_or_model(tmp_path, monkeypatch):
    monkeypatch.setattr(fid.trainer, "Checkpoint", lambda *_: (_ for _ in ()).throw(AssertionError("model")))
    monkeypatch.setattr(fid.trainer, "FrozenTarget", lambda *_: (_ for _ in ()).throw(AssertionError("model")))
    ids = list(range(6))
    hidden = torch.eye(6, 8, dtype=torch.bfloat16)
    manifest, hf_dir, native_dir = write_case(tmp_path, ids, hidden, hidden[-4:].clone())
    report = fid.offline_report(manifest, hf_dir, native_dir)
    neighbors = report["rows"][0]["neighbors"]
    assert report["numeric_passed"] and report["aligned_best_histories"] == 1
    assert report["neighbor_better_histories"] == 0 and report["optimizer_updates"] == 0
    assert [item["position"] for item in neighbors] == [2, 3, 4, 5]
    assert neighbors[0]["minus"] is not None and neighbors[-1]["plus"] is None
    assert all(0 <= item["position"] + item["best_cosine_offset"] < 6 for item in neighbors)
    swapped = hidden[-4:].clone()
    swapped[0] = hidden[3]
    torch.save({"teacher_rows": swapped, "prefix": torch.tensor(ids)}, hf_dir / "a.pt")
    report = fid.offline_report(manifest, hf_dir, native_dir)
    assert report["rows"][0]["errors"]["pass"] is False
    assert report["neighbor_better_cosine_rows"] >= 1 and report["aligned_best_histories"] == 0
    assert report["rows"][0]["neighbors"][0]["best_cosine_offset"] == 1
    assert report["aligned_best_cosine_rows"] != report["neighbor_better_cosine_rows"]
    code = fid.main(["offline", "--manifest", str(manifest), "--hf-dir", str(hf_dir),
                     "--native-dir", str(native_dir), "--output", str(tmp_path / "out.json")])
    saved = json.loads((tmp_path / "out.json").read_text())
    assert code == 1 and saved["status"] == "teacher_parity_gate_open" and saved["optimizer"] is None


@pytest.mark.parametrize("kind", ["ids", "gap", "dtype"])
def test_offline_refuses_bad_ids_gaps_and_dtype(tmp_path, kind):
    ids = list(range(6))
    hidden = torch.eye(6, 8, dtype=torch.bfloat16)
    kwargs = {}
    if kind == "ids":
        kwargs["hf_prefix"] = ids[:-1] + [99]
    elif kind == "gap":
        kwargs["positions"] = torch.tensor([0, 1, 2, 3, 5, 6])
    else:
        hidden = hidden.float()
    manifest, hf_dir, native_dir = write_case(tmp_path, ids, hidden, torch.eye(6, 8, dtype=torch.bfloat16)[-4:], **kwargs)
    with pytest.raises(ValueError, match={"ids": "ID/prefix", "gap": "gapped", "dtype": "BF16"}[kind]):
        fid.offline_report(manifest, hf_dir, native_dir)


def test_offline_refuses_caps_short_prefix_and_empty_verifier(tmp_path, monkeypatch):
    path = tmp_path / "short.json"
    path.write_text(json.dumps([{"name": "a", "input_ids": [1, 2, 3, 4], "scored_teacher_positions": [0, 1, 2, 3]}]))
    with pytest.raises(ValueError, match="shorter than 5"):
        fid.load_entries(path)
    monkeypatch.setattr(fid, "MAX_ENTRIES", 1)
    path.write_text(json.dumps([entry(list(range(5)), "a"), entry(list(range(5)), "b")]))
    with pytest.raises(ValueError, match="1..1"):
        fid.load_entries(path)
    monkeypatch.setattr(fid, "MAX_TOKENS", 5)
    path.write_text(json.dumps([entry(list(range(6)))]))
    with pytest.raises(ValueError, match="exceeds"):
        fid.load_entries(path)
    monkeypatch.undo()
    hidden = torch.eye(6, 8, dtype=torch.bfloat16)
    manifest, hf_dir, native_dir = write_case(tmp_path / "v", list(range(6)), hidden, hidden[-4:].clone())
    with pytest.raises(ValueError, match="verifier histories required"):
        fid.offline_report(manifest, hf_dir, native_dir, tmp_path / "empty-verifier")
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    torch.save({"teacher_rows": hidden[-4:].clone(), "prefix": torch.tensor(list(range(6))),
                "positions": torch.arange(2, 6)}, verifier / "a.pt")
    report = fid.offline_report(manifest, hf_dir, native_dir, verifier)
    assert report["verifier_histories"] == 1 and report["verifier_numeric_passed"] is True


def test_hf_cli_exact_cache_sequence_and_replay_forwarding(tmp_path, monkeypatch):
    ids = list(range(6))
    class Text:
        def __init__(self):
            self.calls, self.pasts = [], []
        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            past = {"id": len(self.calls), "prev": kwargs["past_key_values"]}
            self.pasts.append(past)
            hidden = torch.full((1, kwargs["input_ids"].shape[1], 4), float(len(self.calls)), dtype=torch.bfloat16)
            return SimpleNamespace(last_hidden_state=hidden, past_key_values=past)

    text = Text()
    teacher = SimpleNamespace(device="cpu", text=text, calls=[], grad_enabled=None, head_rows=[],
                              sentinel=torch.arange(16, dtype=torch.bfloat16).reshape(1, 4, 4) + 1)

    def replay(record, roots, proposals):
        teacher.calls.append((record, roots, proposals))
        teacher.grad_enabled = torch.is_grad_enabled()
        teacher.replayed_before_text = not text.calls
        return teacher.sentinel

    def head(rows):
        teacher.head_rows.append(rows.detach().cpu().clone())
        return torch.zeros(rows.shape[0], 3, device=rows.device)

    teacher.replay, teacher.head = replay, head
    syncs = []
    monkeypatch.setattr(fid.trainer, "Checkpoint", lambda path: SimpleNamespace(path=path, raw={}))
    monkeypatch.setattr(fid.trainer, "FrozenTarget", lambda checkpoint, device: teacher)
    monkeypatch.setattr(fid.trainer, "synchronized_time", lambda device: syncs.append(device) or 0.0)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([entry(ids)]))
    code = fid.main(["hf", "--model", str(tmp_path / "model"), "--manifest", str(manifest),
                     "--output-dir", str(tmp_path / "out"), "--device", "cpu"])
    saved = torch.load(tmp_path / "out" / "a.pt", weights_only=True)
    report = json.loads((tmp_path / "out" / "fidelity.json").read_text())
    record, roots, proposals = teacher.calls[0]
    assert teacher.replayed_before_text and teacher.grad_enabled is False
    assert torch.equal(record["input_ids"], torch.tensor(ids[:-3])) and roots == [1]
    assert proposals.shape == (1, 4) and torch.equal(proposals[0, :3], torch.tensor(ids[-3:]))
    assert proposals[0, 3].item() == 0 and torch.equal(saved["full_rows"], teacher.sentinel[0])
    assert len(text.calls) == 5 and text.calls[0]["use_cache"] is True and text.calls[0]["past_key_values"] is None
    assert text.calls[0]["input_ids"].tolist() == [ids[:-4]]
    assert text.calls[0]["position_ids"].tolist() == [list(range(2))]
    for index, call in enumerate(text.calls[1:], start=1):
        assert call["input_ids"].tolist() == [[ids[index + 1]]]
        assert call["position_ids"].tolist() == [[index + 1]]
        assert call["use_cache"] is True and call["past_key_values"] is text.pasts[index - 1]
    assert torch.equal(saved["cached_rows"], torch.tensor([[2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5]],
                                                          dtype=torch.bfloat16))
    assert len(teacher.head_rows) == 2 and syncs == ["cpu"] * 4
    assert code == 1 and report["optimizer_updates"] == 0 and report["argmax_passed"] is True
    assert {item["symbol"] for item in report["kernels"]["functions"]} >= {
        "causal_conv1d_update", "torch_recurrent_gated_delta_rule"}
    assert all(str(item["runtime_file"]).endswith(("modeling_qwen3_5.py", "fla.py")) or "fla" in str(item["runtime_file"])
               for item in report["kernels"]["functions"])
    assert report["kernels"]["cache"]["class"] == "dict"
