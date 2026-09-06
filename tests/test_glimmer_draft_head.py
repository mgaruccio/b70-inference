"""CPU/offline checks for the bounded Glimmer draft-head probes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "scripts/experimental/glimmer_draft_head.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("glimmer_draft_head", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["glimmer_draft_head"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def draft_head():
    return _load_module()


def test_disabled_mode_is_a_strict_noop(draft_head, monkeypatch):
    monkeypatch.delenv("GLIMMER_DRAFT_HEAD", raising=False)
    assert draft_head.parse_mode() is None

    class Model:
        pass

    model = Model()
    assert draft_head.attach_draft_head(model, {}) is None
    assert not hasattr(model, "glimmer_draft_head")


def test_mode_and_graph_guards_are_explicit(draft_head, monkeypatch):
    assert draft_head.parse_mode("int4") == "int4"
    assert draft_head.parse_mode("off") is None
    with pytest.raises(draft_head.DraftHeadError, match="capture|int4|shortlist"):
        draft_head.parse_mode("surprise")

    monkeypatch.delenv("VLLM_XPU_ENABLE_XPU_GRAPH", raising=False)
    with pytest.raises(draft_head.UnsupportedDraftHeadConfiguration, match="GRAPH=0"):
        draft_head.require_graphs_off()
    monkeypatch.setenv("VLLM_XPU_ENABLE_XPU_GRAPH", "0")
    draft_head.require_graphs_off()


def test_contract_refusals_are_fail_closed(draft_head):
    model = SimpleNamespace()
    with pytest.raises(draft_head.UnsupportedDraftHeadConfiguration, match="tensor_parallel"):
        draft_head._validate_existing_contract(
            model,
            SimpleNamespace(
                parallel_config=SimpleNamespace(tensor_parallel_size=2)
            ),
            vocab_size=10,
        )
    with pytest.raises(draft_head.UnsupportedDraftHeadConfiguration, match="mapping"):
        draft_head._validate_existing_contract(
            SimpleNamespace(token_id_mapping=[0, 1]),
            SimpleNamespace(),
            vocab_size=10,
        )
    with pytest.raises(draft_head.UnsupportedDraftHeadConfiguration, match="logit_scale"):
        draft_head._validate_existing_contract(
            SimpleNamespace(logit_scale=0), SimpleNamespace(), vocab_size=10
        )
    with pytest.raises(draft_head.UnsupportedDraftHeadConfiguration, match="greedy"):
        draft_head._validate_existing_contract(
            SimpleNamespace(),
            SimpleNamespace(draft_sample_method="probabilistic"),
            vocab_size=10,
        )
    with pytest.raises(draft_head.UnsupportedDraftHeadConfiguration, match="nonstandard"):
        draft_head._validate_existing_contract(
            SimpleNamespace(),
            SimpleNamespace(rejection_sample_method="block"),
            vocab_size=10,
        )


def test_shortlist_validation_rejects_wrong_size_order_duplicates_and_range(draft_head):
    assert draft_head.validate_shortlist_ids(
        [1, 3, 7], vocab_size=10, expected_size=3
    ) == (1, 3, 7)
    for bad in ([1, 3], [3, 1, 7], [1, 1, 7], [-1, 3, 7], [1, 3, 10]):
        with pytest.raises(ValueError):
            draft_head.validate_shortlist_ids(bad, vocab_size=10, expected_size=3)


def test_json_shortlist_artifact_is_read_only_and_validated(draft_head, tmp_path):
    path = tmp_path / "shortlist.json"
    path.write_text(json.dumps({"token_ids": [1, 4, 9]}), encoding="utf-8")
    assert draft_head.load_shortlist_ids(path, vocab_size=10, expected_size=3) == (1, 4, 9)
    assert json.loads(path.read_text(encoding="utf-8"))["token_ids"] == [1, 4, 9]


def test_calibration_split_rejects_overlap_without_torch(draft_head, tmp_path):
    calibration = tmp_path / "calibration.txt"
    calibration.write_text("shared\n", encoding="utf-8")
    heldout = tmp_path / "heldout.txt"
    heldout.write_text("shared\n", encoding="utf-8")
    assert draft_head._read_label_set(calibration) == {"shared"}
    with pytest.raises(draft_head.DraftHeadError, match="overlap"):
        draft_head.calibrate_shortlist(
            tmp_path,
            calibration_labels={"shared"},
            heldout_labels={"shared"},
            vocab_size=10,
            expected_size=3,
        )


def test_packing_mapping_and_capture_are_cpu_only_when_torch_is_available(
    draft_head, tmp_path
):
    torch = pytest.importorskip("torch")
    weight = torch.linspace(-1, 1, 4 * 128, dtype=torch.float16).reshape(4, 128)
    original = weight.clone()
    identity = id(weight)
    packed = draft_head.pack_int4_weight(weight, chunk_rows=2)
    assert id(weight) == identity
    assert torch.equal(weight, original)
    assert tuple(packed.qweight.shape) == (16, 4)
    assert tuple(packed.qweight.stride()) == (1, 16)
    assert packed.scales.dtype == torch.float16
    assert packed.scales.is_contiguous()
    unpacked = draft_head.unpack_int4_weight(packed.qweight, packed.scales)
    assert tuple(unpacked.shape) == tuple(weight.shape)
    assert int(packed.zero_point.item()) == 8
    assert int(unpacked.min()) >= -2
    assert int(unpacked.max()) <= 2

    shortlist = draft_head.ShortlistDraftHead(weight, (1, 3))
    hidden = torch.zeros((2, 128), dtype=torch.float16)
    hidden[0, 0] = 1
    assert shortlist(hidden).tolist() == [3, 1]
    assert torch.equal(weight, original)

    label = tmp_path / "label.txt"
    label.write_text("calibration-a\n", encoding="utf-8")
    recorder = draft_head.CaptureRecorder(
        artifact_dir=tmp_path / "captures",
        label_file=label,
        max_steps=1,
        sample_stride=1,
        max_rows=2,
        topk=3,
    )
    dense = torch.arange(4 * 128, dtype=torch.float16).reshape(4, 128)

    def compute_logits(hidden_states):
        return torch.nn.functional.linear(hidden_states, dense)

    capture = draft_head.CaptureDraftHead(compute_logits, recorder)
    hidden = torch.randn((2, 128), dtype=torch.float16)
    expected = compute_logits(hidden).argmax(dim=-1)
    actual = capture(hidden)
    assert torch.equal(actual, expected)
    files = sorted((tmp_path / "captures").glob("*.pt"))
    assert len(files) == 1
    payload = torch.load(files[0], map_location="cpu")
    assert payload["label"] == "calibration-a"
    assert payload["hidden_states"].device.type == "cpu"
    assert payload["baseline_top1"].tolist() == expected.tolist()


def test_calibration_uses_calibration_labels_only_when_torch_is_available(
    draft_head, tmp_path
):
    torch = pytest.importorskip("torch")
    captures = tmp_path / "captures"
    captures.mkdir()
    # Keep this intentionally small and exercise the function with a reduced
    # expected size; production CLI defaults remain exactly 32,768.
    for label, offset in (("cal", 0), ("heldout", 100)):
        payload = {
            "label": label,
            "hidden_states": torch.zeros((1, 128), dtype=torch.float16),
            "baseline_top1": torch.tensor([offset]),
            "baseline_topk_ids": torch.tensor([[offset, offset + 1, offset + 2]]),
            "baseline_topk_logits": torch.ones((1, 3), dtype=torch.float16),
        }
        torch.save(payload, captures / f"{label}.pt")
    artifact = draft_head.calibrate_shortlist(
        captures,
        calibration_labels={"cal"},
        heldout_labels={"heldout"},
        vocab_size=10,
        expected_size=3,
    )
    assert artifact["calibration_labels"] == ["cal"]
    assert artifact["heldout_labels"] == ["heldout"]
    assert artifact["token_ids"] == [0, 1, 2]
    assert artifact["not_spec_vocab_replication"] is True


def test_capture_labels_appear_after_load_and_keep_independent_caps(draft_head, tmp_path):
    torch = pytest.importorskip("torch")
    label = tmp_path / "label.txt"
    recorder = draft_head.CaptureRecorder(
        artifact_dir=tmp_path / "captures", label_file=label,
        max_steps=1, sample_stride=1, topk=2,
    )
    hidden = torch.ones((2, 4), dtype=torch.float16)
    logits = torch.tensor([[1., 2., 3.], [3., 2., 1.]])
    recorder.record(hidden, logits)  # Dummy initialization has no live label.
    assert not (tmp_path / "captures").exists()
    for name in ("cal", "heldout", "cal"):
        label.write_text(name)
        recorder.record(hidden, logits)
    files = sorted((tmp_path / "captures").glob("*.pt"))
    assert len(files) == 2
    assert {torch.load(p, weights_only=True)["label"] for p in files} == {"cal", "heldout"}


def test_checkpoint_head_is_cast_like_fp16_server(draft_head, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    monkeypatch.setattr(draft_head, "VOCAB_SIZE", 4)
    monkeypatch.setattr(draft_head, "HIDDEN_SIZE", 128)
    source = torch.arange(512, dtype=torch.bfloat16).reshape(4, 128)
    path = tmp_path / "head.safetensors"
    safetensors.save_file({"lm_head.weight": source}, str(path))
    loaded = draft_head._load_safetensors_head(path, "lm_head.weight")
    assert loaded.dtype == torch.float16
    assert torch.equal(loaded, source.to(torch.float16))
    assert safetensors.load_file(str(path))["lm_head.weight"].dtype == torch.bfloat16


def test_pinned_dflash_mapping_and_config_fields_are_guarded(draft_head):
    with pytest.raises(draft_head.UnsupportedDraftHeadConfiguration, match="mapping"):
        draft_head._validate_existing_contract(
            SimpleNamespace(draft_id_to_target_id=[0, 1]), SimpleNamespace(), vocab_size=10,
        )
    with pytest.raises(draft_head.UnsupportedDraftHeadConfiguration, match="vocab size"):
        draft_head._validate_existing_contract(
            SimpleNamespace(config=SimpleNamespace(draft_vocab_size=9)),
            SimpleNamespace(), vocab_size=10,
        )


@pytest.mark.parametrize("case", ["valid", "empty", "wrong_count", "wrong_dense", "nonfinite"])
def test_probe_references_fail_closed(draft_head, tmp_path, monkeypatch, case):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(draft_head, "VOCAB_SIZE", 4)
    monkeypatch.setattr(draft_head, "HIDDEN_SIZE", 128)
    weight = torch.linspace(-1, 1, 512, dtype=torch.float16).reshape(4, 128)
    monkeypatch.setattr(draft_head, "_load_safetensors_head", lambda *_: weight)
    hidden = torch.ones((0 if case == "empty" else 2, 128), dtype=torch.float16)
    saved = torch.nn.functional.linear(hidden, weight).argmax(dim=-1)
    if case == "wrong_count":
        saved = saved[:1]
    elif case == "wrong_dense":
        saved = torch.full_like(saved, -1)
    elif case == "nonfinite":
        hidden[0, 0] = float("nan")
    torch.save({
        "label": "cal", "hidden_states": hidden, "baseline_top1": saved,
        "baseline_topk_ids": saved.reshape(-1, 1),
    }, tmp_path / "capture.pt")
    args = SimpleNamespace(
        device="cpu", model_index="unused", head_key="lm_head.weight",
        chunk_rows=2, captures=str(tmp_path), heldout_label_file=None, shortlist=None,
        m_values="2", k_values="1", warmup=0, repeats=1,
    )
    if case == "valid":
        report = draft_head._probe(args)
        assert report["rows"] == 2 and report["numerical_screen_passed"]
        assert set(report["timing"]) == {"dense", "int4"}
    else:
        with pytest.raises(draft_head.DraftHeadError):
            draft_head._probe(args)
