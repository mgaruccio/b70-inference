"""Offline fixture checks for the Qwen B70 P0 tracer overlay."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PATCH_DIR = (
    ROOT / "patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b"
)
PATCHER_PATH = PATCH_DIR / "patch_p0_tracer.py"
TRACER_PATH = PATCH_DIR / "p0_tracer.py"
COMPARATOR_PATH = ROOT / "scripts/compare_qwen38_b70_p0_traces.py"
FIXTURE_DIR = ROOT / "tests/fixtures/qwen38_b70_p0_tracer"
MOCK_MODULES_PATH = FIXTURE_DIR / "mock_vllm_modules.py"
TRACE_LEFT = FIXTURE_DIR / "trace_left.jsonl"
TRACE_RIGHT = FIXTURE_DIR / "trace_right.jsonl"


def _spec_events(path: Path) -> list[dict]:
    events = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if "drafted_ids" in event:
            events.append(event)
    return events


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def patcher():
    return _load_module(PATCHER_PATH, "qwen38_b70_p0_tracer_patch")


@pytest.fixture()
def tracer():
    return _load_module(TRACER_PATH, "qwen38_b70_p0_tracer")


@pytest.fixture()
def comparator():
    return _load_module(COMPARATOR_PATH, "qwen38_b70_p0_trace_compare")


def test_cli_dry_run_apply_idempotence_and_rollback(patcher, tmp_path: Path) -> None:
    target = tmp_path / "sitecustomize.py"
    target.write_bytes(b"")

    dry_run = subprocess.run(
        [sys.executable, str(PATCHER_PATH), "--path", str(target), "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    assert "would patch" in dry_run.stdout
    assert target.read_bytes() == b""

    apply = subprocess.run(
        [sys.executable, str(PATCHER_PATH), "--path", str(target)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert apply.returncode == 0, apply.stderr
    assert "patched" in apply.stdout
    patched = target.read_bytes()
    assert patcher.OVERLAY_MARKER.encode("utf-8") in patched
    compile(patched.decode("utf-8"), str(target), "exec")

    repeat = subprocess.run(
        [sys.executable, str(PATCHER_PATH), "--path", str(target)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert repeat.returncode == 0, repeat.stderr
    assert "already patched" in repeat.stdout
    assert target.read_bytes() == patched

    rollback = subprocess.run(
        [sys.executable, str(PATCHER_PATH), "--path", str(target), "--reverse"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rollback.returncode == 0, rollback.stderr
    assert "reversed" in rollback.stdout
    assert target.read_bytes() == b""


def test_cli_refuses_vllm_source_target(patcher, tmp_path: Path) -> None:
    target = tmp_path / "gpu_model_runner.py"
    target.write_text("# fake\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(PATCHER_PATH), "--path", str(target), "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "refusing" in result.stderr


def test_helpers_derive_prefix_divergence_bonus_and_gdn_counts(tracer) -> None:
    drafted = [101, 102, 103, 104]
    target = [101, 102, 999, 104]
    output_row = [101, 102, 999, -1, -1]

    assert tracer.accepted_prefix_len(drafted, target) == 2
    assert tracer.first_divergence_index(drafted, target) == 2
    assert tracer.bonus_used(drafted, target) is False
    assert tracer.derive_bonus_token(drafted, target, 555) is None
    assert tracer.gdn_state_num_accepted_tokens(output_row) == 3

    full_match = [1, 2, 3]
    assert tracer.bonus_used(full_match, full_match) is True
    assert tracer.derive_bonus_token(full_match, full_match, 777) == 777

    event = tracer.build_event(
        request_id="single-8k",
        step=0,
        context_len=8459,
        drafted_ids=drafted,
        target_ids=target,
        output_row=output_row,
        bonus_token_id=555,
    )
    assert event["schema"] == "qwen38-b70-p0/v1"
    assert event["num_accepted_tokens"] == 2
    assert event["gdn_state_num_accepted_tokens"] == 3
    assert event["bonus_used"] is False
    assert event["first_divergence_index"] == 2


def test_row_draft_ids_preserves_padding_and_limits_to_n(tracer) -> None:
    draft_flat = [10, 11, -1, 12]
    assert tracer.row_draft_ids(draft_flat, [2], [2], 0) == [10, 11]
    assert tracer.row_draft_ids(draft_flat, [0], [0], 0) == []


def test_comparator_reports_first_divergence(comparator) -> None:
    left = comparator._load_trace(TRACE_LEFT)
    right = comparator._load_trace(TRACE_RIGHT)
    divergence = comparator.compare_traces(left, right)
    assert divergence is not None
    assert divergence["step"] == 1
    assert divergence["context_len"] == 8462
    assert divergence["index"] == 1
    assert divergence["field"] == "target_ids"

    cli = subprocess.run(
        [
            sys.executable,
            str(COMPARATOR_PATH),
            str(TRACE_LEFT),
            str(TRACE_RIGHT),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert cli.returncode == 1
    assert "step=1 context_len=8462 index=1" in cli.stdout


def test_mock_install_emits_one_event_and_preserves_tensor_identity(
    tracer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_modules = _load_module(MOCK_MODULES_PATH, "mock_vllm_modules")
    _Tensor = mock_modules._Tensor
    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv(tracer.TRACE_ENV, str(trace_path))

    tracer._hooks_installed = False
    tracer._original_sample = None
    tracer._original_rejection_sample = None
    tracer._original_rejection_forward = None
    tracer._install_hooks_impl(mock_modules, mock_modules)
    runner = mock_modules.GPUModelRunner()

    draft = _Tensor([0, 1, 2, 3])
    target_logits = _Tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    bonus = _Tensor([5])
    num_drafts = _Tensor([4])
    cu_drafts = _Tensor([4])
    output = _Tensor([[0, 1, 9, -1, -1]])
    rejection_kwargs = {
        "draft_token_ids": draft,
        "draft_probs": None,
        "target_logits": target_logits,
        "bonus_token_ids": bonus,
        "num_draft_tokens": num_drafts,
        "cu_num_draft_tokens": cu_drafts,
        "sampled_token_ids": output,
    }

    result = mock_modules.rejection_sample(
        draft,
        None,
        target_logits,
        bonus,
        num_drafts,
        cu_drafts,
        sampled_token_ids=output,
    )
    assert result is output
    assert _spec_events(trace_path) == []

    runner.pending_rejection = rejection_kwargs
    runner._sample()
    assert len(mock_modules._calls) == 2

    events = _spec_events(trace_path)
    assert len(events) == 1
    event = events[0]
    assert event["request_id"] == "single-8k"
    assert event["context_len"] == 8459
    assert event["drafted_ids"] == [0, 1, 2, 3]
    assert event["target_ids"] == [0, 1, 9, 3]
    assert event["num_accepted_tokens"] == 2
    assert event["gdn_state_num_accepted_tokens"] == 3

    def exploding_emit(_event):
        raise OSError("disk full")

    monkeypatch.setattr(tracer, "_emit_event", exploding_emit)
    runner.pending_rejection = rejection_kwargs
    safe_result = runner._sample()
    assert safe_result is output

    if tracer._original_sample is not None:
        mock_modules.GPUModelRunner._sample = tracer._original_sample
    if tracer._original_rejection_sample is not None:
        mock_modules.rejection_sample = tracer._original_rejection_sample
    tracer._hooks_installed = False


def test_positional_vllm_signature_emits_event(
    tracer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    mock_modules = _load_module(MOCK_MODULES_PATH, "mock_vllm_modules_positional")
    _Tensor = mock_modules._Tensor
    trace_path = tmp_path / "trace-positional.jsonl"
    monkeypatch.setenv(tracer.TRACE_ENV, str(trace_path))

    draft = _Tensor([0, 1, 2, 3])
    target_logits = _Tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    bonus = _Tensor([5])
    num_drafts = _Tensor([4])
    cu_drafts = _Tensor([4])
    output = _Tensor([[0, 1, 9, -1, -1]])

    def rejection_sample(
        draft_token_ids,
        num_draft_tokens,
        max_spec_len,
        cu_num_draft_tokens,
        draft_probs,
        target_logits,
        bonus_token_ids,
        sampling_metadata=None,
        synthetic_mode=False,
        synthetic_conditional_rates=None,
        use_fp64_gumbel=False,
    ):
        return output

    module = SimpleNamespace()

    class GPUModelRunner:
        def __init__(self) -> None:
            self.input_batch = SimpleNamespace(
                req_ids=["single-8k"],
                num_computed_tokens_cpu=_Tensor([8459]),
            )

        def _sample(self, *args, **kwargs):
            return module.rejection_sample(
                draft,
                num_drafts,
                4,
                cu_drafts,
                None,
                target_logits,
                bonus,
                None,
            )

    class RejectionSampler:
        def forward(self, *args, **kwargs):
            return output

    module.GPUModelRunner = GPUModelRunner
    module.rejection_sample = rejection_sample
    module.RejectionSampler = RejectionSampler
    tracer._hooks_installed = False
    tracer._original_sample = None
    tracer._original_rejection_sample = None
    tracer._original_rejection_forward = None
    tracer._install_hooks_impl(module, module)

    result = module.GPUModelRunner()._sample()
    assert result is output
    events = _spec_events(trace_path)
    assert len(events) == 1
    event = events[0]
    assert event["drafted_ids"] == [0, 1, 2, 3]
    assert event["target_ids"] == [0, 1, 9, 3]
    assert event["num_accepted_tokens"] == 2
    assert event["gdn_state_num_accepted_tokens"] == 3

    if tracer._original_sample is not None:
        module.GPUModelRunner._sample = tracer._original_sample
    if tracer._original_rejection_sample is not None:
        module.rejection_sample = tracer._original_rejection_sample
    tracer._hooks_installed = False
