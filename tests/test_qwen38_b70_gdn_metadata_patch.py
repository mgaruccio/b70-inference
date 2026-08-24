"""Offline fixture checks for the versioned Qwen B70 GDN overlay."""
from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PATCHER_PATH = (
    ROOT
    / "patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b"
    / "patch_gdn_metadata.py"
)
FIXTURE_PATH = (
    ROOT
    / "tests/fixtures/qwen38_b70_gdn_metadata"
    / "gdn_attn_after_mtp_boundary.py"
)


def _load_patcher():
    spec = importlib.util.spec_from_file_location("qwen38_b70_gdn_metadata", PATCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def fixture_text() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


def test_metadata_overlay_is_narrow_idempotent_and_reversible(fixture_text: str) -> None:
    patcher = _load_patcher()

    patched = patcher.patch_text(fixture_text)
    compile(patched, str(FIXTURE_PATH), "exec")
    assert patcher.OVERLAY_MARKER in patched
    assert "self.spec_token_arange[:expected_spec_token_size]" in patched or (
        "self.spec_token_arange[\n                        :expected_spec_token_size\n                    ]"
        in patched
    )
    assert "if non_spec_token_indx.numel() > 0:" in patched
    assert "if spec_token_indx.data_ptr() == self.spec_token_arange.data_ptr():" in patched
    assert "spec_sequence_masks_cpu, : self.num_spec + 1" not in patched
    assert "torch.empty(\n                        0, dtype=torch.int32" not in patched
    assert "gpu_model_runner" not in patched
    assert "_num_accepted_tokens_req_id_to_index" not in patched

    assert patcher.patch_text(patched) == patched
    assert patcher.unpatch_text(patched) == fixture_text
    assert patcher.unpatch_text(fixture_text) == fixture_text

def test_overlay_has_capacity_guard_before_spec_slice(fixture_text: str) -> None:
    patcher = _load_patcher()
    patched = patcher.patch_text(fixture_text)
    tree = ast.parse(patched)

    guards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if (
            isinstance(test.left, ast.Name)
            and test.left.id == "expected_spec_token_size"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Gt)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Call)
            and isinstance(test.comparators[0].func, ast.Attribute)
            and test.comparators[0].func.attr == "numel"
        ):
            guards.append(node)

    assert len(guards) == 1
    guard = guards[0]
    assert any(
        isinstance(statement, ast.Raise)
        and isinstance(statement.exc, ast.Call)
        and isinstance(statement.exc.func, ast.Name)
        and statement.exc.func.id == "RuntimeError"
        for statement in guard.body
    )
    slice_line = patched.splitlines().index(
        "                    spec_token_indx = self.spec_token_arange["
    ) + 1
    assert guard.lineno < slice_line

def test_overlay_fails_closed_when_boundary_anchor_changes(fixture_text: str) -> None:
    patcher = _load_patcher()
    broken = fixture_text.replace("B70_MTP_PARTIAL_FINAL_GROUP", "B70_MTP_PARTIAL_FINAL_GROUP_BROKEN", 1)

    with pytest.raises(ValueError, match="boundary"):
        patcher.patch_text(broken)


def test_cli_dry_run_apply_idempotence_and_rollback(
    fixture_text: str, tmp_path: Path
) -> None:
    target = tmp_path / "gdn_attn.py"
    target.write_text(fixture_text, encoding="utf-8")

    dry_run = subprocess.run(
        [sys.executable, str(PATCHER_PATH), "--path", str(target), "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    assert "would patch" in dry_run.stdout
    assert target.read_text(encoding="utf-8") == fixture_text

    apply = subprocess.run(
        [sys.executable, str(PATCHER_PATH), "--path", str(target)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert apply.returncode == 0, apply.stderr
    assert "patched" in apply.stdout
    patched = target.read_text(encoding="utf-8")

    repeat = subprocess.run(
        [sys.executable, str(PATCHER_PATH), "--path", str(target)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert repeat.returncode == 0, repeat.stderr
    assert "already patched" in repeat.stdout
    assert target.read_text(encoding="utf-8") == patched

    rollback = subprocess.run(
        [
            sys.executable,
            str(PATCHER_PATH),
            "--path",
            str(target),
            "--reverse",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rollback.returncode == 0, rollback.stderr
    assert "reversed" in rollback.stdout
    assert target.read_text(encoding="utf-8") == fixture_text
