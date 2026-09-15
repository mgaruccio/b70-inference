"""Synthetic-only tests for the bounded private Pi trace corpus converter."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/experiments/qwen38_mtp_trace_corpus.py"
SPEC = importlib.util.spec_from_file_location("qwen38_mtp_trace_corpus", SCRIPT)
assert SPEC and SPEC.loader
corpus = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = corpus
SPEC.loader.exec_module(corpus)

NOW = datetime(2026, 1, 10, 12, tzinfo=timezone.utc)
OLD = NOW - timedelta(days=3)
LONG_USER = "synthetic user request " + ("x" * 130)


def _write_session(root: Path, name: str, rows: list[dict], *, timestamp: datetime = OLD, old_mtime: bool = True) -> Path:
    path = root / name
    header = rows[0]
    header.setdefault("timestamp", timestamp.isoformat())
    header.setdefault("cwd", "/synthetic/project")
    header.setdefault("parentSession", None)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    if old_mtime:
        mtime = (timestamp - timedelta(hours=1)).timestamp()
        os.utime(path, (mtime, mtime))
    return path


def _header(session_id: str, *, parent: str | None = None, cwd: str = "/synthetic/project") -> dict:
    return {
        "type": "session",
        "id": session_id,
        "parentSession": parent,
        "cwd": cwd,
        "timestamp": OLD.isoformat(),
    }


def _message(entry_id: str, parent: str, role: str, content, **extra) -> dict:
    message = {"role": role, "content": content, **extra}
    return {"type": "message", "id": entry_id, "parentId": parent, "message": message}


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _candidate_files(extract: Path) -> list[Path]:
    return sorted((extract / "candidates").glob("*.json"))


def _load_candidates(extract: Path) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in _candidate_files(extract)]


def test_real_fork_paths_keep_cross_day_family_together_and_exclude_future_tools(tmp_path):
    sessions = tmp_path / "sessions"
    first, second, excluded = [sessions / name for name in ("first", "second", "excluded")]
    for directory in (first, second, excluded):
        directory.mkdir(parents=True)
    parent = _write_session(first, "parent.jsonl", [
        _header("parent"),
        _message("u", "parent", "user", LONG_USER),
        _message("a", "u", "assistant", [{"type": "toolCall", "id": "future",
                                         "name": "unsupported_future_tool", "arguments": {}}]),
    ])
    child_header = _header("child", parent=str(parent), cwd="/synthetic/another-project")
    child_header["timestamp"] = (OLD + timedelta(days=1)).isoformat()
    _write_session(second, "child.jsonl", [child_header,
        _message("cu", "child", "user", LONG_USER + " second"),
        _message("ca", "cu", "assistant", "Excluded future answer"),
    ])
    (excluded / "do-not-read.jsonl").write_text("invalid unapproved source")
    output = tmp_path / "extract"
    result = corpus.extract(sessions, [first, second], output, now=NOW)
    rows = _load_candidates(output)
    assert result["sessions"] == 2 and result["skipped_invalid"] == 0
    assert len(rows) == 2
    assert len({row["source_group"] for row in rows}) == 1
    assert all(len(row["messages"]) == 1 and row["tools"] == [] for row in rows)


def test_extract_cli_uses_branch_ancestry_drops_thinking_and_pairs_tools(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    rows = [
        _header("session-1"),
        {"type": "model_change", "id": "model-root", "parentId": None, "model": "synthetic"},
        {"type": "thinking_level_change", "id": "thinking-off", "parentId": "model-root", "thinkingLevel": "off"},
        _message("u1", "thinking-off", "user", LONG_USER),
        _message("sibling", "u1", "assistant", "sibling must not enter branch"),
        _message(
            "branch-answer",
            "u1",
            "assistant",
            [{"type": "thinking", "thinking": "private reasoning"}, {"type": "text", "text": "branch output"}],
        ),
        {"type": "thinking_level_change", "id": "thinking-high", "parentId": "branch-answer", "thinkingLevel": "high"},
        _message("u2", "thinking-high", "user", LONG_USER + " second turn"),
        _message(
            "call",
            "u2",
            "assistant",
            [],
            tool_calls=[
                {
                    "id": "provider-specific-call-id",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"printf synthetic"}'},
                }
            ],
        ),
        _message(
            "result",
            "call",
            "tool",
            "synthetic stdout",
            tool_call_id="provider-specific-call-id",
            name="bash",
        ),
        _message("final", "result", "assistant", "target output is excluded"),
    ]
    _write_session(sessions, "synthetic.jsonl", rows)
    extract = tmp_path / "private-extract"

    result = _run_cli(
        "extract",
        "--session-root",
        str(sessions),
        "--allow-root",
        str(tmp_path),
        "--output",
        str(extract),
        "--now",
        NOW.isoformat(),
    )

    assert result.returncode == 0, result.stderr
    records = _load_candidates(extract)
    assert len(records) == 3
    branch = next(record for record in records if record["thinking"] is False)
    assert [message["role"] for message in branch["messages"]] == ["user"]
    assert "sibling" not in json.dumps(branch)
    assert "branch output" not in json.dumps(branch)

    tool_record = next(record for record in records if record["tools"] == ["bash"])
    assert tool_record["thinking"] is True
    assert tool_record["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "tool_0001",
        "name": "bash",
        "content": "synthetic stdout",
    }
    assert tool_record["messages"][1]["tool_calls"][0]["id"] == "tool_0001"
    assert "target output is excluded" not in json.dumps(tool_record)
    assert "private reasoning" not in json.dumps(records)
    assert stat_mode(extract) == 0o700
    assert all(stat_mode(path) == 0o600 for path in _candidate_files(extract))


def test_extract_rejects_unsupported_sensitive_and_incomplete_contexts(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    # The image and extra-argument branches are deliberately placed in context
    # before a later target, so they cannot be mistaken for excluded targets.
    image_rows = [
        _header("image-session"),
        _message("u", "image-session", "user", LONG_USER),
        _message("image", "u", "assistant", [{"type": "image", "url": "synthetic"}]),
        _message("target", "image", "assistant", "should be dropped"),
    ]
    extra_arg_rows = [
        _header("extra-session"),
        _message("u", "extra-session", "user", LONG_USER),
        _message(
            "call",
            "u",
            "assistant",
            [],
            tool_calls=[
                {
                    "id": "call-extra",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"printf ok","extra":1}'},
                }
            ],
        ),
        _message("target", "call", "assistant", "should be dropped"),
    ]
    sensitive_rows = [
        _header("sensitive-session"),
        _message("u", "sensitive-session", "user", LONG_USER),
        _message(
            "call",
            "u",
            "assistant",
            [],
            tool_calls=[
                {
                    "id": "call-secret",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"/tmp/.env"}'},
                }
            ],
        ),
        _message("result", "call", "tool", "[REDACTED]", tool_call_id="call-secret", name="read"),
        _message("target", "result", "assistant", "should be dropped"),
    ]
    orphan_rows = [
        _header("orphan-session"),
        _message("u", "orphan-session", "user", LONG_USER),
        _message("orphan", "u", "tool", "orphan", tool_call_id="missing", name="bash"),
        _message("target", "orphan", "assistant", "should be dropped"),
    ]
    for name, rows in (
        ("image.jsonl", image_rows),
        ("extra.jsonl", extra_arg_rows),
        ("sensitive.jsonl", sensitive_rows),
        ("orphan.jsonl", orphan_rows),
    ):
        _write_session(sessions, name, rows)

    extract = tmp_path / "private-extract"
    result = _run_cli(
        "extract",
        "--root",
        str(sessions),
        "--allow-root",
        str(tmp_path),
        "--out",
        str(extract),
        "--now",
        NOW.isoformat(),
    )
    assert result.returncode == 0, result.stderr
    records = _load_candidates(extract)
    assert len(records) == 1
    assert ".env" not in json.dumps(records)
    assert "[REDACTED]" not in json.dumps(records)


def test_extract_skips_recent_active_malformed_cyclic_and_symlink_inputs(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    recent_rows = [_header("recent"), _message("u", "recent", "user", LONG_USER), _message("a", "u", "assistant", "recent")]
    _write_session(sessions, "recent.jsonl", recent_rows, timestamp=NOW - timedelta(hours=2), old_mtime=False)
    active_rows = [_header("active"), _message("u", "active", "user", LONG_USER), _message("a", "u", "assistant", "active")]
    active_rows[0]["active"] = True
    _write_session(sessions, "active.jsonl", active_rows)
    cyclic_rows = [
        _header("cyclic"),
        _message("u", "a", "user", LONG_USER),
        _message("a", "u", "assistant", "cyclic"),
    ]
    _write_session(sessions, "cyclic.jsonl", cyclic_rows)
    missing_parent = _message("u", "missing", "user", LONG_USER)
    missing_parent.pop("parentId")
    _write_session(
        sessions,
        "missing-parent.jsonl",
        [_header("missing"), missing_parent, _message("a", "u", "assistant", "missing")],
    )
    (sessions / "malformed.jsonl").write_text("not-json\n", encoding="utf-8")
    old = (OLD - timedelta(hours=1)).timestamp()
    os.utime(sessions / "malformed.jsonl", (old, old))

    extract = tmp_path / "private-extract"
    result = _run_cli(
        "extract",
        "--root",
        str(sessions),
        "--allow-root",
        str(tmp_path),
        "--output",
        str(extract),
        "--now",
        NOW.isoformat(),
    )
    assert result.returncode == 0, result.stderr
    assert _candidate_files(extract) == []

    symlink_target = tmp_path / "outside.jsonl"
    symlink_target.write_text("{}\n", encoding="utf-8")
    os.symlink(symlink_target, sessions / "symlink.jsonl")
    rejected = _run_cli(
        "extract",
        "--root",
        str(sessions),
        "--allow-root",
        str(tmp_path),
        "--output",
        str(tmp_path / "second-extract"),
        "--now",
        NOW.isoformat(),
    )
    assert rejected.returncode == 2


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def _candidate(candidate_id: str, group: str, content: str) -> dict:
    return {
        "id": candidate_id,
        "source_group": group,
        "messages": [{"role": "user", "content": content}],
        "tools": [],
        "thinking": True,
    }


def _private_stage(path: Path, candidates: list[dict]) -> Path:
    path.mkdir(mode=0o700)
    candidate_dir = path / "candidates"
    candidate_dir.mkdir(mode=0o700)
    for candidate in candidates:
        (candidate_dir / f"{candidate['id']}.json").write_text(
            json.dumps(candidate, separators=(",", ":")), encoding="utf-8"
        )
    for candidate_path in candidate_dir.iterdir():
        os.chmod(candidate_path, 0o600)
    return path


def test_filter_scan_deduplicates_groups_splits_stably_and_removes_flagged_copy(tmp_path: Path) -> None:
    duplicate_a = _candidate("a", "group-z", LONG_USER)
    duplicate_b = _candidate("b", "group-a", LONG_USER)
    distinct = _candidate("c", "group-c", LONG_USER + " distinct")
    stage = _private_stage(tmp_path / "stage", [duplicate_a, duplicate_b, distinct])
    report = stage / "gitleaks.json"
    # Flag only the distinct candidate; the duplicate pair must unify to group-a.
    report.write_text(json.dumps([{"File": "candidates/c.json"}]), encoding="utf-8")
    os.chmod(report, 0o600)
    output = tmp_path / "dataset"

    result = corpus.filter_scan(stage, report, output)
    assert result["kept"] == 1
    assert result["rejected"] == 1
    assert not (stage / "candidates" / "c.json").exists()
    assert not report.exists()
    assert stat_mode(output) == 0o700
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines() if line]
    rows += [json.loads(line) for line in (output / "dev.jsonl").read_text().splitlines() if line]
    rows += [json.loads(line) for line in (output / "test.jsonl").read_text().splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["id"] == "b"
    assert rows[0]["source_group"] == "group-a"
    assert all(stat_mode(output / f"{name}.jsonl") == 0o600 for name in ("train", "dev", "test"))


def test_filter_scan_fails_closed_for_unknown_paths_and_keeps_inputs(tmp_path: Path) -> None:
    candidate = _candidate("known", "group", LONG_USER)
    stage = _private_stage(tmp_path / "stage", [candidate])
    report = stage / "gitleaks.json"
    report.write_text(json.dumps([{"File": "candidates/unknown.json"}]), encoding="utf-8")
    os.chmod(report, 0o600)
    output = tmp_path / "dataset"

    with pytest.raises(corpus.CorpusError):
        corpus.filter_scan(stage, report, output)
    assert report.exists()
    assert (stage / "candidates" / "known.json").exists()
    assert not output.exists()


def test_outputs_reject_git_and_symlink_boundaries_and_external_report_is_not_deleted(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    _write_session(
        sessions,
        "one.jsonl",
        [_header("one"), _message("u", "one", "user", LONG_USER), _message("a", "u", "assistant", "answer")],
    )
    git_parent = tmp_path / "git-parent"
    git_parent.mkdir()
    (git_parent / ".git").mkdir()
    with pytest.raises(corpus.CorpusError):
        corpus.extract(sessions, [tmp_path], git_parent / "out", now=NOW)

    candidate = _candidate("known", "group", LONG_USER)
    stage = _private_stage(tmp_path / "stage", [candidate])
    external_report = tmp_path / "external-report.json"
    external_report.write_text("[]", encoding="utf-8")
    os.chmod(external_report, 0o600)
    output = tmp_path / "dataset"
    corpus.filter_scan(stage, external_report, output)
    assert external_report.exists()

    symlink_stage = tmp_path / "symlink-stage"
    _private_stage(symlink_stage, [candidate])
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(candidate), encoding="utf-8")
    os.chmod(outside, 0o600)
    os.symlink(outside, symlink_stage / "candidates" / "link.json")
    bad_report = symlink_stage / "bad.json"
    bad_report.write_text("[]", encoding="utf-8")
    os.chmod(bad_report, 0o600)
    with pytest.raises(corpus.CorpusError):
        corpus.filter_scan(symlink_stage, bad_report, tmp_path / "bad-dataset")
