#!/usr/bin/env python3
"""Focused CPU-only tests for analyze-target.py."""
from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any


RESULT_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("analyze_target", RESULT_DIR / "analyze-target.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load analyze-target.py")
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


API_NAME = ANALYZER.API_NAME


def _documents(*, gpu_duration: float = 2.0) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two submissions prove that the measured suffix, not the prefix, is used."""
    trace_events = [
        {
            "ph": "X",
            "cat": "gpu_op",
            "pid": 999,
            "tid": 999,
            "name": "warmup_kernel",
            "ts": 100.0,
            "dur": 1.0,
            "args": {"id": "1"},
        },
        {
            "ph": "X",
            "cat": "gpu_op",
            "pid": 999,
            "tid": 999,
            "name": "gemm_kernel[SIMD16 {128; 1; 1} {16; 2; 8}]",
            "ts": 200.0,
            "dur": gpu_duration,
            "args": {"id": "2"},
        },
        {
            "ph": "s",
            "cat": "Flow_H2D_1_285",
            "pid": 285,
            "tid": 285,
            "name": "dep",
            "ts": 100.0,
            "id": 1,
        },
        {
            "ph": "s",
            "cat": "Flow_H2D_2_285",
            "pid": 285,
            "tid": 285,
            "name": "dep",
            "ts": 200.0,
            "id": 2,
        },
        {
            "ph": "X",
            "cat": "cpu_op",
            "pid": 285,
            "tid": 285,
            "name": API_NAME,
            "ts": 99.0,
            "dur": 2.0,
            "id": 0,
        },
        {
            "ph": "X",
            "cat": "cpu_op",
            "pid": 285,
            "tid": 285,
            "name": API_NAME,
            "ts": 199.0,
            "dur": 2.0,
            "id": 0,
        },
    ]
    step_events = [
        {
            "index": 0,
            "duration_ms": 3.0,
            "event_error": None,
            "target_boundary_control": {"classification": "target", "replay_index": 0},
        }
    ]
    step = {
        "format": "fixture",
        "events": step_events,
        "errors": [],
        "counts": {
            "capture_replays_skipped": 0,
            "graph_events_emitted": 1,
            "graph_replays_dropped_at_bound": 0,
            "graph_replays_seen": 1,
        },
        "host_scopes": [],
    }
    return {"traceEvents": trace_events}, step


def _write_case(trace: dict[str, Any], step: dict[str, Any]) -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
    directory = tempfile.TemporaryDirectory()
    root = Path(directory.name)
    trace_path = root / "trace.json"
    step_path = root / "step.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    step_path.write_text(json.dumps(step), encoding="utf-8")
    return directory, trace_path, step_path


def _analyze(trace: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    directory, trace_path, step_path = _write_case(trace, step)
    try:
        return ANALYZER.analyze(trace_path, step_path, enforce_archival_guard=False)
    finally:
        directory.cleanup()


class AnalyzeTargetTests(unittest.TestCase):
    def test_flow_prefix_and_measured_suffix_own_target(self) -> None:
        trace, step = _documents()
        report = _analyze(trace, step)
        self.assertEqual(report["integrity"]["target_gpu_ops"], 1)
        self.assertEqual(report["target_replays"][0]["all_api_submission_ordinal"], 1)
        self.assertEqual(report["kernel_groups"][0]["kernel_name"], "gemm_kernel[SIMD16 {128; 1; 1} {16; 2; 8}]")

    def test_rejects_missing_flow_ownership(self) -> None:
        trace, step = _documents()
        trace["traceEvents"] = [
            event for event in trace["traceEvents"] if event.get("id") != 2
        ]
        with self.assertRaisesRegex(ANALYZER.AnalysisError, "ownership mismatch"):
            _analyze(trace, step)

    def test_rejects_ambiguous_duplicate_ownership(self) -> None:
        trace, step = _documents()
        apis = [event for event in trace["traceEvents"] if event.get("name") == API_NAME]
        apis[-1]["ts"] = 99.5
        apis[-1]["dur"] = 3.0
        step["events"].append(copy.deepcopy(step["events"][0]))
        step["counts"]["graph_events_emitted"] = 2
        step["counts"]["graph_replays_seen"] = 2
        with self.assertRaisesRegex(ANALYZER.AnalysisError, "overlap|duplicate"):
            _analyze(trace, step)

    def test_rejects_nonpositive_target_gpu_duration(self) -> None:
        trace, step = _documents(gpu_duration=0.0)
        with self.assertRaisesRegex(ANALYZER.AnalysisError, "duration"):
            _analyze(trace, step)

    def test_rejects_timing_errors_or_drops(self) -> None:
        trace, step = _documents()
        step["errors"] = [{"message": "fixture failure"}]
        with self.assertRaisesRegex(ANALYZER.AnalysisError, "errors"):
            _analyze(trace, step)

        trace, step = _documents()
        step["counts"]["graph_replays_dropped_at_bound"] = 1
        with self.assertRaisesRegex(ANALYZER.AnalysisError, "dropped_at_bound"):
            _analyze(trace, step)


if __name__ == "__main__":
    unittest.main()
