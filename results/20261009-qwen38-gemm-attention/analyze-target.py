#!/usr/bin/env python3
"""Reconcile target-replay GPU hotspots from one archived native trace.

This is intentionally an offline, development-only report.  It does not run a
model, contact a host, or make a throughput/speedup claim.  Ownership comes
from the unitrace Flow_H2D source records and the measured API submission
intervals; names are reported literally rather than mapped to projections.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


FORMAT = "b70-gemm-hotspot-accounting-v1"
WORKER_PID = 285
FLOW_PREFIX = "Flow_H2D"
API_NAME = "zeCommandListImmediateAppendCommandListsExp"

DEFAULT_TRACE = Path(
    "/tmp/b70-native64k-analysis-ajgs_8f3/native-boundary64k-20261009-100052-ff9f86/"
    "native/unitrace/python3.285.json"
)
DEFAULT_STEP_TIMING = Path(
    "/tmp/b70-native64k-analysis-ajgs_8f3/native-boundary64k-20261009-100052-ff9f86/"
    "native/step-timing/step-timing-session-rank0-session1.json"
)
DEFAULT_OUTPUT = Path(__file__).with_name("hotspots.json")
ARCHIVAL_COMMAND = "python3 -B /tmp/b70-fetch-boundary-artifacts.py"
ARCHIVAL_ARCHIVE = "native-public-fill64k-01.tar.gz"
ARCHIVAL_EXTRACTION_ROOT = "/tmp/b70-native64k-analysis-ajgs_8f3/"

EXPECTED_TARGET_ROWS = 42
EXPECTED_GPU_OPS_PER_TARGET = 977
EXPECTED_TRUE_TOTAL_MS = 1913.950520
EXPECTED_GPU_WORK_MS = 1827.095850
EXPECTED_GEMM_NAME = "gemm_kernel[SIMD16 {128; 1; 1} {16; 2; 8}]"
EXPECTED_HEADLINES = {
    "fmha": (672, 649.777227),
    "reduce_split_k": (672, 27.569510),
    "gdn": (4032, 56.001976),
}


class AnalysisError(RuntimeError):
    """Raised when the archival accounting contract cannot be established."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _positive(value: Any, label: str) -> float:
    if not _finite(value) or float(value) <= 0:
        raise AnalysisError(f"{label} must be a positive finite number: {value!r}")
    return float(value)


def _nonnegative(value: Any, label: str) -> float:
    if not _finite(value) or float(value) < 0:
        raise AnalysisError(f"{label} must be a finite non-negative number: {value!r}")
    return float(value)


def _identifier(value: Any, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise AnalysisError(f"{label} must be a string or integer: {value!r}")
    result = str(value)
    if not result:
        raise AnalysisError(f"{label} must not be empty")
    return result


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read JSON {path}: {exc}") from exc


def _file_metadata(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AnalysisError(f"cannot read input {path}: {exc}") from exc
    return {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _events(document: Any, path: Path) -> list[Mapping[str, Any]]:
    if not isinstance(document, Mapping) or not isinstance(document.get("traceEvents"), list):
        raise AnalysisError(f"trace has no traceEvents list: {path}")
    result: list[Mapping[str, Any]] = []
    for index, event in enumerate(document["traceEvents"]):
        if not isinstance(event, Mapping):
            raise AnalysisError(f"trace event {index} is not an object")
        result.append(event)
    return result


def _step_events(document: Any, path: Path) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    if not isinstance(document, Mapping) or not isinstance(document.get("events"), list):
        raise AnalysisError(f"step timing has no events list: {path}")
    rows: list[Mapping[str, Any]] = []
    for index, event in enumerate(document["events"]):
        if not isinstance(event, Mapping):
            raise AnalysisError(f"step event {index} is not an object")
        _positive(event.get("duration_ms"), f"step event {index} duration_ms")
        if event.get("event_error") is not None:
            raise AnalysisError(f"step event {index} has event_error: {event['event_error']!r}")
        control = event.get("target_boundary_control")
        if not isinstance(control, Mapping):
            raise AnalysisError(f"step event {index} has no target_boundary_control object")
        rows.append(event)

    errors = document.get("errors")
    if errors != []:
        raise AnalysisError(f"step timing contains errors: {errors!r}")
    counts = document.get("counts")
    if not isinstance(counts, Mapping):
        raise AnalysisError("step timing has no counts object")
    for key in ("graph_events_emitted", "graph_replays_seen"):
        if counts.get(key) != len(rows):
            raise AnalysisError(
                f"step timing {key}={counts.get(key)!r} does not equal event count {len(rows)}"
            )
    for key in ("capture_replays_skipped", "graph_replays_dropped_at_bound"):
        value = counts.get(key)
        if value != 0:
            raise AnalysisError(f"step timing {key} is not zero: {value!r}")
    return rows, counts


def _gpu_and_flows(events: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    gpu: dict[str, Mapping[str, Any]] = {}
    flow: list[dict[str, Any]] = []
    flow_ids: set[str] = set()
    for index, event in enumerate(events):
        category = event.get("cat")
        phase = event.get("ph")
        if category == "gpu_op" and phase == "X":
            args = event.get("args")
            if not isinstance(args, Mapping) or "id" not in args:
                raise AnalysisError(f"gpu_op event {index} has no args.id")
            key = _identifier(args["id"], f"gpu_op event {index} args.id")
            if key in gpu:
                raise AnalysisError(f"duplicate GPU event id: {key}")
            timestamp = event.get("ts")
            if not _finite(timestamp):
                raise AnalysisError(f"gpu_op {key} timestamp is not finite: {timestamp!r}")
            _nonnegative(event.get("dur"), f"gpu_op {key} duration")
            if not isinstance(event.get("name"), str) or not event["name"]:
                raise AnalysisError(f"gpu_op {key} has no kernel name")
            gpu[key] = event
        if (
            isinstance(category, str)
            and category.startswith(FLOW_PREFIX)
            and phase == "s"
            and event.get("pid") == WORKER_PID
            and event.get("tid") == WORKER_PID
        ):
            if "id" not in event:
                raise AnalysisError(f"{category} flow has no id")
            key = _identifier(event["id"], f"{category} flow id")
            if key in flow_ids:
                raise AnalysisError(f"duplicate Flow_H2D source id: {key}")
            timestamp = event.get("ts")
            if not _finite(timestamp):
                raise AnalysisError(f"Flow_H2D {key} timestamp is not finite: {timestamp!r}")
            flow_ids.add(key)
            flow.append({"id": key, "ts": float(timestamp), "category": category})

    if not gpu:
        raise AnalysisError("trace contains no gpu_op/ph=X events with args.id")
    if not flow:
        raise AnalysisError("trace contains no worker Flow_H2D/ph=s sources")
    gpu_ids = set(gpu)
    if gpu_ids != flow_ids:
        missing = sorted(gpu_ids - flow_ids)[:8]
        extra = sorted(flow_ids - gpu_ids)[:8]
        raise AnalysisError(f"GPU/Flow_H2D ownership mismatch; missing={missing}, extra={extra}")
    flow.sort(key=lambda row: row["ts"])
    return gpu, flow


def _api_submissions(events: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = [
        event
        for event in events
        if event.get("cat") == "cpu_op"
        and event.get("ph") == "X"
        and event.get("pid") == WORKER_PID
        and event.get("tid") == WORKER_PID
        and event.get("name") == API_NAME
    ]
    for index, event in enumerate(rows):
        timestamp = event.get("ts")
        _positive(event.get("dur"), f"API submission {index} duration")
        if not _finite(timestamp):
            raise AnalysisError(f"API submission {index} timestamp is not finite: {timestamp!r}")
    rows.sort(key=lambda event: float(event["ts"]))
    previous_ts: float | None = None
    for index, event in enumerate(rows):
        timestamp = float(event["ts"])
        if previous_ts is not None and timestamp <= previous_ts:
            raise AnalysisError("API submissions are not strictly ordered by timestamp")
        previous_ts = timestamp
    return rows


def _category(kernel_name: str) -> str:
    if "ReduceSplitK" in kernel_name:
        return "reduce_split_k"
    if "XeFMHAFwdSplitKVKernel" in kernel_name:
        return "fmha"
    if kernel_name.startswith("gdn::"):
        return "gdn"
    if kernel_name.startswith("gemm_kernel["):
        return "gemm"
    return "other"


def _round(value: float) -> float:
    return round(float(value), 6)


def _close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6):
        raise AnalysisError(f"archival guard {label}: observed {actual:.9f}, expected {expected:.9f}")


def analyze(
    trace_path: Path = DEFAULT_TRACE,
    step_timing_path: Path = DEFAULT_STEP_TIMING,
    *,
    enforce_archival_guard: bool = True,
) -> dict[str, Any]:
    """Build the compact report from the trace and step-timing JSON files."""
    trace_path = Path(trace_path)
    step_timing_path = Path(step_timing_path)
    trace_document = _read_json(trace_path)
    step_document = _read_json(step_timing_path)
    trace_events = _events(trace_document, trace_path)
    step_events, counts = _step_events(step_document, step_timing_path)
    gpu, flows = _gpu_and_flows(trace_events)
    submissions = _api_submissions(trace_events)
    if len(submissions) < len(step_events):
        raise AnalysisError(
            f"only {len(submissions)} API submissions for {len(step_events)} step events"
        )
    suffix_start = len(submissions) - len(step_events)
    measured_submissions = submissions[suffix_start:]
    flow_timestamps = [row["ts"] for row in flows]
    flow_by_id = {row["id"]: row for row in flows}

    owners: dict[str, int] = {}
    assignments: list[dict[str, Any]] = []
    previous_end: float | None = None
    for ordinal, (step_event, submission) in enumerate(zip(step_events, measured_submissions)):
        start = float(submission["ts"])
        duration = _positive(submission.get("dur"), f"API submission {ordinal} duration")
        end = start + duration
        if not _finite(end):
            raise AnalysisError(f"API submission {ordinal} end is not finite")
        if previous_end is not None and start <= previous_end:
            raise AnalysisError("measured API intervals overlap or touch; ownership is ambiguous")
        previous_end = end
        left = bisect.bisect_left(flow_timestamps, start)
        right = bisect.bisect_right(flow_timestamps, end)
        ids = [flows[index]["id"] for index in range(left, right)]
        if not ids:
            raise AnalysisError(f"API submission {ordinal} owns no Flow_H2D sources")
        for key in ids:
            if key in owners:
                raise AnalysisError(f"duplicate measured ownership for GPU id {key}")
            if key not in gpu or key not in flow_by_id:
                raise AnalysisError(f"missing GPU/Flow_H2D record for owned id {key}")
            owners[key] = ordinal
        assignments.append(
            {
                "ordinal": ordinal,
                "all_api_ordinal": suffix_start + ordinal,
                "step_event": step_event,
                "submission": submission,
                "ids": ids,
                "start": start,
                "end": end,
            }
        )

    target_assignments = [
        assignment
        for assignment in assignments
        if assignment["step_event"]["target_boundary_control"].get("classification") == "target"
    ]
    if not target_assignments:
        raise AnalysisError("step timing contains no target_boundary_control.classification=target rows")
    target_ids: set[str] = set()
    target_replays: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    true_total_ms = 0.0
    linked_work_ms = 0.0
    zero_duration_gpu_ops = 0
    target_zero_duration_gpu_ops = 0

    for target_ordinal, assignment in enumerate(target_assignments):
        step_event = assignment["step_event"]
        ids = assignment["ids"]
        if len(set(ids)) != len(ids):
            raise AnalysisError("target row contains duplicate GPU ownership")
        if target_ids.intersection(ids):
            raise AnalysisError("target rows have duplicate GPU ownership")
        target_ids.update(ids)
        row_work_ms = 0.0
        for key in ids:
            event = gpu[key]
            duration = _positive(event.get("dur"), f"target GPU {key} duration")
            row_work_ms += duration / 1000.0
            name = str(event["name"])
            category = _category(name)
            group = groups.setdefault(
                (category, name),
                {"category": category, "kernel_name": name, "trace_category": "gpu_op", "ops": 0, "total_ms": 0.0},
            )
            group["ops"] += 1
            group["total_ms"] += duration / 1000.0
        step_duration_ms = _positive(step_event.get("duration_ms"), f"target step {target_ordinal}")
        true_total_ms += step_duration_ms
        linked_work_ms += row_work_ms
        target_replays.append(
            {
                "replay_ordinal": target_ordinal,
                "step_event_index": assignment["ordinal"],
                "target_boundary_replay_index": step_event["target_boundary_control"].get("replay_index"),
                "measured_api_ordinal": assignment["ordinal"],
                "all_api_submission_ordinal": assignment["all_api_ordinal"],
                "step_duration_ms": _round(step_duration_ms),
                "gpu_ops": len(ids),
                "gpu_work_ms": _round(row_work_ms),
            }
        )

    zero_duration_gpu_ops = sum(float(event.get("dur")) == 0 for event in gpu.values())
    target_zero_duration_gpu_ops = sum(float(gpu[key].get("dur")) == 0 for key in target_ids)
    if target_zero_duration_gpu_ops:
        raise AnalysisError("target ownership includes zero-duration GPU records")

    if enforce_archival_guard:
        if len(target_assignments) != EXPECTED_TARGET_ROWS:
            raise AnalysisError(
                f"archival guard target rows: observed {len(target_assignments)}, expected {EXPECTED_TARGET_ROWS}"
            )
        bad_rows = [row["gpu_ops"] for row in target_replays if row["gpu_ops"] != EXPECTED_GPU_OPS_PER_TARGET]
        if bad_rows:
            raise AnalysisError(
                f"archival guard GPU ops per target row: observed {bad_rows}, expected {EXPECTED_GPU_OPS_PER_TARGET}"
            )
        _close(true_total_ms, EXPECTED_TRUE_TOTAL_MS, "true_total_ms")
        _close(linked_work_ms, EXPECTED_GPU_WORK_MS, "linked_gpu_work_ms")
        main_gemm = groups.get(("gemm", EXPECTED_GEMM_NAME))
        if main_gemm is None:
            raise AnalysisError("archival guard missing expected main GEMM kernel")
        if main_gemm["ops"] != 10752:
            raise AnalysisError(f"archival guard main GEMM ops: {main_gemm['ops']}")
        _close(main_gemm["total_ms"], 1026.985887, "main GEMM total_ms")
        category_totals_for_guard: dict[str, tuple[int, float]] = {}
        for (category, _name), group in groups.items():
            ops, total = category_totals_for_guard.get(category, (0, 0.0))
            category_totals_for_guard[category] = (ops + group["ops"], total + group["total_ms"])
        for category, (expected_ops, expected_ms) in EXPECTED_HEADLINES.items():
            observed = category_totals_for_guard.get(category)
            if observed is None or observed[0] != expected_ops:
                raise AnalysisError(f"archival guard {category} ops: observed {observed}, expected {expected_ops}")
            _close(observed[1], expected_ms, f"{category} total_ms")

    def with_shares(row: dict[str, Any]) -> dict[str, Any]:
        total = float(row["total_ms"])
        return {
            **row,
            "total_ms": _round(total),
            "share_of_linked_gpu_work_pct": _round(100.0 * total / linked_work_ms),
            "share_of_true_replay_pct": _round(100.0 * total / true_total_ms),
        }

    kernel_groups = [with_shares(group) for group in groups.values()]
    kernel_groups.sort(key=lambda row: (-row["total_ms"], row["category"], row["kernel_name"]))
    category_accumulator: dict[str, dict[str, Any]] = {}
    for group in groups.values():
        category = group["category"]
        category_row = category_accumulator.setdefault(
            category, {"category": category, "ops": 0, "total_ms": 0.0}
        )
        category_row["ops"] += group["ops"]
        category_row["total_ms"] += group["total_ms"]
    category_totals = [with_shares(row) for row in category_accumulator.values()]
    category_totals.sort(key=lambda row: (-row["total_ms"], row["category"]))

    return {
        "format": FORMAT,
        "status": "ok",
        "scope": "bounded reproducible offline hotspot analysis",
        "development_only": True,
        "claims": {
            "share_only": True,
            "speedup_or_gain_claim": False,
            "projection_names_inferred": False,
        },
        "archival": {
            "archive": ARCHIVAL_ARCHIVE,
            "archival_command": ARCHIVAL_COMMAND,
            "extraction_root": ARCHIVAL_EXTRACTION_ROOT,
            "source_note": "Inputs are the retained native 64K archival cell; no remote or GPU work is performed by this report.",
        },
        "inputs": {
            "trace": _file_metadata(trace_path),
            "step_timing": _file_metadata(step_timing_path),
        },
        "integrity": {
            "trace_event_count": len(trace_events),
            "gpu_op_count": len(gpu),
            "flow_h2d_count": len(flows),
            "all_api_submission_count": len(submissions),
            "measured_api_submission_count": len(measured_submissions),
            "measured_suffix_gpu_ops": len(owners),
            "unmeasured_gpu_ops": len(gpu) - len(owners),
            "step_event_count": len(step_events),
            "target_row_count": len(target_assignments),
            "target_gpu_ops": len(target_ids),
            "target_zero_duration_gpu_ops": target_zero_duration_gpu_ops,
            "trace_zero_duration_gpu_ops": zero_duration_gpu_ops,
            "timing_errors": 0 if not step_document.get("errors") else len(step_document["errors"]),
            "timing_dropped_at_bound": counts.get("graph_replays_dropped_at_bound"),
        },
        "accounting": {
            "true_total_ms": _round(true_total_ms),
            "linked_gpu_work_ms": _round(linked_work_ms),
            "target_gpu_ops": len(target_ids),
            "share_basis": "Kernel shares are percentages of linked target GPU work and unchanged target replay elapsed time; they are not gains.",
        },
        "archival_cell_guard": {
            "passed": True,
            "expected_target_rows": EXPECTED_TARGET_ROWS,
            "expected_gpu_ops_per_target_row": EXPECTED_GPU_OPS_PER_TARGET,
            "expected_true_total_ms": EXPECTED_TRUE_TOTAL_MS,
            "expected_linked_gpu_work_ms": EXPECTED_GPU_WORK_MS,
        },
        "target_replays": target_replays,
        "category_totals": category_totals,
        "kernel_groups": kernel_groups,
        "limitations": [
            "Generic GEMM kernel names are retained literally; no projection names are inferred.",
            "This is trace accounting with device-work shares, not a throughput comparison or optimization gain.",
            "GPU events are owned from Flow_H2D source timestamps inside measured API submission intervals; pre-suffix submissions are excluded.",
        ],
    }


def _write_report(report: Mapping[str, Any], path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    except OSError as exc:
        raise AnalysisError(f"cannot write report {path}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--step-timing", type=Path, default=DEFAULT_STEP_TIMING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = analyze(args.trace, args.step_timing)
        _write_report(report, args.output)
    except AnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.output} ({report['accounting']['target_gpu_ops']} target GPU ops)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
