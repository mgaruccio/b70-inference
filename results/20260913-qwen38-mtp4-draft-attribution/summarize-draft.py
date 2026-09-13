#!/usr/bin/env python3
"""Summarize one bounded MTP4 draft-attribution trace.

The input may be a profiler trace (JSON/JSON.GZ) or a profile run directory.
Eager device kernels are assigned once to the innermost recorded draft phase by
PyTorch's CPU ``External id``.  Canonical XPU graph-event timings and the
campaign's dispatcher metadata are reported separately; neither CPU-inclusive
scope time nor summed device work is presented as critical-path latency.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping


PHASE_PREFIX = "b70_draft/phase:"
FORMAT = "b70-mtp4-draft-attribution-summary-v1"


class SummaryError(RuntimeError):
    """A malformed or incomplete trace/attribution artifact."""


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _duration(value: Any, *, label: str) -> float:
    if not _finite(value) or float(value) < 0:
        raise SummaryError(f"{label} has a non-finite or negative duration: {value!r}")
    return float(value)


def _read_json(path: Path) -> Any:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as source:
                return json.load(source)
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SummaryError(f"cannot read JSON artifact {path}: {exc}") from exc


def _trace_path(value: Path) -> tuple[Path, Path | None]:
    value = value.resolve()
    if value.is_file():
        run_dir = None
        for candidate in (value.parent, value.parent.parent, value.parent.parent.parent):
            if (candidate / "step-timing").is_dir() or (candidate / "draft-attribution").is_dir():
                run_dir = candidate
                break
        return value, run_dir
    if not value.is_dir():
        raise SummaryError(f"input does not exist: {value}")
    traces = sorted(
        path
        for path in value.rglob("*.pt.trace.json*")
        if path.is_file() and path.name.startswith("rank")
    )
    if not traces:
        traces = sorted(path for path in value.rglob("*.trace.json*") if path.is_file())
    if not traces:
        raise SummaryError(f"no compressed/native profiler trace under {value}")
    if len(traces) > 1:
        raise SummaryError(f"expected one rank trace for bounded profile, found {traces}")
    return traces[0], value


def _safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _safe_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(child) for child in value]
    return str(value)


def _event_end(event: Mapping[str, Any]) -> float:
    return float(event.get("ts", 0)) + float(event.get("dur", 0))


def _scope_events(events: Iterable[Mapping[str, Any]]) -> tuple[dict[tuple[Any, Any], list[Mapping[str, Any]]], collections.Counter[str]]:
    by_thread: dict[tuple[Any, Any], list[Mapping[str, Any]]] = collections.defaultdict(list)
    counts: collections.Counter[str] = collections.Counter()
    for event in events:
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        category = event.get("cat")
        if category == "user_annotation" and name.startswith(PHASE_PREFIX):
            label = name[len(PHASE_PREFIX):]
            _duration(event.get("dur"), label=f"scope {name}")
            by_thread[event.get("pid"), event.get("tid")].append(event)
            counts[label] += 1
    for rows in by_thread.values():
        rows.sort(key=lambda event: (float(event.get("ts", 0)), -float(event.get("dur", 0))))
    return by_thread, counts


def _innermost_scope(
    event: Mapping[str, Any],
    scopes: list[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    start = float(event.get("ts", 0))
    end = _event_end(event)
    containing = [
        scope
        for scope in scopes
        if float(scope.get("ts", 0)) <= start
        and end <= _event_end(scope) + 0.001
    ]
    if not containing:
        return None
    return min(containing, key=lambda scope: (float(scope.get("dur", 0)), -float(scope.get("ts", 0))))


def _external_id(event: Mapping[str, Any]) -> Any:
    args = event.get("args")
    return args.get("External id") if isinstance(args, Mapping) else None


def _input_dims(event: Mapping[str, Any]) -> Any:
    args = event.get("args")
    return args.get("Input Dims") if isinstance(args, Mapping) else None


def _load_trace(path: Path) -> tuple[list[Mapping[str, Any]], str]:
    document = _read_json(path)
    events = document.get("traceEvents") if isinstance(document, Mapping) else None
    if not isinstance(events, list):
        raise SummaryError(f"trace has no traceEvents list: {path}")
    rows = [event for event in events if isinstance(event, Mapping)]
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_launches(events: Iterable[Mapping[str, Any]]) -> dict[Any, Mapping[str, Any]]:
    launches: dict[Any, Mapping[str, Any]] = {}
    for event in events:
        if event.get("cat") != "xpu_runtime":
            continue
        args = event.get("args")
        correlation = args.get("correlation") if isinstance(args, Mapping) else None
        if correlation is None:
            continue
        if correlation in launches:
            raise SummaryError(f"duplicate XPU runtime correlation: {correlation!r}")
        launches[correlation] = event
    return launches


def _phase_for_runtime_launch(
    launch: Mapping[str, Any],
    scopes_by_thread: Mapping[tuple[Any, Any], list[Mapping[str, Any]]],
) -> str | None:
    rows = scopes_by_thread.get((launch.get("pid"), launch.get("tid")), [])
    scope = _innermost_scope(launch, rows)
    if scope is None:
        return None
    name = str(scope.get("name", ""))
    return name[len(PHASE_PREFIX):] if name.startswith(PHASE_PREFIX) else None


def _aggregate_row(rows: Mapping[str, Mapping[str, float | int]], steps: int, *, key_name: str) -> list[dict[str, Any]]:
    result = []
    for name, value in sorted(rows.items(), key=lambda item: (-float(item[1]["total_us"]), item[0])):
        total = float(value["total_us"])
        result.append(
            {
                "name": name,
                "count": int(value["count"]),
                "total_us": total,
                "ms_per_generation_step": total / (1000.0 * steps),
            }
        )
    return result


def summarize_trace(path: Path) -> dict[str, Any]:
    events, trace_sha256 = _load_trace(path)
    scopes_by_thread, phase_counts = _scope_events(events)
    steps = phase_counts.get("propose", 0)
    if not steps:
        raise SummaryError("trace contains no b70_draft/phase:propose scopes")

    lookup: dict[tuple[Any, Any], Mapping[str, Any]] = {}
    by_external: dict[Any, Mapping[str, Any]] = {}
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "cpu_op":
            continue
        external = _external_id(event)
        if external is None:
            continue
        key = (event.get("pid"), external)
        if key in lookup and lookup[key] is not event:
            raise SummaryError(f"duplicate CPU External id: {key!r}")
        lookup[key] = event
        if external in by_external and by_external[external] is not event:
            raise SummaryError(f"cross-process CPU External id collision: {external!r}")
        by_external[external] = event

    launches = _runtime_launches(events)
    aggregate: dict[str, dict[str, dict[str, float | int]]] = {
        name: collections.defaultdict(lambda: {"count": 0, "total_us": 0.0})
        for name in ("phases", "operators", "kernels")
    }
    all_kernel_us = draft_kernel_us = missing_kernel_us = outside_us = 0.0
    kernel_count = missing_count = 0
    queues: set[tuple[Any, Any]] = set()
    missing_audit: collections.defaultdict[str, dict[str, float | int]] = collections.defaultdict(
        lambda: {"count": 0, "kernel_ms": 0.0}
    )

    for event in events:
        if event.get("cat") != "kernel":
            continue
        duration_us = _duration(event.get("dur"), label=f"kernel {event.get('name')}")
        all_kernel_us += duration_us
        kernel_count += 1
        queues.add((event.get("pid"), event.get("tid")))
        external = _external_id(event)
        cpu = by_external.get(external) if external is not None else None
        # Kernel and CPU event pids differ; resolve the CPU row by External id,
        # then use that CPU row's own process/thread for containment.
        if cpu is None:
            missing_kernel_us += duration_us
            missing_count += 1
            args = event.get("args")
            correlation = args.get("correlation") if isinstance(args, Mapping) else None
            audit = "unmapped_cpu_external_id"
            launch = launches.get(correlation)
            if launch is not None:
                phase = _phase_for_runtime_launch(launch, scopes_by_thread)
                audit = phase if phase is not None else "outside_draft"
            missing_audit[audit]["count"] += 1
            missing_audit[audit]["kernel_ms"] += duration_us / 1000.0
            outside_us += duration_us
            continue
        cpu_scopes = scopes_by_thread.get((cpu.get("pid"), cpu.get("tid")), [])
        scope = _innermost_scope(cpu, cpu_scopes)
        if scope is None:
            outside_us += duration_us
            continue
        name = str(scope.get("name", ""))
        phase = name[len(PHASE_PREFIX):] if name.startswith(PHASE_PREFIX) else None
        if phase is None:
            outside_us += duration_us
            continue
        draft_kernel_us += duration_us
        aggregate["phases"][phase]["count"] += 1
        aggregate["phases"][phase]["total_us"] += duration_us
        operator_name = f"{phase} / {event.get('name')} {json.dumps(_input_dims(cpu), sort_keys=True)}"
        aggregate["operators"][operator_name]["count"] += 1
        aggregate["operators"][operator_name]["total_us"] += duration_us
        kernel_name = f"{phase} / {event.get('name')}"
        aggregate["kernels"][kernel_name]["count"] += 1
        aggregate["kernels"][kernel_name]["total_us"] += duration_us
    phase_rows = []
    for name, value in sorted(aggregate["phases"].items(), key=lambda item: (-float(item[1]["total_us"]), item[0])):
        total = float(value["total_us"])
        phase_rows.append(
            {
                "name": name,
                "count": int(value["count"]),
                "total_us": total,
                "ms_per_generation_step": total / (1000.0 * steps),
            }
        )

    scope_timing = []
    for name, count in sorted(phase_counts.items()):
        durations = [
            _duration(event.get("dur"), label=f"scope {name}") / 1000.0
            for event in events
            if event.get("ph") == "X"
            and event.get("cat") == "user_annotation"
            and str(event.get("name", "")) == PHASE_PREFIX + name
        ]
        scope_timing.append(
            {
                "phase": name,
                "count": count,
                "inclusive_host_ms": sum(durations),
                "median_inclusive_host_ms": statistics.median(durations) if durations else None,
            }
        )

    return {
        "trace_path": str(path),
        "trace_sha256": trace_sha256,
        "draft_generation_steps": steps,
        "phase_counts": dict(phase_counts),
        "scope_timing": scope_timing,
        "kernel_count": kernel_count,
        "kernel_queues": sorted(queues, key=str),
        "all_kernel_ms": all_kernel_us / 1000.0,
        "draft_kernel_ms": draft_kernel_us / 1000.0,
        "outside_draft_or_unmapped_ms": outside_us / 1000.0,
        "unmapped_cpu_external_id": {
            "count": missing_count,
            "kernel_ms": missing_kernel_us / 1000.0,
        },
        "unmapped_kernel_runtime_correlation_audit": {
            key: dict(value) for key, value in sorted(missing_audit.items())
        },
        "phases": phase_rows,
        "operators": _aggregate_row(aggregate["operators"], steps, key_name="operators"),
        "kernels": _aggregate_row(aggregate["kernels"], steps, key_name="kernels"),
        "attribution_contract": {
            "cpu_external_id_mapping": "each eager kernel is assigned once to its innermost draft phase",
            "graph_replay_kernels": "usually hidden by XPU graph replay and not guessed into eager phases",
            "cpu_scope_time": "inclusive host intervals; not added to device totals",
            "summed_device_work": "not critical-path latency",
        },
    }


def _mode_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        name = value.get("name")
        return str(name) if name is not None else str(value.get("value"))
    return str(value) if value is not None else None


def _descriptor_key(value: Any) -> str:
    return json.dumps(_safe_json(value), sort_keys=True, separators=(",", ":"))


def _stack_role(stack: Any) -> str:
    if not isinstance(stack, list):
        return "unknown"
    text = " ".join(
        f"{row.get('module', '')} {row.get('function', '')} {row.get('class', '')}"
        for row in stack
        if isinstance(row, Mapping)
    )
    if any(token in text for token in ("Qwen3_5MTP", "EagleProposer", "SpecDecodeBaseProposer", "spec_decode")):
        return "draft"
    if "XPUModelRunner" in text or "gpu_model_runner" in text:
        return "target"
    return "unknown"


def _stage(requested: Any, descriptor: Any) -> str:
    try:
        if int(requested) == 5:
            return "first_five_tokens"
        if int(requested) == 1:
            return "later_one_token"
    except (TypeError, ValueError):
        pass
    if isinstance(descriptor, Mapping):
        try:
            if int(descriptor.get("num_tokens")) == 5:
                return "five_tokens_padded_or_exact"
            if int(descriptor.get("num_tokens")) == 1:
                return "one_token_padded_or_exact"
        except (TypeError, ValueError):
            pass
    return "other"


def _discover_run_dir(input_path: Path, explicit: Path | None, trace_path: Path) -> Path | None:
    if explicit is not None:
        return explicit.resolve()
    if input_path.is_dir():
        return input_path.resolve()
    for candidate in (trace_path.parent, trace_path.parent.parent, trace_path.parent.parent.parent):
        if (candidate / "step-timing").is_dir() or (candidate / "draft-attribution").is_dir():
            return candidate.resolve()
    return None


def _annotation_documents(run_dir: Path | None) -> list[tuple[Path, Mapping[str, Any]]]:
    if run_dir is None:
        return []
    root = run_dir / "draft-attribution"
    if not root.is_dir():
        return []
    result = []
    for path in sorted(root.rglob("*.json")):
        value = _read_json(path)
        if isinstance(value, Mapping) and value.get("format") == "b70-mtp4-draft-attribution-v1":
            result.append((path, value))
    return result


def _timing_documents(run_dir: Path | None) -> list[tuple[Path, Mapping[str, Any]]]:
    if run_dir is None:
        return []
    root = run_dir / "step-timing"
    if not root.is_dir():
        return []
    result = []
    for path in sorted(root.rglob("*.json")):
        value = _read_json(path)
        if isinstance(value, Mapping) and value.get("format") == "b70-step-timing-overlay-v1":
            result.append((path, value))
    return result


def _draft_span_summary(
    annotation_docs: list[tuple[Path, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Summarize deferred current-stream timings around complete draft calls."""
    rows: list[dict[str, Any]] = []
    observed = False
    for _path, document in annotation_docs:
        raw_rows = document.get("draft_spans")
        if not isinstance(raw_rows, list):
            continue
        observed = True
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                continue
            row = dict(raw)
            duration = row.get("duration_ms")
            if duration is not None:
                _duration(duration, label="draft span")
            rows.append(row)
    durations = [
        float(row["duration_ms"])
        for row in rows
        if row.get("duration_ms") is not None
    ]
    finite_metrics = all(
        row.get("duration_ms") is not None
        and _finite(row.get("duration_ms"))
        and float(row["duration_ms"]) >= 0
        for row in rows
    )
    return {
        "observed": observed,
        "events": rows,
        "event_count": len(rows),
        "finite_duration_count": len(durations),
        "finite_metrics": finite_metrics,
        "total_ms": sum(durations),
        "median_ms": statistics.median(durations) if durations else None,
        "source": "campaign torch.xpu.Event around complete EagleProposer.propose span",
        "cpu_scope_time_is_not_included": True,
    }

def _graph_rows(
    annotation_docs: list[tuple[Path, Mapping[str, Any]]],
    timing_docs: list[tuple[Path, Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    dispatches: list[dict[str, Any]] = []
    graph_events: list[dict[str, Any]] = []
    errors: list[str] = []
    if annotation_docs:
        for _path, document in annotation_docs:
            errors.extend(str(error) for error in document.get("errors", []) if error)
            for row in document.get("dispatches", []):
                if isinstance(row, Mapping):
                    dispatches.append(dict(row))
            for row in document.get("graph_replays", []):
                if isinstance(row, Mapping):
                    graph_events.append(dict(row))
    # The campaign document copies the canonical events, but fall back to the
    # canonical artifact if serialization was unavailable or yielded no rows.
    if not graph_events:
        for _path, document in timing_docs:
            errors.extend(str(error) for error in document.get("errors", []) if error)
            for row in document.get("events", []):
                if isinstance(row, Mapping):
                    graph_events.append(dict(row))
    return dispatches, graph_events, errors


def _attribute_graph_events(
    dispatches: list[Mapping[str, Any]],
    graph_events: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed: dict[tuple[str, str | None, str], list[tuple[int, Mapping[str, Any]]]] = collections.defaultdict(list)
    for index, row in enumerate(dispatches):
        role = str(row.get("role", "unknown"))
        mode = _mode_key(row.get("returned_runtime_mode"))
        descriptor = row.get("returned_batch_descriptor")
        indexed[(role, mode, _descriptor_key(descriptor))].append((index, row))

    attributed: list[dict[str, Any]] = []
    for event_index, event in enumerate(graph_events):
        context = event.get("campaign_graph_context")
        context = context if isinstance(context, Mapping) else {}
        descriptor = context.get("batch_descriptor") or event.get("descriptor_fields")
        mode = context.get("runtime_mode")
        if mode is None:
            mode = event.get("runtime_mode")
        role = str(event.get("role") or _stack_role(event.get("caller_stack")))
        if role == "unknown":
            role = _stack_role(event.get("caller_stack"))
        candidates = indexed.get((role, _mode_key(mode), _descriptor_key(descriptor)), [])
        match_index: int | None = None
        match: Mapping[str, Any] | None = None
        if candidates:
            match_index, match = candidates.pop(0)
        if match is None and role == "unknown":
            # Do not guess a role from graph size, but permit an exact key
            # match when an older overlay omitted its caller classification.
            for key, rows in indexed.items():
                if key[1] == _mode_key(mode) and key[2] == _descriptor_key(descriptor) and rows:
                    match_index, match = rows.pop(0)
                    role = str(match.get("role", "unknown"))
                    break
        requested = match.get("requested_num_tokens") if match is not None else None
        matched_stage = match.get("stage") if match is not None else None
        stage = (
            str(matched_stage)
            if isinstance(matched_stage, str) and matched_stage not in {"", "None"}
            else _stage(requested, descriptor)
        )
        duration = event.get("duration_ms")
        if duration is not None:
            _duration(duration, label=f"graph event {event_index}")
        attributed.append(
            {
                "index": event.get("index", event_index),
                "role": role,
                "stage": stage,
                "duration_ms": duration,
                "host_elapsed_ns": event.get("host_elapsed_ns"),
                "graph_id": event.get("graph_id"),
                "graph_type": event.get("graph_type"),
                "manager_type": event.get("manager_type"),
                "runtime_mode": mode,
                "wrapper_runtime_mode": context.get("wrapper_runtime_mode"),
                "batch_descriptor": descriptor,
                "graph_entry_state": context.get("graph_entry_state"),
                "dispatch_match_index": match_index,
                "classification_reason": event.get("classification_reason"),
            }
        )
    return attributed, [dict(row) for row in dispatches]


def _graph_summary(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str | None, str], list[float]] = collections.defaultdict(list)
    counts: collections.Counter[tuple[str, str, str | None, str]] = collections.Counter()
    for event in events:
        key = (
            str(event.get("role", "unknown")),
            str(event.get("stage", "other")),
            _mode_key(event.get("runtime_mode")),
            _descriptor_key(event.get("batch_descriptor")),
        )
        counts[key] += 1
        if event.get("duration_ms") is not None:
            groups[key].append(float(event["duration_ms"]))
    result = []
    for key, count in sorted(counts.items()):
        durations = groups.get(key, [])
        result.append(
            {
                "role": key[0],
                "stage": key[1],
                "runtime_mode": key[2],
                "batch_descriptor": json.loads(key[3]),
                "count": count,
                "finite_duration_count": len(durations),
                "durations_ms": durations,
                "median_duration_ms": statistics.median(durations) if durations else None,
            }
        )
    return result


def summarize(input_path: Path, *, run_dir: Path | None = None) -> dict[str, Any]:
    trace_path, inferred_run_dir = _trace_path(input_path)
    run_dir = _discover_run_dir(input_path, run_dir, trace_path) or inferred_run_dir
    eager = summarize_trace(trace_path)
    annotation_docs = _annotation_documents(run_dir)
    timing_docs = _timing_documents(run_dir)
    draft_spans = _draft_span_summary(annotation_docs)
    dispatches, graph_events, annotation_errors = _graph_rows(annotation_docs, timing_docs)
    attributed_graph, dispatch_rows = _attribute_graph_events(dispatches, graph_events)
    finite_graph = all(
        event.get("duration_ms") is not None and _finite(event.get("duration_ms")) and float(event["duration_ms"]) >= 0
        for event in attributed_graph
    )
    if not attributed_graph:
        finite_graph = True
    return {
        "format": FORMAT,
        "input": str(input_path.resolve()),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "eager_trace": eager,
        "graph_replay_attribution": {
            "events": attributed_graph,
            "groups": _graph_summary(attributed_graph),
            "finite_metrics": finite_graph,
            "source": "canonical deferred current-stream XPU events",
            "cpu_scope_time_is_not_included": True,
        },
        "draft_span_attribution": draft_spans,
        "dispatcher": {
            "events": dispatch_rows,
            "source": "CudagraphDispatcher.dispatch observed only inside bounded profile window",
            "actual_graph_keys_and_runtime_modes": True,
        },
        "annotation_artifacts": [str(path) for path, _ in annotation_docs],
        "timing_artifacts": [str(path) for path, _ in timing_docs],
        "annotation_errors": annotation_errors,
        "limitations": [
            "XPU graph replay commonly hides inner kernels from the native torch trace.",
            "CPU record_function scopes are inclusive host intervals and are not additive device time.",
            "Summed eager device work and graph-event intervals are not throughput or critical-path claims.",
            "No historical 54.88 ms proxy or performance outcome is inferred from this attribution.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="one trace file or the completed profile directory")
    parser.add_argument("--run-dir", type=Path, help="profile run directory when input is a trace file")
    parser.add_argument("--out", type=Path, help="write JSON here instead of stdout only")
    args = parser.parse_args(argv)
    try:
        result = summarize(args.input, run_dir=args.run_dir)
    except SummaryError as exc:
        parser.error(str(exc))
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
