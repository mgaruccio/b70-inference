#!/usr/bin/env python3
"""Summarize GDN visibility in a bounded native profiler trace.

The report consumes the ordinary PyTorch chrome trace produced by vLLM's
native profiler.  It matches kernel events to CPU ``External id`` records,
retains GDN operator names/input shapes, and groups device time conservatively
as GDN, matmul, attention, draft, or other.  A graph replay that hides inner
operators is reported as insufficient visibility; it is never converted into
a speed claim.
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


FORMAT = "b70-gdn-locality-summary-v1"
ANNOTATION_PREFIX = "b70_gdn/"


class SummaryError(RuntimeError):
    """A malformed or incomplete native trace."""


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _duration(value: Any, label: str) -> float:
    if not _finite(value) or float(value) < 0:
        raise SummaryError(f"{label} has a non-finite or negative duration: {value!r}")
    return float(value)


def classify_operator(name: Any, annotation: Any = "") -> str:
    """Classify a trace operator without relying on a runtime import."""
    text = f"{name or ''} {annotation or ''}".lower()
    compact = text.replace("_", "").replace("-", "")
    if any(marker in compact for marker in ("gdnattention", "gdnattn", "gateddelta")) or "gdn" in text:
        return "gdn"
    if any(marker in text for marker in ("eagle", "spec_decode", "specdecode", "draft", "mtp")):
        return "draft"
    if any(marker in text for marker in ("flash_attn", "flashattention", "varlen_fwd", "self_attn", "attention")):
        return "attention"
    if any(marker in text for marker in ("matmul", "mm.", "mm_", "::mm", "gemm", "addmm", "linear")):
        return "matmul"
    return "other"


def _read_json(path: Path) -> Any:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as source:
                return json.load(source)
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SummaryError(f"cannot read JSON trace {path}: {exc}") from exc


def _find_trace(value: Path) -> Path:
    value = value.resolve()
    if value.is_file():
        return value
    if not value.is_dir():
        raise SummaryError(f"input does not exist: {value}")
    traces = sorted(
        path for path in value.rglob("*.pt.trace.json*")
        if path.is_file() and path.name.startswith("rank")
    )
    if not traces:
        traces = sorted(path for path in value.rglob("*.trace.json*") if path.is_file())
    if not traces:
        raise SummaryError(f"no native profiler trace under {value}")
    if len(traces) > 1:
        raise SummaryError(f"expected one bounded rank trace, found {traces}")
    return traces[0]


def _load_events(path: Path) -> tuple[list[Mapping[str, Any]], str]:
    document = _read_json(path)
    events = document.get("traceEvents") if isinstance(document, Mapping) else None
    if not isinstance(events, list):
        raise SummaryError(f"trace has no traceEvents list: {path}")
    rows = [event for event in events if isinstance(event, Mapping)]
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def _external_id(event: Mapping[str, Any]) -> Any:
    args = event.get("args")
    return args.get("External id") if isinstance(args, Mapping) else None


def _event_end(event: Mapping[str, Any]) -> float:
    return float(event.get("ts", 0)) + float(event.get("dur", 0))


def _innermost(event: Mapping[str, Any], scopes: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    start = float(event.get("ts", 0))
    end = _event_end(event)
    containing = [
        scope for scope in scopes
        if float(scope.get("ts", 0)) <= start and end <= _event_end(scope) + 0.001
    ]
    if not containing:
        return None
    return min(containing, key=lambda scope: (float(scope.get("dur", 0)), -float(scope.get("ts", 0))))


def _stage(annotation: str | None) -> str:
    return "draft" if annotation and "/draft/" in annotation else "target"


def summarize_trace(path: Path) -> dict[str, Any]:
    events, trace_sha256 = _load_events(path)
    scopes_by_thread: dict[tuple[Any, Any], list[Mapping[str, Any]]] = collections.defaultdict(list)
    gdn_scopes: list[dict[str, Any]] = []
    for event in events:
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        if event.get("cat") == "user_annotation" and name.startswith(ANNOTATION_PREFIX):
            duration = _duration(event.get("dur"), f"annotation {name}")
            scopes_by_thread[(event.get("pid"), event.get("tid"))].append(event)
            gdn_scopes.append(
                {
                    "name": name,
                    "stage": _stage(name),
                    "duration_ms": duration / 1000.0,
                    "args": event.get("args", {}),
                }
            )

    cpu_by_external: dict[Any, Mapping[str, Any]] = {}
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "cpu_op":
            continue
        external = _external_id(event)
        if external is None:
            continue
        if external in cpu_by_external:
            raise SummaryError(f"duplicate CPU External id: {external!r}")
        cpu_by_external[external] = event

    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    operator_evidence: list[dict[str, Any]] = []
    kernel_count = 0
    missing_cpu_count = 0
    missing_cpu_us = 0.0
    all_kernel_us = 0.0

    for event in events:
        if event.get("cat") != "kernel":
            continue
        duration_us = _duration(event.get("dur"), f"kernel {event.get('name')}")
        kernel_count += 1
        all_kernel_us += duration_us
        cpu = cpu_by_external.get(_external_id(event))
        if cpu is None:
            missing_cpu_count += 1
            missing_cpu_us += duration_us
            continue
        scope = _innermost(cpu, scopes_by_thread.get((cpu.get("pid"), cpu.get("tid")), []))
        annotation = str(scope.get("name", "")) if scope is not None else ""
        category = classify_operator(cpu.get("name", ""), annotation)
        stage = _stage(annotation) if annotation else ("draft" if category == "draft" else "target")
        key = (category, stage)
        row = aggregate.setdefault(
            key,
            {"count": 0, "kernel_us": 0.0, "durations_us": [], "operator_names": set()},
        )
        row["count"] += 1
        row["kernel_us"] += duration_us
        row["durations_us"].append(duration_us)
        row["operator_names"].add(str(cpu.get("name", "")))
        if category == "gdn" and len(operator_evidence) < 128:
            args = cpu.get("args")
            operator_evidence.append(
                {
                    "stage": stage,
                    "operator": cpu.get("name"),
                    "input_dims": args.get("Input Dims") if isinstance(args, Mapping) else None,
                    "external_id": _external_id(cpu),
                    "cpu_duration_us": _duration(cpu.get("dur", 0), "GDN CPU operator"),
                    "annotation": annotation or None,
                    "kernel": event.get("name"),
                    "kernel_duration_us": duration_us,
                }
            )

    categories = []
    for (category, stage), row in sorted(aggregate.items()):
        categories.append(
            {
                "category": category,
                "stage": stage,
                "count": row["count"],
                "kernel_ms": row["kernel_us"] / 1000.0,
                "median_kernel_us": statistics.median(row["durations_us"]),
                "operator_names": sorted(row["operator_names"]),
            }
        )

    gdn_kernel_count = sum(row["count"] for (category, _), row in aggregate.items() if category == "gdn")
    gdn_scope_count = len(gdn_scopes)
    visible = gdn_kernel_count > 0 or bool(operator_evidence)
    return {
        "format": FORMAT,
        "status": "ok" if visible else "insufficient_gdn_visibility",
        "trace_path": str(path.resolve()),
        "trace_sha256": trace_sha256,
        "trace_event_count": len(events),
        "gdn_scope_count": gdn_scope_count,
        "gdn_operator_event_count": len(operator_evidence),
        "gdn_operator_events": operator_evidence,
        "gdn_scopes": gdn_scopes[:128],
        "kernel_count": kernel_count,
        "all_kernel_ms": all_kernel_us / 1000.0,
        "categories": categories,
        "unmapped_cpu_external_id": {"count": missing_cpu_count, "kernel_ms": missing_cpu_us / 1000.0},
        "graph_inner_ops_may_be_hidden": True,
        "eager_profile_required_if_gdn_not_visible": not visible,
        "limitations": [
            "Graph replay can hide inner GDN, matmul, and attention kernels from a native trace.",
            "CPU record_function scopes are inclusive host intervals and are not device critical-path time.",
            "Kernel categories are attribution evidence, not a throughput or speedup claim.",
        ],
    }


def summarize(input_path: Path) -> dict[str, Any]:
    return summarize_trace(_find_trace(input_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="one native trace file or completed profile directory")
    parser.add_argument("--out", type=Path, help="write JSON here as well as stdout")
    args = parser.parse_args(argv)
    try:
        result = summarize(args.input)
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
