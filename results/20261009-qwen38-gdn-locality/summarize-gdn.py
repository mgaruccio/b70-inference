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


FORMAT = "b70-gdn-locality-summary-v2"
UNITRACE_FORMAT = "b70-gdn-locality-unitrace-v1"
ANNOTATION_PREFIX = "b70_gdn/"
FUSION_SHARE_THRESHOLD = 0.05


class SummaryError(RuntimeError):
    """A malformed or incomplete native trace."""


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _duration(value: Any, label: str) -> float:
    if not _finite(value) or float(value) < 0:
        raise SummaryError(f"{label} has a non-finite or negative duration: {value!r}")
    return float(value)


def _annotation_stage(annotation: Any) -> str:
    text = str(annotation or "").lower()
    if text in {"draft", "target"}:
        return text
    if "/draft/" in text or text.startswith("draft/"):
        return "draft"
    if "/target/" in text or text.startswith("target/"):
        return "target"
    return "unknown"


def classify_operator(name: Any, annotation: Any = "") -> str:
    """Classify an operator without letting provenance names change its kind."""
    name_text = str(name or "").lower()
    annotation_text = str(annotation or "").lower()
    if annotation_text.startswith("b70_gdn/"):
        annotation_text = annotation_text.split("/", 2)[-1]
    compact = f"{name_text} {annotation_text}".replace("_", "").replace("-", "")
    if any(marker in compact for marker in ("gdnattention", "gdnattn", "gateddelta")) or "gdn" in compact:
        return "gdn"
    # A target module can contain qwen3_5_mtp.  Draft markers are therefore
    # read from the operator name only; stage/root provenance is reported
    # separately by _annotation_stage.
    if any(marker in name_text for marker in ("flash_attn", "flashattention", "varlen_fwd", "self_attn", "attention")):
        return "attention"
    if any(marker in name_text for marker in ("matmul", "mm.", "mm_", "::mm", "gemm", "addmm", "linear")):
        return "matmul"
    if any(marker in name_text for marker in ("eagle", "spec_decode", "specdecode", "draft", "mtp")):
        return "draft"
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
    return _annotation_stage(annotation)


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
        stage = _stage(annotation) if annotation else "unknown"
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


def fusion_decision(
    *,
    evidence_mode: str,
    graph_coverage: str,
    whole_replay_ms: Any,
    replay_reconciled: bool,
    gdn_kernel_ms: Any,
    gdn_provenance: str = "unknown",
) -> dict[str, Any]:
    """Apply the no-proxy gate for a possible locality/fusion decision."""
    reasons: list[str] = []
    whole = float(whole_replay_ms) if _finite(whole_replay_ms) and float(whole_replay_ms) > 0 else None
    gdn = float(gdn_kernel_ms) if _finite(gdn_kernel_ms) and float(gdn_kernel_ms) >= 0 else None
    if evidence_mode != "unitrace-profile":
        reasons.append("only normal graph-enabled unitrace evidence can establish a fusion share")
    if evidence_mode == "eager-profile":
        reasons.append("eager-only timing is diagnostic and cannot select or reject fusion")
    if graph_coverage != "complete":
        reasons.append("target graph kernel coverage is not complete")
    if whole is None:
        reasons.append("whole target graph replay timing is missing")
    if not replay_reconciled:
        reasons.append("unitrace kernel timestamps are not reconciled with whole replay timing")
    if gdn is None:
        reasons.append("GDN kernel duration is missing")
    if gdn_provenance != "target":
        reasons.append("target GDN provenance is not explicit")

    # Do not even publish a GDN/whole-replay ratio unless every attribution
    # prerequisite is satisfied.  Eager, partial, and stage-unknown traces may
    # contain useful operator rows, but they cannot establish a share or a
    # fusion decision.
    measurement_ready = (
        evidence_mode == "unitrace-profile"
        and graph_coverage == "complete"
        and whole is not None
        and replay_reconciled
        and gdn is not None
        and gdn_provenance == "target"
    )
    share = (gdn / whole) if measurement_ready else None
    if share is not None and share < FUSION_SHARE_THRESHOLD:
        reasons.append(f"GDN share is below the {FUSION_SHARE_THRESHOLD:.0%} candidate gate")
    eligible = measurement_ready and not reasons
    return {
        "decision_status": "eligible" if eligible else "inconclusive",
        "fusion_selection_allowed": eligible,
        "candidate_gate": f">={FUSION_SHARE_THRESHOLD:.0%} of reconciled target graph replay device time",
        "evidence_mode": evidence_mode,
        "graph_coverage": graph_coverage,
        "whole_replay_ms": whole,
        "gdn_kernel_ms": gdn,
        "gdn_provenance": gdn_provenance,
        "gdn_share_of_whole_replay": share,
        "replay_reconciled": bool(replay_reconciled),
        "reasons": reasons,
    }

def _find_unitrace_trace(value: Path) -> Path:
    value = value.resolve()
    if value.is_file():
        return value
    if not value.is_dir():
        raise SummaryError(f"unitrace input does not exist: {value}")
    traces = sorted(
        path for path in value.rglob("*.json")
        if path.is_file() and path.name not in {"summary.json", "effective-config.json", "launch-metadata.json"}
    )
    if not traces:
        raise SummaryError(f"no unitrace JSON trace under {value}")
    if len(traces) > 1:
        raise SummaryError(f"expected one bounded unitrace JSON trace, found {traces}")
    return traces[0]


def _unitrace_kernel_rows(events: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = []
    for event in events:
        category = str(event.get("cat", "")).lower()
        name = str(event.get("name", "")).lower()
        if "kernel" not in category and "kernel" not in name and "device" not in category:
            continue
        if not _finite(event.get("dur")) or float(event.get("dur", 0)) < 0:
            continue
        rows.append(event)
    return rows


def _is_gdn_kernel(event: Mapping[str, Any]) -> bool:
    text = str(event.get("name", "")).lower().replace("_", "")
    return any(marker in text for marker in ("gdn", "causalconv", "gateddelta", "deltarule"))


def _unitrace_stage(event: Mapping[str, Any]) -> str:
    args = event.get("args")
    values: list[Any] = [event.get("name")]
    if isinstance(args, Mapping):
        values.extend(args.get(key) for key in ("stage", "provenance", "annotation", "module"))
    for value in values:
        stage = _annotation_stage(value)
        if stage != "unknown":
            return stage
    return "unknown"

def summarize_unitrace(
    input_path: Path,
    *,
    graph_coverage: str = "unknown",
    whole_replay_ms: Any = None,
    replay_reconciled: bool = False,
) -> dict[str, Any]:
    path = _find_unitrace_trace(input_path)
    document = _read_json(path)
    events = document.get("traceEvents") if isinstance(document, Mapping) else None
    if not isinstance(events, list):
        raise SummaryError(f"unitrace trace has no traceEvents list: {path}")
    rows = _unitrace_kernel_rows(event for event in events if isinstance(event, Mapping))
    all_kernel_ms = sum(float(event["dur"]) for event in rows) / 1000.0
    gdn_rows = [event for event in rows if _is_gdn_kernel(event)]
    target_gdn_rows = [event for event in gdn_rows if _unitrace_stage(event) == "target"]
    gdn_kernel_ms = sum(float(event["dur"]) for event in gdn_rows) / 1000.0
    target_gdn_kernel_ms = sum(float(event["dur"]) for event in target_gdn_rows) / 1000.0
    decision = fusion_decision(
        evidence_mode="unitrace-profile",
        graph_coverage=graph_coverage,
        whole_replay_ms=whole_replay_ms,
        replay_reconciled=replay_reconciled,
        gdn_kernel_ms=target_gdn_kernel_ms if target_gdn_rows else None,
        gdn_provenance="target" if target_gdn_rows else "unknown",
    )
    return {
        "format": UNITRACE_FORMAT,
        "status": "ok" if rows else "insufficient_unitrace_visibility",
        "trace_path": str(path),
        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "trace_event_count": len(events),
        "kernel_count": len(rows),
        "gdn_kernel_count": len(gdn_rows),
        "target_gdn_kernel_count": len(target_gdn_rows),
        "all_kernel_ms": all_kernel_ms,
        "gdn_kernel_ms": gdn_kernel_ms if gdn_rows else None,
        "target_gdn_kernel_ms": target_gdn_kernel_ms if target_gdn_rows else None,
        "gdn_kernels": [
            {
                "name": event.get("name"),
                "duration_us": event.get("dur"),
                "ts": event.get("ts"),
                "stage": _unitrace_stage(event),
            }
            for event in gdn_rows[:256]
        ],
        "fusion_decision": decision,
        "attribution_status": decision["decision_status"],
        "limitations": [
            "Unitrace graph support is an evolving proof-of-concept and must be validated on this exact stack.",
            "Partial or missing graph records are not evidence that the hidden work is absent.",
            "Kernel names alone do not establish target/draft provenance; target-root evidence must be explicit.",
            "Eager timing and torch XPU Event boundaries cannot establish this share.",
        ],
    }


def summarize(input_path: Path, *, evidence_mode: str = "unknown", graph_coverage: str = "unknown", whole_replay_ms: Any = None, replay_reconciled: bool = False) -> dict[str, Any]:
    result = summarize_trace(input_path)
    result["fusion_decision"] = fusion_decision(
        evidence_mode=evidence_mode,
        graph_coverage=graph_coverage,
        whole_replay_ms=whole_replay_ms,
        replay_reconciled=replay_reconciled,
        gdn_kernel_ms=None,
    )
    result["attribution_status"] = result["fusion_decision"]["decision_status"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="one native or unitrace trace file/directory")
    parser.add_argument("--source", choices=("torch", "unitrace"), default="torch")
    parser.add_argument("--evidence-mode", choices=("unknown", "profile", "eager-profile", "unitrace-profile"), default="unknown")
    parser.add_argument("--graph-coverage", choices=("unknown", "partial", "complete"), default="unknown")
    parser.add_argument("--whole-replay-ms", type=float)
    parser.add_argument("--replay-reconciled", action="store_true")
    parser.add_argument("--out", type=Path, help="write JSON here as well as stdout")
    args = parser.parse_args(argv)
    try:
        if args.source == "unitrace":
            result = summarize_unitrace(
                args.input,
                graph_coverage=args.graph_coverage,
                whole_replay_ms=args.whole_replay_ms,
                replay_reconciled=args.replay_reconciled,
            )
        else:
            result = summarize(
                args.input,
                evidence_mode=args.evidence_mode,
                graph_coverage=args.graph_coverage,
                whole_replay_ms=args.whole_replay_ms,
                replay_reconciled=args.replay_reconciled,
            )
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
