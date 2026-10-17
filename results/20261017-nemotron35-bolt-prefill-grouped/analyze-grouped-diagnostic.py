#!/usr/bin/env python3
"""Parse host-side grouped IQ2 diagnostic markers from a llama.cpp log.

The parser consumes only lines containing ``[SYCL_GROUPED_IQ2_DIAG]``.  It
pairs each caller begin marker with the immediately following inner result
marker (allowing unrelated log lines between them), validates the host-side
routing invariants, and reports grouped/fallback decisions without claiming
performance.  Use ``--window START:END`` to summarize request-sized line
slices separately (line numbers are one-based and inclusive).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

MARKER = "[SYCL_GROUPED_IQ2_DIAG]"
SCHEMA = "grouped_iq2_diag/v1"
TAIL_LIMIT = 16
TILE_WIDTH = 32


class DiagnosticError(ValueError):
    """Raised when a diagnostic marker violates the host-side contract."""


@dataclass(frozen=True)
class MarkerRecord:
    line: int
    data: dict[str, Any]


@dataclass(frozen=True)
class DiagnosticCall:
    begin: MarkerRecord
    result: MarkerRecord | None


def _json_marker(line: str, line_number: int) -> dict[str, Any] | None:
    """Decode one marker line, tolerating arbitrary log prefixes/suffixes."""
    marker_at = line.find(MARKER)
    if marker_at < 0:
        return None
    payload = line[marker_at + len(MARKER) :].lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(payload)
    except json.JSONDecodeError as exc:
        raise DiagnosticError(f"line {line_number}: malformed diagnostic JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise DiagnosticError(f"line {line_number}: diagnostic JSON must be an object")
    if value.get("schema") != SCHEMA:
        raise DiagnosticError(f"line {line_number}: unsupported schema {value.get('schema')!r}")
    return value


def read_markers(path: Path) -> list[MarkerRecord]:
    records: list[MarkerRecord] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            value = _json_marker(line, line_number)
            if value is not None:
                records.append(MarkerRecord(line_number, value))
    return records


def _int(data: dict[str, Any], key: str, *, line: int) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DiagnosticError(f"line {line}: {key} must be an integer")
    return value


def _bool(data: dict[str, Any], key: str, *, line: int) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise DiagnosticError(f"line {line}: {key} must be a boolean")
    return value


def _rows(data: dict[str, Any], *, line: int) -> list[int]:
    values = data.get("row_counts")
    if not isinstance(values, list) or any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise DiagnosticError(f"line {line}: row_counts must be an integer array")
    if any(v < 0 for v in values):
        raise DiagnosticError(f"line {line}: row_counts cannot be negative")
    return values


def _validate_common(data: dict[str, Any], *, line: int) -> dict[str, Any]:
    experts = _int(data, "experts", line=line)
    total_rows = _int(data, "total_rows", line=line)
    n_active = _int(data, "n_active", line=line)
    M = _int(data, "M", line=line)
    K = _int(data, "K", line=line)
    rows = _rows(data, line=line)
    if experts <= 0 or len(rows) != experts:
        raise DiagnosticError(f"line {line}: row_counts length does not equal experts")
    if M <= 0 or K <= 0:
        raise DiagnosticError(f"line {line}: M and K must be positive")
    if total_rows <= 0:
        raise DiagnosticError(f"line {line}: total_rows must be positive")
    if sum(rows) != total_rows:
        raise DiagnosticError(f"line {line}: row_counts sum {sum(rows)} != total_rows {total_rows}")
    expected_active = sum(value > 0 for value in rows)
    if n_active != expected_active:
        raise DiagnosticError(f"line {line}: n_active {n_active} != {expected_active}")
    if n_active <= 0:
        raise DiagnosticError(f"line {line}: n_active must be positive")

    expected_tiles = sum((value + TILE_WIDTH - 1) // TILE_WIDTH for value in rows)
    if data.get("tiles_bn32") != expected_tiles:
        raise DiagnosticError(
            f"line {line}: tiles_bn32 {data.get('tiles_bn32')!r} != ceil(row_counts/32) sum {expected_tiles}"
        )
    expected_tails = [value % TILE_WIDTH for value in rows]
    if data.get("tail_rows_bn32") != expected_tails:
        raise DiagnosticError(f"line {line}: tail_rows_bn32 does not match row_counts modulo 32")
    if any(value > TAIL_LIMIT for value in expected_tails):
        raise DiagnosticError(f"line {line}: row-count tail exceeds {TAIL_LIMIT}: {expected_tails}")
    if data.get("valid_columns") != total_rows:
        raise DiagnosticError(f"line {line}: valid_columns must equal total_rows")

    return {
        "M": M,
        "K": K,
        "experts": experts,
        "total_rows": total_rows,
        "n_active": n_active,
        "row_counts": rows,
        "tiles_bn32": expected_tiles,
        "tail_rows_bn32": expected_tails,
    }


def _validate_begin(record: MarkerRecord) -> dict[str, Any]:
    data = record.data
    if data.get("event") != "caller_begin":
        raise DiagnosticError(f"line {record.line}: expected caller_begin, got {data.get('event')!r}")
    common = _validate_common(data, line=record.line)
    outer_eligible = _bool(data, "outer_eligible", line=record.line)
    outer_contiguous = _bool(data, "outer_contiguous", line=record.line)
    outer_type = _bool(data, "outer_type_ok", line=record.line)
    outer_stride = _bool(data, "outer_stride_ok", line=record.line)
    reasons = data.get("outer_reasons")
    if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
        raise DiagnosticError(f"line {record.line}: outer_reasons must be a string array")
    expected_reasons = [
        reason
        for reason, ok in (
            ("contiguity", outer_contiguous),
            ("type", outer_type),
            ("stride", outer_stride),
        )
        if not ok
    ]
    if reasons != expected_reasons:
        raise DiagnosticError(f"line {record.line}: outer_reasons do not match outer gate fields")
    if outer_eligible != (not expected_reasons):
        raise DiagnosticError(f"line {record.line}: outer_eligible disagrees with outer_reasons")
    if data.get("padded_columns") is not None or data.get("n_tiles") is not None:
        raise DiagnosticError(f"line {record.line}: caller begin cannot report selected tiles")
    if data.get("selected_combo") is not None or data.get("BN") is not None or data.get("SG_ROWS") is not None:
        raise DiagnosticError(f"line {record.line}: caller begin cannot report a selected combo")
    common.update(
        {
            "tensor": data.get("tensor"),
            "outer_eligible": outer_eligible,
            "outer_reasons": reasons,
        }
    )
    return common


def _nullable_int(data: dict[str, Any], key: str, *, line: int) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise DiagnosticError(f"line {line}: {key} must be an integer or null")
    return value


def _validate_result(record: MarkerRecord, begin: dict[str, Any]) -> dict[str, Any]:
    data = record.data
    if data.get("event") != "inner_result":
        raise DiagnosticError(f"line {record.line}: expected inner_result, got {data.get('event')!r}")
    common = _validate_common(data, line=record.line)
    for key in ("M", "K", "experts", "total_rows", "n_active", "row_counts", "tiles_bn32", "tail_rows_bn32"):
        if common[key] != begin[key]:
            raise DiagnosticError(f"line {record.line}: inner {key} disagrees with caller begin")

    decision = data.get("decision")
    reason = data.get("reason")
    if decision not in {"grouped", "fallback"} or not isinstance(reason, str):
        raise DiagnosticError(f"line {record.line}: invalid inner decision/reason")
    allowed_reasons = {
        "dynamic_precision",
        "type_mask",
        "unsupported_layout_type",
        "invalid_shape",
        "average_gt_64",
        "unsupported_combo",
        "launch_true",
        "launch_false",
    }
    if reason not in allowed_reasons:
        raise DiagnosticError(f"line {record.line}: unknown inner reason {reason!r}")
    if decision == "grouped" and reason != "launch_true":
        raise DiagnosticError(f"line {record.line}: grouped result must use launch_true")
    if decision == "fallback" and reason == "launch_true":
        raise DiagnosticError(f"line {record.line}: launch_true cannot be a fallback")
    if reason == "average_gt_64" and common["total_rows"] <= common["n_active"] * 64:
        raise DiagnosticError(f"line {record.line}: average_gt_64 reason is not justified by totals")

    combo = _nullable_int(data, "selected_combo", line=record.line)
    BN = _nullable_int(data, "BN", line=record.line)
    SG_ROWS = _nullable_int(data, "SG_ROWS", line=record.line)
    n_tiles = _nullable_int(data, "n_tiles", line=record.line)
    padded = _nullable_int(data, "padded_columns", line=record.line)
    if decision == "grouped" or reason == "launch_false":
        if combo is None or BN is None or SG_ROWS is None or n_tiles is None:
            raise DiagnosticError(f"line {record.line}: selected launch result lacks combo/tile fields")
        if combo < 0 or BN <= 0 or SG_ROWS <= 0 or n_tiles <= 0:
            raise DiagnosticError(f"line {record.line}: selected combo/tile fields are invalid")
        expected_tiles = sum((value + BN - 1) // BN for value in common["row_counts"])
        if n_tiles != expected_tiles:
            raise DiagnosticError(f"line {record.line}: n_tiles {n_tiles} != per-expert ceil(rows/BN) {expected_tiles}")
        if padded != n_tiles * BN:
            raise DiagnosticError(f"line {record.line}: padded_columns does not equal n_tiles * BN")
        if padded < common["total_rows"]:
            raise DiagnosticError(f"line {record.line}: padded_columns is smaller than valid_columns")
    else:
        if any(value is not None for value in (combo, BN, SG_ROWS, n_tiles, padded)):
            raise DiagnosticError(f"line {record.line}: gate fallback cannot report selected tiles")

    result = dict(common)
    result.update(
        {
            "decision": decision,
            "reason": reason,
            "selected_combo": combo,
            "BN": BN,
            "SG_ROWS": SG_ROWS,
            "n_tiles": n_tiles,
            "padded_columns": padded,
            "padding_fraction": (padded - common["total_rows"]) / padded if padded else None,
        }
    )
    return result


def pair_markers(records: Sequence[MarkerRecord]) -> list[DiagnosticCall]:
    """Pair begin/result markers, allowing outer-gate fallbacks without a result."""
    calls: list[DiagnosticCall] = []
    pending: MarkerRecord | None = None
    for record in records:
        event = record.data.get("event")
        if event == "caller_begin":
            if pending is not None:
                raise DiagnosticError(f"line {record.line}: caller begin precedes result for line {pending.line}")
            begin = _validate_begin(record)
            pending = record
            if not begin["outer_eligible"]:
                calls.append(DiagnosticCall(record, None))
                pending = None
        elif event == "inner_result":
            if pending is None:
                raise DiagnosticError(f"line {record.line}: inner result has no preceding caller begin")
            begin = _validate_begin(pending)
            if not begin["outer_eligible"]:
                raise DiagnosticError(f"line {record.line}: inner result follows an outer-ineligible caller")
            calls.append(DiagnosticCall(pending, record))
            pending = None
        else:
            raise DiagnosticError(f"line {record.line}: unknown diagnostic event {event!r}")
    if pending is not None:
        raise DiagnosticError(f"line {pending.line}: outer-eligible caller has no inner result")
    return calls


def _call_summary(call: DiagnosticCall) -> dict[str, Any]:
    begin = _validate_begin(call.begin)
    result = _validate_result(call.result, begin) if call.result is not None else None
    rows = begin["row_counts"]
    detail: dict[str, Any] = {
        "begin_line": call.begin.line,
        "result_line": call.result.line if call.result else None,
        "tensor": begin.get("tensor"),
        "M": begin["M"],
        "K": begin["K"],
        "experts": begin["experts"],
        "total_rows": begin["total_rows"],
        "n_active": begin["n_active"],
        "row_counts": rows,
        "row_histogram": dict(sorted(Counter(rows).items())),
        "tiles_bn32": begin["tiles_bn32"],
        "tail_rows_bn32": begin["tail_rows_bn32"],
        "outer_eligible": begin["outer_eligible"],
        "outer_reasons": begin["outer_reasons"],
        "decision": "fallback" if result is None else result["decision"],
        "reason": "outer_" + "+".join(begin["outer_reasons"]) if result is None else result["reason"],
        "selected_combo": None if result is None else result["selected_combo"],
        "BN": None if result is None else result["BN"],
        "SG_ROWS": None if result is None else result["SG_ROWS"],
        "n_tiles": None if result is None else result["n_tiles"],
        "valid_columns": begin["total_rows"],
        "padded_columns": None if result is None else result["padded_columns"],
        "padding_fraction": None if result is None else result["padding_fraction"],
        "padding_fraction_bn32": (
            (begin["tiles_bn32"] * TILE_WIDTH - begin["total_rows"]) / (begin["tiles_bn32"] * TILE_WIDTH)
        ),
    }
    return detail


def _counter_dict(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def summarize_calls(calls: Sequence[DiagnosticCall], *, source: str | None = None) -> dict[str, Any]:
    details = [_call_summary(call) for call in calls]
    shape_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        shape_groups[(detail["M"], detail["K"])].append(detail)

    by_shape: dict[str, Any] = {}
    for (M, K), group in sorted(shape_groups.items()):
        by_shape[f"{M}x{K}"] = {
            "M": M,
            "K": K,
            "calls": len(group),
            "grouped": sum(item["decision"] == "grouped" for item in group),
            "fallback": sum(item["decision"] == "fallback" for item in group),
            "total_rows": sum(item["total_rows"] for item in group),
            "n_active": sum(item["n_active"] for item in group),
            "tiles_bn32": sum(item["tiles_bn32"] for item in group),
            "observed_tiles": sum(item["n_tiles"] or 0 for item in group),
            "padding_fraction_mean": (
                sum(item["padding_fraction_bn32"] for item in group) / len(group)
            ),
            "reasons": _counter_dict(item["reason"] for item in group),
        }

    grouped = [item for item in details if item["decision"] == "grouped"]
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "source": source,
        "marker_records": len(details) + sum(item["result_line"] is not None for item in details),
        "calls": len(details),
        "grouped_calls": len(grouped),
        "fallback_calls": len(details) - len(grouped),
        "outer_fallback_reasons": _counter_dict(
            reason for item in details if not item["outer_eligible"] for reason in item["outer_reasons"]
        ),
        "inner_fallback_reasons": _counter_dict(
            item["reason"] for item in details if item["result_line"] is not None and item["decision"] == "fallback"
        ),
        "row_histogram": dict(sorted(Counter(row for item in details for row in item["row_counts"]).items())),
        "tiles_bn32": sum(item["tiles_bn32"] for item in details),
        "observed_tiles": sum(item["n_tiles"] or 0 for item in details),
        "n_active": sum(item["n_active"] for item in details),
        "padding_fraction_mean_bn32": (
            sum(item["padding_fraction_bn32"] for item in details) / len(details) if details else None
        ),
        "by_M_K": by_shape,
        "calls_detail": details,
    }
    return summary


def _window_spec(spec: str) -> tuple[int, int]:
    try:
        start_text, end_text = spec.split(":", 1)
        start, end = int(start_text), int(end_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(f"invalid window {spec!r}; expected START:END") from exc
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError(f"invalid window {spec!r}; expected 1 <= START <= END")
    return start, end


def analyze_path(path: Path, windows: Sequence[tuple[int, int]] = ()) -> dict[str, Any]:
    records = read_markers(path)
    calls = pair_markers(records)
    result = summarize_calls(calls, source=str(path))
    result["request_window_log_slices"] = []
    if windows:
        slices: list[dict[str, Any]] = []
        for start, end in windows:
            selected = [
                call
                for call in calls
                if start <= call.begin.line <= end
                and (call.result is None or call.result.line <= end)
            ]
            excluded = [
                call
                for call in calls
                if start <= call.begin.line <= end
                and call.result is not None
                and call.result.line > end
            ]
            if excluded:
                raise DiagnosticError(f"window {start}:{end} cuts through a caller/result pair")
            window_summary = summarize_calls(selected, source=str(path))
            slices.append({"start_line": start, "end_line": end, "summary": window_summary})
        result["request_window_log_slices"] = slices
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="llama.cpp stderr/server log containing diagnostic markers")
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        type=_window_spec,
        metavar="START:END",
        help="also summarize this inclusive one-based log-line slice; repeatable",
    )
    args = parser.parse_args(argv)
    try:
        report = analyze_path(args.log, args.window)
    except (OSError, DiagnosticError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
