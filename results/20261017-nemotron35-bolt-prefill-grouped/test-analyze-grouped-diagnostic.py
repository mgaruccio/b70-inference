#!/usr/bin/env python3
"""CPU-only tests for analyze-grouped-diagnostic.py."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("grouped_diagnostic", ROOT / "analyze-grouped-diagnostic.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load analyze-grouped-diagnostic.py")
ANALYZER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYZER
SPEC.loader.exec_module(ANALYZER)


def _common(rows: list[int]) -> dict[str, Any]:
    total = sum(rows)
    return {
        "schema": ANALYZER.SCHEMA,
        "M": 4096,
        "K": 4096,
        "experts": len(rows),
        "total_rows": total,
        "n_active": sum(value > 0 for value in rows),
        "row_counts": rows,
        "tiles_bn32": sum((value + 31) // 32 for value in rows),
        "tail_rows_bn32": [value % 32 for value in rows],
        "valid_columns": total,
    }


def _begin(rows: list[int], *, eligible: bool = True) -> dict[str, Any]:
    value = _common(rows)
    value.update(
        {
            "event": "caller_begin",
            "tensor": "blk\\\"iq2",
            "outer_eligible": eligible,
            "outer_contiguous": True,
            "outer_type_ok": eligible,
            "outer_stride_ok": True,
            "outer_reasons": [] if eligible else ["type"],
            "padded_columns": None,
            "n_tiles": None,
            "selected_combo": None,
            "BN": None,
            "SG_ROWS": None,
        }
    )
    return value


def _result(rows: list[int], *, decision: str = "grouped", reason: str = "launch_true") -> dict[str, Any]:
    value = _common(rows)
    tiles = sum((item + 31) // 32 for item in rows)
    value.update(
        {
            "event": "inner_result",
            "decision": decision,
            "reason": reason,
            "selected_combo": 0,
            "combo_name": "f16xf16 8x16x16 sg16",
            "BN": 32,
            "SG_ROWS": 16,
            "n_tiles": tiles,
            "padded_columns": tiles * 32,
            "reordered": False,
            "src1_precision": 0,
        }
    )
    if decision == "fallback" and reason != "launch_false":
        for key in ("selected_combo", "combo_name", "BN", "SG_ROWS", "n_tiles", "padded_columns"):
            value[key] = None
    return value


def _write_log(records: list[dict[str, Any]]) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    directory = tempfile.TemporaryDirectory()
    path = Path(directory.name) / "server.log"
    lines = ["ordinary runtime line\n"]
    for record in records:
        lines.append("prefix " + ANALYZER.MARKER + " " + json.dumps(record) + " suffix\n")
        lines.append("unrelated line\n")
    path.write_text("".join(lines), encoding="utf-8")
    return directory, path


class GroupedDiagnosticTests(unittest.TestCase):
    def test_pairs_markers_and_aggregates_shape_histogram(self) -> None:
        rows = [32, 16, 0]
        directory, path = _write_log([_begin(rows), _result(rows)])
        try:
            report = ANALYZER.analyze_path(path, windows=[(1, 5)])
        finally:
            directory.cleanup()
        self.assertEqual(report["calls"], 1)
        self.assertEqual(report["grouped_calls"], 1)
        self.assertEqual(report["row_histogram"], {0: 1, 16: 1, 32: 1})
        self.assertEqual(report["by_M_K"]["4096x4096"]["tiles_bn32"], 2)
        self.assertEqual(report["request_window_log_slices"][0]["summary"]["calls"], 1)
        self.assertAlmostEqual(report["calls_detail"][0]["padding_fraction"], 16 / 64)

    def test_outer_fallback_is_retained_without_inner_marker(self) -> None:
        rows = [16, 16]
        directory, path = _write_log([_begin(rows, eligible=False)])
        try:
            report = ANALYZER.analyze_path(path)
        finally:
            directory.cleanup()
        self.assertEqual(report["fallback_calls"], 1)
        self.assertEqual(report["outer_fallback_reasons"], {"type": 1})
        self.assertEqual(report["calls_detail"][0]["reason"], "outer_type")

    def test_rejects_total_mismatch_and_large_tail(self) -> None:
        rows = [16, 16]
        bad_total = _begin(rows)
        bad_total["total_rows"] += 1
        directory, path = _write_log([bad_total])
        try:
            with self.assertRaisesRegex(ANALYZER.DiagnosticError, "row_counts sum"):
                ANALYZER.analyze_path(path)
        finally:
            directory.cleanup()

        bad_tail = _begin([17, 16])
        bad_tail["tail_rows_bn32"] = [17, 16]
        directory, path = _write_log([bad_tail])
        try:
            with self.assertRaisesRegex(ANALYZER.DiagnosticError, "tail"):
                ANALYZER.analyze_path(path)
        finally:
            directory.cleanup()

    def test_rejects_cut_window_and_unpaired_inner_result(self) -> None:
        rows = [16, 16]
        directory, path = _write_log([_begin(rows), _result(rows)])
        try:
            with self.assertRaisesRegex(ANALYZER.DiagnosticError, "cuts through"):
                ANALYZER.analyze_path(path, windows=[(2, 2)])
        finally:
            directory.cleanup()

        directory, path = _write_log([_result(rows)])
        try:
            with self.assertRaisesRegex(ANALYZER.DiagnosticError, "no preceding"):
                ANALYZER.analyze_path(path)
        finally:
            directory.cleanup()


if __name__ == "__main__":
    unittest.main()
