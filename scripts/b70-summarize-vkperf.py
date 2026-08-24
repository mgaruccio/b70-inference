#!/usr/bin/env python3
"""Parse llama.cpp GGML_VK_PERF_LOGGER dumps and rank ops per generate tick."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


BLOCK_RE = re.compile(r"^----------------\s*$")
HEADER_RE = re.compile(r"^Vulkan Timings:\s*$")
TOTAL_RE = re.compile(r"^Total time:\s+([0-9.]+)\s+us\.?\s*$")
LINE_RE = re.compile(
    r"^(?P<name>.+):\s+(?P<count>\d+)\s+x\s+(?P<avg>[0-9.]+)\s+us\s+=\s+(?P<total>[0-9.]+)\s+us"
    r"(?:\s+\((?P<gflops>[0-9.]+)\s+GFLOPS/s\))?\s*$"
)
TICK_RE = re.compile(
    r"b70tick verify ms=(?P<ms>[0-9.]+)\s+n_tokens=(?P<n_tokens>\d+)\s+"
    r"unique_seq=(?P<unique_seq>\d+)"
)
TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]")


FAMILY_RULES = (
    ("flash_attn", re.compile(r"FLASH_ATTN", re.I)),
    ("mul_mat_q4", re.compile(r"MUL_MAT.*\bq4", re.I)),
    ("mul_mat_vec", re.compile(r"MUL_MAT_VEC", re.I)),
    ("mul_mat", re.compile(r"MUL_MAT", re.I)),
    ("rms_norm", re.compile(r"RMS_NORM", re.I)),
    ("softmax", re.compile(r"SOFT_MAX|SOFTMAX", re.I)),
    ("elementwise", re.compile(r"\b(ADD|MUL|DIV|SCALE|CPY|CONT|ROPE|GLU|SILU|GELU|UNARY)\b", re.I)),
    ("get_rows", re.compile(r"GET_ROWS", re.I)),
    ("set_rows", re.compile(r"SET_ROWS", re.I)),
    ("barrier_or_empty", re.compile(r"barrier|empty|NOOP", re.I)),
)


def family_of(name: str) -> str:
    for label, pat in FAMILY_RULES:
        if pat.search(name):
            return label
    return "other"


def parse_blocks(text: str) -> list[dict]:
    lines = text.splitlines()
    blocks: list[dict] = []
    i = 0
    while i < len(lines):
        if BLOCK_RE.match(lines[i]) and i + 1 < len(lines) and HEADER_RE.match(lines[i + 1]):
            ops = []
            total_us = None
            j = i + 2
            while j < len(lines):
                if BLOCK_RE.match(lines[j]):
                    break
                m_total = TOTAL_RE.match(lines[j])
                if m_total:
                    total_us = float(m_total.group(1))
                    j += 1
                    break
                m = LINE_RE.match(lines[j])
                if m:
                    ops.append(
                        {
                            "name": m.group("name"),
                            "count": int(m.group("count")),
                            "avg_us": float(m.group("avg")),
                            "total_us": float(m.group("total")),
                            "gflops": float(m.group("gflops")) if m.group("gflops") else None,
                            "family": family_of(m.group("name")),
                        }
                    )
                j += 1
            if ops:
                if total_us is None:
                    total_us = sum(op["total_us"] for op in ops)
                ops_sorted = sorted(ops, key=lambda o: o["total_us"], reverse=True)
                top = ops_sorted[0]
                blocks.append(
                    {
                        "line": i + 1,
                        "total_us": total_us,
                        "n_ops": len(ops),
                        "top_name": top["name"],
                        "top_us": top["total_us"],
                        "top_share": top["total_us"] / total_us if total_us else 0.0,
                        "ops": ops_sorted,
                    }
                )
            i = j
            continue
        i += 1
    return blocks


def parse_verifies(text: str) -> list[dict]:
    out = []
    for idx, line in enumerate(text.splitlines(), start=1):
        m = TICK_RE.search(line)
        if not m:
            continue
        out.append(
            {
                "line": idx,
                "ms": float(m.group("ms")),
                "n_tokens": int(m.group("n_tokens")),
                "unique_seq": int(m.group("unique_seq")),
            }
        )
    return out


def nearest_verify(block_line: int, verifies: list[dict]) -> dict | None:
    after = [v for v in verifies if v["line"] >= block_line]
    if after:
        return min(after, key=lambda v: v["line"] - block_line)
    before = [v for v in verifies if v["line"] < block_line]
    if before:
        return max(before, key=lambda v: v["line"])
    return None


def generate_filter(label: str, verify: dict | None) -> bool:
    if verify is None:
        return False
    if label.startswith("c1") and verify["unique_seq"] == 1 and verify["n_tokens"] in (3, 4):
        return True
    if "n1" in label and verify["unique_seq"] == 4 and verify["n_tokens"] in (8, 9):
        return True
    if "n2" in label and "c4" in label and verify["unique_seq"] == 4 and verify["n_tokens"] in (12, 13):
        return True
    return False


def summarize_subset(blocks: list[dict]) -> dict:
    if not blocks:
        return {"n": 0}
    totals = [b["total_us"] for b in blocks]
    family_us: dict[str, list[float]] = defaultdict(list)
    name_us: dict[str, list[float]] = defaultdict(list)
    for b in blocks:
        fam_sum: dict[str, float] = defaultdict(float)
        for op in b["ops"]:
            fam_sum[op["family"]] += op["total_us"]
            name_us[op["name"]].append(op["total_us"])
        for fam, us in fam_sum.items():
            family_us[fam].append(us)
    top_names = sorted(
        (
            {
                "name": name,
                "n": len(vals),
                "p50_us": statistics.median(vals),
                "mean_us": statistics.mean(vals),
                "share_of_median_total": statistics.median(vals) / statistics.median(totals)
                if statistics.median(totals)
                else 0.0,
            }
            for name, vals in name_us.items()
        ),
        key=lambda x: x["p50_us"],
        reverse=True,
    )
    top_fams = sorted(
        (
            {
                "family": fam,
                "n": len(vals),
                "p50_us": statistics.median(vals),
                "mean_us": statistics.mean(vals),
                "share_of_median_total": statistics.median(vals) / statistics.median(totals)
                if statistics.median(totals)
                else 0.0,
            }
            for fam, vals in family_us.items()
        ),
        key=lambda x: x["p50_us"],
        reverse=True,
    )
    return {
        "n": len(blocks),
        "total_p50_us": statistics.median(totals),
        "total_p50_ms": statistics.median(totals) / 1000.0,
        "total_mean_us": statistics.mean(totals),
        "top_ops": top_names[:12],
        "top_families": top_fams,
        "example_block": {
            "line": blocks[len(blocks) // 2]["line"],
            "total_us": blocks[len(blocks) // 2]["total_us"],
            "top": blocks[len(blocks) // 2]["ops"][:8],
        },
    }


def summarize_log(path: Path, label: str) -> dict:
    text = path.read_text(errors="replace")
    blocks = parse_blocks(text)
    verifies = parse_verifies(text)
    for b in blocks:
        b["verify"] = nearest_verify(b["line"], verifies)
    gen = [b for b in blocks if generate_filter(label, b.get("verify"))]
    large = sorted(blocks, key=lambda b: b["total_us"], reverse=True)[:8]
    return {
        "label": label,
        "log": str(path),
        "n_blocks": len(blocks),
        "n_verifies": len(verifies),
        "generate": summarize_subset(gen),
        "all": summarize_subset(blocks),
        "largest_blocks": [
            {
                "line": b["line"],
                "total_us": b["total_us"],
                "top_name": b["top_name"],
                "top_share": b["top_share"],
                "verify": b.get("verify"),
                "top": b["ops"][:6],
            }
            for b in large
        ],
    }


def compare(c1: dict, c4n1: dict, c4n2: dict) -> dict:
    def top(cell: dict) -> dict:
        gen = cell.get("generate") or {}
        ops = gen.get("top_ops") or []
        fams = gen.get("top_families") or []
        return {
            "total_p50_ms": gen.get("total_p50_ms"),
            "n": gen.get("n"),
            "top_op": ops[0] if ops else None,
            "top_family": fams[0] if fams else None,
            "families": fams[:6],
        }

    sick = top(c4n2)
    healthy_n1 = top(c4n1)
    healthy_c1 = top(c1)
    named = None
    if sick.get("top_op") and sick["top_op"]["share_of_median_total"] >= 0.5:
        named = {"kind": "op", **sick["top_op"]}
    elif sick.get("top_family") and sick["top_family"]["share_of_median_total"] >= 0.5:
        named = {"kind": "family", **sick["top_family"]}
    same_as_healthy = False
    if named and named["kind"] == "op":
        for other in (healthy_n1, healthy_c1):
            if other.get("top_op") and other["top_op"]["name"] == named.get("name"):
                if other["top_op"]["share_of_median_total"] >= 0.4:
                    same_as_healthy = True
    if named and named["kind"] == "family":
        for other in (healthy_n1, healthy_c1):
            if other.get("top_family") and other["top_family"]["family"] == named.get("family"):
                if other["top_family"]["share_of_median_total"] >= 0.4:
                    same_as_healthy = True
    return {
        "c1": healthy_c1,
        "c4n1": healthy_n1,
        "c4n2": sick,
        "named_bottleneck": named,
        "same_as_healthy_control": same_as_healthy,
        "pass": bool(named) and not same_as_healthy,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--c1")
    ap.add_argument("--c4n1")
    ap.add_argument("--c4n2")
    args = ap.parse_args()
    out = Path(args.out)
    if args.compare:
        payload = compare(
            json.loads(Path(args.c1).read_text()),
            json.loads(Path(args.c4n1).read_text()),
            json.loads(Path(args.c4n2).read_text()),
        )
    else:
        payload = summarize_log(Path(args.log), args.label)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
