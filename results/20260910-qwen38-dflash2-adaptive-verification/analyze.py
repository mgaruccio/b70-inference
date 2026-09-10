#!/usr/bin/env python3
"""Reproduce development comparisons from retained API evidence, without a server."""
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "common", ROOT.parent / "20260909-qwen38-dflash2-rtn-standard/analyze.py")
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)


def load(path):
    return json.loads(path.read_text())


def adaptive_summary(path):
    requests = [json.loads(match.group(1)) for match in re.finditer(
        r"B70_DFLASH2_ADAPTIVE (\{.*\})", (path / "server.log").read_text())]
    depths, switches, decisions = {}, {}, {}
    for request in requests:
        for field, total in (("scheduled_depths", depths), ("switches", switches), ("decisions", decisions)):
            for key, count in request[field].items():
                total[key] = total.get(key, 0) + count
    return {"scope": "all requests served by this cell, including canaries and warmups; scheduled, not executed, depths",
            "request_count": len(requests), "scheduled_depths": depths,
            "switches": switches, "decisions": decisions, "requests": requests}


def main():
    cells, raw_points = {}, {}
    for line in (ROOT / "cell-exits.tsv").read_text().splitlines():
        name, code = line.split("\t")
        if name in cells:
            raise ValueError(f"duplicate cell: {name}")
        path = ROOT / name
        summary = load(path / "summary.json")
        result = {key: summary.get(key) for key in (
            "status", "error", "context", "verification_cap", "speculative_tokens",
            "adaptive_verification",
            "image", "max_num_batched_tokens", "cache_group_size",
            "launcher_unchanged", "power_unchanged", "glimmer_running_after")}
        result["exit_code"] = int(code)
        if summary.get("adaptive_verification"):
            result["adaptive"] = adaptive_summary(path)
        if not (result["launcher_unchanged"] and result["power_unchanged"]
                and result["glimmer_running_after"] == "false"):
            raise ValueError(f"cleanup invariant failed: {name}")
        result["points"] = []
        raw_points[name] = {}
        cold_path = path / "long-context/summary.json"
        if cold_path.exists():
            cold = load(cold_path)
            result["cold_status"] = cold["status"]
            result["cold_errors"] = cold["errors"]
            for point in cold["points"]:
                length = point["requested_length"]
                raw_points[name][length] = point
                rows = point["measurements"]
                valid = [r for r in rows if r.get("valid")]
                aggregate = {}
                for row in valid:
                    delta = common.spec_delta(
                        "\n".join(row["metrics_before"]["speculative_position_counter_lines"]),
                        "\n".join(row["metrics_after"]["speculative_position_counter_lines"]))
                    for key, value in delta.items():
                        aggregate[key] = aggregate.get(key, 0) + value
                spec = common.spec_summary(aggregate)
                steps = spec["draft_steps"]
                spec["mean_proposed_per_round"] = spec["proposed"] / steps if steps else None
                spec["client_ms_per_round_proxy"] = (
                    1000 * sum(r["decode_elapsed_s"] for r in valid) / steps if steps else None)
                result["points"].append({
                    "input_tokens": length, "status": point["status"],
                    "measured": len(rows), "valid_measured": len(valid),
                    "warmup_valid": (point.get("warmup") or {}).get("valid"),
                    "statistics": {metric: common.distribution([r[metric] for r in valid])
                                   for metric in ("decode_tps_post_first", "ttft_s", "e2e_tps")},
                    "speculation": spec})
        cells[name] = result
    comparisons = {}
    for name, points in raw_points.items():
        if name == "baseline":
            continue
        matches = []
        for length, candidate in points.items():
            baseline = raw_points.get("baseline", {}).get(length)
            if not baseline or candidate["status"] != "complete" or baseline["status"] != "complete":
                continue
            lhs = [baseline["warmup"], *baseline["measurements"]]
            rhs = [candidate["warmup"], *candidate["measurements"]]
            if len(lhs) != len(rhs):
                raise ValueError(f"unmatched trial count: {name} {length}")
            identical_payloads = identical_text = 0
            for a, b in zip(lhs, rhs):
                request_a = load(ROOT / "baseline/long-context" / a["request_path"])
                request_b = load(ROOT / name / "long-context" / b["request_path"])
                if request_a != request_b:
                    raise ValueError(f"unmatched request: {name} {length}")
                identical_payloads += 1
                if "text" not in a["stream"] or "text" not in b["stream"]:
                    raise ValueError("missing decoded text for identity comparison")
                identical_text += a["stream"]["text"] == b["stream"]["text"]
            row = {"input_tokens": length, "identical_payloads": identical_payloads,
                   "identical_decoded_outputs": identical_text, "trials_including_warmup": len(lhs)}
            a = next(p for p in cells["baseline"]["points"] if p["input_tokens"] == length)
            b = next(p for p in cells[name]["points"] if p["input_tokens"] == length)
            row["median_delta_percent"] = {
                metric: 100 * (b["statistics"][metric]["median"] / a["statistics"][metric]["median"] - 1)
                for metric in ("decode_tps_post_first", "ttft_s", "e2e_tps")}
            matches.append(row)
        comparisons[name] = matches
    print(json.dumps({"tier": "development", "cells": cells, "comparisons": comparisons,
                      "limitations": ["Sequential cells; no significance claim",
                                      "Client milliseconds per round are not GPU component profiles",
                                      "Fixed caps do not adapt generation and retain Kmax7 allocation",
                                      "Decoded text identity is not a broad quality evaluation"]},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
