#!/usr/bin/env python3
"""Reproduce this completed capacity report: python3 analyze.py > analysis.json.

Reuses the previous campaign's committed, standard-library-only reducers.
Raw files are read, never modified. Warmups are excluded from measurements.
"""
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "20260909-qwen38-dflash2-rtn-standard" / "analyze.py"
CELLS = ("int4-48k", "int4-64k", "bf16-48k", "bf16-64k")


def main():
    spec = importlib.util.spec_from_file_location("previous_campaign_analysis", REFERENCE)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    common.ROOT = ROOT
    cells = {}
    for name in CELLS:
        path = ROOT / name
        cell = json.loads((path / "summary.json").read_text())
        raw = json.loads((path / "long-context/summary.json").read_text())
        assert cell["status"] == "long_context_client_completed" and raw["status"] == "completed"
        result = common.cold(name)
        result.update(context=cell["context"], started_utc=cell["started_utc"],
                      finished_utc=cell["finished_utc"], draft_quantization=cell["draft_quantization"],
                      launcher_unchanged=cell["launcher_unchanged"], power_unchanged=cell["power_unchanged"],
                      glimmer_running_after=cell["glimmer_running_after"], peak_workload_device_memory=None)
        for point, source in zip(result["points"], raw["points"]):
            point["measured_distributions"] = {
                metric: common.distribution([row[metric] for row in source["measurements"] if row.get("valid")])
                for metric in ("decode_tps_post_first", "ttft_s", "e2e_tps")
            }
        log = (path / "server.log").read_text()
        patterns = {
            "model_loading_gib": r"Model loading took ([0-9.]+) GiB memory",
            "available_kv_gib": r"Available KV cache memory: ([0-9.]+) GiB",
            "reported_kv_tokens_not_tested_maximum": r"GPU KV cache size: ([0-9,]+) tokens",
        }
        result["startup_memory"] = {}
        for key, pattern in patterns.items():
            values = re.findall(pattern, log)
            assert len(values) == 1, (name, key, values)
            result["startup_memory"][key] = float(values[0]) if "gib" in key else int(values[0].replace(",", ""))
        result["largest_successful_prompt_tokens"] = max(p["requested_length"] for p in raw["points"] if p["status"] == "complete")
        cells[name] = result
    comparisons = {}
    for suffix, prompt in (("48k", 49024), ("64k", 65408)):
        pair = {arm: next(p for p in cells[f"{arm}-{suffix}"]["points"] if p["requested_length"] == prompt)
                for arm in ("bf16", "int4")}
        comparisons[suffix] = {"prompt_tokens": prompt, "output_tokens": 128,
                              "int4_decode_median_delta_percent": 100 * (
                                  pair["int4"]["statistics"]["median"]["decode_tps_post_first"] /
                                  pair["bf16"]["statistics"]["median"]["decode_tps_post_first"] - 1)}
    print(json.dumps({"tier": "development", "method": "Sequential cells; six measured trials per supported point; medians, inclusive IQR and sample CV. No significance or quality-equivalence claim.",
                      "cells": cells, "near_limit_comparisons": comparisons}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
