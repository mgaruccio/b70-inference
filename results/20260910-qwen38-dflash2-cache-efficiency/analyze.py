#!/usr/bin/env python3
"""Reproduce this development report from retained raw files; no server calls."""
import importlib.util
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "20260909-qwen38-dflash2-rtn-standard/analyze.py"


def main():
    spec = importlib.util.spec_from_file_location("previous_campaign_analysis", REFERENCE)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    common.ROOT = ROOT
    cells = {}
    for line in (ROOT / "cell-exits.tsv").read_text().splitlines():
        name, exit_code = line.split("\t")
        assert name not in cells, f"duplicate cell: {name}"
        path = ROOT / name
        summary = json.loads((path / "summary.json").read_text())
        result = {key: summary.get(key) for key in (
            "status", "error", "context", "started_utc", "finished_utc", "max_num_batched_tokens",
            "cache_group_size", "launcher_unchanged", "power_unchanged", "glimmer_running_after")}
        result["exit_code"] = int(exit_code)
        result["peak_workload_device_memory"] = None
        assert result["launcher_unchanged"] and result["power_unchanged"]
        assert result["glimmer_running_after"] == "false"
        log = (path / "server.log").read_text()
        patterns = {
            "model_loading_gib": r"Model loading took ([0-9.]+) GiB memory",
            "available_kv_gib": r"Available KV cache memory: ([0-9.]+) GiB",
            "consumed_weights_and_runtime_gib": r"Actual usage is ([0-9.]+) GiB for consumed memory",
            "profiled_peak_activation_gib": r"([0-9.]+) GiB for peak activation",
            "graph_gib": r"([0-9.]+) GiB for CUDAGraph memory",
        }
        result["startup_memory"] = {
            key: float(matches[-1]) if (matches := re.findall(pattern, log)) else None
            for key, pattern in patterns.items()}
        result["padding_warnings"] = re.findall(r"Add \d+ padding layers[^\n]*", log)
        rejection = re.search(
            r"max seq len \((\d+)\), \(([0-9.]+) GiB KV cache is needed, which is larger than the available KV cache "
            r"memory \(([0-9.]+) GiB\).*?estimated maximum model length is (\d+)", log)
        if rejection:
            result["startup_capacity_rejection"] = {
                "requested_context": int(rejection[1]), "required_cache_gib": float(rejection[2]),
                "available_for_admission_gib": float(rejection[3]),
                "estimated_max_context_not_tested": int(rejection[4]),
            }
        result["cold"] = common.cold(name)
        cold_path = path / "long-context/summary.json"
        if cold_path.exists():
            raw = json.loads(cold_path.read_text())
            for point, source in zip(result["cold"]["points"], raw["points"]):
                point["measured_distributions"] = {
                    metric: common.distribution([row[metric] for row in source["measurements"] if row.get("valid")])
                    for metric in ("decode_tps_post_first", "ttft_s", "e2e_tps")}
            complete = [p["requested_length"] for p in raw["points"] if p["status"] == "complete"]
            result["largest_successful_prompt_tokens"] = max(complete) if complete else None
            metrics = next(iter(sorted((path / "long-context/points").glob("*/measured-01/metrics-before.raw"))), None)
            if metrics:
                config_lines = [line for line in metrics.read_text().splitlines() if line.startswith("vllm:cache_config_info{")]
                assert len(config_lines) == 1
                labels = dict(re.findall(r'(\w+)="([^"]*)"', config_lines[0]))
                result["cache_config"] = {key: labels[key] for key in (
                    "num_gpu_blocks", "block_size", "cache_dtype", "mamba_cache_mode",
                    "mamba_ssm_cache_dtype", "kv_cache_size_tokens", "kv_cache_max_concurrency")}
                # Pinned page size; grouping patch guards this geometry. These
                # are allocation calculations, not measured peak GPU memory.
                group_size = summary.get("cache_group_size") or 5
                page_bytes = 3407872
                pool_blocks = int(labels["num_gpu_blocks"])
                result["startup_memory"]["allocated_pool_gib"] = pool_blocks * group_size * page_bytes / 2**30
                batch = summary["max_num_batched_tokens"]
                recurrent_groups = math.ceil(48 / group_size)
                full_groups = math.ceil(16 / group_size)
                sliding_groups = math.ceil(5 / group_size)
                sliding_blocks = math.ceil((2048 - 1 + 2 * batch) / 832) + 1
                required_blocks = recurrent_groups * 8 + full_groups * math.ceil(summary["context"] / 1664) + sliding_groups * sliding_blocks
                result["allocation_reconstruction"] = {
                    "required_blocks_for_configured_limit": required_blocks,
                    "required_gib_for_configured_limit": required_blocks * group_size * page_bytes / 2**30,
                    "reported_capacity_reproduces": int(pool_blocks / required_blocks * summary["context"]) == int(labels["kv_cache_size_tokens"]),
                }
        cells[name] = result
    comparisons = {}
    baseline = cells.get("auto-8192-64k", {}).get("cold", {}).get("points", [])
    for name, result in cells.items():
        if name == "auto-8192-64k" or result["context"] != 65536:
            continue
        paired = []
        for candidate in result["cold"].get("points", []):
            base = next((p for p in baseline if p["requested_length"] == candidate["requested_length"]), None)
            if base and base["status"] == candidate["status"] == "complete":
                row = {"prompt_tokens": candidate["requested_length"]}
                for metric in ("decode_tps_post_first", "ttft_s", "e2e_tps"):
                    row[metric + "_median_delta_percent"] = 100 * (candidate["statistics"]["median"][metric] / base["statistics"]["median"][metric] - 1)
                paired.append(row)
        comparisons[name] = paired
    print(json.dumps({"tier": "development", "method": "Sequential cells, one warmup and six measured trials per supported point. Medians, inclusive IQR and sample CV; no significance or quality-equivalence claim.",
                      "cells": cells, "matched_64k_comparisons_vs_baseline": comparisons}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
