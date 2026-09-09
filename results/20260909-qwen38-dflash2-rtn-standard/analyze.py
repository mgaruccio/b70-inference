#!/usr/bin/env python3
"""Reduce this campaign's raw files: python3 analyze.py > analysis.json.

Standard library only. No raw data is modified. Missing cells remain pending;
failed cells and unsupported context probes remain in the output. This does
not assign publication compliance or statistical significance.
"""
import json
from pathlib import Path
import re
import statistics as stats

ROOT = Path(__file__).resolve().parent
CLIENTS = ("bf16-clients", "int4-clients", "mtp4-clients")
COLD = ("int4-long", "bf16-long", "int4-long-32000", "bf16-long-32000", "mtp4-long")


def load(relative):
    return json.loads((ROOT / relative).read_text())


def distribution(values):
    if not values:
        return {"n": 0}
    quartiles = stats.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [values[0]] * 3
    mean = stats.mean(values)
    return {"n": len(values), "median": stats.median(values),
            "iqr": quartiles[2] - quartiles[0],
            "cv_percent": 100 * stats.stdev(values) / mean if len(values) > 1 and mean else None}


def counters(text):
    result = {}
    for line in text.splitlines():
        match = re.match(r'(vllm:spec_decode_\w+_total)(\{[^}]*\})?\s+(\S+)', line)
        if match:
            result[match[1] + (match[2] or "")] = float(match[3])
    return result


def spec_delta(before, after):
    left, right = counters(before), counters(after)
    result = {key: right[key] - left.get(key, 0) for key in right}
    if any(value < 0 for value in result.values()):
        raise ValueError("speculative counter reset; cannot aggregate this interval")
    return result


def spec_summary(values):
    def total(metric):
        return sum(v for k, v in values.items() if k.startswith("vllm:spec_decode_" + metric + "_total{"))
    steps = total("num_drafts")
    proposed = total("num_draft_tokens")
    accepted = total("num_accepted_tokens")
    positions = {}
    for key, value in values.items():
        if "num_accepted_tokens_per_pos_total" in key:
            position = re.search(r'position="(\d+)"', key)
            if position:
                positions[position[1]] = positions.get(position[1], 0) + value
    return {"draft_steps": steps, "proposed": proposed, "accepted": accepted,
            "acceptance_percent": 100 * accepted / proposed if proposed else None,
            "mean_acceptance_length_including_bonus": 1 + accepted / steps if steps else None,
            "accepted_by_zero_based_position": positions}


def clients(cell):
    source = ROOT / cell / "betterbench/results.json"
    if not source.exists():
        return {"status": "pending_or_failed_before_results"}
    data = load(source)
    categories = {}
    for name, rows in data["single_stream"].items():
        categories[name] = {"measured": len(rows), "failed": sum(not row["ok"] for row in rows),
                            "decode_tps": distribution([row["decode_tps"] for row in rows if row["ok"]])}
    weighted = sum(data["config"]["weights"].get(name, 0) * row["decode_tps"].get("median", 0)
                   for name, row in categories.items())
    prefill = [{"nominal_depth": row["target_depth"], "skipped": row.get("skipped", False), "reason": row.get("reason"),
                "prompt_tokens": distribution(row.get("prompt_tokens", [])),
                "ttft_ms": distribution(row.get("ttft_ms", [])),
                "prefill_proxy_tps": distribution(row.get("pp_tps", []))} for row in data["prefill"]]
    concurrency = [{"level": row["level"], "requests": row["requests"], "ok": row["ok"],
                    "aggregate_tps": row["aggregate_tps"], "ttft_ms": distribution(row["ttft_ms"])}
                   for row in data["concurrency"]]
    serving = {}
    for c in (1, 2, 4, 8):
        path = ROOT / cell / f"vllm-bench/c{c}.json"
        if not path.exists():
            serving[str(c)] = {"status": "pending_or_failed"}
            continue
        raw = load(path)
        serving[str(c)] = {key: value for key, value in raw.items()
                           if key.startswith(("spec_decode_", "mean_", "median_", "p25_", "p75_"))
                           or key in ("completed", "failed", "duration", "request_throughput",
                                      "output_throughput", "total_token_throughput")}
        for key in ("input_lens", "output_lens"):
            serving[str(c)][key] = distribution(raw[key])
    phase = ROOT / cell / "spec-metrics"
    speculation = spec_summary(spec_delta((phase / "betterbench-before.prom").read_text(),
                                           (phase / "betterbench-after.prom").read_text()))
    return {"source": str(source.relative_to(ROOT)), "status": load(ROOT / cell / "summary.json")["status"],
            "weighted_category_median_tps": weighted, "categories": categories, "prefill": prefill,
            "concurrency": concurrency, "vllm_serving": serving,
            "betterbench_speculation_including_warmups_all_phases": speculation}


def cold(cell):
    path = ROOT / cell / "long-context/summary.json"
    if not path.exists():
        return {"status": "pending_or_failed_before_results"}
    raw = load(path)
    points = []
    for p in raw["points"]:
        point = {key: p[key] for key in ("requested_length", "status", "classification")}
        point["statistics"] = p.get("summary")
        if p.get("failure"):
            point["failure"] = p["failure"]
        if p.get("unsupported_probe"):
            point["probe_error"] = p["unsupported_probe"].get("error")
        sums = {}
        for row in p["measurements"]:
            if not row.get("valid"):
                continue
            texts = ["\n".join(row[f"metrics_{phase}"]["speculative_position_counter_lines"])
                     for phase in ("before", "after")]
            for key, value in spec_delta(*texts).items():
                sums[key] = sums.get(key, 0) + value
        point["measured_speculation_excluding_warmup"] = spec_summary(sums)
        points.append(point)
    return {"source": str(path.relative_to(ROOT)), "status": raw["status"],
            "errors": raw["errors"], "points": points}


def output_variation():
    a, b = [load(ROOT / name / "betterbench/results.json") for name in CLIENTS[:2]]
    lengths = {}
    for name in a["single_stream"]:
        left, right = a["single_stream"][name], b["single_stream"][name]
        assert len(left) == len(right) == 20
        assert all((x["prompt_id"], x["prompt_tokens"]) == (y["prompt_id"], y["prompt_tokens"])
                   for x, y in zip(left, right))
        lengths[name] = sum(x["completion_tokens"] != y["completion_tokens"] for x, y in zip(left, right))
    cross, within = {}, {}
    for c in (1, 2, 4, 8):
        x, y = [load(ROOT / name / f"vllm-bench/c{c}.json")["generated_texts"] for name in CLIENTS[:2]]
        assert len(x) == len(y) == 48
        cross[str(c)] = sum(a == b for a, b in zip(x, y))
    for name in CLIENTS[:2]:
        base = load(ROOT / name / "vllm-bench/c1.json")["generated_texts"]
        within[name] = {str(c): sum(a == b for a, b in zip(base, load(ROOT / name / f"vllm-bench/c{c}.json")["generated_texts"]))
                        for c in (2, 4, 8)}
    return {"betterbench_output_length_differences_per_20": lengths,
            "vllm_exact_text_matches_across_arms_per_48": cross,
            "vllm_exact_text_matches_vs_same_arm_c1_per_48": within,
            "quality_suite_run": False, "cause_of_variation_established": False}


if __name__ == "__main__":
    print(json.dumps({"method": "Sequential runs; medians and inclusive IQR; sample CV. No significance/quality claim.",
                      "clients": {name: clients(name) for name in CLIENTS},
                      "cold": {name: cold(name) for name in COLD},
                      "output_variation": output_variation()}, indent=2, allow_nan=False))
