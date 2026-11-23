#!/usr/bin/env python3
"""Aggregate aligned candidate XPU and CPU reference artifacts."""
import json
import math
import pathlib
import statistics
import sys

CAND = pathlib.Path(sys.argv[1])
REF = pathlib.Path(sys.argv[2])
ROOT = CAND
CASES = ("raw-short-00", "chat-short-00", "raw-short-01", "chat-short-01", "raw-long-00", "chat-long-00")


def pct(values, fraction):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * fraction
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] * (hi - pos) + values[hi] * (pos - lo)


def stats(values):
    return {"n": len(values), "median": statistics.median(values) if values else None, "p95": pct(values, .95), "max": max(values) if values else None}


def main():
    summary = {"description": "Corrected CPU reference compares four short prompts in full and the first 2048 tokens of two long prompts (raw-long-00 and chat-long-00).", "per_case": {}, "overall": {}}
    all_xpu, all_bf16 = [], []
    counts = {"xpu_top1_match": 0, "bf16_top1_match": 0, "positions": 0, "outside_bf16_noise": 0}
    examples = []
    for case in CASES:
        cp = json.loads((CAND / f"{case}.json").read_text())
        fp = json.loads((REF / "cpu-fp32" / f"{case}.json").read_text())
        bf = json.loads((REF / "cpu-bf16" / f"{case}.json").read_text())
        token_ids = cp["input_ids"]
        xpu_lp = cp["response"]["choices"][0]["prompt_logprobs"]
        cpu_fp_lp, cpu_fp_top = fp["reference_token_logprobs"], fp["top1_ids"]
        cpu_bf_lp, cpu_bf_top = bf["reference_token_logprobs"], bf["top1_ids"]
        n = min(len(fp["reference_token_ids"]), len(bf["reference_token_ids"]), len(token_ids)-1)
        xpu_delta, bf_delta = [], []
        xpu_agree = bf_agree = outside = 0
        outside_positions = []
        for j in range(n):
            pos = j + 1
            entry = xpu_lp[pos]
            token = token_ids[pos]
            xpu_token_lp = entry[str(token)]["logprob"]
            xpu_top_id = min(((int(k), v["rank"]) for k, v in entry.items()), key=lambda pair: pair[1])[0]
            dx = abs(xpu_token_lp - cpu_fp_lp[j])
            db = abs(cpu_bf_lp[j] - cpu_fp_lp[j])
            xpu_delta.append(dx)
            bf_delta.append(db)
            xpu_agree += xpu_top_id == cpu_fp_top[j]
            bf_agree += cpu_bf_top[j] == cpu_fp_top[j]
            if dx > db:
                outside += 1
                outside_positions.append({"position": pos, "xpu_delta": dx, "bf16_delta": db, "excess": dx-db})
        # Long prompt cases 00/01 share identical first-256-token prefixes; count that prefix once overall.
        if case not in ("raw-long-01", "chat-long-01"):
            all_xpu.extend(xpu_delta)
            all_bf16.extend(bf_delta)
            counts["xpu_top1_match"] += xpu_agree
            counts["bf16_top1_match"] += bf_agree
            counts["positions"] += n
            counts["outside_bf16_noise"] += outside
            examples.extend({"case": case, **x} for x in sorted(outside_positions, key=lambda x: x["excess"], reverse=True)[:5])
        summary["per_case"][case] = {
            "positions": n,
            "xpu_vs_fp32_top1_agreement": xpu_agree / n,
            "bf16_vs_fp32_top1_agreement": bf_agree / n,
            "xpu_vs_fp32_abs_logprob_delta": stats(xpu_delta),
            "bf16_vs_fp32_abs_logprob_delta": stats(bf_delta),
            "positions_xpu_delta_exceeds_same_position_bf16_noise": outside,
            "largest_noise_floor_excess_positions": sorted(outside_positions, key=lambda x: x["excess"], reverse=True)[:5],
        }
    summary["overall"] = {
        "positions": counts["positions"],
        "xpu_vs_fp32_top1_agreement": counts["xpu_top1_match"] / counts["positions"],
        "bf16_vs_fp32_top1_agreement": counts["bf16_top1_match"] / counts["positions"],
        "xpu_vs_fp32_abs_logprob_delta": stats(all_xpu),
        "bf16_vs_fp32_abs_logprob_delta": stats(all_bf16),
        "positions_xpu_delta_exceeds_same_position_bf16_noise": counts["outside_bf16_noise"],
        "noise_floor_excess_examples_top5_per_prompt": examples,
    }
    decodes = {}
    for case in ("chat-short-00", "raw-short-01"):
        p = REF / "cpu-fp32" / f"decode-{case}.json"
        if not p.exists():
            p = REF / "cpu-fp32" / f"decode-first-{case}.json"
        if p.exists():
            decodes[case] = json.loads(p.read_text())
    summary["greedy_decode"] = decodes
    out = ROOT / "comparison.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    md = ["# Independent full-model numerical comparison", "", summary["description"], "", "| Prompt | Compared positions | XPU top-1 agreement | XPU median / p95 / max Δlogprob | CPU bf16 median / p95 / max Δlogprob | Outside noise-floor positions |", "|---|---:|---:|---:|---:|---:|"]
    for case, row in summary["per_case"].items():
        x, b = row["xpu_vs_fp32_abs_logprob_delta"], row["bf16_vs_fp32_abs_logprob_delta"]
        md.append(f"| {case} | {row['positions']} | {row['xpu_vs_fp32_top1_agreement']:.4f} | {x['median']:.6g} / {x['p95']:.6g} / {x['max']:.6g} | {b['median']:.6g} / {b['p95']:.6g} / {b['max']:.6g} | {row['positions_xpu_delta_exceeds_same_position_bf16_noise']} |")
    o = summary["overall"]
    x, b = o["xpu_vs_fp32_abs_logprob_delta"], o["bf16_vs_fp32_abs_logprob_delta"]
    md += ["", f"Overall ({o['positions']} scored positions): XPU top-1 agreement {o['xpu_vs_fp32_top1_agreement']:.4f}; CPU bf16 top-1 agreement {o['bf16_vs_fp32_top1_agreement']:.4f}.",
           f"XPU-vs-fp32 |Δlogprob| median/p95/max: {x['median']:.6g} / {x['p95']:.6g} / {x['max']:.6g}.",
           f"CPU-bf16-vs-fp32 noise floor median/p95/max: {b['median']:.6g} / {b['p95']:.6g} / {b['max']:.6g}.",
           f"XPU delta exceeded same-position bf16 noise at {o['positions_xpu_delta_exceeds_same_position_bf16_noise']} positions.", "", "## Greedy decode first divergences", "", "| Prompt | Ref tokens | Candidate tokens | First divergence index |", "|---|---:|---:|---:|"]
    for case, row in decodes.items():
        md.append(f"| {case} | {row.get('reference_tokens', row.get('reference_steps_exercised'))} | {row['candidate_tokens']} | {row['first_divergence']} |")
    (ROOT / "comparison.md").write_text("\n".join(md)+"\n")
    print(json.dumps({"summary_json": str(out), "summary_markdown": str(ROOT / "comparison.md"), "overall": summary["overall"], "decode_first_divergence": {k:v.get("first_divergence") for k,v in decodes.items()}}, indent=2))


if __name__ == "__main__":
    main()
