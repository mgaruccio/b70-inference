#!/usr/bin/env python3
"""Reproduce bounded acceptance diagnostics; not a standard performance benchmark."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
CELLS = ("target-only", "current-dspark", "native-markov-greedy", "native-probabilistic")


def load(path):
    return json.loads(path.read_text())


def summarize():
    result = {"tier": "development", "cells": {}, "greedy_parity": {}, "repeatability": {}}
    for cell in CELLS:
        directory = ROOT / cell
        summary = load(directory / "summary.json")
        assert summary["status"] == "passed" and summary["host_unchanged"]
        rows = [load(p) for p in sorted(directory.glob("experiment-*-result.json"))]
        assert len(rows) == 36
        assert all(r["transport_pass"] and r["usage"]["completion_tokens"] == 512 for r in rows)
        groups = []
        for thinking in (False, True):
            for temperature in (0, 1):
                group = [r for r in rows if r["sampling"]["thinking"] == thinking and r["sampling"]["temperature"] == temperature]
                steps = sum(r["draftsteps_delta"] for r in group)
                proposed = sum(r["proposals_delta"] for r in group)
                accepted = sum(r["accepts_delta"] for r in group)
                positions = [sum(r["per_position_accepted_counters"].get(str(i), 0) for r in group) for i in range(7)]
                groups.append({"thinking": thinking, "temperature": temperature, "requests": len(group),
                               "median_decode_tps": statistics.median(r["decode_tps"] for r in group),
                               "draft_steps": steps, "proposed": proposed, "accepted": accepted,
                               "emitted_per_step": 1 + accepted / steps if steps else None,
                               "first_position_acceptance": positions[0] / steps if steps else None,
                               "accepted_per_position": positions})
        result["cells"][cell] = groups
        if cell != "target-only":
            comparisons = []
            for row in rows:
                label = row["label"]
                baseline = load(ROOT / "target-only" / f"{label}-result.json")
                assert load(directory / f"{label}-request.json") == load(ROOT / "target-only" / f"{label}-request.json")
                assert row["prompt_token_ids"] == baseline["prompt_token_ids"]
                if row["sampling"]["temperature"] != 0:
                    continue
                first = next((i for i, (a, b) in enumerate(zip(baseline["token_ids"], row["token_ids"])) if a != b), None)
                comparisons.append({"label": label, "exact": first is None, "first_divergence": first})
            result["greedy_parity"][cell] = {"matched": sum(r["exact"] for r in comparisons), "total": len(comparisons), "rows": comparisons}
    for cell in ("target-repeatability", "dspark-repeatability"):
        result["repeatability"][cell] = load(ROOT / cell / "repeatability.json")
    result["limitations"] = ["Ordered small diagnostic matrix; no interleaved confidence interval or standard benchmark claim.",
                              "Baseline itself is not bitwise stable on repeated greedy prose; no full-output identity or distribution-fidelity claim.",
                              "Weight-quantization causality and full reference draft numerical parity remain unestablished."]
    return result


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
