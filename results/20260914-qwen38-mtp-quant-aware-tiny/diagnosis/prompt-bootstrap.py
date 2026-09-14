"""Exploratory paired prompt-cluster bootstrap of the saved ABBA counters."""
import json
from pathlib import Path
import random

root = Path(__file__).resolve().parent.parent
records = {}
for arm in ("stock", "tuned"):
    for run in (1, 2):
        for line in (root / f"abba-{arm}-{run}" / "raw-results.jsonl").read_text().splitlines():
            row = json.loads(line)
            totals = records.setdefault(row["label"], {"stock": [0, 0], "tuned": [0, 0]})[arm]
            for i, suffix in enumerate(("accepted_tokens", "drafts")):
                totals[i] += sum(v for k, v in row["metric_deltas"].items()
                                 if k.split("{")[0] == f"vllm:spec_decode_num_{suffix}_total")


def relative_delta(names):
    values = {arm: [sum(records[name][arm][i] for name in names) for i in (0, 1)]
              for arm in ("stock", "tuned")}
    return (values["tuned"][0] / values["tuned"][1]) / (values["stock"][0] / values["stock"][1]) - 1


names = sorted(records)
rng = random.Random(42)
samples = sorted(relative_delta(rng.choices(names, k=len(names))) for _ in range(10000))
print(json.dumps({"relative_change_percent": 100 * relative_delta(names),
                  "exploratory_percentile_interval": [100 * samples[250], 100 * samples[9749]],
                  "without_mbpp_515_percent": 100 * relative_delta([n for n in names if n != "mbpp-515"]),
                  "unit": "prompt cluster retaining both runs/head", "draws": 10000, "seed": 42}))
