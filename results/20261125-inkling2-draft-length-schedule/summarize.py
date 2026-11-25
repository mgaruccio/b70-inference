"""Tabulate a hillclimb.sh run directory: one row per arm, medians of the paired trials.

Usage: python3 -I summarize.py RUN_DIR [--json OUT]
"""
import json
import statistics
import sys
from pathlib import Path


def load(path):
    return json.loads(path.read_text()) if path.is_file() else None


def arm_row(d):
    row = {"arm": d.name}
    chat = load(d / "chat-0.json")
    if chat:
        s = chat["summary"]
        row["chat_c1"] = s["median_decode_tok_s"]
        row["chat_c1_al"] = s["mean_acceptance_length"]
    deep = load(d / "chat-30000.json")
    if deep:
        row["chat_c1_30k"] = deep["summary"]["median_decode_tok_s"]
    for c in (8, 32):
        cc = load(d / f"chatc-{c}.json")
        if cc:
            row[f"chat_c{c}"] = cc["aggregate_output_tok_s"]
            row[f"chat_c{c}_al"] = cc["mean_acceptance_length"]
    for n in (1, 4, 8, 16, 32):
        rows = [load(d / f"c{n}-r{t}.summary.json") for t in (1, 2, 3)]
        rows = [r for r in rows if r]
        if rows:
            row[f"rand_c{n}"] = statistics.median(r["output_throughput"] for r in rows)
            row[f"rand_c{n}_ok"] = all(r["passed"] for r in rows)
    par = load(d / "parity.json")
    if par and par.get("comparison"):
        row["parity_exact"] = sum(v["matching_prefix"] == v["tokens"] for v in par["comparison"].values())
        row["parity_n"] = len(par["comparison"])
    for t in (200000, 400000):
        nd = load(d / f"needle-{t}.json")
        if nd:
            row[f"needle_{t // 1000}k"] = f"{'PASS' if nd['passed'] else 'FAIL'}@{nd['prompt_tokens']}"
    st = d / "startup.txt"
    if st.is_file():
        row["startup_s"] = int(st.read_text().split("=")[1])
    return row


def main():
    run = Path(sys.argv[1])
    rows = [arm_row(d) for d in sorted(run.iterdir()) if d.is_dir() and d.name[:2].isdigit()]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    fmt = lambda v: f"{v:.1f}" if isinstance(v, float) else str(v)
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for r in rows:
        print("| " + " | ".join(fmt(r.get(k, "")) for k in keys) + " |")
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
