"""Run vLLM bench serve with concurrent scheduler metric sampling."""
import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

URL = "http://127.0.0.1:18080"
RUNNING = "num_requests_running"
PREEMPT = "num_preemptions_total"
DRAFTS = "spec_decode_num_drafts_total"
PROPOSED = "spec_decode_num_draft_tokens_total"
ACCEPTED = "spec_decode_num_accepted_tokens_total"
TRACK = {RUNNING, PREEMPT, DRAFTS, PROPOSED, ACCEPTED, "kv_cache_usage_perc"}


def snapshot():
    with urlopen(f"{URL}/metrics", timeout=5) as response:
        text = response.read().decode()
    metrics = {}
    for line in text.splitlines():
        if line.startswith("vllm:"):
            metric, value = line.rsplit(maxsplit=1)
            key = metric.split("{", 1)[0].removeprefix("vllm:")
            if key in TRACK:
                metrics[key] = metrics.get(key, 0.0) + float(value)
    if not {RUNNING, PREEMPT} <= metrics.keys():
        raise RuntimeError("required scheduler metrics absent")
    return metrics


def bench(art, model_dir, n, trial):
    name = f"c{n}-r{trial}"
    result = art / f"{name}.json"
    prompts = max(8, 4 * n)
    cmd = [sys.executable, "-B", "-m", "vllm.entrypoints.cli.main", "bench", "serve",
           "--backend", "vllm", "--base-url", URL, "--endpoint", "/v1/completions",
           "--model", "inkling-2-mini", "--tokenizer", model_dir,
           "--dataset-name", "random", "--random-input-len", "512",
           "--random-output-len", "128", "--random-range-ratio", "0",
           "--random-prefix-len", "0", "--skip-chat-template", "--temperature", "0",
           "--ignore-eos", "--num-warmups", str(n if trial == 1 else 1),
           "--num-prompts", str(prompts), "--max-concurrency", str(n),
           "--request-rate", "inf", "--seed", str(100 * n + trial),
           "--percentile-metrics", "ttft,tpot,e2el", "--metric-percentiles", "50,95,99",
           "--save-result", "--save-detailed", "--result-dir", str(art),
           "--result-filename", result.name]
    (art / f"{name}.command.json").write_text(json.dumps(cmd) + "\n")
    before = snapshot()
    peak = 0
    with (
        (art / f"{name}.log").open("w") as log,
        subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT) as child,
    ):
        try:
            deadline = time.monotonic() + 100
            while child.poll() is None:
                peak = max(peak, snapshot()[RUNNING])
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{name}: benchmark timed out")
                time.sleep(0.08)
        finally:
            if child.poll() is None:
                child.kill()
        if child.returncode != 0:
            raise RuntimeError(f"{name} failed, see {name}.log")
    after = snapshot()
    data = json.loads(result.read_text())
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in (PREEMPT, DRAFTS, PROPOSED, ACCEPTED)}
    checks = {"completed": data["completed"], "expected": prompts,
              "total_input_tokens": data["total_input_tokens"],
              "total_output_tokens": data["total_output_tokens"], "peak_running": peak,
              "preemptions_including_warmup": delta[PREEMPT]}
    passed = (checks["completed"] == prompts and checks["total_output_tokens"] == prompts * 128
              and peak == n and delta[PREEMPT] == 0)
    row = {"n": n, "trial": trial, "passed": passed, **checks,
           **{key: data[key] for key in ("output_throughput", "request_throughput",
                 "total_token_throughput", "median_ttft_ms", "p95_ttft_ms", "median_tpot_ms")},
           "drafts_including_warmup": delta[DRAFTS],
           "proposed_including_warmup": delta[PROPOSED],
           "accepted_including_warmup": delta[ACCEPTED],
           "acceptance_length_including_warmup": 1 + delta[ACCEPTED] / delta[DRAFTS] if delta[DRAFTS] else None}
    (art / f"{name}.summary.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)
    return row


def near_limit(art):
    rng = random.Random(327424)
    payload = json.dumps({"model": "inkling-2-mini",
                          "prompt": [rng.randrange(1000, 16000) for _ in range(327424)],
                          "temperature": 0, "max_tokens": 128, "ignore_eos": True}).encode()
    before = snapshot()
    start = time.monotonic()
    with urlopen(Request(f"{URL}/v1/completions", data=payload,
                         headers={"Content-Type": "application/json"}), timeout=300) as resp:
        usage = json.load(resp)["usage"]
    result = {"usage": usage, "elapsed_s": time.monotonic() - start,
              "preemptions": snapshot()[PREEMPT] - before[PREEMPT]}
    (art / "near-limit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if (usage["prompt_tokens"], usage["completion_tokens"], result["preemptions"]) != (327424, 128, 0):
        raise RuntimeError("near-limit request did not pass")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("art", type=Path)
    parser.add_argument("model_dir")
    parser.add_argument("ceiling", type=int)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--resume", action="store_true", help="reuse completed trials from the same server configuration")
    args = parser.parse_args()
    args.art.mkdir(exist_ok=True)
    if args.verify:
        if not bench(args.art, args.model_dir, args.ceiling, 4)["passed"]:
            raise SystemExit("exact-cap verification failed")
        near_limit(args.art)
        return
    summary = []
    for n in (1, 2, 4, 8, 16, 32, 64):
        if n > args.ceiling:
            break
        try:
            rows = []
            for trial in (1, 2, 3):
                saved = args.art / f"c{n}-r{trial}.summary.json"
                if args.resume and saved.is_file():
                    row = json.loads(saved.read_text())
                    if row["n"] != n or row["trial"] != trial:
                        raise RuntimeError(f"unexpected trial data in {saved}")
                    rows.append(row)
                else:
                    rows.append(bench(args.art, args.model_dir, n, trial))
        except (KeyError, OSError, RuntimeError, TimeoutError) as exc:
            summary.append({"n": n, "passed": False, "error": str(exc)})
            break
        entry = {"n": n, "passed": all(row["passed"] for row in rows),
                 **{key: statistics.median(row[key] for row in rows) for key in (
                    "output_throughput", "request_throughput", "total_token_throughput",
                    "median_ttft_ms", "p95_ttft_ms", "median_tpot_ms")}}
        summary.append(entry)
        if not entry["passed"]:
            break
    candidates = [row for row in summary if row["passed"] and row["n"] > 1]
    winner = max(candidates, key=lambda row: (row["output_throughput"], -row["n"])) if candidates else None
    result = {"server_ceiling": args.ceiling, "results": summary, "winner": winner}
    (args.art / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if winner is None:
        raise SystemExit("no verified concurrent configuration")


if __name__ == "__main__":
    main()
