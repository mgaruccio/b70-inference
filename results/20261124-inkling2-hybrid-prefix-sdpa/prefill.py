"""Cold C1 prefill and decode over OpenAI completions, with vLLM metric deltas."""
import argparse
import hashlib
import http.client
import json
import random
import statistics
import time
from pathlib import Path
from urllib.request import urlopen

METRICS = (
    "request_prefill_kv_computed_tokens_sum",
    "request_prefill_time_seconds_sum",
    "request_prefill_time_seconds_count",
    "request_decode_time_seconds_sum",
    "spec_decode_num_drafts_total",
    "spec_decode_num_draft_tokens_total",
    "spec_decode_num_accepted_tokens_total",
)


def snapshot(port, path):
    with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as response:
        text = response.read().decode()
    path.write_text(text)
    totals = dict.fromkeys(METRICS, 0.0)
    for line in text.splitlines():
        if line.startswith("vllm:"):
            metric, value = line.rsplit(maxsplit=1)
            key = metric.split("{", 1)[0].removeprefix("vllm:")
            if key in totals:
                totals[key] += float(value)
    return totals


def run(port, art, depth, trial, output_tokens):
    name = f"{depth}-{trial}"
    # Matched across flag settings, distinct from the first token across trials.
    seed = f"inkling-hybrid-prefix-v1:{name}"
    rng = random.Random(seed)
    prompt = [rng.randrange(1000, 16000) for _ in range(depth)]
    payload = json.dumps({
        "model": "inkling-2-mini", "prompt": prompt, "temperature": 0,
        "max_tokens": output_tokens, "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True},
    })
    before = snapshot(port, art / f"{name}-before.prom")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    done = False
    first = usage = None
    start = time.monotonic()
    try:
        conn.request("POST", "/v1/completions", payload, {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read()[:1000]!r}")
        with (art / f"{name}.sse").open("wb") as stream:
            for line in response:
                stream.write(line)
                if not line.startswith(b"data: "):
                    continue
                if line[6:].strip() == b"[DONE]":
                    done = True
                    break
                event = json.loads(line[6:])
                if event.get("usage"):
                    usage = event["usage"]
                if first is None and any(row.get("text") for row in event["choices"]):
                    first = time.monotonic() - start
    finally:
        conn.close()
    if not (done and first is not None and usage and usage["prompt_tokens"] == depth
            and usage["completion_tokens"] == output_tokens):
        raise RuntimeError(f"{name}: incomplete response or unexpected usage: {usage}")
    after = snapshot(port, art / f"{name}-after.prom")
    delta = {key: after[key] - before[key] for key in METRICS}
    computed, prefill_s, count, decode_s, drafts, proposed, accepted = delta.values()
    if computed != depth or prefill_s <= 0 or count != 1 or decode_s <= 0:
        raise RuntimeError(f"{name}: expected exactly one uncached request: {delta}")
    result = {
        "name": name, "seed": seed, "request_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "prompt_tokens": depth, "output_tokens": output_tokens, "ttft_s": first,
        "prefill_s": prefill_s, "decode_s": decode_s, "prefill_tok_s": depth / prefill_s,
        "decode_tok_s": (output_tokens - 1) / decode_s,
        "acceptance_length": 1 + accepted / drafts if drafts else None,
        "draft_acceptance": accepted / proposed if proposed else None, "metrics": delta,
    }
    (art / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("art", type=Path)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--depths", type=int, nargs="+", default=[65536, 131072, 245760])
    args = parser.parse_args()
    run(args.port, args.art, 16384, "warmup", args.output_tokens)
    summary = {}
    for depth in args.depths:
        rows = [run(args.port, args.art, depth, i, args.output_tokens)
                for i in range(1, args.trials + 1)]
        summary[depth] = {}
        for metric in ("prefill_tok_s", "decode_tok_s", "ttft_s", "acceptance_length"):
            vals = [row[metric] for row in rows if row[metric] is not None]
            if not vals:
                continue
            quartiles = statistics.quantiles(vals, method="inclusive") if len(vals) > 1 else None
            summary[depth][metric] = {
                "median": statistics.median(vals),
                "iqr": quartiles[2] - quartiles[0] if quartiles else None,
            }
    (args.art / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
