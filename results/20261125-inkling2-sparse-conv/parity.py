"""Exercise Inkling's completion API across chunk and prefix-cache boundaries."""

import argparse
import concurrent.futures
import json
import random
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

URL = "http://127.0.0.1:18080"
MODEL = "inkling-2-mini"
METRICS = (
    "num_preemptions_total",
    "spec_decode_num_drafts_total",
    "spec_decode_num_draft_tokens_total",
    "spec_decode_num_accepted_tokens_total",
    "num_requests_running",
)


def snapshot():
    out = dict.fromkeys(METRICS, 0.0)
    with urlopen(f"{URL}/metrics", timeout=3) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("vllm:"):
                metric, value = line.rsplit(maxsplit=1)
                key = metric.split("{", 1)[0][5:]
                if key in out:
                    out[key] += float(value)
    return out


def complete(prompt, output=48, probs=3, seed=42):
    opts = dict(model=MODEL, prompt=prompt, max_tokens=output, temperature=0,
                ignore_eos=True, return_token_ids=True, seed=seed)
    if probs:
        opts["logprobs"] = probs
    req = Request(f"{URL}/v1/completions", json.dumps(opts).encode(),
                  {"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urlopen(req, timeout=300) as r:
            data = json.load(r)
    except HTTPError as e:
        raise RuntimeError(e.read().decode()) from e
    usage = data["usage"]
    choice = data["choices"][0]
    assert usage["prompt_tokens"] == len(prompt), usage
    assert usage["completion_tokens"] == output, usage
    assert len(choice["token_ids"]) == output, choice
    details = usage.get("prompt_tokens_details") or {}
    return dict(choice=choice, usage=usage, cached=details.get("cached_tokens", 0),
                elapsed_s=time.monotonic() - started)


def streamed(prompt, output):
    opts = dict(model=MODEL, prompt=prompt, max_tokens=output, temperature=0,
                ignore_eos=True, return_token_ids=True, stream=True,
                stream_options={"include_usage": True})
    req = Request(f"{URL}/v1/completions", json.dumps(opts).encode(),
                  {"Content-Type": "application/json"})
    started = time.monotonic()
    first = usage = None
    with urlopen(req, timeout=300) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            data = json.loads(payload)
            if first is None and any(c.get("token_ids") for c in data.get("choices", [])):
                first = time.monotonic() - started
            if data.get("usage") is not None:
                usage = data["usage"]
    assert first is not None and usage is not None
    assert (usage["prompt_tokens"], usage["completion_tokens"]) == (len(prompt), output), usage
    return dict(ttft_s=first, elapsed_s=time.monotonic() - started, usage=usage)


def tokens(n, seed):
    rnd = random.Random(seed)
    return [rnd.randrange(1024, 30000) for _ in range(n)]


def suite(directory):
    before = snapshot()
    samples = {}
    for n in (513, 2047, 2048, 2049, 8321, 32771):
        prompt = tokens(n, n)
        for label, ids in (("cold", prompt), ("warm", prompt),
                           ("extend", prompt + tokens(177, n + 1))):
            name = f"{n}-{label}"
            row = complete(ids)
            samples[name] = row
            (directory / f"{name}.json").write_text(json.dumps(row) + "\n")
            print(name, "cached", row["cached"], "seconds", round(row["elapsed_s"], 3),
                  flush=True)
            if label == "cold":
                assert row["cached"] == 0, name
            elif n >= 8321:
                assert row["cached"] >= n - 512, (name, row["cached"])
    after = snapshot()
    delta = {m: after[m] - before[m] for m in METRICS}
    assert delta["num_preemptions_total"] == 0, delta
    assert 0 < delta["spec_decode_num_accepted_tokens_total"] < delta[
        "spec_decode_num_draft_tokens_total"], delta
    (directory / "metrics.json").write_text(json.dumps(delta, indent=2) + "\n")
    return samples


def compare(samples, reference, directory):
    report = {}
    for name, row in samples.items():
        ref = json.loads((reference / f"{name}.json").read_text())
        a, b = ref["choice"], row["choice"]
        ai, bi = a["token_ids"], b["token_ids"]
        assert len(ai) == len(bi), name
        la, lb = a["logprobs"]["token_logprobs"], b["logprobs"]["token_logprobs"]
        same_prefix = next((i for i, (x, y) in enumerate(zip(ai, bi)) if x != y), len(ai))
        lp = [abs(x-y) for x, y in zip(la[:same_prefix], lb[:same_prefix])]
        report[name] = dict(matching_prefix=same_prefix, tokens=len(ai),
                            max_logprob_delta=max(lp, default=None),
                            cached_tokens=[ref["cached"], row["cached"]])
    (directory / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, row in report.items():
        if not (row["matching_prefix"] == row["tokens"] and row["max_logprob_delta"] <= 0.25):
            raise RuntimeError(f"Parity failed: {name}: {row}")


def mixed(directory):
    before = snapshot()
    peak = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = {"long": pool.submit(streamed, tokens(245760, 100), 128)}
        deadline = time.monotonic() + 15
        while snapshot()["num_requests_running"] < 1 and not futures["long"].done():
            if time.monotonic() > deadline:
                raise TimeoutError("long request did not enter scheduler")
            time.sleep(0.1)
        for i in range(4):
            futures[f"short-{i}"] = pool.submit(streamed, tokens(8192, 200+i), 64)
        while not all(f.done() for f in futures.values()):
            peak = max(peak, snapshot()["num_requests_running"])
            time.sleep(0.25)
        rows = {name: f.result() for name, f in futures.items()}
    after = snapshot()
    summary = dict(peak_running=peak, preemptions=after["num_preemptions_total"] -
                   before["num_preemptions_total"], requests=rows)
    (directory / "mixed.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert summary["preemptions"] == 0, summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--mixed", action="store_true")
    args = p.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.mixed:
        mixed(args.directory)
    else:
        samples = suite(args.directory)
        if args.reference is not None:
            compare(samples, args.reference, args.directory)
