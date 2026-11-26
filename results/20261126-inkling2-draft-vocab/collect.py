"""Collect Inkling-generated token IDs for draft-vocabulary frequency counts.

Runs on b70-dev against the live authenticated server. Default sampling
(generation_config) with reasoning on, so counts match real chat traffic.

Usage: python3 -I collect.py PROMPTS_JSON OUT_JSONL [--concurrency 24] [--max-tokens 768]
"""
import argparse
import concurrent.futures
import json
import subprocess
import threading
import time
from urllib.request import Request, urlopen

URL = "http://127.0.0.1:8080/v1/chat/completions"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=768)
    a = ap.parse_args()
    key = subprocess.check_output(["sudo", "-n", "cat", "/etc/b70-flint/api-key"], text=True).strip()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    prompts = json.load(open(a.prompts))
    done = set()
    try:
        with open(a.out) as f:
            done = {json.loads(line)["i"] for line in f}
    except FileNotFoundError:
        pass
    lock, out = threading.Lock(), open(a.out, "a")
    start, total = time.monotonic(), [0]

    def run(i):
        p = prompts[i]
        body = dict(model="inkling-2-mini", messages=[{"role": "user", "content": p["prompt"]}],
                    max_tokens=a.max_tokens, return_token_ids=True)
        with urlopen(Request(URL, json.dumps(body).encode(), headers), timeout=900) as r:
            d = json.load(r)
        ids = d["choices"][0]["token_ids"]
        with lock:
            out.write(json.dumps(dict(i=i, source=p["source"], token_ids=ids)) + "\n")
            out.flush()
            total[0] += len(ids)
            if i % 100 == 0:
                print(f"{i} tokens={total[0]} tok/s={total[0] / (time.monotonic() - start):.0f}", flush=True)

    todo = [i for i in range(len(prompts)) if i not in done]
    with concurrent.futures.ThreadPoolExecutor(a.concurrency) as pool:
        for f in concurrent.futures.as_completed([pool.submit(run, i) for i in todo]):
            f.result()
    print(f"done: {len(todo)} requests, {total[0]} tokens, {time.monotonic() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
