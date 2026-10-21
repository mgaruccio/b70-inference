#!/usr/bin/env python3
"""Exercise the real B70 HTTP endpoint; development checks, not a benchmark suite."""
import argparse
import ast
import json
import re
import time
import urllib.request
from pathlib import Path


def request(url, path, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url + path, data, {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def metrics(url):
    with request(url, "/metrics") as response:
        text = response.read().decode()
    counters = {}
    for line in text.splitlines():
        match = re.match(r"(vllm:spec_decode_[^ {]+)(?:\{[^}]*\})? ([^ ]+)$", line)
        if match:
            counters[match[1]] = counters.get(match[1], 0.0) + float(match[2])
    return text, counters


def chat(url, model, out, label, prompt, count, forced=False):
    payload = {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "seed": 42, "max_tokens": count,
        "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False}, "return_token_ids": True,
        "ignore_eos": forced,
    }
    (out / f"{label}-request.json").write_text(json.dumps(payload, indent=2))
    raw_metrics, before = metrics(url)
    (out / f"{label}-metrics-before.txt").write_text(raw_metrics)
    start = time.monotonic()
    text, token_ids, usage, finish, first, done = "", [], {}, None, None, False
    with request(url, "/v1/chat/completions", payload) as response, (out / f"{label}-sse.jsonl").open("w") as log:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            now = time.monotonic()
            log.write(json.dumps({"elapsed_s": now - start, "data": data}) + "\n")
            if data == "[DONE]":
                done = True
                break
            obj = json.loads(data)
            if obj.get("error"):
                raise RuntimeError(obj["error"])
            usage = obj.get("usage") or usage
            for choice in obj.get("choices", []):
                piece = choice.get("delta", {}).get("content") or ""
                if piece and first is None:
                    first = now
                text += piece
                token_ids.extend(choice.get("token_ids") or [])
                finish = choice.get("finish_reason") or finish
    elapsed = time.monotonic() - start
    raw_metrics, after = metrics(url)
    (out / f"{label}-metrics-after.txt").write_text(raw_metrics)
    n = usage.get("completion_tokens", 0)
    result = {"label": label, "content": text, "token_ids": token_ids, "usage": usage,
              "finish_reason": finish, "elapsed_s": elapsed,
              "ttft_s": None if first is None else first - start,
              "decode_tps": None if first is None or n < 2 else (n - 1) / (start + elapsed - first),
              "metric_deltas": {k: after[k] - before.get(k, 0.0) for k in after}}
    (out / f"{label}-result.json").write_text(json.dumps(result, indent=2))
    if not (done and text.strip() and finish and n > 0 and len(token_ids) == n):
        raise RuntimeError(f"{label}: incomplete stream, empty output or token accounting mismatch")
    if forced and (n != count or finish != "length"):
        raise RuntimeError(f"{label}: requested generation length was not honored")
    print(json.dumps(result), flush=True)
    return result


def wait_for_models(url, timeout):
    """Public-boundary readiness precondition for the end-to-end run."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            with request(url, "/v1/models", timeout=5) as response:
                return json.load(response)
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18099")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--require-draft", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    models = wait_for_models(args.url, 600)
    (args.out / "models.json").write_text(json.dumps(models, indent=2))
    model = models["data"][0]["id"]
    with request(args.url, "/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": "What is 19 + 23? Reply with only the integer."}],
        "temperature": 0, "seed": 42, "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False}, "stream": False,
    }) as response:
        nonstream = json.load(response)
    (args.out / "nonstream.json").write_text(json.dumps(nonstream, indent=2))
    if nonstream["choices"][0]["message"]["content"].strip() != "42":
        raise RuntimeError("non-streaming arithmetic request failed")
    chat(args.url, model, args.out, "warmup", "Count from one to ten.", 32)
    cases = [
        ("math", "What is 19 + 23? Reply with only the integer.", 32, False),
        ("json", 'Reply with only this exact JSON object and no markdown: {"answer":42}', 64, False),
        ("code", "Output only Python source without markdown: define a function add(a, b) that returns a + b.", 96, False),
        ("decode", "Explain how a relational database executes a SQL query, from parsing and planning through indexing, joins, transactions, and returning rows. Write a detailed technical tutorial with concrete examples and tradeoffs.", 128, True),
        ("long", "Read the archive and return only the vault code.\n" + "The archive contains ordinary accounting notes and no other secret codes.\n" * 300 + "The vault code is 91741.\n" + "The archive contains ordinary accounting notes and no other secret codes.\n" * 300 + "Return only the vault code.", 32, False),
    ]
    rows = {label: chat(args.url, model, args.out, label, prompt, n, forced) for label, prompt, n, forced in cases}
    quality = {
        "math": rows["math"]["content"].strip() == "42",
        "json": json.loads(rows["json"]["content"]) == {"answer": 42},
        "code": ast.dump(ast.parse(rows["code"]["content"])) == ast.dump(ast.parse("def add(a,b): return a+b")),
        "long": rows["long"]["content"].strip() == "91741" and rows["long"]["usage"]["prompt_tokens"] >= 7500,
    }
    deltas = {}
    for row in rows.values():
        for name, value in row["metric_deltas"].items():
            deltas[name] = deltas.get(name, 0.0) + value
    summary = {"tier": "development", "quality": quality, "speculative_metrics": deltas}
    if args.compare:
        summary["baseline_token_equality"] = {
            label: row["token_ids"] == json.loads((args.compare / f"{label}-result.json").read_text())["token_ids"]
            for label, row in rows.items()
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if not all(quality.values()):
        raise RuntimeError("functional check failed")
    if args.require_draft:
        for name in ("vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total"):
            if deltas.get(name, 0) <= 0:
                raise RuntimeError(f"no real speculation observed: {name}")
    if args.compare and not all(summary["baseline_token_equality"].values()):
        raise RuntimeError("greedy candidate output differs from baseline; inspect before promotion")


if __name__ == "__main__":
    main()
