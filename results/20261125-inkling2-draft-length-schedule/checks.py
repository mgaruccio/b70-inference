"""Adaptive-verification A/B checks against an isolated loopback engine.

Subcommands (all write JSON under ART):
  parity ART [--reference DIR]   greedy completion token IDs on fixed token fixtures
  chat ART [--depth N --cases K] C1 streaming chat decode on the live_bench prompt panel
  chatc ART --concurrency N      N concurrent chats (2N requests) on the same panel, aggregate tok/s
  needle ART --tokens N          3 exact-match codes at 15/50/85% depth of an ~N-token document

No request asks for logprobs: vLLM rejects output logprobs with adaptive verification.
Token counts come from server usage stats, never from streamed chunks.
"""
import argparse
import http.client
import json
import random
import re
import statistics
import string
import time
from pathlib import Path

HOST, PORT, MODEL = "127.0.0.1", 18080, "inkling-2-mini"
SPEC = ("num_drafts", "num_draft_tokens", "num_accepted_tokens")
PROMPTS = [  # identical to ~/cmp/live/live_bench.py
    'Python coding exercise: merge overlapping intervals. Implement merge(intervals) with a doc comment and three unit tests.',
    'Go coding exercise: implement a bounded worker pool that honors context cancellation. Implement Process(ctx, jobs, workers).',
    'Rust coding exercise: parse timestamps from rows, keep good rows and report invalid lines. Implement parse_rows(input).',
    'Explain the difference between a mutex and a semaphore, with a short example of each in C.',
    'A train leaves at 09:40 and arrives at 13:05. It stops 3 times for 7 minutes each. What is its average speed in km/h '
    'over 260 km of track, counting only time in motion?',
    'Write a SQL query that returns the top 3 customers by total order value per country, and explain it.',
]


def metrics():
    conn = http.client.HTTPConnection(HOST, PORT, timeout=15)
    try:
        conn.request("GET", "/metrics")
        text = conn.getresponse().read().decode()
    finally:
        conn.close()
    out = {}
    for name in SPEC + ("num_preemptions",):
        m = re.search(rf"^vllm:(?:spec_decode_)?{name}_total{{[^}}]*}} ([0-9.e+]+)$", text, re.M)
        out[name] = float(m.group(1)) if m else 0.0
    return out


def delta(before, after):
    return {k: after[k] - before[k] for k in before}


def post(path, body, *, stream=False, timeout=900):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    start = time.monotonic()
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read(4096)!r}")
    if not stream:
        data = json.load(resp)
        conn.close()
        return data, time.monotonic() - start
    return conn, resp, start


def fixture(n, seed):  # identical to sparse-conv parity.py tokens()
    rnd = random.Random(seed)
    return [rnd.randrange(1024, 30000) for _ in range(n)]


def parity(art, reference):
    before, rows = metrics(), {}
    for n in (513, 2049, 8321, 32771):
        prompt = fixture(n, n)
        for label in ("cold", "warm"):
            data, elapsed = post("/v1/completions", dict(
                model=MODEL, prompt=prompt, max_tokens=64, temperature=0, ignore_eos=True,
                return_token_ids=True, seed=42))
            ids = data["choices"][0]["token_ids"]
            assert data["usage"]["completion_tokens"] == 64 == len(ids), data["usage"]
            rows[f"{n}-{label}"] = dict(token_ids=ids, elapsed_s=elapsed, usage=data["usage"])
    spec = delta(before, metrics())
    out = dict(rows=rows, spec=spec)
    if reference:
        ref = json.loads((reference / "parity.json").read_text())["rows"]
        out["comparison"] = {}
        for name, row in rows.items():
            a, b = ref[name]["token_ids"], row["token_ids"]
            prefix = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
            out["comparison"][name] = dict(matching_prefix=prefix, tokens=len(a))
    (art / "parity.json").write_text(json.dumps(out) + "\n")
    print(json.dumps({"spec": spec, "comparison": out.get("comparison")}), flush=True)


def filler(depth):
    if depth <= 0:
        return ""
    lines = [f"Log {i:05d}: sensor array {i % 17} reported nominal voltage and the operator signed off."
             for i in range(depth // 20)]
    return "Background log (ignore unless asked):\n" + "\n".join(lines) + "\n\nTask: "


def stream_chat(content, max_tokens, *, temperature=0, ignore_eos=True, effort=None):
    body = dict(model=MODEL, messages=[{"role": "user", "content": content}], max_tokens=max_tokens,
                temperature=temperature, ignore_eos=ignore_eos, stream=True,
                stream_options={"include_usage": True})
    if effort:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    before = metrics()
    conn, resp, start = post("/v1/chat/completions", body, stream=True)
    times, usage, text, reasoning = [], None, [], []
    try:
        while line := resp.readline():
            line = line.strip()
            if line == b"data: [DONE]":
                break
            if not line.startswith(b"data: "):
                continue
            frame = json.loads(line[6:])
            if frame.get("usage"):
                usage = frame["usage"]
            for ch in frame.get("choices", ()):
                d = ch.get("delta") or {}
                piece = d.get("content") or ""
                think = d.get("reasoning") or d.get("reasoning_content") or ""
                if piece or think:
                    times.append(time.monotonic() - start)
                    text.append(piece)
                    reasoning.append(think)
    finally:
        conn.close()
    n = usage["completion_tokens"]
    decode_s = times[-1] - times[0]
    return dict(prompt_tokens=usage["prompt_tokens"], completion_tokens=n, ttft_s=times[0],
                decode_s=decode_s, decode_tok_s=(n - 1) / decode_s if decode_s > 0 else None,
                spec=delta(before, metrics()), content="".join(text), reasoning="".join(reasoning))


def chat(art, depth, cases, tokens):
    stream_chat(filler(depth) + "Say hello in five words.", 32)  # warmup, not scored
    rows = []
    for i, prompt in enumerate(PROMPTS[:cases]):
        row = dict(case=i, depth=depth, **stream_chat(filler(depth) + prompt, tokens))
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("content", "reasoning")}), flush=True)
    spec = {k: sum(r["spec"][k] for r in rows) for k in SPEC}
    summary = dict(tier="development", depth=depth, tokens=tokens,
                   median_decode_tok_s=statistics.median(r["decode_tok_s"] for r in rows),
                   median_ttft_s=statistics.median(r["ttft_s"] for r in rows),
                   per_case=[round(r["decode_tok_s"], 2) for r in rows], spec=spec,
                   mean_acceptance_length=1 + spec["num_accepted_tokens"] / spec["num_drafts"]
                   if spec["num_drafts"] else None)
    (art / f"chat-{depth}.json").write_text(json.dumps(dict(summary=summary, rows=rows)) + "\n")
    print(json.dumps(summary), flush=True)


def chatc(art, n, tokens, rounds=2):
    """n concurrent streaming chats over the prompt panel; aggregate = tokens / wall time."""
    import concurrent.futures
    jobs = [f"Request {i}: " + PROMPTS[i % len(PROMPTS)] for i in range(n * rounds)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(lambda p: stream_chat(p, 16), jobs[:n]))  # warmup, not scored
        before, start = metrics(), time.monotonic()
        rows = list(pool.map(lambda p: stream_chat(p, tokens), jobs))
        wall = time.monotonic() - start
    spec = delta(before, metrics())
    total = sum(r["completion_tokens"] for r in rows)
    assert total == tokens * len(jobs), total
    summary = dict(tier="development", concurrency=n, requests=len(jobs), tokens=tokens, wall_s=wall,
                   aggregate_output_tok_s=total / wall,
                   median_request_decode_tok_s=statistics.median(r["decode_tok_s"] for r in rows),
                   median_ttft_s=statistics.median(r["ttft_s"] for r in rows), spec=spec,
                   mean_acceptance_length=1 + spec["num_accepted_tokens"] / spec["num_drafts"]
                   if spec["num_drafts"] else None)
    (art / f"chatc-{n}.json").write_text(json.dumps(summary) + "\n")
    print(json.dumps(summary), flush=True)


def needle(art, tokens, seed):
    rng = random.Random(seed)
    names = ["Avery", "Blake", "Casey", "Devon", "Emery", "Finley", "Gray"]
    lines = [f"Log {i:06d}: array {i % 17} read {rng.randint(100, 999)} mV; operator {names[i % 7]} "
             f"logged shift {rng.randint(1, 9)} as nominal." for i in range(tokens // 24)]
    codes = {v: "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(8))
             for v in "ABC"}
    for v, frac in zip("ABC", (0.15, 0.50, 0.85)):
        lines.insert(int(len(lines) * frac), f"IMPORTANT: the access code for vault {v} is {codes[v]}.")
    content = ("Read the maintenance log below.\n\n" + "\n".join(lines) +
               "\n\nWhat are the access codes for vaults A, B and C? Answer exactly in the form "
               "A=<code>, B=<code>, C=<code> and nothing else.")
    row = stream_chat(content, 1024, ignore_eos=False, effort="low")
    found = {v: c in row["content"] for v, c in codes.items()}
    row.update(codes=codes, found=found, passed=all(found.values()), requested_tokens=tokens)
    (art / f"needle-{tokens}.json").write_text(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "reasoning"}), flush=True)
    if not row["passed"]:
        raise SystemExit(f"needle gate failed at {row['prompt_tokens']} tokens: {found}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("parity", "chat", "chatc", "needle"))
    ap.add_argument("art", type=Path)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--cases", type=int, default=len(PROMPTS))
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=409)
    a = ap.parse_args()
    a.art.mkdir(parents=True, exist_ok=True)
    if a.cmd == "parity":
        parity(a.art, a.reference)
    elif a.cmd == "chat":
        chat(a.art, a.depth, a.cases, a.tokens)
    elif a.cmd == "chatc":
        chatc(a.art, a.concurrency, a.tokens)
    else:
        needle(a.art, a.tokens, a.seed)


if __name__ == "__main__":
    main()
