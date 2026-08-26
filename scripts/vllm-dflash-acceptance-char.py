#!/usr/bin/env python3
"""Multi-prompt DFlash acceptance characterization at the public OpenAI boundary.

Does not replace vllm-dflash-instrument.py (the single-prompt gate harness).
Use >=256-token windows; 128-token windows overstate acceptance.

Usage:
  python3 vllm-dflash-acceptance-char.py [num_reps] [max_tokens] [out.json]
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "muse-glimmer-gptq"

PROMPTS = (
    {
        "id": "sky-rayleigh",
        "kind": "reasoning",
        "prompt": (
            "Explain in detail why the sky is blue. Cover Rayleigh scattering, "
            "wavelength dependence, and why sunsets look red. Write complete sentences."
        ),
    },
    {
        "id": "python-heap",
        "kind": "coding",
        "prompt": (
            "Write a complete Python 3 implementation of a binary min-heap with "
            "push, pop, and heapify. Include a short correctness argument and a "
            "few doctest-style examples. Do not skip methods."
        ),
    },
    {
        "id": "aime-style",
        "kind": "math",
        "prompt": (
            "Solve step by step: a contest has 12 problems. Each problem is worth "
            "either 1, 2, or 3 points. There are twice as many 2-point problems as "
            "1-point problems, and the total score of all problems is 25. How many "
            "3-point problems are there? Show all algebra."
        ),
    },
    {
        "id": "kv-cache-systems",
        "kind": "long-form",
        "prompt": (
            "Explain paged KV cache, prefix caching, and why speculative decoding "
            "changes the verify batch shape. Cover memory layout, fragmentation, "
            "and what breaks if max-num-seqs is 1. Write a thorough technical note."
        ),
    },
    {
        "id": "tool-json",
        "kind": "tool-like",
        "prompt": (
            "Produce a JSON schema and two valid example payloads for a calendar "
            "tool with actions create, update, and cancel. Include timezone, "
            "recurrence, and attendee fields. Then explain validation failure modes."
        ),
    },
    {
        "id": "es-history",
        "kind": "multilingual",
        "prompt": (
            "Explica en espanol, con oraciones completas, la diferencia entre "
            "la Reconquista y la unificacion de Castilla y Aragon. Incluye fechas "
            "clave y por que importa para la historia moderna de Espana."
        ),
    },
    {
        "id": "debug-race",
        "kind": "coding",
        "prompt": (
            "A Python asyncio service randomly drops websocket messages under "
            "load. Walk through a debugging plan, likely causes (backpressure, "
            "task cancellation, shared mutable state), and a patched sketch."
        ),
    },
    {
        "id": "agent-plan",
        "kind": "agentic",
        "prompt": (
            "Plan a local-only research agent that searches files, writes a "
            "report, and asks for confirmation before deleting anything. Detail "
            "tools, failure recovery, and a 12-step runbook for a first task."
        ),
    },
)


def chat(prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": MODEL,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "seed": 42,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = None
    usage = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if ch:
                d = ch[0].get("delta") or {}
                piece = (
                    (d.get("content") or "")
                    + (d.get("reasoning") or "")
                    + (d.get("reasoning_content") or "")
                )
                if piece and first is None:
                    first = time.perf_counter()
    t1 = time.perf_counter()
    ct = (usage or {}).get("completion_tokens")
    return {
        "ttft_s": None if first is None else round(first - t0, 4),
        "completion_tokens": ct,
        "decode_tok_s": None if not (first and ct) else round(ct / (t1 - first), 3),
        "e2e_tok_s": None if not ct else round(ct / (t1 - t0), 3),
        "decode_wall_s": None if first is None else round(t1 - first, 4),
    }


def scrape_spec_counters() -> dict[str, float]:
    want = re.compile(r"spec_decode|accept", re.I)
    totals: dict[str, float] = {}
    with urllib.request.urlopen(f"{BASE}/metrics", timeout=10) as r:
        text = r.read().decode("utf-8", "replace")
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split(" ")[0]
        if not want.search(name):
            continue
        try:
            val = float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
        totals[name] = totals.get(name, 0.0) + val
    return totals


def counter_deltas(pre: dict[str, float], post: dict[str, float]) -> dict[str, float]:
    names = set(pre) | set(post)
    return {
        name: post.get(name, 0.0) - pre.get(name, 0.0)
        for name in sorted(names)
        if abs(post.get(name, 0.0) - pre.get(name, 0.0)) > 0.5
    }


def acceptance_stats(deltas: dict[str, float]) -> dict[str, float | None]:
    acc = deltas.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    drafts = deltas.get("vllm:spec_decode_num_drafts_total", 0.0)
    draft_tok = deltas.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    out: dict[str, float | None] = {
        "accepted_tokens": acc or None,
        "draft_runs": drafts or None,
        "draft_tokens": draft_tok or None,
        "accepted_per_draft_run": (acc / drafts) if drafts else None,
        "accepted_per_draft_token": (acc / draft_tok) if draft_tok else None,
        "draft_tokens_per_run": (draft_tok / drafts) if drafts else None,
    }
    return out


def median(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return round(statistics.median(vals), 4) if vals else None


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    out_path = sys.argv[3] if len(sys.argv) > 3 else (
        f"/tmp/dflash-acceptance-char-{int(time.time())}.json"
    )
    if max_tokens < 256:
        raise SystemExit("refuse max_tokens < 256; short windows overstate acceptance")

    prompt_results = []
    print(f"== characterization prompts={len(PROMPTS)} reps={reps} max_tokens={max_tokens} ==")
    for item in PROMPTS:
        print(f"-- {item['id']} ({item['kind']}) warmup --", flush=True)
        chat(item["prompt"], max_tokens)
        pre = scrape_spec_counters()
        print(f"-- {item['id']} measured --", flush=True)
        rows = [chat(item["prompt"], max_tokens) for _ in range(reps)]
        post = scrape_spec_counters()
        deltas = counter_deltas(pre, post)
        acc = acceptance_stats(deltas)
        summary = {
            "id": item["id"],
            "kind": item["kind"],
            "rows": rows,
            "median_decode_tok_s": median([r["decode_tok_s"] for r in rows]),
            "median_e2e_tok_s": median([r["e2e_tok_s"] for r in rows]),
            "median_ttft_s": median([r["ttft_s"] for r in rows]),
            "counter_deltas": deltas,
            "acceptance": acc,
        }
        prompt_results.append(summary)
        print(
            item["id"],
            "decode", summary["median_decode_tok_s"],
            "e2e", summary["median_e2e_tok_s"],
            "acc/run", None if acc["accepted_per_draft_run"] is None
            else round(acc["accepted_per_draft_run"], 3),
            flush=True,
        )

    decodes = [p["median_decode_tok_s"] for p in prompt_results if p["median_decode_tok_s"] is not None]
    accs = [
        p["acceptance"]["accepted_per_draft_run"]
        for p in prompt_results
        if p["acceptance"]["accepted_per_draft_run"] is not None
    ]
    out = {
        "reps": reps,
        "max_tokens": max_tokens,
        "model": MODEL,
        "temperature": 1.0,
        "top_p": 0.95,
        "seed": 42,
        "prompts": prompt_results,
        "across_prompt_median_decode_tok_s": median(decodes),
        "across_prompt_median_accepted_per_draft_run": median(accs),
        "across_prompt_min_accepted_per_draft_run": round(min(accs), 4) if accs else None,
        "across_prompt_max_accepted_per_draft_run": round(max(accs), 4) if accs else None,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("across_prompt_median_decode_tok_s", out["across_prompt_median_decode_tok_s"])
    print("across_prompt_median_accepted_per_draft_run", out["across_prompt_median_accepted_per_draft_run"])
    print("across_prompt_min_accepted_per_draft_run", out["across_prompt_min_accepted_per_draft_run"])
    print("across_prompt_max_accepted_per_draft_run", out["across_prompt_max_accepted_per_draft_run"])
    print("saved", out_path)


if __name__ == "__main__":
    main()
