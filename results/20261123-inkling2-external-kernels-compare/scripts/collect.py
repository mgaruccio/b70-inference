#!/usr/bin/env python3
"""(cmp copy: OUT dir from argv, six reference cases only) Capture teacher-forced prompt logprobs from the vLLM XPU OpenAI API."""
import json
import pathlib
import sys
import urllib.request

BASE = "http://127.0.0.1:18080"
OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

SHORT = [
    "Explain why the sky appears blue in one concise paragraph.",
    "Write a Python function that returns the factorial of a nonnegative integer.",
    "What are two practical ways to reduce household food waste?",
    "Translate into French: The train leaves at half past seven.",
    "Give a three-item checklist for reviewing a small software patch.",
    "Why do plants need sunlight? Answer for a curious child.",
    "Summarize the difference between TCP and UDP in two sentences.",
    "Continue this story: The lighthouse blinked once, though the sea was calm.",
]
LONG_BASE = (
    "A useful engineering report separates observations from assumptions. "
    "It records the inputs, explains the method, and preserves enough detail "
    "for another person to reproduce the result. When measurements disagree, "
    "the report should describe the disagreement rather than hide it. "
)
PROMPTS = []
for i, text in enumerate(SHORT[:4]):
    PROMPTS.append({"id": f"raw-short-{i:02d}", "kind": "raw", "text": text})
    PROMPTS.append({"id": f"chat-short-{i:02d}", "kind": "chat", "text": text})
for i in range(4):
    text = (LONG_BASE * (75 + i * 8)) + f"\nQuestion: In case {i}, state the main reproducibility principle and one reason it matters."
    PROMPTS.append({"id": f"raw-long-{i:02d}", "kind": "raw", "text": text})
    PROMPTS.append({"id": f"chat-long-{i:02d}", "kind": "chat", "text": text})


def post(path, payload):
    req = urllib.request.Request(BASE + path, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response)

CASES = {"raw-short-00","chat-short-00","raw-short-01","chat-short-01","raw-long-00","chat-long-00"}
for item in [p for p in PROMPTS if p["id"] in CASES]:
    if item["kind"] == "chat":
        tok = post("/tokenize", {"model": "inkling-2-mini", "messages": [{"role": "user", "content": item["text"]}], "add_generation_prompt": True})
    else:
        tok = post("/tokenize", {"model": "inkling-2-mini", "prompt": item["text"], "add_special_tokens": True})
    ids = tok.get("tokens") or tok.get("token_ids")
    if not ids:
        raise RuntimeError(f"No tokens for {item['id']}: {tok}")
    response = post("/v1/completions", {"model": "inkling-2-mini", "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 20})
    row = {**item, "input_ids": ids, "prompt_token_count": len(ids), "response": response}
    (OUT / f"{item['id']}.json").write_text(json.dumps(row) + "\n")
    print(json.dumps({"id": item["id"], "tokens": len(ids), "status": "saved"}), flush=True)
