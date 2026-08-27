#!/usr/bin/env python3
"""Dump target hidden states via vLLM extract_hidden_states (non-streaming).

Uses the shared characterization prompts. Writes a manifest next to the
safetensors files. extract_hidden_states is currently blocked on this Muse/XPU
cell; keep this driver in sync anyway.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dflash_char_prompts import PROMPTS

BASE = "http://127.0.0.1:8000"
MODEL = "muse-glimmer-gptq"


def complete(prompt: str, max_tokens: int, save_path: str) -> dict:
    body = json.dumps({
        "model": MODEL,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "seed": 42,
        "messages": [{"role": "user", "content": prompt}],
        "kv_transfer_params": {
            "hidden_states_path": save_path,
            "include_output_tokens": True,
        },
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        obj = json.loads(r.read().decode())
    elapsed = round(time.perf_counter() - t0, 3)
    usage = obj.get("usage") or {}
    kv = obj.get("kv_transfer_params") or {}
    choice = (obj.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = (msg.get("content") or "") + (msg.get("reasoning") or "") + (
        msg.get("reasoning_content") or ""
    )
    return {
        "elapsed_s": elapsed,
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "hidden_states_path": kv.get("hidden_states_path") or save_path,
        "finish_reason": choice.get("finish_reason"),
        "text_chars": len(text),
    }


def main() -> None:
    host_dir = sys.argv[1] if len(sys.argv) > 1 else (
        "/home/mike/b70-evals/muse-glimmer/20260826T-vllm-dflash-hidden-capture/hidden_states"
    )
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    container_dir = "/hidden_states"
    rows = []
    for item in PROMPTS:
        name = item["id"]
        save_path = f"{container_dir}/{name}.safetensors"
        print(f"-- {name} -> {save_path}", flush=True)
        row = {
            "id": name,
            "suite": item.get("suite"),
            **complete(item["prompt"], max_tokens, save_path),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    manifest = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "aux_layer_ids": [2, 14, 26, 38, 50],
        "include_output_tokens": True,
        "container_dir": container_dir,
        "host_dir": host_dir,
        "rows": rows,
    }
    path = f"{host_dir}/manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print("saved", path)


if __name__ == "__main__":
    main()
