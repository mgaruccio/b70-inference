#!/usr/bin/env python3
"""Summarize a vllm-dflash-instrument receipt JSON."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/mike/b70-evals/muse-glimmer/20260826T-vllm-dflash-draft-gptq-c1/gptq-draft-8x256.json"
)
d = json.load(open(path))
print("reps:", d.get("reps"), "max_tokens:", d.get("max_tokens"))
print("median_decode_tok_s:", d.get("median_decode_tok_s"))
print("median_e2e_tok_s:", d.get("median_e2e_tok_s"))
print("median_ttft_s:", d.get("median_ttft_s"))
pre = d.get("counters_pre", {})
post = d.get("counters_post", {})
for k in sorted(set(pre) | set(post)):
    delta = post.get(k, 0) - pre.get(k, 0)
    if abs(delta) > 0.5:
        print(f"{k} delta={delta:.0f}")
# acceptance ratio if counters present
acc = post.get("vllm:spec_decode_num_accepted_tokens_total", 0) - pre.get(
    "vllm:spec_decode_num_accepted_tokens_total", 0
)
drafts = post.get("vllm:spec_decode_num_drafts_total", 0) - pre.get(
    "vllm:spec_decode_num_drafts_total", 0
)
draft_tok = post.get("vllm:spec_decode_num_draft_tokens_total", 0) - pre.get(
    "vllm:spec_decode_num_draft_tokens_total", 0
)
if drafts:
    print(f"accepted/draft_run={acc/drafts:.3f} draft_tokens/draft={draft_tok/drafts:.1f}")
    print(f"accepted/draft_token={acc/draft_tok:.4f}")