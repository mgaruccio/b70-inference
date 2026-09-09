#!/usr/bin/env python3
"""Opt-in top-two probability-ratio relaxation, NOT distribution-preserving.

Inspired by the relaxed-acceptance proposal https://github.com/vllm-project/vllm/pull/45229,
not the MoE adaptive-K Cascade scheduler. Pinned Qwen3.8 single-user research only.
"""
from pathlib import Path

MARKER = "B70_CASCADE_ACCEPTANCE_RESEARCH"
ANCHOR = "        target_argmax = target_logits.argmax(dim=-1)\n"
INJECTION = '''        # B70_CASCADE_ACCEPTANCE_RESEARCH: only all-greedy standard sampling.
        if B70_CASCADE_RATIO > 0 and sampling_metadata.all_greedy and not synthetic_mode:
            target_argmax = relax_argmax(target_logits, draft_token_ids, target_argmax, B70_CASCADE_RATIO)
'''
HELPER = '''\
"""Explicitly lossy top-two acceptance. Zero ratio disables the experiment."""
import math
import os

import torch

B70_CASCADE_RATIO = float(os.environ.get("B70_CASCADE_ALPHA", "0"))
if not math.isfinite(B70_CASCADE_RATIO) or not 0 <= B70_CASCADE_RATIO <= 1:
    raise ValueError("B70_CASCADE_ALPHA must be a finite probability ratio in [0,1]")
if B70_CASCADE_RATIO:
    print(f"B70_CASCADE_ACCEPTANCE_RESEARCH=top2 ratio={B70_CASCADE_RATIO} greedy-only LOSSY", flush=True)


def relax_argmax(logits, draft_ids, argmax, ratio):
    if ratio == 0 or not logits.shape[0]:
        return argmax
    if logits.shape[-1] != 248320:
        raise ValueError("relaxed verifier is restricted to Qwen3.8's 248320 vocabulary")
    # Keep the exact argmax for fallback/ties. Never claim that a top-k index is
    # the target argmax. Negative padded drafts must fail, never gather at -1.
    safe_ids = draft_ids.clamp(0, logits.shape[-1] - 1).long()
    candidate = logits.gather(1, safe_ids[:, None]).squeeze(1)
    best = logits.gather(1, argmax[:, None]).squeeze(1)
    top_two = logits.topk(2, dim=-1).indices
    in_top_two = (top_two == safe_ids[:, None]).any(dim=-1)
    # All special/control/padding tokens in this tokenizer lie at >=248044.
    # Conservatively keep that entire tail strict, in BOTH draft and target.
    ordinary = (draft_ids >= 0) & (draft_ids < 248044) & (argmax < 248044)
    near = candidate >= best + math.log(ratio)
    accept = ordinary & in_top_two & near & torch.isfinite(candidate) & torch.isfinite(best)
    # The existing prefix-truncation kernel emits this draft id when relaxed,
    # keeping accepted prefixes and their KV state aligned. Bonus sampling and
    # logprob reporting remain on the real, unmodified target logits.
    return torch.where(accept, safe_ids.to(argmax.dtype), argmax)
'''


def patch(root):
    path = Path(root) / "v1/sample/rejection_sampler.py"
    text = path.read_text()
    if MARKER not in text:
        if text.count(ANCHOR) != 1 or text.count("import torch\n") != 1:
            raise RuntimeError("pinned rejection sampler anchors changed")
        text = text.replace(ANCHOR, ANCHOR + INJECTION)
        text = text.replace("import torch\n", "import torch\nfrom vllm.v1.sample.b70_cascade_acceptance import B70_CASCADE_RATIO, relax_argmax\n", 1)
    compile(text, str(path), "exec")
    compile(HELPER, "b70_cascade_acceptance.py", "exec")
    (path.parent / "b70_cascade_acceptance.py").write_text(HELPER)
    path.write_text(text)
    print(f"{MARKER}: {path}", flush=True)


if __name__ == "__main__":
    import vllm

    patch(Path(vllm.__file__).parent)
