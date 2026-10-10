# Twitter draft: long-context shared-KV update

Draft only; nothing has been posted. Attach the first image to the main post and
the second to the reply (or attach both to the main post).

## Main post

```text
~10% faster long-context decode on one Intel Arc Pro B70 at 230 W.

Our experimental shared-KV path for Qwen3.8-27B: 46.75 → 51.46 tok/s at a full 212,992-token window.

Matched C1, n=5. Quality not yet cleared.

https://github.com/mgaruccio/b70-inference#experimental-shared-kv-recipe
```

Image: [full-context PNG](images/qwen38-shared-kv-full-context.png)

**Alt text:** Qwen3.8-27B on one Intel Arc Pro B70 at 230 W. Native MTP4 achieves
46.75 decode tokens per second; shared-KV MTP4 achieves 51.46, a 10.08% improvement
(10.1% rounded). Context is 212,864 input plus 128 output tokens. Same prompts,
five measured samples per arm, C1, INT4 target and draft, FP8 KV. Experimental;
quality not yet cleared. Decode-only, not a measured whole-agent speedup.

## Reply

```text
The decode gain holds as context grows:

~128K: +7.9%
160K: +9.3%
192K: +10.3%
Full window: +10.1%

Useful territory for agents carrying long histories. This measures generation with a large KV cache—not whole-agent speedup. Multi-turn performance is still unmeasured.
```

Image: [context/gain PNG](images/qwen38-shared-kv-context-gains.png)

**Alt text:** Horizontal bar chart of shared-KV decode throughput gains over
matched native MTP4 at 230 W. At 130,944 input tokens: 61.18 to 66.03 tok/s,
+7.92%; at 163,840: 50.36 to 55.07, +9.35%; at 196,608: 49.24 to 54.29, +10.25%;
at 212,864: 46.75 to 51.46, +10.08%. All requests generate 128 tokens, with five
measured samples per point. The ~128K result is from the preceding matched run.
Experimental; quality not yet cleared.

## Expanded post — Lithos attribution and try-it link

Long-form draft; requires long-post support or splitting into a thread. The
inspiration is the community's megakernel work, particularly **lithos-metal**,
not the MLXFast challenge. No original X handle or post URL has been verified;
credit the project rather than inventing a personal attribution.

```text
The community's megakernel work—particularly lithos-metal on Apple Silicon—got us looking at what similar data-locality ideas could do for inference on Intel's Arc Pro B70.

We didn't port their engine or build a whole-model megakernel. We profiled our vLLM/XPU setup and pursued a narrower optimization: sharing KV-cache work across MTP verification positions.

MTP4 verification processes five query positions. Our custom attention path groups those queries so the kernel can reuse K/V data across them, while preserving each position's causal mask.

The result becomes more interesting as the context grows:
• ~128K: +7.9% decode throughput
• 160K: +9.3%
• 192K: +10.3%
• Full configured window: +10.1%

At 212,864 input tokens + 128 generated tokens, Qwen3.8-27B went from 46.75 to 51.46 tok/s on one B70 at 230 W.

Same machine, model, prompts and serving settings. Five measured prompts per point.

This is why long-context optimization matters for agents: the accumulated code, conversation and tool history stays relevant while generating subsequent tokens. Faster decode with that large context is worth pursuing independently of initial prompt-ingestion time.

Still experimental: quality qualification isn't cleared, and we haven't measured whole-agent task speedup.

Credit to the lithos-metal community for the work that prompted this investigation:
https://github.com/lithos-ai/lithos-metal

Experimental recipe, setup checks and native fallback:
https://github.com/mgaruccio/b70-inference#experimental-shared-kv-recipe
```

The recipe is an advanced, host-adapted integration guide, not a portable
one-command installer. The [research update](qwen38-shared-kv-long-context.md)
links the methodology and raw evidence. All drafts remain **unposted**.

## Posting notes

- Both PNGs are **1600×900**, with editable SVG siblings.
- Lead with decode at a large resident context, not cold-prefill/short-output
  wall-time ratios. No measured agent-task completion speedup is claimed.
- This is a matched local native-versus-shared-KV comparison, **not** a claimed
  win over an identically reproduced community serving setup.
- The linked update discloses the earlier HumanEval+ result (136/164 shared-KV
  versus 139/164 native), unresolved qualification, output differences and n=5
  sequential runs. Do not describe this as quality-neutral or production-ready.
- The data are pinned in commit `7927610d`; the image generator reads the
  committed JSON directly. See [the public update](qwen38-shared-kv-long-context.md)
  for source and reproduction links.
