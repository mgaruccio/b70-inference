# Twitter draft: long-context shared-KV update

Draft only; nothing has been posted. Attach the first image to the main post and
the second to the reply (or attach both to the main post).

## Main post

```text
~10% faster long-context decode on one Intel Arc Pro B70 at 230 W.

Our experimental shared-KV path for Qwen3.8-27B: 46.75 → 51.46 tok/s at a full 212,992-token window.

Matched C1, n=5. Quality not yet cleared.

https://github.com/mgaruccio/b70-inference/blob/main/docs/qwen38-shared-kv-long-context.md
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
