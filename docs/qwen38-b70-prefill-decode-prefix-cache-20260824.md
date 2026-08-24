# B70 prefill vs decode, and prefix cache — 2026-08-24

Do not report `completion_tokens / wall` as decode. That hid a ~100 s
98k prefill behind “0.64 tok/s.”

## Live service

`~/inference/launchers/start-qwen38.sh` now uses
`--enable-prefix-caching --mamba-cache-mode align`. Logs show
`enable_prefix_caching: True`. Still BF16 MTP-4, GPTQ target, FP8 KV, 131k.

## Historical rows, split

| Measurement | Prefill | Decode |
| --- | --- | --- |
| Short warm BF16 (old C1) | tiny prompt | **~44 tok/s** e2e ≈ decode |
| ~98k + 64 tokens, ~100 s | **~98k in ~100 s ≈ 980 prompt-tok/s** | not isolated; the 0.64 figure is e2e |
| ~98k + 2048 tokens (draft-INT4 only) | same ~100 s prefill | incremental **~77 tok/s** |

## New streaming split (prefix cache on)

Short decode (59-token prompt, 8 completions):

- TTFT 1.29 s
- **decode 62.4 tok/s**
- e2e 5.7 tok/s (useless as a decode number)

Shared-prefix reuse, **7,621 prompt tokens**, 32 completions:

| Turn | TTFT | Prefill | Decode |
| --- | ---: | ---: | ---: |
| Cold | 4.24 s | 1,799 prompt-tok/s | **57.5 tok/s** |
| Warm (same prefix, new question) | **1.51 s** | 5,034 prompt-tok/s | **57.6 tok/s** |

Warm TTFT is **36% of cold**. Decode does not change. Server log after the
7.6k pair: **Prefix cache hit rate: 27.3%**. That is the expected APC shape:
prefill reuse only.

A 1,504-token pair was too short to show a hit (TTFT 0.81 vs 0.79 s, log
0.0%). Use multi-k shared prefixes (Pi system/tool text) when checking cache.

Artifacts:

```text
~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260824T-prefix-on/
~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260824T-kv-offload/
~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260824T-ssd-32g/
```

## Where the prefix cache lives

APC is not a separate file. Completed blocks stay in the **GPU KV pool**
(HBM) until evicted. This engine reports **5.68 GiB GPU KV = 141,882
tokens** (~1.08× a 131k request).

To keep several prefixes after GPU eviction, the live launcher now uses
`--kv-offloading-backend native --kv-offloading-size 8` plus
`TieringOffloadingSpec` with an `fs` secondary tier.

Tiers:

- **GPU HBM:** 5.68 GiB / 141,882 tokens (active + hot prefix)
- **CPU RAM:** 8 GiB pinned host (`--ipc=host` so `/dev/shm` can hold it)
- **SSD:** 32 GiB ext4 loop at `~/b70-evals/kv-offload-ssd`
  (`/dev/loop0`, fstab `loop,nofail`). Engine: `Created secondary tier #0 (fs)`.

Host box is 31 GiB RAM. SSD cap is the loop filesystem, not a vLLM quota.

MTP draft attention groups 0–3: trailing speculative chunk is **not**
offloaded (volatility). Target/prefix blocks are.

8k reuse with all three tiers: cold TTFT 5.09 s → warm **1.52 s**
(ratio 0.30), decode ~58 tok/s. After that smoke the SSD already held
**387 MiB** of block files.

## Sources that set the flags

- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) — APC cuts prefill, not decode.
- [vLLM KV offloading](https://github.com/vllm-project/vllm/blob/main/docs/features/kv_offloading_usage.md) — native CPU tier, XPU supported.
- B70 cookbook Qwen recipe: default characterization is cache-off; agentic/concurrent launch adds `--enable-prefix-caching`.
- This pin exposes `mamba_cache_mode` in `{all, align, none}` (`align` for hybrid GDN) and `kv_offloading_size` in GiB.
