# Glimmer B70 — Vulkan op inside C4 n_max=2 verify — 2026-08-24

Host: `inference-host`, Intel Arc Pro B70, public `POST /v1/chat/completions` `stream=true`.
Binary: llama.cpp **#27342** `64f765f` + packed `process()` + `b70tick`. Official DFlash.
Runtime only: `GGML_VK_PERF_LOGGER=1` `GGML_VK_PERF_LOGGER_CONCURRENT=1`. No rebuild. Inject not touched.
Receipts: `~/b70-evals/muse-glimmer/20260824T153825-vkperf/`.
Decode = log `eval time`. Card W = Xe hwmon. Never `completion_tokens / wall`.

## Headline

The 12-token C4 `n_max=2` 30B verify is **not** flash-attn, not inject, and not a host stall.

It falls off `ggml_vk_mul_mat_vec_q_f16` onto `ggml_vk_mul_mat_q_f16` because

```text
mul_mat_vec_max_cols = 8
```

C4 n1 verifies **8** tokens (`4 × (1 + 1)`) and stays on **MUL_MAT_VEC** (83.5% of a 94 ms graph).
C4 n2 verifies **12** tokens (`4 × (1 + 2)`) and spends **98.2%** of a **691 ms** graph in **MUL_MAT** GEMM at `n=12`.

Same weights, same 2800 MHz, CCS pegged. Power drops 216 W → **131 W**. Cheap/wrong Q4/Q5/Q6 GEMM dispatch, not “needs more XMX”.

## Matched cells (64 new tokens, one warmup, sky prompt, seed 42)

| Cell | Decode (log eval) | Acc | Card W | CCS p50 | Verify p50 | Full-graph dump p50 |
|---|---:|---:|---:|---:|---:|---:|
| C1 n2, `-np 2` | **43.7** tok/s | 0.95 | **255** | 91% | 56.0 ms | 51.7 ms |
| C4 n1, 8×32k load4 | **16.4** | 0.88–0.94 | **216** | 91% | 98.4 ms | 93.7 ms |
| C4 n2, 8×32k load4 | **3.67** | 0.91–0.95 | **131** | **100%** | **696** ms | **691** ms |

C1 is a short-gen 64-token row (previous 128-token C1 was 38.7). The C4 cliff is unchanged: n2 is still ~4.5× slower than n1 at half the watts. Acceptance stays healthy.

ms / verified token (verify p50 / n_tokens): C1 **18.7**, C4 n1 **12.3**, C4 n2 **58.0**. Same shape as the earlier tick profile (17.2 / 12.0 / 57.8).

## Ranked ops — one full 30B verify graph

Dumps kept only when `Vulkan Timings` total matches the following `b70tick verify` within 15%. That drops draft / inject scraps and prefill.

### C4 n2 — sick 12-token verify (20 graphs)

Dump p50 **691.2 ms** vs verify p50 **696.4 ms**.

| Family | p50 | Share |
|---|---:|---:|
| **MUL_MAT** (non-VEC) | **678.5 ms** | **98.2%** |
| FLASH_ATTN_EXT | 8.0 ms | 1.2% |
| RMS_NORM | 3.2 ms | 0.5% |
| other / SET_ROWS / GET_ROWS | 1.6 ms | 0.2% |

Top named ops, all `n=12`:

| Op | p50 | Share |
|---|---:|---:|
| `MUL_MAT q6_K m=6656 n=12 k=19968` | 152.0 ms | 22.0% |
| `MUL_MAT q5_K m=6656 n=12 k=19968` | 108.6 ms | 15.7% |
| `MUL_MAT q4_K m=6656 n=12 k=19968` | 57.1 ms | 8.3% |
| fused `MUL_MAT q5_K m=19968 n=12 k=6656` ×2 | 42.7 ms | 6.2% |
| `MUL_MAT q6_K m=6656 n=12 k=4096` | 40.2 ms | 5.8% |

No single kernel is ≥50%. The **tight family** is: quantized `MUL_MAT` GEMM at `n=12` after missing the VEC cutoff. That family is **98%** of the 694 ms verify.

Typical generate line:

```text
b70tick verify ms=695.88 n_tokens=12 unique_seq=4
Vulkan Total time: 691000 us
MUL_MAT q6_K m=6656 n=12 k=19968: 22 x 6910 us = 152000 us
```

### C4 n1 — healthy 8-token verify (31 graphs)

Dump p50 **93.7 ms** vs verify p50 **98.4 ms**.

| Family | p50 | Share |
|---|---:|---:|
| **MUL_MAT_VEC** | **78.3 ms** | **83.5%** |
| FLASH_ATTN_EXT | 10.3 ms | 10.9% |
| RMS_NORM | 3.0 ms | 3.2% |
| MUL_MAT (non-VEC) | 0.8 ms | 0.8% |

Top op: `MUL_MAT_VEC q6_K m=6656 n=8 k=19968` (13.1%). Same FFN shape as the sick row, **n=8**, still on the VEC kernel.

### C1 n2 — healthy 3-token verify (27 graphs)

Dump p50 **51.7 ms** vs verify p50 **56.0 ms**.

| Family | p50 | Share |
|---|---:|---:|
| **MUL_MAT_VEC** | **43.7 ms** | **84.6%** |
| FLASH_ATTN_EXT | 3.5 ms | 6.7% |
| RMS_NORM | 2.8 ms | 5.4% |
| MUL_MAT (non-VEC) | 0.6 ms | 1.1% |

Top op: `MUL_MAT_VEC q6_K m=6656 n=3 k=19968` (12.4%).

## Why n=8 is fine and n=12 dies

`ggml-vulkan.cpp`:

```text
static constexpr uint32_t mul_mat_vec_max_cols = 8;
```

Dispatch (`ggml_vk_mul_mat`):

```text
else if ((dst->ne[1] == 1 || (dst->ne[1] <= mul_mat_vec_max_cols && src1->ne[2] * src1->ne[3] == 1))
         && (F32 || F16 || BF16 || quantized))
    ggml_vk_mul_mat_vec_q_f16(...);   // DMMV, n = 1..8
else
    ggml_vk_mul_mat_q_f16(...);       // full GEMM
```

| Verify | `dst.ne[1]` | Path | Graph |
|---|---:|---|---:|
| C1 n2 | 3 | VEC | 52 ms / 255 W |
| C4 n1 | 8 | VEC (exactly at the cap) | 94 ms / 216 W |
| C4 n2 | 12 | **GEMM** | **691 ms / 131 W** |

Dynamic-Q4_K_XL is mixed Q4_K / Q5_K / Q6_K. All three take the same branch. Flash-attn on the 4×3 speculative batch is 8 ms. It is not the tax.

This is the “cheap/wrong Q4 dispatch” story from the last handoff, named: **`ggml_vk_mul_mat_q_f16` at n=12**, not `MUL_MAT_VEC`, not FA.

## Killed this session

- Flash-attn / non-causal mask on 4×3 — 1.2% of the sick graph
- Many tiny unfused nodes — one 691 ms dump matches the 696 ms `llama_decode` wall
- Barriers / empty submits — not in the logger
- Inject / `process()` rewrite — still ~9 ms from the prior tick split; not re-opened
- `iaprof` / patched NEO — still the wrong API

`draft()` at ~90 ms is still a real secondary tax. It is still not this session.

## Restore

Production cell is back: official DFlash `n_max=2`, `-np 2`, 131k, port 18099, tmux `muse`.
No `GGML_VK_PERF_LOGGER`, no `--log-timestamps` / `--perf`. `b70tick` binary left in place. Pack not reverted.

## Next

Stay on the existing VEC kernel. Do not write a new Q4 GEMM first.

1. Raise `mul_mat_vec_max_cols` to **12 or 16** on the `64f765f` tree and re-run only C4 n2 vs C4 n1. Pass: C4 n2 verify dump is `MUL_MAT_VEC` at n=12 and decode returns toward the n1 16 tok/s / ~220 W signature.
2. If VEC at n=12 is broken or slower, split the 4×3 verify so each `llama_decode` stays at `n≤8` (two 4×2-ish graphs). That is a batching change, not an inject rewrite.
3. Only then consider a Battlemage Q4/Q5/Q6 GEMM for n=9–16.

C4 production remains official DFlash **`n_max=1`**, 8×32k. That cell is exactly at the VEC cap.

## Sources

- Receipts: `~/b70-evals/muse-glimmer/20260824T153825-vkperf/`
- Rank: `vkperf-fullgraph-rank.json` in that directory
- Runner: `scripts/glimmer-vkperf-run.sh`
- Parser: `scripts/b70-summarize-vkperf.py`
- Prior tick split: `docs/glimmer-b70-profiling-20260824.md`
- Logger: `GGML_VK_PERF_LOGGER` in `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- VEC cap: `mul_mat_vec_max_cols = 8` in the same file
