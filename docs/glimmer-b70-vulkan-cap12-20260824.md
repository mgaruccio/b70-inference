# Glimmer B70 — Vulkan `mul_mat_vec` cap=12 result — 2026-08-24

Host experiment on `inference-host` (Intel Arc Pro B70 `8086:e223`, `xe`) using llama.cpp `64f765f` plus the pre-existing packed DFlash/b70tick dirty diff. Full receipt: `~/b70-evals/muse-glimmer/20260824T163018-cap12/`.

## Decision

**Not promoted; restored to cap=8.** The only source change was:

```text
static constexpr uint32_t mul_mat_vec_max_cols = 8;
```

to `12` in `ggml/src/ggml-vulkan/ggml-vulkan.cpp:389`. Nominal C4 `n=12` stayed on `MUL_MAT_VEC` and improved the observed decode path, but the raw C4 n=2 stream also contained an `n=13` GEMM fallback. That fallback means the nominal n=12 dispatch result is not evidence that the whole batched graph avoided GEMM.

The prior total-verify `≤1.25×` gate is withdrawn as **invalid**: it mixed per-request verify timing with aggregate public-boundary behavior and did not use a correctly weighted acceptance measure. The 130.8 ms and 98.2 ms values remain raw per-request observations, not a decisive promotion gate. Two bounded alternating n=1/n=2 blocks reproduced the exploratory result. Cap=12 was not promoted and cap=16 was not attempted.
## Matched public-boundary experiment

Real streaming `POST /v1/chat/completions` on isolated port `18100`; fixed sky prompt, seed 42, temperature 1.0, top-p 0.95, top-k 64, 64 output tokens, one discarded warmup, official DFlash, `-ub 512`. Decode in the table is **per-request server-log `eval time`**; aggregate e2e is separate; never use `completion_tokens / wall` as decode.

Instrumentation commands:

```bash
gputop -d 0.1 -n 3000 > <receipt>/<cell>.gputop.raw
python3 ~/inference/launchers/glimmer-phase0-instrument.py \
  --base http://127.0.0.1:18100/v1 --concurrency 4 --reps 3 \
  --max-tokens 64 --warmup --seed 42 --log <server-log> \
  --label <cell> --out <receipt>/<cell>.json
```
| Cell | Per-request log-eval tok/s p50 | Aggregate e2e tok/s p50 | Per-request verify p50 (ms) | Card W p50 | Per-request acceptance median | Weighted acceptance* |
|---|---:|---:|---:|---:|---:|---:|
| Cap=8 C4 n=1 | 14.130 | 40.933 | 98.445 | 202.839 | 0.89394 | 0.89822 |
| Cap=8 C4 n=2 | 3.550 | 13.186 | 697.530 | 128.627 | 0.95349 | 0.93474 |
| Cap=12 C4 n=1 | 16.435 | 42.600 | 98.210 | 211.902 | 0.93750 | 0.91753 |
| Cap=12 C4 n=2 | 18.005 | 45.642 | 130.800 | 209.779 | 0.90909 | 0.91651 |
| Cap=12 alternating C4 n=1 | 16.440 | 42.539 | 98.325 | 214.968 | 0.93129 | 0.92248 |
| Cap=12 alternating C4 n=2 | 18.210 | 45.600 | 130.590 | 212.843 | 0.93129 | 0.93103 |

`Per-request log-eval` and verify are server-log observations. `Aggregate e2e` is the receipt `aggregate_e2e_tok_s` p50 across waves. *Weighted acceptance is `sum accepted / sum generated` over the same request rows; it is not the per-request acceptance median.

The nominal candidate C4 n=2 was 5.07× the cap=8 C4 n=2 by per-request log-eval p50 (18.005 vs 3.550), while aggregate e2e wave p50 was 45.642 vs 13.186. Candidate n=2 weighted acceptance was 0.91651 versus 0.93474 for the cap=8 baseline; candidate n=2 power was 0.99× candidate n=1. No assert, device loss, NaN, or invalid output occurred. The only startup `failed` lines were the known DFlash memory-fit probe warning before successful initialization.
## Dispatch and first-use evidence

- C3 n=2 (`-np 6 -c 196608`, 3 streams, n=9): all requests completed; first verify 134.99 ms, steady last-20 p50 105.03 ms; timings show `MUL_MAT_VEC ... n=9`.
- C4 n=2 (`-np 8 -c 262144`, nominal n=12): first verify 141.69 ms, steady last-20 p50 131.18 ms; timings show `MUL_MAT_VEC ... n=12`, but the raw stream also contains an `n=13` GEMM fallback. The absence of an exact-n=12 GEMM line is not a no-fallback proof.
- C4 n=1 stayed on `MUL_MAT_VEC ... n=8` with 98.21 ms p50.
- Candidate SSE completed with `[DONE]`; restore C1 32-token SSE completed with `[DONE]` (33 data lines). Restore `/health` was `{"status":"ok"}` and `/v1/models` returned `muse-glimmer-30b`.

## Restore

The candidate was stopped, the source cap was restored to 8, and `ninja -C ~/inference/src/llama.cpp-dflash2/build -v llama-server` completed successfully. Relevant restored hashes exactly match the preflight backups:

- `llama-server`: `d001f92857adda2d2dc3ef14a93ca4676a6026419bff07459f01265e27acfaad`
- `libggml-vulkan.so.0.20.2`: `c7ddbb1c24e62f2faa5d35dd2f5f67df8ea6701143e4cb127383da2b9c66d67e`

Normal `muse` is up on port `18099` with official DFlash `n_max=2`, no perf logger, no `--log-timestamps`, and no `--perf`. The side tree final diff still contains only the pre-existing `common/speculative.cpp` and `tools/server/server-context.cpp` changes. Candidate/rollback verbose logs note that CMake/Ninja relinked additional unchanged libraries during reconfigure; no generated shader source changed.

## Research and next branch

- [Vulkan dispatch source at `64f765f`](https://github.com/ggml-org/llama.cpp/blob/64f765f5adefa4620dddda436ce56f1430435536/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L389): cap=8 is host-side dispatch/pipeline policy.
- [`NUM_COLS` specialization constant](https://github.com/ggml-org/llama.cpp/blob/64f765f5adefa4620dddda436ce56f1430435536/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_base.glsl#L91): no literal shader ceiling at 8.
- [llama.cpp PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342) ([files](https://github.com/ggml-org/llama.cpp/pull/27342/files)): no Vulkan change.

Do not promote cap=12 from this exploratory receipt, rewrite inject/process, start a custom Q4 GEMM, try DFlash2 on C4, or run cap=16 without a demonstrated 13–16 cap-bound need. The old total-verify `≤1.25×` gate is invalid; use the separated per-request/aggregate and weighted-acceptance labels above. The current native state and paused next action are in `docs/glimmer-b70-native-result-handoff-20260825.md`.
