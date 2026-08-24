# Glimmer B70 — Vulkan `mul_mat_vec` cap=12 result — 2026-08-24

Host experiment on `inference-host` (Intel Arc Pro B70 `8086:e223`, `xe`) using llama.cpp `64f765f` plus the pre-existing packed DFlash/b70tick dirty diff. Full receipt: `~/b70-evals/muse-glimmer/20260824T163018-cap12/`.

## Decision

**Inconclusive for promotion; restored to cap=8.** The only source change was:

```text
static constexpr uint32_t mul_mat_vec_max_cols = 8;
```

to `12` in `ggml/src/ggml-vulkan/ggml-vulkan.cpp:389`. It successfully kept C4 n=12 on `MUL_MAT_VEC`, but steady C4 n=2 verify was 130.8 ms versus the C4 n=1 guard at 98.2 ms (1.33×), missing the decisive ≤1.25× verify guard. Two bounded alternating n=1/n=2 blocks reproduced the result. Cap=12 was not promoted and cap=16 was not attempted.

## Matched public-boundary experiment

Real streaming `POST /v1/chat/completions` on isolated port `18100`; fixed sky prompt, seed 42, temperature 1.0, top-p 0.95, top-k 64, 64 output tokens, one discarded warmup, official DFlash, `-ub 512`. Decode is the server log `eval time`; never `completion_tokens / wall`.

Instrumentation commands:

```bash
gputop -d 0.1 -n 3000 > <receipt>/<cell>.gputop.raw
python3 ~/inference/launchers/glimmer-phase0-instrument.py \
  --base http://127.0.0.1:18100/v1 --concurrency 4 --reps 3 \
  --max-tokens 64 --warmup --seed 42 --log <server-log> \
  --label <cell> --out <receipt>/<cell>.json
```

| Cell | log eval tok/s p50 | verify ms p50 | card W p50 | acceptance median |
|---|---:|---:|---:|---:|
| Cap=8 C4 n=1 | 14.130 | 98.445 | 202.839 | 0.89394 |
| Cap=8 C4 n=2 | 3.550 | 697.530 | 128.627 | 0.95349 |
| Cap=12 C4 n=1 | 16.435 | 98.210 | 211.902 | 0.93750 |
| Cap=12 C4 n=2 | 18.005 | 130.800 | 209.779 | 0.90909 |
| Cap=12 alternating C4 n=1 | 16.440 | 98.325 | 214.968 | 0.93750 |
| Cap=12 alternating C4 n=2 | 18.210 | 130.590 | 212.843 | 0.90909 |

Cap=12 C4 n=2 was 5.07× cap=8 C4 n=2 and 1.096× the matched C4 n=1 decode. Candidate n=2 power was 0.99× candidate n=1; acceptance was 95.3% of cap=8 n=2. No assert, device loss, NaN, or invalid output occurred. The only startup `failed` lines were the known DFlash memory-fit probe warning before successful initialization.

## Dispatch and first-use evidence

- C3 n=2 (`-np 6 -c 196608`, 3 streams, n=9): all requests completed; first verify 134.99 ms, steady last-20 p50 105.03 ms; timings show `MUL_MAT_VEC ... n=9`.
- C4 n=2 (`-np 8 -c 262144`, n=12): first verify 141.69 ms, steady last-20 p50 131.18 ms; timings show `MUL_MAT_VEC ... n=12` and no exact-n=12 GEMM lines in candidate graphs.
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

Do not promote cap=12, rewrite inject/process, start a custom Q4 GEMM, try DFlash2 on C4, or run cap=16 without a demonstrated 13–16 cap-bound need. Next bounded branch is split/native investigation; same-GGUF SYCL is the fair native fallback, while OpenVINO remains a separate IR/single-flight comparison. Do not use `scripts/glimmer-vkperf-run.sh` unattended.
