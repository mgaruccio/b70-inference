# DFlash packed-QKV context precompute — profiling result (2026-08-26 evening)

## Summary

The packed-QKV KV-only W4A16 slice from the handoff was **profiled and
abandoned**: it cannot reach the ≥5% throughput gate. The evidence shows the
draft context precompute is ~1.4% of the per-step time; even eliminating it
entirely would not move the gate.

## What was measured

- Server: `muse-vllm-xpu-c1` on `inference-host:8000` (target GPTQ + assistant
  GPTQ, XPU graph mode, DFlash n=20), candidate overlay v1 (per-layer packed
  QKV fallback) + timing instrumentation (patch v2, `DFLASH_KV_MODE=timing`).
- Workload: public OpenAI streaming API, 8 reps × 256 tokens, fixed prompt,
  seed 42 (`vllm-dflash-instrument.py 8 256`).

## Method

`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` calls
`precompute_and_store_context_kv` **eagerly every step** ("Runs eagerly
outside the captured graph because the context shape varies per step"), so
the v2 overlay wraps both `precompute_and_store_context_kv` and
`_project_context_kv` with `torch.xpu.synchronize()` + `perf_counter`
instrumentation (marker `DFLASH_GPTQ_CONTEXT_KV_TIMING`), logging cumulative
summaries every 100 calls.

## Per-step evidence (12 summaries, ~1099 calls)

| Quantity | Value |
|---|---|
| steady-state `num_ctx` | 21.6/step (buckets: 1×[1-2), 1088×[16-32), 10×[64-128), 1×[512+) dummy 2048) |
| `_project_context_kv` (`proj`) | **0.41 ms/step** (5 per-layer `qkv_proj` W4A16 GEMMs, M≈21, K=6656, N=6144) |
| k-norm + RoPE + cache writes (`rest`) | **0.25 ms/step** |
| whole context precompute (`tot`) | **0.66 ms/step** |
| clean step time | 4.84 s / 111 steps ≈ **43.6 ms/step** |
| projection share of step | **~0.9%** |
| precompute share of step | **~1.4%** |

## Gate verdict

- Realistic fused KV-only gain: 5 launches → 1 + N 6144 → 2048 ≈
  0.2-0.25 ms/step ≈ **0.5%**.
- Gate: ≥5% over 52.90 tok/s. **Not close → not implemented.**
- The v2 patch script contains the full `DFLASH_KV_MODE=kvonly` implementation
  (fused 5-layer KV-only `int4_gemm_w4a16`, lazy build from processed packed
  params, one-shot numeric self-check vs the per-layer fallback) but it was
  **not deployed**; only AST + container patch dry-runs were validated.

## Secondary findings

- **Instrumentation perturbs acceptance**: same seed 42, clean runs give
  exactly 1.560 accepted/draft-run (1248/800) reproducibly; the instrumented
  run gave 1.306 (1160/888) — device syncs change step cadence / sampling
  stream. Never compare instrumented runs to clean baselines.
- **Clean runs are reproducible**: 52.903 vs 52.922 decode tok/s; TTFT
  142.6 vs 142.5 ms; identical acceptance.
- **Dominant per-step cost is the 30B GPTQ target verify** (~21 tokens × ~30B
  params ≈ 1.3 TFLOP/step in ~43.6 ms → ~29 TFLOPS effective), not the draft
  context path. Recommended next levers: acceptance (real-target GPTQ
  recalibration of the draft; accepted/step 1.31-1.56 of 20 proposed = 7-8%)
  and target-verify kernel efficiency.

## Receipts

- `~/b70-evals/muse-glimmer/20260826T-vllm-dflash-ctxkv-profile/ctxkv-timing-run.json`
- `~/b70-evals/muse-glimmer/20260826T-vllm-dflash-ctxkv-profile/ctxkv-timing-logs.txt`
- `~/b70-evals/muse-glimmer/20260826T-vllm-dflash-ctxkv-profile/clean-baseline-verify.json`
- Server restored to v1-equivalent clean state; patch log shows
  `restored pristine original ... mode=base` + `XPUwNa16LinearKernel`.

## Artifacts

- `scripts/patch-vllm-dflash-gptq-context-kv.py` v2: modes `none` (v1
  behavior, default via launcher), `timing`, `kvonly`, `kvonly+timing`;
  `.orig` backup/restore built in.
- `scripts/start-muse-vllm-dflash-c1-graph-draft-gptq.sh`: passes
  `DFLASH_KV_MODE` (default `none` = prior v1 state).
- `scripts/wait-vllm-health.sh`: health wait helper (host `/tmp` wiped by
  reboot).
- `scripts/summarize-instrument-receipt.py`: receipt summarizer with
  acceptance ratios.