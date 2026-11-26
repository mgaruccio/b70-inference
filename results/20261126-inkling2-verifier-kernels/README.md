# Inkling 2 Mini verifier kernels — Arc Pro B70

Development tier; not a standard-tier benchmark. Control: sparse-69444, plugin main `7884405`, live 409000/C32 FP8 KV, CUTLASS MoE, hybrid prefix SDPA, sparse conv history, DSpark schedule `[[1,2,7],[3,32,3]]`. Keep the drafter fixed. For C1 the prior weight-traffic estimate is ~160–230 tok/s at ~560 GB/s; 109 chat tok/s is 47–68% of that estimate, before deep KV traffic. Target-specific bounds require a new profile.

## Test process

- Use the existing idle/lock/restore harness from `../20261125-inkling2-draft-length-schedule/` on b70-dev, placing `hillclimb.sh`, `checks.py` and `profile_cases.py` together in `~/inkling-vllm-xpu/verify-kernels`. Baseline service is on :8080; experiments use loopback :18080 without auth. Do not modify the persistent launcher.
- `timeout 1200 bash hillclimb.sh profile-graph profile-eager` runs production decode graphs first, then eager for attribution only. Both add a bounded XPU profiler. Warm fixed token-ID prompts via `/v1/completions` (128 and 30080 tokens C1; 128 tokens C8/C32); start profiler, delay eight iterations, collect four, stop. Inspect the traces for evidence of steady verify shapes. Profiled request timings are not throughput results. Retain raw trace and launch/env on guest, summary here.
- For a kernel candidate: run an on-device parity and latency test, then unprofiled interleaved control/candidate/control through the existing harness (`QUICK=1`), compare acceptance and greedy IDs, chat and random throughputs. Full confirmation adds +30K chat and 200K/375K needles; enforce both gates on the candidate. No promotion without approval. The harness stops its loopback engine and verifies authenticated smoke and unauthenticated 401 on restoration.
- References: [vLLM profiling](https://docs.vllm.ai/en/stable/contributing/profiling/) (bound the engine iterations, disable frontend), [PyTorch profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) (time XPU device kernels, not host launch scopes).

## Profiles: run-20261126-070911.5tcZMK

Both modes completed all four workloads. Graph mode: 8.23 GiB KV / 412,065 tokens. Eager: 8.64 GiB / 432,912. The original service passed the authenticated smoke and unauthenticated 401 after the run.

| Workload (graph) | Sum of kernel durations, 4 iterations | NVFP4 expert GEMMs | Generic dense GEMMs* | Relative attention |
|---|---:|---:|---:|---:|
| C1 128 | 108.07 ms | 32.9% | 41.5% | 2.5% |
| C1 30080 | 127.37 ms | 28.9% | 37.8% | 13.0% |
| C8 128 | 190.96 ms | 46.7% | 27.7% | 2.7% |
| C32 128 | 328.89 ms | 47.8% | 20.6% | 4.4% |

\*Dense name spans several target and draft operations; not an actionable single kernel. Device time is summed, not wall time. The native NVFP4 kernel is identifiable, including its policy (`m_8` at C1/C8, `m_16` at C32). Its share agrees within ~1.5 percentage points in eager. A 20% reduction in this kernel's duration alone would reduce short-context summed kernel time by ~6.6% at C1 or ~9.3% at C8; actual wall gain may be smaller. Distinct routed experts are not recorded in these traces, so a precise GB/s figure from model size would be spurious. Long-context attention is a separate secondary target.
The 24 checkpoint layers are not 24 executions: `loop_config` repeats layers 6–17 three times. Each target forward thus executes **44 NVFP4 layer visits and 4 MXFP8 visits**, two projections per visit. Four iterations yielded precisely 352 NVFP4 and 32 MXFP8 calls in every trace. This confirms the attributed expert calls are target forwards. The older 160–230 tok/s ceiling counts static weights, not these repeated visits; it is optimistic, not a hard utilization estimate.

`profiles/` retains exact launch, host, versions, workload usage, full kernel aggregates. Full Kineto traces stay on guest. First isolated probes: AOT compile identical source with prefetch distance 6 (same-toolchain control), 3 and 9; arithmetic unchanged. [Intel GPU optimization guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/overview.html): tune memory access on target hardware. Timing and parity, not the prefetch hint alone, determine whether a variant advances.

## Prefetch-distance probes

`build-probes.sh 6 3 9` rebuilt the three extensions using the pinned headers (`vllm-xpu-kernels e9b6e4b`, `sycl-tla 7b56b93`). `PRETEST='bash check-probes.sh' timeout 600 bash hillclimb.sh pretest-only` ran seven real B70 FP64-reference tests at each distance (21 passes, no skips in the selected tests). This includes padded rows and deterministic repeat outputs. Run `run-20261126-071912.QlY46I`; compact pytest reports in `unit/`. The reference-bound tests alone do not establish bitwise equality to stock.

Interleaved timing command: `PRETEST='bash run-micro.sh stock pf6 pf3 pf6 pf9 pf6 stock' timeout 900 bash hillclimb.sh pretest-only`. Uses `scripts/bench_verify_nvfp4.py` from plugin commit `a25cf43`, copied alongside these scripts. All 128 experts populated; eight fixtures exercise synthetic concentrated and spread routes. Compare exact BF16 output digests separately from synchronized graph latency. Logical bytes per call are not DRAM counters.

`run-20261126-072205.2yWgsJ`: all eight digests matched across all seven processes. See `micro-prefetch/`. **Reject distance 9**: slower than both adjacent controls on every fixture. Distance 3 reduced C8 concentrated latency to 238.43 µs (244.24 / 242.31 controls), and C8 spread to 1052.05 µs (1056.51 / 1056.72). Largest stable reduction ~3% on 128 concentrated rows. C1 concentrated was noisy (stock varied 142–170 µs); no C1 benefit established there. This is at best a small lever; advance distance 3 to an unprofiled serving test targeting C8, `QUICK=1 timeout 3600 bash hillclimb.sh pf6 pf3 pf6`. No serving gain inferred from the microbench.

**Serving verdict: reject distance 3.** Initial attempt failed before timing because the staged plugin lacked `.dist-info` entry points; restoration passed. Restaged from production metadata, then ran `pf6 pf3 pf6` at 409K. Chat C8 (target): **329.3 / 322.6 / 331.9 tok/s**. Random C8: 230.8 / 240.1 / 248.7 tok/s, i.e. between controls. Random C4 rose to 180.4 from 154.5 / 170.4, but this was not the declared target and the controls varied. All 8 greedy fixtures matched, chat acceptance length was unchanged at reporting precision. No needles on rejected candidates. Full development summary in `prefetch-serving.json`, guest `run-20261126-072540.37WNVn`. Service restored.

## Coalesced scale loads

Next single lever (plugin probe `1d10e1c`): prepack the original FP8 scale pairs at load to `[E,K/32,N,2]` and fetch adjacent pairs for neighbor output channels without changing accumulation. Build one isolated DSO via `build-layout.sh`. Test `PRETEST='bash check-layout.sh' timeout 600 bash hillclimb.sh pretest-only`: layout validation and exact model-path parity, timed isolated w13/w2 projections with alternating layouts. If promising, use `layout-ctl layout-cand layout-ctl` on the same binary at stock prefetch distance, targeting chat C8. Production and drafter stay fixed.

**Reject scale transposition.** Both tests passed on B70, both projection results matched stock bitwise at 1/8/32/128 rows, but all eight timings were worse. At 32 rows, w13 rose from 635.3 to 883.0 µs and w2 from 313.0 to 428.3 µs. Do not spend a serving comparison on this layout. Results: `unit/layout-times.txt` and `unit/layout.pytest.txt`, guest `run-20261126-075529.W7787z`. Probe remains off main.

## Static tile assignment

Next independent lever: replace the native NVFP4 persistent GEMM's per-tile atomic increment + local barrier with a deterministic grid-stride walk of output tiles. No change to per-tile math or K order; may reduce scheduling overhead or worsen tail imbalance. `build-probes.sh static` on unmodified production sources (stock scale layout, prefetch 6); compare via `run-micro.sh pf6 static pf6`, exact digests first. Target C8 throughput if advanced to serving.

`run-20261126-080005.GFMu7p`: all graph/eager and control/candidate digests matched. At 32 rows static wins for 8 distinct experts (238.28 vs 242.79 / 243.88 µs) but loses for 128 (1075.37 vs 1056.16 / 1055.92). Short-context single-request spread routing also lost. Not enough to select a winner on synthetic fixtures.

Measured actual route unions in a separate stock eager probe (`run-20261126-080344.lCC3cT`): six loopback completion workloads, 8 sampled steps × 48 layer visits per full-width case. No output text, expert IDs or token IDs logged; `routing.jsonl` columns are case, visit ID, step, nominal rows, unpadded rows, highest expert count, distinct expert count. Ignore non-full-width rows for these figures. At C1/C8/C32 the median distinct experts per visit were **29/76/95** for natural-language completions and **25/39/54** for random prompts. The C8 scenario sits between both synthetic extremes. Static scheduling therefore gets a real `QUICK=1 ... pf6 static pf6` serving comparison, not an extrapolation from either endpoint. Timing of the instrumented eager run is not representative.

**Static scheduling: not accepted as a default.** Real C8 chat hit 325.5 vs 314.9 / 313.9 tok/s (+3.4–3.7%), with all 8 greedy fixtures exact, but C1 chat fell to 107.5 from 109.1 / 109.2, and random C4 at 160.9 lay between highly variable controls (157.0 / 192.3; −16% against the latter). Random C8 reached 242.0 vs 228.9 / 241.1. This single ABA does not resolve control drift or establish a regression-free serving win; do not change the default. See `static-serving.json`, run `run-20261126-080614.dI65fM`.

## Long verifier attention

Opt-in probe `209f36c` avoids mask and relative-bias loads on verifier tiles entirely preceding the bias extent; KV contents, softmax update order and split layout remain unchanged. [FlashAttention mask-free loop](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp). `PRETEST='bash check-attn.sh' timeout 600 bash hillclimb.sh pretest-only` checks bitwise partial and final outputs and times 128/30,080 tokens in XPU graphs. If advanced, target long C1 chat: `QUICK=1 timeout 3600 bash hillclimb.sh attn-ctl attn-cand attn-ctl` at 409K with target-only flag toggled. For these arms the harness also tests depth=32000 and enforces actual API `prompt_tokens >= 30000` (nominal 30K chat renders only ~28.5K).
