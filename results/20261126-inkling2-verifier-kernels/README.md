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
