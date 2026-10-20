# IQ2 four-token batching R1 — correctness passed, speed objective failed

**Not promoted.** The original launchers and operational service remain unchanged.

## Real-system verification

The frozen SDK rebuilt all 105 GGML-SYCL objects with the captured compiler/link rules. Only `libggml-sycl.so.0` changed among the six checked artifacts; original source/binaries remained intact. See `build-attempt-r1/`.

The clean A1/W3a/W3b/A2 run and a separate independent restoration/analysis invocation completed successfully on `inference-host`:

- All **100 public checks** and **24 measured responses** passed exact output comparisons.
- Every W3 short response retained **129 proposed / 84 accepted**; long retained **133 / 82**.
- Actual retained native logs were independently rechecked: Q8 **82 cases / 54 byte-exact cases**, IQ2 **34 cases / 16 byte-exact cases** per arm. All existing negative/canary/alias checks remained frozen. Each W3 IQ2 log requires exactly two truthful batched parent records; A arms contain none.
- Independent live original identity/source/health/sole-GPU ownership restoration passed with no errors.
- Full eight-position drafter, 7×8 noncausal lattice, rejected-KV semantics, max3 and W7 exclusion remained unchanged. Serving diagnostics were off.

Exact run commands, in separate invocations:

```sh
cd /home/mike/b70-evals/20261020-lilicorr-iq2-batch4
python3 -B run-performance.py --lead-run-timing
python3 -B run-performance.py --lead-after-timing-restore
```

## Clean performance

Geometric means of six measured responses per arm/point. Development-tier evidence; no significance claim.

| Prompt | NoDraft decode tok/s | W3 decode tok/s | W3 / NoDraft | NoDraft end-to-end tok/s | W3 end-to-end tok/s | W3 / NoDraft |
|---|---:|---:|---:|---:|---:|---:|
| Short, 55 tokens | 44.8785 | 40.6767 | 0.9064 | 37.9548 | 34.6340 | 0.9125 |
| Long, 8,270 tokens | 43.0817 | 38.2483 | 0.8878 | 11.1006 | 10.1511 | 0.9145 |

W3 decode improved about 2.4% relative to the earlier clean Q8 W3 capture, but remains **8.6–8.7% slower end-to-end than NoDraft** in this run. This does not meet the requested net, dramatic speedup. `clean-summary-r1.json` retains every sample, sample standard deviation, range, native recheck and accounting calculation. Server decode uses the existing 127-step denominator for a 128-token output; no rate definition was changed.

## Critical attribution finding

Existing, uninstrumented cumulative `draft-dflash` statistics were differenced across the actual measured requests and checked against call/proposal/acceptance counts. They put `draft()` at **303.588 ms short / 331.613 ms long**, about **9.72% / 9.99%** of decode wall.

That is **not the full drafter cost**: `common_speculative_process()` calls `impl->process()` separately, outside `t_draft_us`. Feature reads, KV-cache injection and associated synchronization must be measured separately. Feature reads can also drain outstanding **target** work; remaining time cannot be labeled pure target compute or pure drafter overhead.

A deliberately optimistic counterfactual removing only the timed `draft()` and holding all other work fixed reaches just **1.0040× NoDraft short / 0.9863× long** decode rate. This is conditional accounting, not a universal physical ceiling. It rules out relying on a faster draft-token kernel alone while ignoring process/verification costs.

## Bounded follow-up, not another blind optimization

Source and actual traces show Q8 drafter width8 already uses one quantizer and one reordered multi-column kernel (`MMVQ_REORDERED`, precision30). The apparent 2..8 singleton recursion is an F32 router-only control, not this drafter path.

The existing graph backend re-records every call and lacks established reusable scratch ownership. It is not a safe cached-replay shortcut. A finite diagnostic is being prepared to distinguish process/submission/drain intervals and actual selected SYCL kernel events without changing arithmetic, adding waits, or using device-wide profiling state.

Fresh primary sources constraining that diagnostic:

- [SYCL event profiling](https://github.khronos.org/SYCL_Reference/iface/event.html): device `command_start/end` differ from host submission wall; queries can wait, so defer them until existing completion points.
- [USM lifetime](https://github.khronos.org/SYCL_Reference/iface/usm_allocations.html) and [Intel graph usage](https://intel.github.io/llvm/syclgraph/SYCLGraphUsageGuide.html): freeing/recycling storage and graph updates require explicit lifetime/order correctness; no graph enable/replay change is selected.
- [IGT intel_gpu_top manual](https://gitlab.freedesktop.org/drm/igt-gpu-tools/-/raw/master/man/intel_gpu_top.rst): current IGT does not support Xe through this tool. Installed CLI help is not capability evidence.
- [IGT gputop](https://gitlab.freedesktop.org/drm/igt-gpu-tools/-/raw/master/tools/gputop.c) and [profiling-state handling](https://gitlab.freedesktop.org/drm/igt-gpu-tools/-/raw/master/lib/igt_profiling.c): monitoring changes device profiling knobs; it is not used here, and ownership guards are not relaxed.

## Retention

Remote originals: `/home/mike/b70-evals/20261020-lilicorr-iq2-batch4/candidate-r1/`. Local copy: `candidate-r1/`. All **2,480 files / 977,183,171 bytes** individually matched remote SHA-256 and size; see `retention-candidate-r1.json`. Large raw logs/JSON use lossless gzip sidecars for Git while raw local/remote originals remain unchanged. The completed declaration/code files are not edited to retroactively change their recorded hashes.
