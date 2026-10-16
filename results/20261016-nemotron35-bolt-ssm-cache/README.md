# Bolt SYCL SSM_SCAN → cache CPY fusion

**Development patch, not a performance result or a promoted configuration.** Incremental on the accepted **original-weight NVFP4 R1** tree at `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-r1` (upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb`). No ReLU candidate, weight, sampling, cache-policy, launcher, or precision changes.

## Change

`ssm-cache.patch` changes only:

- `ggml/src/ggml-sycl/ssm_scan.{cpp,hpp}`: pass a separate final-state destination to the **existing** scan kernel. The recurrence, reduction, token order, F32 arithmetic, and K>1 snapshot code are unchanged. The ordinary entry point still writes the original tail. Remove `restrict` from the state input because an aligned input-cache row may now be overwritten.
- `ggml/src/ggml-sycl/ggml-sycl.cpp`: match the first executable operation after SSM_SCAN, require the exact final-tail → cache-view CPY, redirect the final-state write, and skip only that CPY and intervening non-executed/view nodes. Reuse `GGML_SYCL_ENABLE_FUSION` and the existing `scope_op_debug_print` facility.
- `tests/test-backend-ops.cpp`: extend the existing cache-fusion test with a deterministic, separately selectable native suite. Existing cases/selectors remain unchanged.

Eligibility is intentionally narrow: Mamba-2, F32, `d_state=128`, `K=1`, one sequence, a supported head dimension, contiguous state, and the exact cache-row layout. Views wholly within `y`, including output views, remain valid. Graph use counts and view extents reject additional tail/full-scan consumers. Scan/tail/CPY output flags, intervening launches, padded destination strides, K>1, multiple sequences, and other state sizes fall back. Destination and scan must have the same backend buffer type; destination overlap with scan output or non-state inputs is rejected. State-input overlap is allowed only for a complete aligned row, safe for the single sequence's independent per-coordinate state updates.

For Bolt's 40 Mamba-2 layers (`128 × 64 × 128` F32 state, 4 MiB/layer), an eligible decode graph should remove **40 cache copies and about 320 MiB/token of temporary-state write/read traffic**. This is byte/launch accounting, **not a measured speedup**. The graph's temporary allocation is not removed. K>1 and multi-sequence acceleration are non-goals.

## Native regression suite

Selector:

```sh
test-backend-ops test -b SYCL0 -o SSM_SCAN_CACHE_FUSION -p 'bolt_ssm_cache='
```

Exactly **25 cases**; no unsupported `d_state=96` case is selected:

- 9 actual Bolt-state cases: 1/4/32 tokens × independent state / in-place cache row / gather from another cache row.
- 2 small eligible controls: 4 heads, 1/4 tokens.
- 10 guarded cases: padded cache stride, shifted state overlap, activation-input overlap, scan-output overlap, extra tail consumer, tail output, full scan output, full scan consumer, CPY output, and a real intervening cache read.
- 2 K=4 fallbacks: 1 and 8 tokens, including unwritten rollback slots and padded slot spacing.
- 1 `d_state=256` fallback; 1 two-sequence fallback.

Each new case runs a forced-unfused scan first (its full output is marked OUTPUT), before the candidate can change aliased inputs. It compares **actual `y` and final/copied state arrays** against the native CPU reference using the unchanged default **1e-7 NMSE** limit. Separate full-array deltas require **exact zero** for candidate versus same-backend unfused `y`, copied state, and the entire cache, including canary rows, offsets, padding and unused slots. No scalar reduction can conceal a wrong element. Live tail/full-scan consumers are also checked. The deliberate scan-output-overlap fallback checks the post-CPY observable `y`, which that CPY intentionally overwrites.

Expected debug dispatch count with fusion on: **11** `ggml_sycl_ssm_scan_fused_cache` entries; with fusion off: **0**. The other 14 candidate scans and all 25 forced references must stay unfused. Count entry markers ending in `:` only, not the matching `done` records.

## Verification and handoff

Worker access was read-only SSH; **no remote source writes, builds, containers, GPU jobs, or server changes were performed**. Local verification uses a temporary source copy beneath the worktree and removes it afterward.

Observed worker checks:

- Unified-diff hunk counts and `git apply --whitespace=error --check`, apply, and reverse-check against the exact R1 source: passed.
- The existing recurrence/reduction/snapshot/final-store block is byte-identical to R1; only its final-state base address changes: passed source comparison (not a runtime correctness result).
- Full patched native test translation unit: `/usr/bin/c++ -std=c++17 -fsyntax-only -I<scratch>/ggml/include <scratch>/tests/test-backend-ops.cpp`: passed with GCC 16.2.1 (20260810). An initial bare `c++` invocation in the isolated evaluator could not locate `cc1plus`; using the absolute compiler path resolved it.
- SYCL kernel compilation, native execution and serving E2E: **pending the lead's real runtime checks**. Local syntax/application checks do not substitute for these.

Lead commands, in a disposable candidate copy of R1 and its existing configured oneAPI/oneDNN environment (not the live R1 source):

```sh
# P is this patch's absolute path; D is the disposable candidate source root.
cd "$D"
git apply --check "$P"
git apply "$P"
cmake -S . -B build-sycl -DLLAMA_BUILD_TESTS=ON
cmake --build build-sycl --config Release -j6 --target llama-server test-backend-ops

# Run sequentially in the existing B70 runtime environment; keep the GPU dedicated to Bolt.
set -o pipefail
mkdir -p "$R/native"
for fusion in 0 1; do
    GGML_SYCL_ENABLE_FUSION="$fusion" GGML_SYCL_DEBUG=1 \
        ./build-sycl/bin/test-backend-ops test -b SYCL0 \
        -o SSM_SCAN_CACHE_FUSION -p 'bolt_ssm_cache=' \
        2>&1 | tee "$R/native/fusion-$fusion.log"
done
```

Require **25/25 passing, zero unsupported/failed cases in each run**, plus the 0/11 dispatch-entry counts. Then rerun the lead's unchanged **112-case legacy suite**. Preserve compiler output and native logs, including failures; do not relax tolerances to obtain a pass. Debug logging is for correctness/dispatch only, not throughput timing.

The lead owns the already-declared real serving journey: unchanged original-weight R1 baseline versus this candidate, fixed 20 API guards **before timing**, matched short 55-token and long 8270-token prompts → 128 output tokens, one warm-up plus three measured repetitions, and exact 8K retrieval. Keep weights, sampling, cache, workload and launch settings identical apart from this patch; retain commands, environment, outputs, raw timing results and failures under the lead's run directory. Clean up only candidate test processes/artifacts; do not restore another model or alter persistent launchers. This reduced process is **development-tier**, not the repository's standard publication suite. No speed or quality claim is made before the real runs.

## External source consulted

Fresh upstream CUDA implementation (read 2026-10-16):

https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-cuda/ssm-scan.cu

It separates the state destination from `y` and preserves the group-scan recurrence. This patch adopts only that addressing principle; it does not port CUDA SSD, change math, or introduce rollback-slot fusion. The local R1 CUDA matcher and SYCL GDN cache matcher supplied the graph-shape/dispatch pattern, strengthened here with tail-consumer and alias checks.
