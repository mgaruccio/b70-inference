# Bolt SYCL SSM_SCAN → cache CPY fusion

**Development patch, not a performance result or a promoted configuration.** Incremental on the accepted **original-weight NVFP4 R1** tree at `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-r1` (upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb`). No ReLU candidate, weight, sampling, cache-policy, launcher, or precision changes.

## Change

`ssm-cache.patch` changes only:

- `ggml/src/ggml-sycl/ssm_scan.{cpp,hpp}`: pass a separate final-state destination to the **existing** scan kernel. The recurrence, reduction, token order, F32 arithmetic, and K>1 snapshot code are unchanged. The ordinary entry point still writes the original tail. Remove `restrict` from the state input because an aligned input-cache row may now be overwritten.
- `ggml/src/ggml-sycl/ggml-sycl.cpp`: match the first executable operation after SSM_SCAN, require the exact final-tail → cache-view CPY, redirect the final-state write, and skip only that CPY and intervening non-executed/view nodes. Reuse `GGML_SYCL_ENABLE_FUSION` and the existing `scope_op_debug_print` facility.
- `tests/test-backend-ops.cpp`: extend the existing cache-fusion test with a deterministic, separately selectable native suite. Existing cases/selectors remain unchanged.

Eligibility is intentionally narrow: Mamba-2, F32, `d_state=128`, `K=1`, one sequence, a supported head dimension, contiguous state, and the exact cache-row layout. Live views wholly within `y` remain valid. `ggml_set_output(y)` also marks the full scan OUTPUT, so explicitly exported `y` views conservatively fall back. Graph use counts and view extents reject additional tail/full-scan consumers. Scan/tail/CPY output flags, intervening launches, padded destination strides, K>1, multiple sequences, and other state sizes fall back. Destination and scan must have the same backend buffer type; destination overlap with scan output or non-state inputs is rejected. State-input overlap is allowed only for a complete aligned row, safe for the single sequence's independent per-coordinate state updates.

For Bolt's 40 Mamba-2 layers (`128 × 64 × 128` F32 state, 4 MiB/layer), an eligible decode graph should remove **40 cache copies and about 320 MiB/token of temporary-state write/read traffic**. This is byte/launch accounting, **not a measured speedup**. The graph's temporary allocation is not removed. K>1 and multi-sequence acceleration are non-goals.

## Native regression suite

Selector:

```sh
test-backend-ops test -b SYCL0 -o SSM_SCAN_CACHE_FUSION -p 'bolt_ssm_cache='
```

Exactly **26 cases**; no unsupported `d_state=96` case is selected:

- 9 actual Bolt-state cases: 1/4/32 tokens × independent state / in-place cache row / gather from another cache row.
- 2 small eligible controls: 4 heads, 1/4 tokens.
- 11 guarded cases: padded cache stride, shifted state overlap, activation-input overlap, scan-output overlap, extra tail consumer, y output, tail output, full scan output, full scan consumer, CPY output, and a real intervening cache read.
- 2 K=4 fallbacks: 1 and 8 tokens, including unwritten rollback slots and padded slot spacing.
- 1 `d_state=256` fallback; 1 two-sequence fallback.

Each new case runs a forced-unfused scan first (its full output is marked OUTPUT), before the candidate can change aliased inputs. It compares **actual `y` and final/copied state arrays** against the native CPU reference using the unchanged default **1e-7 NMSE** limit. Separate full-array deltas require **exact zero** for candidate versus same-backend unfused `y`, copied state, and the entire cache, including canary rows, offsets, padding and unused slots. No scalar reduction can conceal a wrong element. Live tail/full-scan consumers are also checked. The deliberate scan-output-overlap fallback checks the post-CPY observable `y`, which that CPY intentionally overwrites.

Expected debug dispatch count with fusion on: **11** `ggml_sycl_ssm_scan_fused_cache` entries; with fusion off: **0**. The other 15 candidate scans and all 26 forced references must stay unfused. Count entry markers ending in `:` only, not the matching `done` records.

## Verification and handoff

Worker access was read-only SSH; **no remote source writes, builds, containers, GPU jobs, or server changes were performed**. Local verification uses a temporary source copy beneath the worktree and removes it afterward.

Observed worker checks:

- Unified-diff hunk counts and `git apply --whitespace=error --check`, apply, and reverse-check against the exact R1 source: passed.
- The existing recurrence/reduction/snapshot/final-store block is byte-identical to R1; only its final-state base address changes: passed source comparison (not a runtime correctness result).
- Full patched native test translation unit: `/usr/bin/c++ -std=c++17 -fsyntax-only -I<scratch>/ggml/include <scratch>/tests/test-backend-ops.cpp`: passed with GCC 16.2.1 (20260810). An initial bare `c++` invocation in the isolated evaluator could not locate `cc1plus`; using the absolute compiler path resolved it.
- At worker handoff, SYCL compilation and runtime validation were pending. Lead results below supersede that source-only status.

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

Require **26/26 passing, zero unsupported/failed cases in each run**, plus the 0/11 dispatch-entry counts. Then rerun the lead's unchanged **112-case legacy suite**. Preserve compiler output and native logs, including failures; do not relax tolerances to obtain a pass. Debug logging is for correctness/dispatch only, not throughput timing.

The lead owns the already-declared real serving journey: unchanged original-weight R1 baseline versus this candidate, fixed 20 API guards **before timing**, matched short 55-token and long 8270-token prompts → 128 output tokens, one warm-up plus three measured repetitions, and exact 8K retrieval. Keep weights, sampling, cache, workload and launch settings identical apart from this patch; retain commands, environment, outputs, raw timing results and failures under the lead's run directory. Clean up only candidate test processes/artifacts; do not restore another model or alter persistent launchers. This reduced process is **development-tier**, not the repository's standard publication suite. No speed or quality claim is made before the real runs.

## Lead runtime results

Evidence: `../20261016-nemotron35-bolt-sycl/ssm-cache-r1-validated/`, using `run-bolt-cell-ssm-r1-v2.py --name ssm-cache-r1-validated --source /home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-cache-r1 --measure` on `inference-host`.

- Initial SYCL build exposed the worker's test-local `ggml_is_view_op` reference; replaced it with explicit backend view-op checks. Original failed patch/log retained.
- Initial native run passed 25 cases but activated **zero** fusions. The dispatch-count gate correctly blocked serving. `ggml_set_output(y)` propagates OUTPUT to the scan, so all cases had exercised fallback. Fixed the test, not the conservative matcher: ordinary `y` remains live through graph consumers, and an additional explicit `y_output` case verifies fallback. Rejection diagnostics use existing `GGML_SYCL_DEBUG` only.
- Rebuilt successfully with oneAPI 2026.0; dependency precheck passed and oneDNN hash unchanged. **164 native cases pass**: 112 legacy plus 26 fusion-off and 26 fusion-on. Exactly 0/11 fused entry markers, unchanged CPU tolerance, exact same-backend output/state/cache deltas.
- Public API: **20/20 guards**, exact `BOLT-7319-CEDAR` 8K retrieval, all measured requests uncached and 128 output tokens. Short prompt 55 tokens; long prompt 8270 tokens; one warm-up and three measured requests each.
- Median client decode: **40.17000 short / 38.67530 long tok/s**, versus earlier accepted R1 **39.43410 / 38.16702** (`fusion-r1-baseline-b`): provisional **+1.87% / +1.33%**. Long prefill **960.31663 tok/s**, TTFT **8.69906 s**. This is not an interleaved A/B comparison or a bandwidth-saturation measurement; no production promotion.
- `ssm-cache-r1-dispatch` (smoke-only, not timed) confirms **1,280 actual-model fused starts**, including **840 single-token starts =40 ×21 decode evaluations**, with zero standalone SSM scan calls and zero matcher rejections. All observed Mamba-2 layers take the cache-write path.
- Bolt remains serving in the labelled experimental container. No other-model restoration or persistent launcher/weight changes.

## External source consulted

Fresh upstream CUDA implementation (read 2026-10-16):

https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-cuda/ssm-scan.cu

It separates the state destination from `y` and preserves the group-scan recurrence. This patch adopts only that addressing principle; it does not port CUDA SSD, change math, or introduce rollback-slot fusion. The local R1 CUDA matcher and SYCL GDN cache matcher supplied the graph-shape/dispatch pattern, strengthened here with tail-consumer and alias checks.
