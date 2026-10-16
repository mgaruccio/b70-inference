# Bolt R1: nine-expert weighted reduction

**Development candidate, source-only; not yet SYCL-built or runtime-validated. No speedup claim.** Standalone on accepted R1, not stacked with SSM/ReLU experiments. No remote files, builds, GPU work, launchers, model restores, or canaries were changed/run by this worker.

Base: upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb` plus accepted R1, read from `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-r1` on `mike@inference-host` (non-Git source). The patch changes only `ggml/src/ggml-sycl/topk-moe.cpp` and `tests/test-backend-ops.cpp`.

## Contract

Fuse only F32 contiguous `[width,9,tokens,1] * [1,9,tokens,1]`, followed by all nine correctly offset/strided views and eight left-associated ADDs. One work-item computes one output; padded columns are bounds-checked. A prior expert-scale operation remains separate. No routing, top-k, matmul, precision, or graph-construction changes.

- Scoped `#pragma clang fp contract(off) reassociate(off)` preserves separately rounded products and ADD order; no global compiler flags change.
- `ggml_can_fuse_subgraph` protects compute/output flags, external consumers, and view sources. In-place intermediates and unsupported layouts fall back. Final output/input byte ranges must not overlap, **including leaf inputs** (the existing top-k memory helper skips `GGML_OP_NONE`, so it is not reused).
- Existing `GGML_SYCL_ENABLE_FUSION` controls dispatch. Existing debug facility emits `ggml_sycl_op_moe_weighted_reduce`; exactly 17 following nodes are consumed.

Primary sources read on 2026-10-16: [pinned CUDA reduction](https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-cuda/moe-weighted-reduction.cu) confirms expert order; [Clang FP pragmas](https://clang.llvm.org/docs/LanguageExtensions.html#extensions-to-specify-floating-point-flags) documents scoped contraction/reassociation controls. Clang warns that explicit `-ffp-contract=fast` overrides pragmas: do not introduce that flag. The R1 recorded command uses oneAPI 2026.0 `icpx -O3 -fsycl`, with no explicit fast-math/contract flag. Actual device codegen/numerics still require the sensitive native cases below.

## Native gate (lead, inside the existing isolated R1 build/runtime environment)

Use a **disposable copy**, never the accepted R1 source itself. `SRC`, `BUILD`, `PATCH`, and `RUN` below are absolute candidate-source, configured-build, patch, and ordinary scratch-log paths. Keep the baseline stack/configuration unchanged.

```bash
set -euo pipefail
patch --dry-run --batch -p1 -d "$SRC" < "$PATCH"
patch --batch -p1 -d "$SRC" < "$PATCH"
cmake --build "$BUILD" --target test-backend-ops llama-server --parallel 2
mkdir -p "$RUN"
for f in 1 0; do
    env GGML_SYCL_ENABLE_GRAPH=0 GGML_SYCL_ENABLE_FUSION="$f" GGML_SYCL_DEBUG=1 \
        "$BUILD/bin/test-backend-ops" test -b SYCL0 -o '^MOE_REDUCE_9$' -j 1 \
        > "$RUN/native-$f.log" 2>&1
    grep -F '20/20 tests passed' "$RUN/native-$f.log"
    test "$(grep -cF 'MOE_REDUCE_9 full-array: mismatches=0 max_abs=0 ' "$RUN/native-$f.log" || true)" -eq 20
done
test "$(grep -cF '[SYCL][OP] call ggml_sycl_op_moe_weighted_reduce:' "$RUN/native-1.log" || true)" -eq 8
test "$(grep -cF '[SYCL][OP] call ggml_sycl_op_moe_weighted_reduce:' "$RUN/native-0.log" || true)" -eq 0
```

Exactly **20 cases**, no unsupported/skipped cases accepted:

- Eight eligible: `(width,tokens) = (1024,1), (1024,128), (1024,1024), (1031,3)`, each with deterministic ordinary and cancellation/FMA-sensitive values.
- Twelve fallbacks at `(1024,17)`: extra MUL/view/ADD consumers; MUL/view/ADD output flags; strided experts; strided weights; in-place final ADD; shuffled expert views; right-associated ADDs; eight experts.
- Each graph contains a separate output-marked, forced-unfused control. The complete candidate/control arrays must match bitwise on the **same backend**, with reported maximum absolute delta zero. CPU comparison retains the existing **1e-7 NMSE** tolerance; protected extra outputs and sentinels are also checked. No test-framework API or tolerance changes.

Then run the lead's unchanged legacy 112-case native selector. A selector matching zero cases, missing the eight eligible markers, or passing only with fusion disabled is not acceptance.

## Public-boundary gate and comparison (lead-owned, not run here)

Development tier only. Compare accepted R1 against R1 plus this patch alone on the same B70, original weights, 64K context, batch/ubatch 1024, q8 KV, F32 recurrent state, and unchanged remaining settings. First run the existing fixed 20-request API quality gate and exact 8K retrieval check. Record the baseline's intermittent `drawer` → `rewrad` failure rather than relaxing the gate; do not publish timing as a successful improvement while quality is red.

After quality passes, reuse the existing matched short (55 input tokens) and long (8270 input tokens) API requests, 128 output tokens, greedy decoding, one warm-up and three measured repetitions, unchanged cache policy. Retain exact launch/request commands, environment, complete outputs, failures, and raw timings in the lead's normal run logs. Keep Bolt resident between experiments; no unrelated restores/canaries. This reduced process does not satisfy the repository's standard-publication checklist. Cleanup only disposable source/build/log scratch when the lead is finished.

## Worker close gate

Observed locally with GCC `16.2.1 20260810`:

```bash
g++ -std=c++17 -Wall -Wextra -Werror -fsyntax-only \
    -I .pi/moe-reduce/candidate/ggml/include \
    .pi/moe-reduce/candidate/tests/test-backend-ops.cpp
patch --dry-run --batch -p1 -d .pi/moe-reduce/base \
    < results/20261016-nemotron35-bolt-moe-reduce/moe-reduce.patch
git apply --check --whitespace=error --directory=.pi/moe-reduce/base \
    results/20261016-nemotron35-bolt-moe-reduce/moe-reduce.patch
```

All passed. Copies under ignored `.pi/moe-reduce/` are worker scratch, not deliverables. **These are syntax/patch checks, not a SYCL compilation or numerical/performance result.** OneAPI compilation, 20-case fusion-on/off runs, 112 legacy cases, public API quality/retrieval, and timings remain required before accepting/promoting this candidate.
