# Bolt R1: Mamba post-scan fusion

**Development candidate; source/patch checks only. Not SYCL-built, GPU-tested, accepted, or timed. No speedup claim.** Standalone on accepted R1; no SSM-cache, MoE-reduction, ReLU, model, precision, state, routing, or launcher changes. The worker performed no remote writes, builds, or GPU work.

Base: upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb` plus accepted R1, copied read-only from `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-r1` on `mike@inference-host` (non-Git source). Deliverables are only this README and `mamba-post.patch`. The patch changes:

- `ggml/src/ggml-sycl/element_wise.cpp`: guarded matcher and one F32 kernel;
- `ggml/src/ggml-sycl/element_wise.hpp`: dispatch declaration;
- `ggml/src/ggml-sycl/ggml-sycl.cpp`: dispatch at CONT;
- `tests/test-backend-ops.cpp`: separately selectable native cases.

## Contract and source checks

Match **CONT(z), optional VIEW(y), MUL(x,D), ADD(y,mul), split SWIGLU(cont,add)** in that order. This is the supplied actual-model trace and the expression built at `src/models/mamba-base.cpp:282–303`; the scan, cache CPY and subsequent grouped RMS norm stay unchanged. Four executable launches become one when eligible (potentially 120 fewer launches per token across 40 layers, not a measured speedup).

- Only F32 `[64,128,nt,1]`, `1 <= nt <= 1024`; D is contiguous `[1,128,1,1]`. Head rows of x/z must be packed, with a non-overlapping, float-aligned token stride. y and all four compute outputs must be contiguous. Actual two-token strides are z `[4,256,74240,148480]`, x `[4,256,40960,81920]`, y/output `[4,256,32768,65536]`, D `[4,4,512,512]` (bytes).
- Exact source positions, `GGML_GLU_OP_SWIGLU`, and GLU parameter 1 equal to zero are required. `ggml_glu_impl` writes those parameters; split CPU/SYCL GLU applies SiLU to src0 and multiplies by src1. The unchanged `op_silu(float)` is `x / (1 + sycl::exp(-x))`.
- Kernel arithmetic is a separately rounded F32 product, then `y + product`, then **the existing `op_silu(z) * sum`**. Scoped `#pragma clang fp contract(off) reassociate(off)` prevents the new MUL/ADD contraction/reassociation; no global flags or tolerances change.
- The real `ggml_can_fuse_subgraph_ext` checks compute/output flags and consumers of CONT/MUL/ADD. Only those compute nodes and the final GLU enter its index list: the optional y VIEW is an external input, not an elided result. It remains usable by other consumers or as an output. No other intervening node is skipped.
- Elided in-place writes, reads from elided intermediates, unsupported layouts/shapes and GLU variants fall back. Integer stride products and pointer-range endpoints are checked. The final output must not overlap **any** live input, including `GGML_OP_NONE` leaves; the top-k helper that skips leaves is not used.
- Existing `GGML_SYCL_ENABLE_FUSION` controls dispatch. Existing debug logging emits `ggml_sycl_op_mamba_post`. The matcher consumes three following nodes, or four with the non-executable y VIEW.

Fresh official sources consulted on 2026-10-16:

- [Pinned SYCL helper](https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/element_wise.hpp): confirms the precise SiLU expression; also checked against the copied R1 header.
- [Pinned element-wise implementation](https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/element_wise.cpp), researched in the lead handoff: SwiGLU passes `op_silu` to unary-gated dispatch. The copied R1 implementation and the remote CPU `ops.cpp`/`vec.cpp` confirm operand order and split/swapped parameter behavior.
- [Clang floating-point pragmas](https://clang.llvm.org/docs/LanguageExtensions.html#extensions-to-specify-floating-point-flags): scoped contraction/reassociation controls. An explicit `-ffp-contract=fast` can override the pragma; do not introduce fast-math/contract flags. Device compilation and the sensitive cases remain mandatory.

Remote reads used `ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host "bash -lc '…'"`. Scratch copies are under ignored `.pi/mamba-post/{base,candidate}/`, not deliverables.

## Native gate — lead-owned, not run by worker

Use a disposable R1 source copy, never modify the accepted source. `SRC`, `BUILD`, `PATCH`, and `RUN` below are absolute disposable candidate-source, configured isolated oneAPI build, patch, and ordinary run-log paths. Keep the accepted R1 build flags/libraries unchanged; save the build output too.

```bash
set -euo pipefail
patch --dry-run --batch -p1 -d "$SRC" < "$PATCH"
patch --batch -p1 -d "$SRC" < "$PATCH"
mkdir -p "$RUN"
cmake --build "$BUILD" --target test-backend-ops llama-server --parallel 2 \
    > "$RUN/build.log" 2>&1
for f in 0 1; do
    env GGML_SYCL_ENABLE_GRAPH=0 GGML_SYCL_ENABLE_ESIMD=1 \
        GGML_SYCL_ENABLE_FUSION="$f" GGML_SYCL_DEBUG=1 \
        "$BUILD/bin/test-backend-ops" test -b SYCL0 -o '^MAMBA_POST$' -j 1 \
        > "$RUN/native-$f.log" 2>&1
    grep -F '34/34 tests passed' "$RUN/native-$f.log"
    test "$(grep -cF 'MAMBA_POST full-array: mismatches=0 max_abs=0 ' "$RUN/native-$f.log" || true)" -eq 34
done
test "$(grep -cF '[SYCL][OP] call ggml_sycl_op_mamba_post:' "$RUN/native-0.log" || true)" -eq 0
test "$(grep -cF '[SYCL][OP] call ggml_sycl_op_mamba_post:' "$RUN/native-1.log" || true)" -eq 11
```

Exactly **34/34 supported passing cases per run**, with exactly **0/11** off/on dispatch entries:

- Eight eligible cases: nt **1, 2, 17, 1024**, each with deterministic ordinary and cancellation/FMA-sensitive values, using the actual x/z token strides and nonzero view offsets. nt=17 covers an irregular token count; fixed 64×128 planes are multiples of the 256-item block, so this deliberately narrow kernel has no partial element-block case.
- Three more eligible cases at nt=17: y VIEW emitted before CONT (no intervening VIEW), external y output, external y consumer. A nonconstant precomputed source with a tail models the scan allocation; the scan/cache kernels themselves are not part of this test or patch.
- Twenty-three fallbacks: extra consumers and output flags on each of CONT/MUL/ADD; in-place MUL/ADD; output aliasing a direct leaf z or an offset z view; padded x/z head strides, y stride, or D stride; ReGLU, reversed split operands, swapped parameter; intervening independent compute; head dimension 32; two sequences; reads of elided CONT as x or y; nt=1025.

Every case contains an independent same-backend control forced unfused by output flags on **its** CONT/MUL/ADD, not the eligible candidate. Flattened candidate/control arrays must match bit-for-bit, with maximum absolute delta zero. The existing **1e-7 CPU NMSE** threshold is unchanged; protected side outputs and sentinels are checked. Consumer SCALE receives CONT(view), respecting the CPU layout requirement. `ggml_set_output(y)` recursively marks its external source, not any elided compute node.

Also run the lead's unchanged **112-case legacy selector**: expected total **180** native passes (112 + 34 off + 34 on). Zero selected cases, unsupported/skipped selected cases, wrong marker counts, or success only with fusion disabled do not pass the gate. Confirm the same dispatch entry occurs in the actual model's decode and prefill graph before serving validation; synthetic eligibility alone is insufficient.

## Public-boundary gate and comparison — lead-owned, pending

**Development tier**, accepted R1 versus R1 plus this patch alone, same B70 and original Nemotron/Bolt weights. No stacking with the SSM/MoE/ReLU experiments. The lead's existing real HTTP API process is the acceptance journey, not the native test substitute:

1. Build/load the standalone candidate in the existing isolated runtime. Keep context **65536**, batch/ubatch **1024**, q8 KV, **F32 recurrent state**, ESIMD **1**, graphs **0**, greedy sampling and seed **42**, and all other accepted settings unchanged. Keep debug tracing out of timed requests; preserve a separate dispatch trace. Do not change persistent launchers or run unrelated restores/canaries.
2. Send the existing **fixed 20 API quality requests** through the real server, using the existing fixtures/checker. Require **20/20 before any timing or exact 8K retrieval**. A known intermittent baseline `drawer` → `rewrad` failure is still a failure: retain it, do not relax the gate or retry the identical failed variant into acceptance.
3. Only after that gate passes, run the existing exact 8K retrieval fixture (expected `BOLT-7319-CEDAR`) and matched **55/8270 input-token → 128 output-token** API requests, unchanged cache policy, one warm-up and three measured repetitions. Check actual input/output counts and cache status, preserve complete outputs, and report medians, not best runs. Pair/interleave with fresh R1 when comparing small deltas.
4. Retain exact launch/request commands, environment, build/native/dispatch logs, API results, raw timing results and every failure under the lead's ordinary run directory. Cleanup only disposable source/build/log scratch when finished; do not promote the candidate without all gates.

No public-boundary run or performance comparison was performed here. The standard-publication checklist in `BENCHMARKING_STANDARDS.md` is incomplete; neither the patch nor future reduced development timings are standard-compliant publishable results.

## Worker close gate

Observed locally with GCC `16.2.1 20260810`:

```bash
g++ -std=c++17 -Wall -Wextra -Werror -fsyntax-only \
    -I .pi/mamba-post/candidate/ggml/include \
    .pi/mamba-post/candidate/tests/test-backend-ops.cpp
patch --dry-run --batch -p1 -d .pi/mamba-post/base \
    < results/20261016-nemotron35-bolt-mamba-post/mamba-post.patch
git apply --check --whitespace=error --directory=.pi/mamba-post/base \
    results/20261016-nemotron35-bolt-mamba-post/mamba-post.patch
git apply --reverse --check --whitespace=error --directory=.pi/mamba-post/candidate \
    results/20261016-nemotron35-bolt-mamba-post/mamba-post.patch
```

All passed. These establish test-translation-unit syntax and clean forward/reverse patch applicability, **not SYCL compilation, numerical correctness, model fusion activation, or performance**. A raw repository `git diff --cached --check` reports four single-space blank-context rows in the unified patch; these are patch syntax, not source trailing whitespace. The source-addition `git apply --whitespace=error` checks and README whitespace check pass. Only the two requested artifact files are committed. Remaining risks are oneAPI device code generation, exact same-backend arithmetic, model-allocator eligibility, and the mandatory native/API gates; the lead owns those checks.
