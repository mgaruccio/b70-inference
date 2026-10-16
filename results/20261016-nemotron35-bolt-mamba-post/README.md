# Bolt R1: Mamba post-scan fusion

**Development-only result: volatile variant passes native/API gates and measures +0.84%/+0.59% versus fresh R1; it remains slower than standalone MoE. Earlier variants failed parity or were performance-neutral. Combining with MoE failed the API gate and was not timed.** Standalone on accepted R1; no SSM-cache, MoE-reduction, ReLU, model, precision, state, routing, or launcher changes. The worker performed no remote writes, builds, or GPU work.

Base: upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb` plus accepted R1, copied read-only from `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-r1` on `mike@inference-host` (non-Git source). Deliverables are only this README and `mamba-post.patch`. The patch changes:

- `ggml/src/ggml-sycl/element_wise.cpp`: guarded matcher and one F32 kernel;
- `ggml/src/ggml-sycl/element_wise.hpp`: dispatch declaration;
- `ggml/src/ggml-sycl/ggml-sycl.cpp`: dispatch at CONT;
- `tests/test-backend-ops.cpp`: separately selectable native cases.

## Contract and source checks

Match **CONT(z), optional VIEW(y), MUL(x,D), ADD(y,mul), split SWIGLU(cont,add)** in that order. This is the supplied actual-model trace and the expression built at `src/models/mamba-base.cpp:282–303`; the scan, cache CPY and subsequent grouped RMS norm stay unchanged. Fuse only CONT/MUL/ADD, materializing both CONT and ADD outputs; run the original SwiGLU kernel afterward. Three launches become one, so the complete four-launch chain becomes two (potentially 80 fewer launches per token across 40 layers, not a measured speedup).

- Only F32 `[64,128,nt,1]`, `1 <= nt <= 1024`; D is contiguous `[1,128,1,1]`. Head rows of x/z must be packed, with a non-overlapping, float-aligned token stride. y and all four compute outputs must be contiguous. Actual two-token strides are z `[4,256,74240,148480]`, x `[4,256,40960,81920]`, y/output `[4,256,32768,65536]`, D `[4,4,512,512]` (bytes).
- Exact source positions, `GGML_GLU_OP_SWIGLU`, and GLU parameter 1 equal to zero are required. `ggml_glu_impl` writes those parameters; split CPU/SYCL GLU applies SiLU to src0 and multiplies by src1. The unchanged `op_silu(float)` is `x / (1 + sycl::exp(-x))`.
- Kernel arithmetic is a separately rounded F32 product, then `y + product`, plus a bit-preserving copy of z. **SwiGLU remains in its original kernel/compiler context**, not inlined into this kernel. A volatile F32 product store/load preserves multiply rounding before the add in the tested cases, without the explicit Intel rounding calls used by the parity-passing but neutral variant. The scoped FP pragma remains but proved insufficient alone. The volatile replacement passed all37 exact native cases in both modes. No global flags or tolerances change.
- The real `ggml_can_fuse_subgraph_ext` checks compute/output flags and consumers of CONT/MUL/ADD. Only those compute nodes and the final GLU enter its index list: the optional y VIEW is an external input, not an elided result. It remains usable by other consumers or as an output. No other intervening node is skipped.
- Elided in-place writes, reads from elided intermediates, unsupported layouts/shapes and GLU variants fall back. Integer stride products and pointer-range endpoints are checked. CONT/ADD destinations cannot overlap each other. CONT may reuse only its own packed z input; ADD may reuse a packed, same-shape input at the identical byte range, with all per-coordinate reads preceding writes. Shifted/strided overlaps and repeated D overlap remain rejected, including leaf inputs. The later, unchanged GLU may reuse its normal buffers after the fused launch completes.
- Existing `GGML_SYCL_ENABLE_FUSION` controls dispatch. Existing debug logging emits `ggml_sycl_op_mamba_post`. The matcher consumes two following nodes, or three with the non-executable y VIEW.

Fresh official sources consulted on 2026-10-16:

- [Pinned SYCL helper](https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/element_wise.hpp): confirms the precise SiLU expression; also checked against the copied R1 header.
- [Pinned element-wise implementation](https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/element_wise.cpp), researched in the lead handoff: SwiGLU passes `op_silu` to unary-gated dispatch. The copied R1 implementation and the remote CPU `ops.cpp`/`vec.cpp` confirm operand order and split/swapped parameter behavior.
- [Clang floating-point pragmas](https://clang.llvm.org/docs/LanguageExtensions.html#extensions-to-specify-floating-point-flags): scoped contraction/reassociation controls. An explicit `-ffp-contract=fast` can override the pragma; do not introduce fast-math/contract flags. Device compilation and the sensitive cases remain mandatory.
- [Intel rounding intrinsics](https://raw.githubusercontent.com/intel/llvm/sycl/sycl/include/sycl/ext/intel/math/imf_rounding_math.hpp): explicit nearest-even `fmul_rn`/`fadd_rn`; definitions also verified in the installed oneAPI2026.0 header. Used by the retained RN experiment; the current volatile variant no longer requires that header.

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
    grep -F '37/37 tests passed' "$RUN/native-$f.log"
    test "$(grep -cF 'MAMBA_POST full-array: mismatches=0 max_abs=0 ' "$RUN/native-$f.log" || true)" -eq 37
done
test "$(grep -cF '[SYCL][OP] call ggml_sycl_op_mamba_post:' "$RUN/native-0.log" || true)" -eq 0
test "$(grep -cF '[SYCL][OP] call ggml_sycl_op_mamba_post:' "$RUN/native-1.log" || true)" -eq 16
```

Exactly **37/37 supported passing cases per run**, with exactly **0/16** off/on dispatch entries:

- Eight eligible cases: nt **1, 2, 17, 1024**, each with deterministic ordinary and cancellation/FMA-sensitive values, using the actual x/z token strides and nonzero view offsets. nt=17 covers an irregular token count; fixed 64×128 planes are multiples of the 256-item block, so this deliberately narrow kernel has no partial element-block case.
- Three more eligible cases at nt=17: y VIEW emitted before CONT (no intervening VIEW), external y output, external y consumer. A nonconstant precomputed source with a tail models the scan allocation; the scan/cache kernels themselves are not part of this test or patch.
- Five additional eligible cases: late eager-GLU output aliasing a leaf/offset z view, plus allocator-style exact-coordinate reuse of ADD←y (nt17), ADD←x (nt1), and CONT←z (nt1). The reuse tests redirect candidate data/buffer pointers only after initialization; control inputs remain independent.
- Twenty-one fallbacks: extra consumers and output flags on each of CONT/MUL/ADD; in-place MUL/ADD; padded x/z head strides, y stride, or D stride; ReGLU, reversed split operands, swapped parameter; intervening independent compute; head dimension 32; two sequences; reads of elided CONT as x or y; nt=1025.

Every case contains an independent same-backend control forced unfused by output flags on **its** CONT/MUL/ADD, not the eligible candidate. Flattened candidate/control arrays must match bit-for-bit, with maximum absolute delta zero. The existing **1e-7 CPU NMSE** threshold is unchanged; protected side outputs and sentinels are checked. Consumer SCALE receives CONT(view), respecting the CPU layout requirement. `ggml_set_output(y)` recursively marks its external source, not any elided compute node.

Also run the lead's unchanged **112-case legacy selector**: expected total **186** native passes (112 + 37 off + 37 on). Zero selected cases, unsupported/skipped selected cases, wrong marker counts, or success only with fusion disabled do not pass the gate. Confirm the same dispatch entry occurs in the actual model's decode and prefill graph before serving validation; synthetic eligibility alone is insufficient.

## Public-boundary gate and comparison — lead-owned

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

## Lead experiments

Evidence is under `../20261016-nemotron35-bolt-sycl/`; all runs use the existing isolated oneAPI2026.0/B70 runtime and unchanged fixed model/API settings.

- Initial full fusion compiled, but `mamba-post-r1-dispatch` passed34/34 only with fusion off; all11 eligible fused cases failed bitwise parity (maximum delta1.90734863e-6). No server/API run.
- Narrowing to CONT/MUL/ADD while retaining the original SwiGLU kernel still failed the11 eligible cases (`mamba-post-r1-partial-dispatch`, maximum delta9.53674316e-7). No tolerance relaxation.
- Explicit `fmul_rn`/`fadd_rn` passed180 native cases (112+34+34), but `mamba-post-r1-rounded-dispatch` activated zero fusions in the model. Rejection diagnostics identified allocator buffer reuse, not a shape mismatch.
- Exact-coordinate reuse support plus three new allocator-style tests passed **186 native cases** (112+37+37), with0/16 entries and37 full-array bitwise matches per mode. `mamba-post-r1-reuse-dispatch` records **873 actual-model fused starts**, including **840 one-token starts=40×21 decode evaluations**, plus33 prefill starts. All1,280 SwiGLU calls remain separate;407 unsafe prefill overlaps fall back. This removes80 launches per decode token, not a measured speedup.
- RN variant `mamba-post-r1-validated`: **20/20 API checks and exact8K retrieval pass**, all six measured requests match fresh R1 exactly (55/8270 input tokens,128 uncached outputs). Median **39.40859 short /38.11326 long tok/s**, versus39.44741/38.13176 R1: **−0.10%/−0.05%**, effectively neutral. Long prefill956.12120 tok/s, TTFT8.74141s. **Not selected as a speed improvement.**
- Volatile variant `mamba-post-r1-volatile`: **186 native cases,20/20 API and exact8K retrieval pass**. Median **39.77873/38.35798 tok/s**, **+0.84%/+0.59%** versus freshR1; all six requests match. LongTTFT8.70489s. It does not beat standalone MoE40.18873/38.84961.
- MoE+volatile post stack: `moe-post-r1-dispatch` passed **226 native cases** and smoke; actual decode shows both fusions840 times each, with expected prefill fallbacks. `moe-post-r1-validated` failed **19/20 API**, returning `rewrad` instead of `reward`. **No timing/retrieval, no promotion, no identical retry.** Neither SSM-cache nor ReLU was included.
- Focused read-only review found no blocking defect in rounding, exact-coordinate aliases, or pointer-override tests. Volatile rounding is toolchain-specific, not a portable IEEE fence. ADD←z follows the same injective rule but lacks a dedicated native case; CONT←z is synthetic rather than a gallocr in-place path. The completed MoE+volatile dispatch run independently confirms the final kernel's actual-model activation, despite its later API rejection.
