# EXPERIMENTAL Bolt NVFP4 reordered GEMV

**Patch artifact only. No GPU/runtime correctness PASS or performance claim.**
Tier: development, potentially quality-sensitive because floating-point reduction
order can change. Not a standard-compliant benchmark publication.

Target: llama.cpp `b6334e37e6ef286fc5091a41b6561b2113d915fb`.
Repository slice base: `74b2e709d3a8fe2416b899265be596406bb5cd04`.
Only `nvfp4-reorder.patch`, this README, and `test_layout.py` are delivered.
No live source, launcher, model, server, or host configuration was changed.

## Scope and layout

This is a deliberately model-scoped experiment, **not general NVFP4 reorder
support**. Initial reorder requires an owned, contiguous, two-dimensional NVFP4
matrix in a regular (not split) SYCL buffer, 16-byte-aligned storage, a width
multiple of 64, and an element count within the existing integer-offset limit.
The weight name must end in `.ssm_in.weight` or `.ssm_out.weight`, and the call
must be dense single-token MUL_MAT with the existing optimization gate enabled.
The pinned Mamba graph uses these immutable weights only as whole-matrix
projections. Ordinary NVFP4 tensors, views and MoE experts remain unreordered.

The official Bolt projections motivating this experiment are `[4096,18560]`
and `[8192,4096]`, across 40 Mamba2 layers. Their estimated weight-traffic share
motivates investigation; it does **not** establish a latency or bandwidth
bottleneck, nor predict a speedup.

For B blocks, storage changes in place from `[d[4], qs[32]] * B` to
`[qs[32] * B][d[4] * B]`, with no extra persistent weight allocation. The existing
reorder staging-buffer lifecycle is reused; failed temporary allocation leaves
the original layout and flag intact. Each lane handles two independently scaled
16-value subblocks, using aligned 16-byte weight/activation loads, the existing
E2M1 SWAR decoder, and the existing integer dot-product primitive.

**NVFP4 scales are UE4M3, not MXFP4 E8M0.** The existing UE4M3 helper already
includes the half factor needed by the doubled integer E2M1 table; this patch
adds no second half factor and does not change scale conversion or accumulation
precision.

A single-token eligibility check alone would be insufficient: the tensor stays
reordered for its allocation lifetime. Therefore the patch also adds:

- NVFP4 reorder metadata, traits, dispatch and layout conversion;
- a reordered GEMV for **each** column of subsequent 2–8-token MMVQ calls;
- layout-aware FP16 and FP32 dequantization for subsequent library-GEMM/prefill
  calls, including the explicit-F32-accumulation test variant.

Multi-column execution deliberately reuses the one-column kernel rather than
adding a second optimized kernel. It may regress small-batch throughput.
At this pin NVFP4 is absent from DMMV, reordered ESIMD, MMQ and fused-dequant-GEMM
support, so those paths cannot consume its reordered bytes. The patch does not
change their gates, Q8 tuning, IQ2 MoE, graph capture, fusion policy, weights,
KV/recurrent-state precision, or runtime defaults.

This restriction relies on the pinned Mamba weight-use contract. It does not
make reordered storage safe for arbitrary tensor readback, copying, aliases,
GET_ROWS, weight mutation or training. Do not broaden the naming/eligibility
gate without implementing and testing those consumers.

## Primary sources affecting the approach

Fresh research was supplied by the lead; the pinned code was also fetched
through HTTPS during this slice:

- [Block layout](https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-common.h): 64 values, four UE4M3 scales, 32 packed bytes.
- [CPU scale convention](https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-impl.h): doubled E2M1 table compensation and zero/sentinel handling.
- [Dot helpers](https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/vecdotq.hpp) and [MMVQ](https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/mmvq.cpp): native NVFP4 subblock mapping, reusable SWAR conversion and reordered GEMV template.
- [Mamba graph](https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/src/models/mamba-base.cpp): whole-matrix `ssm_in`/`ssm_out` consumers.
- [Official model](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF) and [SYCL build documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md): unchanged model and native backend/build path.

## Local checks actually executed

Environment: linked worktree `pi/bolt-nvfp4-reorder`; Linux
`7.2.9-2-cachyos x86_64`; Python `3.14.7`; Git `2.56.0`;
GCC `16.2.1 20260810`. No runtime dependencies installed, binaries built, model
loads, GPU activity, host downloads/builds/writes, or service changes.
The only compiler invocation was local `-fsyntax-only` on the test source.

Pinned files were fetched with Python stdlib `urllib.request.urlopen` from the
URLs above into `.nvfp4-scratch/{base,candidate}`. Six touched SYCL files matched
byte-for-byte SHA-256 output from read-only SSH inspection of
`/home/mike/inference/src/llama.cpp-bolt-b6334e37` on `mike@inference-host`.
SSH used `-o BatchMode=yes -o ConnectTimeout=10`. That directory has no `.git`:
the initial `git rev-parse HEAD` failed, so source comparison used the pinned
HTTPS files instead. Initial discovery also requested nonexistent `reorder.cpp`
and `reorder.hpp` (HTTP 404); reorder implementation is in `ggml-sycl.cpp`.

Commands from the worktree root:

```sh
python3 results/20261016-nemotron35-bolt-nvfp4-reorder/test_layout.py
c++ -std=c++17 -fsyntax-only -I.nvfp4-scratch/include \
  .nvfp4-scratch/candidate/tests/test-backend-ops.cpp
git apply --check --whitespace=error --directory=.nvfp4-scratch/verify \
  results/20261016-nemotron35-bolt-nvfp4-reorder/nvfp4-reorder.patch
git apply --whitespace=error --directory=.nvfp4-scratch/verify \
  results/20261016-nemotron35-bolt-nvfp4-reorder/nvfp4-reorder.patch
git apply --reverse --check --directory=.nvfp4-scratch/verify \
  results/20261016-nemotron35-bolt-nvfp4-reorder/nvfp4-reorder.patch
git apply --reverse --directory=.nvfp4-scratch/verify \
  results/20261016-nemotron35-bolt-nvfp4-reorder/nvfp4-reorder.patch
```

Observed: **4 Python tests passed** (0.398 seconds on the initial run); C++ test
syntax check exited 0 without diagnostics; apply/reverse checks exited 0.
Python byte comparisons confirmed the applied tree equaled the candidate and
the reversed tree equaled the pinned base (11 scratch source files checked).
The syntax check initially lacked `gguf.h`; fetching that pinned public header
resolved it. A deprecated precision-setter warning was fixed using the current
API before the clean final check. The syntax include set was `ggml.h`,
`ggml-alloc.h`, `ggml-backend.h`, `ggml-cpp.h`, and `gguf.h`, all from the same pin.
An exploratory AST scan did not provide a clean SYCL syntax verdict because
unexpanded upstream macros/preprocessor branches generate parser errors; it is
not counted as compilation evidence. Scratch sources were removed before commit.

The Python checks cover all 65,536 four-lane E2M1 code combinations, all packed
bytes and scale-byte codes, subblock/activation mapping, layout round trips,
small multi-column matrix algebra, and real-shape offset/alignment bounds.
They model the proposed algebra; **they do not execute the patched C++** or
validate GPU reduction, vector loads, queue ordering, or FP16 conversion.

## Parent-only runtime acceptance (not executed here)

Use a **separate** pinned source/build; never apply to the live source. Apply
with `git apply --check /absolute/path/nvfp4-reorder.patch` then `git apply`.
Compile with the baseline SYCL build configuration and retain exact configure,
build commands and full diagnostics. No promotion is authorized by this artifact.

The patch includes 56 focused cases in the real `test-backend-ops` executable:

```sh
GGML_SYCL_ENABLE_OPT=1 "$PROTO_BUILD/bin/test-backend-ops" test \
  -b SYCL0 -o MUL_MAT -p 'nvfp4_reorder=1'
GGML_SYCL_ENABLE_OPT=0 "$PROTO_BUILD/bin/test-backend-ops" test \
  -b SYCL0 -o MUL_MAT -p 'nvfp4_reorder=1'
```

Use the backend name reported by the prototype if it differs. Require executed
cases, not skipped/unsupported cases, and retain stdout/stderr and exit status.
`OPT=0` is the original NVFP4 path control in the same binary. Each case uses a
single weight allocation and explicit dependencies for **prefill → decode →
later call**, comparing all three matrix outputs with CPU through the existing
backend harness. This is necessary because ordinary `test_mul_mat` weights are
named `a` and would never exercise this patch. Cases cover odd row tails, widths
64/192, columns 1–9/128, the real two projection dimensions at columns 1/8/9,
all 128 low-seven-bit UE4M3 codes and all E2M1 codes, both default/F32 accumulation,
and non-SSM eligibility controls. They inherit MUL_MAT's existing `5e-4` NMSE
threshold; do not relax it to hide failures. Both FP16-build/default and F32
fallback behavior must be exercised. Any failure blocks a correctness claim.

After numerical checks, run the lead's already-defined real API process:

- Baseline: current pinned Bolt runtime, exact official weights, one B70, 65,536
  context, chosen identical batch/microbatch, Q8 KV, F32 recurrent state, no
  speculation, temperature 0, same template/cache policy; ESIMD=1, FUSION=1,
  VMM=1, GRAPH=0 unless an independently selected baseline is explicitly recorded.
- Candidate intentionally differs only by this patch, not by the parent's Q8
  ESIMD/MMVQ or runtime-overhead experiments.
- Through actual `/v1/chat/completions`, verify READY, arithmetic result 323,
  Python `add`, and exact `BOLT-7319-CEDAR` retrieval at 8,238 rendered tokens.
- Run warmed short 55-input/128-output and cold 8,270-input/128-output repeated
  timings with matched inputs/settings. Retain environment, exact argv, payloads,
  outputs, raw timings, errors/retries and first-reorder/warm-up observations in
  the parent's existing result directory. Separate one-time reorder cost from
  steady-state measurements; check later prefill after decode as well.
- Restore original Qwen through the reviewed controller and stop the isolated
  prototype. Leave persistent launchers unchanged.

Same-input, temperature-zero baseline continuations already vary, so exact
model-output identity is not the sole numerical criterion. Outstanding risks:
SYCL compile/device support, vector-load code generation/alignment in practice,
subgroup reduction drift, actual FP16/F32 fallback correctness, asynchronous
staging lifetime, first-decode cost, and small multi-column/performance regressions.
These require the native tests and real API run, not the scalar model checks.
The repository publication checklist remains incomplete until the parent runs
and records the required benchmark/quality process; no speedup is asserted.


## Subsequent parent development validation

The parent completed host SYCL compilation and dependency checks, then executed **56/56 native cases with OPT=1 and 56/56 with OPT=0**, at the unchanged NMSE gate. Reordered dispatch was observed only in OPT=1. Two independent real API launches with original weights, 65,536 configured context and batch/microbatch 1024 each passed **20/20 fixed checks plus exact 8K retrieval**, then restored unchanged Qwen.

Measured short decode was **39.4662 / 39.4535 tok/s**, versus **28.5548** on the unmodified explicit-default baseline; after 8K it was **38.1589 / 38.1335**, versus **27.8931**. The ~38% short / ~37% long decode gain repeats. Cold prefill remained approximately955–956 tok/s. Short TTFT varied; no general TTFT improvement is asserted. Raw commands, frozen controllers, responses, timing samples, library/source hashes, failures and limitations are in [`../20261016-nemotron35-bolt-sycl/execution.md`](../20261016-nemotron35-bolt-sycl/execution.md) and its adjacent cell directories.

These observations supersede the initial pending-device handoff status above, **not** the publication caveats: development only, actual prompts up to8K, no full-context/BetterBench/comprehensive quality result and no persistent service promotion.
