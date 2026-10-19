# Phase-scoped Q8/IQ2 singleton reuse — native experiment preparation

**Development tier; unqualified hypothesis, not a root cause or fidelity fix.**
No native SYCL build, GPU/model/HTTP run, timing or promotion was done by this worker.
Baseline is the exact callsite parent below; the only intentional difference is
`verify-singletons.patch`. Model, launch, sampling, tolerances and serving configuration
are unchanged. The lead's preceding callsite runs were A 25/25, W3 23/25, W7 21/25;
all natural-stop functional cases passed. Those are **parent**, not candidate, results.
The proven F32 router callsite and its existing singleton backend path are untouched.

- Parent: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-callsite-r1`
- NEW candidate: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2`
- NEW remote artifacts: `/home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-verify-singletons`

## Exact intervention / adapter interfaces

Only these source files may differ from the parent (exact hashes in `source-hashes.json`):

1. `ggml/src/ggml-sycl/ggml-sycl.cpp`: retain the existing `*_single2`/`latent_dmmv2`
   names and n=2 behavior; additionally admit **marked n=3..8**. Q8 retains every
   ESIMD/type/name/device/contiguity/alignment/reorder/F16 guard and the actual
   once-per-column DMMV callback (the callback itself ignores ncols). IQ2 uses
   fixed capacity-eight descriptor arrays, preserves parent strides, preflights
   **all** columns before enqueueing, then invokes the existing n=1 helper n times.
   A failed post-preflight dispatch aborts, never partially dispatches then falls back.
   Wider paths check bounded products, buffer ranges/device ownership and nonaliasing
   destination spans; IQ2's padded ID stride multiplication is checked against SIZE_MAX/n.
2. `src/llama-graph.cpp`: mark direct Q8 attention and IQ2 expert-up/down producers
   in `build_lora_mm` / `build_lora_mm_id`, including the precomputed-router path.
   Require the Nemotron MoE architecture, local token extent 3..8 equal to graph
   `n_tokens`, and `n_outputs == n_tokens`. Exact operands, F32 activations/results,
   supported weight types/names, unchanged Hadamard input identity, and direct op
   identity are required. Scale/LoRA wrappers and cached-expert slots fail closed.
   Target `blk.<il>` names require `il < n_layer`; the appended MTP layer is excluded.
   DFlash/other architectures receive no new marks.
3. `src/models/nemotron-h.cpp`: mark the two actual direct latent Q8 producers with
   the same phase/type/identity guard, **not** their unrelated router. The latent
   input/weight operands are unchanged; no LoRA/rotation wrapper is introduced.
4. `tests/test-backend-ops.cpp`: add isolated fixtures only. All inherited tests,
   F32 router/N9 controls and their tolerances remain byte-for-byte source-identical.

The existing `ggml_prec_set_acc(..., GGML_PREC_F32)` request remains in word 0
(route `accum_prec=10`). **Independently**, target producers annotate existing
`op_params[15]` with `0x4e535631`; both conditions are required for n=3..8.
F32 alone is not target identity: the parent already requests it for GLM4/GLM4_MOE/
JAIS2 attention output (`llama-graph.cpp:3218–3222,3305–3309`). Those operations
remain unannotated and cannot enter the widened Q8/IQ2 singleton paths.
Activation precision remains unchanged (`src1_prec=0` under the frozen settings).
No API/header, CLI/env flag, precision-bit, tensor-name or tensor-flag encoding is
added. N1/N2 ignore the annotation entirely; unannotated wider operations, N9 and rejected layouts
keep their original dispatch. Partial-output checkpoint prefill stays unmarked;
a small **full-output prompt can satisfy this predicate** (not a universal phase classifier).

For the lead's native/public trace adapter:

- Q8: existing `DMMV_ESIMD` record retains the **wide** src1/dst shape/name and
  accum_prec=10; its existing callback loop makes n singleton launches with explicit
  offsets. Do not expect n new Q8 route records. Native byte equality checks the result.
- IQ2: n existing `MOE_SINGLE_MMVQ_PLAIN_Q8_1` records share the wide destination
  name, but `src1_ne2=dst_ne2=ids_ne1=1`; parent strides still encode n columns.
- F32 router records/interfaces are unchanged. No public/trace/N9 adapter is copied
  here; the lead must retain its newer Env/bind-order fixes and fresh RCA2-style names.
- `check-source.check(parent, candidate, patch)` remains callable and read-only;
  `candidate=None` checks parent/patch identity. The builder separately checks the
  complete non-build source maps, not just these four files.

## Exact r2 correction

R1 reproduced six `q8-p{0..5}-n4-m1-l2` `MMVQ_REORDERED` NaN cases; the required
Q8 byte comparisons were zero-mismatch failures, so this is a fresh qualification
experiment rather than a claimed fix. Relative to the r1 patch, the only candidate
source change is the non-contiguous RHS staging call in `ggml-sycl.cpp`: it now uses
`quantize_row_q8_1_sycl<quantize_f>` instead of the hardcoded `quantize_q8_1`.
The contiguous call already selected `quantize_f`; copy, quantization, and compute
remain on the same stream. No route, weight, threshold, or test-case change is made.
The four consumed C++ source files and expected three binary deltas remain unchanged
in scope; only the ggml-sycl candidate hash changes relative to r1.

## Private operation-word audit

This is a pinned-source internal experiment, not a new serving contract. Word 15
(byte offset 60) is unused **for MUL_MAT/MUL_MAT_ID**, not globally unused:

- `ggml/include/ggml.h:226,700` allocates 64 bytes / 16 int32 words. New tensor
  allocation zeroes all words (`ggml/src/ggml.c:1835`). `ggml_mul_mat:3362` and
  `ggml_mul_mat_id:3413` create fresh results and set only the operation/operands.
- The matmul parameter writers use word 0 for accumulation (`ggml_prec_set_acc:3303`,
  `ggml_mul_mat_set_prec:3379`), word 1 for the Hadamard hint (`ggml_mul_mat_set_hint:3389`),
  and word 3 for source-1 precision (`ggml_prec_set_src:3327`, only idx=1 accepted).
  These writers do not copy/clear or reinterpret word 15. `src/` has only three
  pre-existing direct parameter reads, all Hadamard word 1 in `llama-context.cpp`.
- SYCL matmul consumers read accumulation word 0, hint word 1 and activation word 3
  (`ggml-sycl.cpp:3296,5679`, `common.hpp:114`); CPU matmul reads hint word 1
  (`ggml-cpu/ggml-cpu.c:1262`), not word 15. The source-wide word-15 scan, including
  pointer casts/aliases, finds only RoPE `n_offs` producers/consumers; RoPE is unchanged.
- Backend graph duplication copies the operation **and all 64 parameter bytes**
  (`ggml-backend.cpp:2278–2280`). IQ2 lowering copies complete output descriptors
  before changing only pointers/extents; Q8 dispatch retains the original descriptor.
  The compiled checks verify accumulation/scope preservation in IQ2 singleton copies.

The annotated producer/consumer/test constants are checked together: **3 producer
sites, 2 guards, 2 fixture classes**, word 15 / `0x4e535631`. The source audit and
unchanged-core hashes are retained in `worker-checks.log`; re-audit on any source
refresh rather than treating this slot as a portable ABI reservation.

## Native fixtures and CPU-only command printing

```bash
python3 -B test-native.py -v
python3 -B native.py --print-native-command > native-command-r2.sh
# LEAD only, after copying this package to its NEW remote root:
PARENT=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-callsite-r1
CANDIDATE=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2
bash build-candidate.sh
python3 -B check-source.py --candidate "$CANDIDATE"
python3 -B test-guards.py --parent "$PARENT" --candidate "$CANDIDATE"
```

The printed **inner Bash** command first runs the exact inherited F32 router command
(30/70/5, 61/402/35, 2/4/0 cases/records/splits), then new selectors
`^verify_singletons=1,case=q8-` and `^verify_singletons=1,case=iq2-` against the real
`/src/build-sycl/bin/test-backend-ops test -b SYCL0`. Mount NEW candidate `/src:ro`,
oneDNN `/opt/onednn:ro`, this package `/verify-singletons:ro`, inherited phase package
`/router-verify-phase:ro`, and a **fresh** writable native attempt directory `/capture`.
Lead owns container/device exclusivity and restoration. Do not run inside interactive Pi.

- **82 Q8 cases**: original 78 retained (latent down/up and attention q/k/v/output
  n=1..9, unmarked n4, marked n4 offset/padded activations, full attention reshape
  marked3..8); add F32 attention-output n4/n8 with absent/malformed scope markers.
- **34 IQ2 cases**: original 26 retained (both expert shapes n=1..9, unmarked n4/n8,
  marked n4 strided weights/padded token layouts); add both F32 expert shapes at
  n4/n8 with absent/malformed markers. Cyclic IDs exercise all 16 experts by n2.
- Existing fixture IDs are unchanged. New negative IDs append `-a0` (absent) or
  `-a2` (malformed) to `*-pP-nN-m1-l0`; these still request F32. The parser requires
  Q8 `MMVQ_REORDERED` and IQ2 original grouped/per-expert fallback, never the widened
  singleton path. The marker itself is not a new route-log field.
- Each graph computes an actual wide op and separate default-precision n=1 ops on
  identical weights, activation columns and IDs, then concatenates the singleton
  outputs. `BOLT_SINGLETON case=... bytes=... required=... mismatches=...` reports a
  direct backend byte comparison. **54 Q8 + 16 IQ2 cases require zero byte differences**;
  negative controls report differences without imposing singleton equivalence.
  Existing 5e-4 CPU-reference NMSE is also required, never relaxed. Native failure,
  skip, duplicate/missing case, missing/wrong route/mark/stride, or required byte
  mismatch fails `native.py --kind q8|iq2 LOG`. Synthetic parser inputs are only unit tests.
- N9 is deliberately marked in the new backend guard fixture; real graph N9 is
  unmarked. This does **not** replace the inherited seeded F32 M496/N9 unchanged-path
  comparison. Lead reruns that exact control against its reference and NEW candidate
  with fresh paths; its prior 17,856-byte match is not inherited. Preserve its separate
  CPU-tolerance error as an error, not a newly relaxed or relabeled native pass.

## Builder protections

Narrowly derived from the callsite builder. It protects accepted, widths-single2,
columns8, verify-phase **and callsite** ancestors, complete source maps and all six
actual captured binaries before/after and on failure. It refuses existing candidate,
attempt and container names; applies with fuzz=0; retains **4 GiB + 1 GiB** reserve,
`/src`, read-only oneDNN, pinned Intel2026 image
`sha256:aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0`,
disposable-container dependency bootstrap, no-GPU/offline ldd precheck, and
`cmake --build /src/build-sycl --config Release -j6 --target llama-server test-backend-ops`.
No image pull, host install, service changes or CMake reconfiguration. The bootstrap
still uses apt/network; this is not a bit-reproducible dependency claim.

Actual hashes of all six candidate/parent binaries are saved **before** asserting
exactly `libggml-sycl.so.0`, `libllama.so.0.7.0`, `test-backend-ops` changed. The other
three must match. No candidate binary hashes are guessed. Logs/manifests stay under
NEW root: `build-attempt-r1/`, `source-check-{parent,candidate}.log`, `source-diff.json`,
`runtime-precheck.log`, `post-build-verification.log`, `{parent,candidate}-binaries.json`.

## Research basis

Fresh official research supplied for this r2 experiment (2026-10-19):

- <https://github.khronos.org/SYCL_Reference/iface/usm_allocations.html>: USM
  allocation APIs do not guarantee initialized contents; no correctness argument may
  rely on fresh allocation contents, and backing data must be explicitly initialized.
- <https://github.khronos.org/SYCL_Reference/iface/queue.html>: in-order dependency
  guarantees are within the same queue; the existing copy, selected quantizer, and
  compute remain submitted to the same stream.
- <https://github.com/oneapi-src/SYCLomatic/blob/4893685d6f2f1c6126a1962be65dfa7688bb4162/clang/runtime/dpct-rt/include/memory.hpp.inc>:
  asynchronous pitched copies preserve byte width/pitches and require completion or
  a valid dependency before consumers; the existing strided-copy contract is retained.

## Required lead E2E — defined, NOT executed here

Run only on `inference-host` with the existing B70/xe/Intel stack, outside interactive
Pi, using the read-only SSH contract (`BatchMode=yes`, `ConnectTimeout=15`,
`ControlMaster=no`, `ControlPath=none`, `ForwardAgent=no`) and a managed background
job for long runtime commands. Preserve the original service and independently restore
it before any analysis. The ordered journey is:

1. Reproduce the targeted six parent/r1 failing Q8 cases with the same test binary;
   preserve the NaN/byte-mismatch failures and do not weaken a gate.
2. Build the fresh r2 candidate in the frozen CPU-only SYCL path, checking all source
   protections and all six binary hashes. Only `libggml-sycl.so.0`, `libllama.so.0.7.0`,
   and `test-backend-ops` may change relative to the callsite parent.
3. Run the corrected six native cases, then all 116 singleton cases (82 Q8 + 34 IQ2)
   and every inherited F32 control unchanged. Require the 70 byte-exact cases and
   preserve any failures; do not claim parity from synthetic parser fixtures.
4. Run the fresh `VSN9B-PARENT`/`VSN9B-CANDIDATE` pair and its independent restoration
   check with the unchanged 17,856-byte marker/control contract.
5. Run the exact CPU freeze gate, including source/reference/binary immutability and
   generated-shell/parser/guard checks, before public capture.
6. Run the public boundary exactly 25 requests for each `VSB-A`, `VSB-W3`, and
   `VSB-W7` (75 total) through `/apply-template`, `/tokenize`, and
   `/v1/chat/completions`; restore the original service, then run independent analysis.
7. Only when public capture is complete, run 46 trace requests (23 each for
   `VSTB-A`/`VSTB-W3`), restore independently, and analyze the raw first frame.

Retain raw commands, source/binary/native/N9/PID/container/request/response/trace and
restoration artifacts under the new root. No timing or promotion is permitted. This
worker ran no native build, GPU/model/API lifecycle, N9, public, or trace execution;
remote CPU/runtime validation remains lead-owned and pending.

## Worker close gate observed

No r1 logs or runtime results were copied into this fresh root. The r2 local close gate
records the exact commands and observed results in the lead handoff: patch/source scope,
strided-quantizer regression, guard/parser fixtures, Python and shell syntax, and fuzz=0
forward/reverse patch checks. No full SYCL compilation, native runtime, GPU/API lifecycle,
N9, public, or trace execution was performed here; those remain lead-owned and pending.
