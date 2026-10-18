# Phase-scoped Q8/IQ2 singleton reuse — native experiment preparation

**Development tier; unqualified hypothesis, not a root cause or fidelity fix.**
No native SYCL build, GPU/model/HTTP run, timing or promotion was done by this worker.
Baseline is the exact callsite parent below; the only intentional difference is
`verify-singletons.patch`. Model, launch, sampling, tolerances and serving configuration
are unchanged. The lead's preceding callsite runs were A 25/25, W3 23/25, W7 21/25;
all natural-stop functional cases passed. Those are **parent**, not candidate, results.
The proven F32 router callsite and its existing singleton backend path are untouched.

- Parent: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-callsite-r1`
- NEW candidate: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r1`
- NEW remote artifacts: `/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-verify-singletons`

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

The mark is the **existing** `ggml_prec_set_acc(..., GGML_PREC_F32)` request for
MUL_MAT/MUL_MAT_ID: `op_params[0]` / route `accum_prec=10`; activation precision is
unchanged (`src1_prec=0` under the frozen settings). No API/header, CLI/env flag or
precision policy is added. N1/N2, unmarked wider operations, N9 and rejected layouts
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

## Native fixtures and CPU-only command printing

```bash
python3 -B test-native.py -v
python3 -B native.py --print-native-command > native-command-r1.sh
# LEAD only, after copying this package to its NEW remote root:
PARENT=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-callsite-r1
CANDIDATE=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r1
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

- **78 Q8 cases**: latent down/up and attention q/k/v/output at n=1..9; unmarked n4;
  marked n4 offset/padded activation controls; full attention reshape at marked3..8.
- **26 IQ2 cases**: both expert shapes at n=1..9; unmarked n4/n8; marked n4 strided
  weight and padded-token-layout controls. Cyclic IDs exercise all 16 experts by n2.
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

## Research and lifetime reasoning

Fresh external research supplied by the lead's researcher (2026-10-18):

- <https://github.khronos.org/SYCL_Reference/iface/defining-kernels.html>: kernel
  parameters are copy-captured; do not retain references to stack descriptors.
- <https://github.khronos.org/SYCL_Reference/iface/queue.html>: in-order queues or
  explicit dependencies are necessary; no implicit ordering across queues.
- <https://github.khronos.org/SYCL_Reference/iface/usm_allocations.html>: USM free
  does not wait; correct-context parent/scratch storage must outlive queued uses.
- <https://github.com/ggml-org/ggml/blob/master/include/ggml.h>: views carry strides;
  consult the actual source's precision API rather than deprecated MUL_MAT-only setters.

Actual parent verification: `common.hpp:411-419` uses the device default queue;
`dpct/helper.hpp:731,793-797` makes that queue in-order. `mmvq.cpp:3050-3067` captures
pointers/scalars by value; its IQ2 switch returns true after launch. Descriptor copies
are host-local only; no parent allocation or pool lifetime changes and no new queue
are introduced. Existing same-queue pool reuse orders scratch reads before later
quantization. These facts **do not establish bitwise parity**; native/public runs must.

## Required lead E2E — defined, NOT executed here

Use inference-host's isolated existing B70 runtime and unchanged frozen model,
launch/API settings; preserve original-service ownership and independent restoration.
After the new build, require actual native positive/negative, inherited F32 and N9 gates.
Then use fresh A/W3/W7 names/output for the inherited **25 real HTTP requests per arm**
(21 natural-stop functional, two fixed-length, two real-ID diagnostics). Compare to
frozen and fresh A, require rejected drafts actually exercised, retain responses/IDs.

Next run fresh A/W3 **23-request-per-role** tracing: require same-role reproduction,
the complete 55-token prefix/common emission1, unchanged partial-output prefill bytes,
and actual target first full-output n4 frame with both the existing F32 router and
intended Q8/IQ2 routes. Bind actual sampler row0 to API emission2 and compare raw F32
rows, not unmatched operands. Retain raw HTTP, IDs, PID/environment/container/native
logs, source/binary manifests and restoration evidence under the NEW artifact root.
Always restore the original service and preserve stopped experiments, including on
failure. No timing/promotion unless fidelity is established; CPU checks are not E2E proof.

## Worker close gate observed

`worker-checks.log` retains exact local commands/environment (Python3.14.7, GCC16.2.1):
source/hash scope plus 11 rejecting patch mutations; **313 compiled actual-guard cases**
with CPU stubs and UBSan (including late preflight failure/no enqueue, pointer offsets,
parent strides, alias/overflow/layout/phase/architecture/MTP controls); three parser
unit tests including 42 rejecting mutations; Python/build/embedded-shell syntax;
GCC C++17 syntax-only checks of both graph files and test-backend-ops; fuzz=0 forward
and reverse patch dry-runs. All passed. No full SYCL compilation was substituted.

`worker-remote-checks.log`: read-only actual-parent `git apply --check --verbose -`
and fuzz=0 patch dry-run passed without offsets; the **exact builder protection
heredoc** passed against five source trees / 30 binary hashes. SSH used BatchMode=yes,
ControlMaster=no, ControlPath=none, ForwardAgent=no and `bash -lc` (host login is Fish).
No remote source/file/service mutation occurred. Temporary fetched source/header
copies were removed from this package before commit. Native build, real routes/byte
parity, candidate binary hashes and the complete public E2E remain lead-owned and pending.
