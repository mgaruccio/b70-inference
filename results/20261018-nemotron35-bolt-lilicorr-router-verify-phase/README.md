# Router small full-output batches — bounded preparation

**Development tier; not a benchmark, proven fidelity fix, or promoted build.**
This slice prepares the runtime patch, build and native gates only. No remote
build, helper import/execution, native/GPU run, public request, launcher change,
or timing run was performed by this worker. The lead owns execution and exact
service restoration; the public adapter is a separate slice.

## Contract and hypothesis

Baseline is `llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-widths-single2-r1`
(`PARENT`), not the rejected unconditional columns8 candidate. The combined
`router-verify-phase.patch` changes exactly:

- `ggml/src/ggml-sycl/ggml-sycl.cpp`: expand the preflight arrays and loops from
  2 to 8 columns. N3..8 additionally requires the existing accumulator tag
  `ggml_get_op_params_i32(dst, 0) == GGML_PREC_F32`. Default/unmarked N3..8 falls
  through to the original dispatch. N2 and every other existing guard remain
  unchanged; dispatch never depends on diagnostics.
- `src/llama-graph.cpp`: request F32 accumulation only for a direct F32
  Nemotron-H-MoE router `MUL_MAT`, with local N3..8, local N equal to graph
  `n_tokens`, and graph `n_outputs == n_tokens`. Check setter success with
  `GGML_ASSERT`. The existing SQRT_SOFTPLUS block remains byte-identical. The
  target SIGMOID model otherwise retains default precision. No output selection,
  global phase state, tensor-name protocol, precision API, or configuration knob
  is changed.
- `tests/test-backend-ops.cpp`: retain the entire old test file, including all
  original 30 `router_single2` fixtures and their 1e-7 tolerance. Add the columns8
  family, explicitly marking N3..8 (including negative name/view/width cases),
  and two separate default-precision M496/N4,N5 `router_prefill` cases.

The API in the supplied `ggml.h` returns `bool` for `ggml_prec_set_acc` and asks
for a **minimum accumulator precision**, not an arbitrary phase hint. F32 is tag
10; MUL_MAT accumulator intent is in op_params[0]. The new graph request is
numerical intent under a bounded operational condition. All-output batches are
**not universally unique to speculative verification**: an explicit all-logits
prompt can satisfy the same condition. Ordinary N4/N5 checkpoint chunks with
zero or one output do not; N1, N2, N9 and larger batches are unchanged by this
new graph condition.

Supplied evidence: the rejected candidate passed all 61 changed-path native
cases at 1e-7 and its unchanged N9 result matched the parent's exact 17,856 bytes.
Its no-draft public A passed 20 quality cases plus the anchor, but short55/128
output diverged at character 197. The retained
`../20261018-nemotron35-bolt-lilicorr-router-columns8/qualification-r3/R8R3-A/server.stderr.log`
(lines 1270–1285) shows prompt55 checkpoint chunks **5,46,4** despite ubatch1024
and `cache_prompt=false`. An N-only split therefore also changes prefill. Actual
public router routes were not captured: this is a candidate explanation, not a
proven root cause or a claim that public fidelity is restored.

## Native gates and CPU regressions

The exact native contracts are:

- `router_single2`: `^router_single2=1,`; **30 cases, 70 route records, 5 split
  cases**. All have accum_prec=0; only N2 splits. The old N3 controls are unsplit
  again. The rejected candidate's 90-route/10-split expectation is not inherited.
- `router_columns8_r3`: **61 cases, 402 routes, 35 splits**. The registration
  family still contains 62 cases and retains its original native tensor names.
  The selector below excludes **only** M496/N9, whose unchanged behavior is
  assessed separately by exact parent/candidate bytes, not a new CPU criterion.
  All N3..8 cases, including unsplit negative controls, require accum_prec=10;
  N1/N2/N9 require 0. Source precision and dynamic precision must both remain 0.
- `router_prefill`: `^router_prefill=1,`; **2 cases, 4 routes, 0 splits**. Names
  are `router_prefill-m496-n4` and `router_prefill-m496-n5`; variable prefixes are
  `router_prefill=1,router_single2=1,`. Both require accum_prec=0.

The 61-case standard C++/ECMAScript regex is unchanged from the previous public
gate (including the comma after `n=9`):

```text
^router_columns8=1,(?!router_single2=1,weight=blk\.48\.ffn_gate_inp\.weight,layout=0,type_a=f32,type_b=f32,m=496,n=9,)
```

`analyze-trace.py` requires the exact numerical OK cases, actual route pairs,
shapes/strides, and precision fields. Raw ANSI is preserved; only exact plain or
colored `OK` is accepted. `test-trace.py` adds CPU parser regressions for marked
N3..8 splits, unmarked N4/N5 and old N3 non-splits, mismatched routes/precision
(in both pair halves), and negative guard cases. Its source test applies the
patch to actual parent files, checks exact hashes, compares the complete runtime
files to explicitly enumerated changes, and proves the whole original test file
survives. Synthetic fixtures do not prove runtime or public fidelity.

## Lead execution (not run here)

Preconditions: case files integrated at the ROOT below, old immutable sources
and case manifests present, Intel 2026 image already installed, existing oneDNN
installation, enough disk reserve, and the inherited exclusive-device/service
restoration procedure available. Do not mutate/rebuild the accepted, parent,
rejected columns8, or existing seeded N9-control trees. Do not toggle CNR or
change the runtime environment to mask a fidelity difference.

CPU preparation on `inference-host`, retaining output in this fresh case:

```bash
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-verify-phase
set -eo pipefail
cd "$ROOT"
python3 -B "$ROOT/test-trace.py" 2>&1 | tee "$ROOT/cpu-native-r1.log"
bash "$ROOT/build-candidate.sh"
python3 -B "$ROOT/run-trace.py" --print-native-command
```

The builder makes a fresh private copy of PARENT at
`/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-verify-phase-r1`,
applies the three-file patch with fuzz=0, and compares exact consumed sources.
It retains the same `intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04`
image, oneDNN, `/src` CMake cache, and disposable-container bootstrap. Build:

```bash
cmake --build /src/build-sycl --config Release -j6 --target llama-server test-backend-ops
```

No GPU is mounted for the build. `build-attempt-r1/{build.log,image.json,exit-code.txt}`,
`runtime-precheck.log`, `post-build-verification.log`, `source-diff.json`, and the
source/binary manifests retain build evidence. Existing attempt/candidate paths
are refused, including failed attempts. The old source and five-artifact pin
contracts are checked read-only before capture. All three protected trees then
receive six-artifact before/after checks in this **new** case. The sixth artifact,
`libllama.so.0.7.0`, contains the model graph and intentionally changes in the new
candidate; SYCL and test binary changes alone are insufficient. Six hashes are
also exposed through `binaries()` for the public adapter's mapped-library proof.

After the CPU tests/build, the lead must:

1. Run the existing private seeded N9r2 test executable against REFERENCE=PARENT
   and this candidate using the same overlay. Require one actual 17,856-byte
   output per arm and literal equality. Retain the inherited CPU numerical
   failure as data; it must not replace the unchanged-path byte criterion.
2. In the inherited exclusive-DRM window, mount candidate `/src:ro`, oneDNN
   `/opt/onednn:ro`, this case `/router-verify-phase:ro`, and a **new** native-r1
   directory `/capture`. Execute the printed `NATIVE_COMMAND` inside the Intel
   container. It runs, in order:

   ```bash
   GGML_SYCL_DEBUG=1 /src/build-sycl/bin/test-backend-ops test -b SYCL0 -o MUL_MAT -p "$selector"
   python3 -B /router-verify-phase/analyze-trace.py --tag "$tag" "/capture/$tag.log"
   ```

   The printed command sets the inherited runtime environment and capture dir,
   selects all three exact selectors above, refuses existing log files, and tees
   raw stdout/stderr into `native-r1/{router_single2,router_columns8_r3,router_prefill}.log`.
   Expected totals are 30/70/5, 61/402/35, and 2/4/0. No changed-path 1e-7
   requirement is relaxed.
3. Run the separate integrated public adapter's unchanged **75 real requests**
   (A/W3/W7: 20 quality + anchor + four identity/fidelity checks each), unchanged
   payloads, frozen plus fresh A controls, and exact actual emitted token IDs.
   A's short55/128 text must return to baseline; actual W3 and W7 must include
   real rejections. Preserve raw requests/responses, IDs, maps, and logs in the
   new qualification attempt, and verify all six mapped artifacts. The adapter's
   `qualification.md` supplies its exact public commands and cleanup.
4. Restore the accepted service exactly after every exclusive window, including
   on failure. Stop only candidate/temporary containers; retain attempts and all
   failures. No timing is eligible until fidelity passes. No promotion is made
   by this preparation slice.

## Fresh primary research supplied before implementation

The researcher completed fresh external research; conclusions used here:

- <https://github.com/ggml-org/llama.cpp/blob/master/src/llama-graph.h> and
  <https://github.com/ggml-org/llama.cpp/blob/master/src/llama-graph.cpp> expose
  graph output counts and ubatch output selection. Use existing per-graph state,
  not a new global phase flag; the supplied actual parent files govern this patch.
- <https://github.com/ggml-org/llama.cpp/pull/23198> distinguishes MTP prompt and
  output processing; small N alone is not a reliable phase discriminator.
- <https://uxlfoundation.github.io/oneMath/spec/domains/blas/gemm_batch.html>
  provides no cross-shape bitwise guarantee.
- <https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/conditional-numerical-reproducibility-control.html>
  addresses repeatability of the same computation, not arbitrary cross-shape
  identity. Do not toggle CNR/environment as part of this fix.

## Worker static close gate

Observed locally in the linked worktree, without helper imports/execution:

- `patch --directory="$WORKTREE/.runtime-applied" --fuzz=0 --batch --forward -p1 < "$CASE/router-verify-phase.patch"`
  applied all three actual parent files without offsets/fuzz. `diff -qr` against
  the independently edited expected source was empty; full-file expected-change
  comparisons and all source/patch hashes passed. Only 82 insertions/8 deletions
  across the three source files; full source copies are not committed.
- Python `ast.parse` and compile-only `compile(..., 'exec')` passed for all three
  Python files and four build heredocs. `bash -n` passed for the builder and the
  AST-extracted native command. None of those helpers was imported or run.
- A `c++ -std=c++17 -fsyntax-only -x c++ -` check against the supplied actual
  `ggml.h` confirmed F32=10, the bool setter return, and the asserted setter call.
  This is a header/API check, **not** a full SYCL or model compilation.
- C++ AST ERROR scans of actual parent and patched files each found 23 inherited
  macro/preprocessor parser nodes, none in the modified regions; llama-graph.cpp
  had none. Full Intel build, 11 CPU tests, native gates, N9 and public fidelity
  are pending lead execution, not claimed green here.
- Static enumeration of the actual registrations and printed selector confirmed
  30/70/5, 61/402/35, and 2/4/0; only M496/N9 is excluded from the 62-case family.
- Whitespace checks use the preceding case's convention: normal files pass
  `git diff --cached --check`; the patch passes that check with
  `-c core.whitespace=-blank-at-eol`. The unconfigured patch check reports only
  its four required blank context lines (`" "`), not added source whitespace.

Temporary actual-source copies are removed before commit. The lead must not
interpret these static results as proof of the candidate causal hypothesis.
