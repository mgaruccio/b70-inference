# Verify-singletons public qualification r1 — preparation, not a result

**Development-only, quality-sensitive, and unpromoted.** This is a bounded mechanical
port of the router-callsite public/N9 lifecycle. It is not a completed runtime adapter,
a native result, a timing run, or a persistent launcher change.

## Baseline and fixed interfaces

- Parent source: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-callsite-r1`.
- Future candidate: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r1`.
- Remote result root: `/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-verify-singletons`.
- Public output: `qualification-r1/`; cells: `VSA-A`, `VSA-W3`, `VSA-W7`.
- N9 output: `n9/run-r1/`; cells: `VSN9-PARENT`, `VSN9-CANDIDATE`.
- The later trace slice uses `trace-short55-r1`, `VST-A`, and `VST-W3`; its public
  P/C/K functions consume this port unchanged.
- The seeded N9 runner/control-source tree remains read-only. Its paired-equivalence
  schema, 17,856-byte marker, F32 controls, and fixed `1e-7` tolerance are unchanged.

The exact singleton source delta, allowed binary manifest, native command, and native
parser are still lead-owned. No model-only whitelist or old phase source proof is a
claim about this candidate.

## Explicit native handoff and fail-closed boundary

Both public and N9 `candidate_proof()` currently raise exactly:

```text
RuntimeError('LEAD_NATIVE_INTEGRATION pending: singleton source/build proof')
```

No live launch can pass that missing proof. The lead must replace the stop only after
the native worker supplies the exact `check-source.check(parent, candidate, patch)`
contract, source delta, binary manifest, build evidence, command, and parser. In
particular, do not run the old phase `check_consumed_sources(PARENT, 'candidate')`
against the callsite parent and assume it proves the singleton change.

The preparation files mark the bounded insertion seams:

- public `make_create`: add the exact singleton native check after the inherited F32
  checks, without changing the existing launch delta;
- public `check_startup`: add the exact singleton marker/parser while retaining every
  inherited F32 startup gate;
- public native preservation: extend the raw artifact list only with exact native
  names supplied by the lead, copying bytes without normalization;
- public tested-file list: add the exact native helper/parser/test/build/source-check
  files once their names are known. The current CPU freeze must not silently omit them.

## Preserved lifecycle and validators

The port retains the latest lifecycle fixtures: unique environment-key mapping,
order-independent bind multiplicity, full Config/HostConfig/mount validation, actual
PID records without rewriting, raw startup/native log immutability, startup-failure
preservation, Docker's atomic published-port conflict cleanup, unchanged 25 request
jobs per role, real emitted prompt IDs, frozen-versus-fresh A comparison, exercised
and rejected-draft requirements, independent restoration before raw analysis, and
`timing_permitted=false` / `promotion_permitted=false`. No tolerance, retry, timing,
or promotion path is added.

Inherited F32 native controls remain mandatory. They are not replaced by guessed
singleton tests. CPU fixtures install process/network/DRM audit guards and use only
synthetic startup/container inputs plus read-only recorded parent inputs.

## Research basis

This mechanical slice relies on the already-recorded official research for SYCL
queue/kernel capture and the public ggml API. It makes no new native-contract claim;
the current pinned source and the native worker's verified contract govern integration.

## Preconditions and ordered end-to-end process (lead only)

Run on `inference-host` with the existing B70/xe/Intel stack, outside interactive Pi.
The Fish login shell requires an explicit Bash block, for example:

```text
ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no \
  -o ControlPath=none -o ForwardAgent=no inference-host \
  'bash -lc "bash -s"'
```

Use a managed background job for long live commands. Do not install packages, run
Docker/model/API/GPU work from this worker, or write remote evidence during this
preparation. Before any live command, the lead must have the exact native contract,
pinned private build, source/binary proof, and the fresh N9 recipe.

First, in a fresh result root, run the independent N9 window and restoration check:

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-verify-singletons
python3 -B "$ROOT/n9/run-control.py" --lead-run-control
python3 -B "$ROOT/n9/run-control.py" --lead-after-restore
```

Then run the CPU preparation. It may run isolated syntax/lifecycle fixtures, but it
must report native build, fresh N9, and public qualification as dependent/not run
until the lead-owned proof exists:

```bash
bash "$ROOT/cpu-public-gate.sh"
```

Only after those checks are green, run capture and analysis as separate invocations;
never overwrite a prior attempt:

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-verify-singletons
test ! -e "$ROOT/qualification-r1"
test ! -e "$ROOT/public-capture-r1.log"
python3 -B "$ROOT/public-gate.py" --lead-run-qualification 2>&1 \
  | tee "$ROOT/public-capture-r1.log"
```

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-verify-singletons
test ! -e "$ROOT/public-analysis-r1.log"
python3 -B "$ROOT/public-gate.py" --lead-after-restore 2>&1 \
  | tee "$ROOT/public-analysis-r1.log"
```

After public restoration, the trace worker runs its existing `trace-short55-r1`
journey with `VST-A` and `VST-W3`, including the full 55-request prefix, unchanged
partial-prefill checks, intended routes, raw F32 controls, and same-role reproduction.
A trace run cannot substitute for missing native or public proof.

## Public journey and expected results

Exercise exactly **25 real generation requests per public role** (75 total): the
inherited 20 quality cases, retrieval anchor, short/long ordinary cases with
55-/8270-token prompts, and short/long 128-token actual-ID diagnostics. Use the real
`/apply-template`, `/tokenize`, and `/v1/chat/completions` boundaries, temperature 0,
seed 42, and `cache_prompt=false`; preserve natural-stop, fixed-length, and diagnostic
payloads exactly.

Fresh `VSA-A` must strictly equal the frozen A results. `VSA-W3` and `VSA-W7` must
exercise 21 functional natural-stop cases, 2 fixed-length cases, 2 diagnostics, and
real rejected-draft evidence. Compare prompt IDs, text, finish, usage, and actual
emitted diagnostic IDs/pieces; do not retokenize output or compare probabilities.
Record a valid wider mismatch while completing all 25 requests without retry; reject
that width. Fresh-A, transport, corrupt-input, native, source/isolation, and
restoration failures abort. An empty qualified-width list is an investigation result,
not evidence that the divergence is fixed.

## Evidence, cleanup, and current status

Retain raw build/source/binary/native/N9/PID/container/argv/startup/restoration files,
all template/tokenize/request/response artifacts, proposal/rejection counts, and
analysis files. Preserve stopped experiments and failures. Cleanup must restore the
exact original container identity/config/HostConfig/canonical mounts, source/binaries,
hardware, sole DRM ownership, health, and READY smoke; do not normalize records, retry,
delete evidence, time, or promote results.

At this commit there is no new native build, N9 pair, public baseline, or trace result.
Only syntax and isolated inherited CPU/helper tests are appropriate. This preparation
claims no singleton runtime success; the lead owns native integration, all integration,
and real execution.
