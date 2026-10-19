# Verify-singletons public qualification r2 — preparation, not a result

**Development-only, quality-sensitive, and unpromoted.** This integrates the
router-callsite public/N9 lifecycle with the reviewed singleton source and native
contracts. Runtime qualification is still pending; this is not a timing run or a
persistent launcher change.

## Baseline and fixed interfaces

- Parent source: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-callsite-r1`.
- Future candidate: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2`.
- Remote result root: `/home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-verify-singletons`.
- Public output: `qualification-r1/`; cells: `VSB-A`, `VSB-W3`, `VSB-W7`.
- N9 output: `n9/run-r1/`; cells: `VSN9B-PARENT`, `VSN9B-CANDIDATE`.
- The later trace slice uses `trace-short55-r1`, `VSTB-A`, and `VSTB-W3`; its public
  P/C/K functions consume this port unchanged.
- The seeded N9 runner/control-source tree remains read-only. Its paired-equivalence
  schema, 17,856-byte marker, F32 controls, and fixed `1e-7` tolerance are unchanged.

## Native contract and fail-closed boundary

Public and N9 gates share the exact source/build proof. It requires successful
build and dependency records, the protected callsite parent, only the four C++
files in `check-source.py`, and the exact three changed artifacts among all six:
`libggml-sycl.so.0`, `libllama.so.0.7.0`, and `test-backend-ops`.

The wider path requires F32 plus private word-15 annotation `0x4e535631`.
N1/N2 remain unchanged. Public startup retains all inherited F32 gates and adds
82 Q8 and 34 IQ2 native cases, including 70 required byte-exact comparisons.
The exact new command suffix comes from `native.py`; only its raw-log destination
and stdout redirection change for public startup. Both raw logs are copied before
requests and after restoration, parsed strictly, and checked byte-for-byte for drift.
A read-only `/verify-singletons` helper mount is the only additional mount.

Missing build, source, binary, native, fresh N9, or CPU-frozen proof fails closed.
The public freeze includes the actual native helpers, tests, patch, builder,
source checker, and trace implementation; no unconditional preparation stub remains.

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
ROOT=/home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-verify-singletons
python3 -B "$ROOT/n9/run-control.py" --lead-run-control
python3 -B "$ROOT/n9/run-control.py" --lead-after-restore
```

Then run the CPU gate. It verifies the successful build and restored fresh N9
pair, checks generated shell syntax, runs isolated lifecycle/parser/frame fixtures,
and freezes the exact code, references, source, and binaries before public capture:

```bash
bash "$ROOT/cpu-public-gate.sh"
```

Only after those checks are green, run capture and analysis as separate invocations;
never overwrite a prior attempt:

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-verify-singletons
test ! -e "$ROOT/qualification-r1"
test ! -e "$ROOT/public-capture-r1.log"
python3 -B "$ROOT/public-gate.py" --lead-run-qualification 2>&1 \
  | tee "$ROOT/public-capture-r1.log"
```

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-verify-singletons
test ! -e "$ROOT/public-analysis-r1.log"
python3 -B "$ROOT/public-gate.py" --lead-after-restore 2>&1 \
  | tee "$ROOT/public-analysis-r1.log"
```

After public restoration, the trace worker runs its existing `trace-short55-r1`
journey with `VSTB-A` and `VSTB-W3`, including the full 55-token prompt prefix, unchanged
partial-prefill checks, intended routes, raw F32 controls, and same-role reproduction.
A trace run cannot substitute for missing native or public proof.

## Public journey and expected results

Exercise exactly **25 real generation requests per public role** (75 total): the
inherited 20 quality cases, retrieval anchor, short/long ordinary cases with
55-/8270-token prompts, and short/long 128-token actual-ID diagnostics. Use the real
`/apply-template`, `/tokenize`, and `/v1/chat/completions` boundaries, temperature 0,
seed 42, and `cache_prompt=false`; preserve natural-stop, fixed-length, and diagnostic
payloads exactly.

Fresh `VSB-A` must strictly equal the frozen A results. `VSB-W3` and `VSB-W7` must
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

Host CPU checks passed: compiled guard stubs, patch-mutation rejections, native
parser regressions, and all ten first-frame fixtures. Native build, fresh N9,
public qualification, and trace outcomes must be reported from their real artifacts;
source review and CPU fixtures alone do not establish runtime success.
