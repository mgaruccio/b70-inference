# Router-columns8 public qualification — preparation, not a result

**Development tier, quality-sensitive. No timing, promotion, or persistent launcher change.**
This adapter reuses the frozen width-sweep Runtime/capture and contracts read-only.
The seven native/build files from `24b07d6f` are unchanged. CPU fixtures, native
execution and public qualification have **not** been run by the preparation worker.

## Hypothesis, baseline and intentional differences

The parent W3/W7 passed 20 quality checks plus the anchor, but short/long ordinary
and actual-ID checks diverged (first actual-ID differences 116/14). The same-width
max3/min4 suppression control retained state3, nout4 and block8: 504 real drafter
generation calls / 11930.648 ms, zero proposals/acceptances, all four requests exact
against frozen and fresh A. This implicates active verification/rollback, not a
proven router cause. Native success alone cannot demonstrate the public fix.

The native patch changes F32 per-column router dispatch from n=2 to n=2..8 only;
all shape/type/layout/name/width guards remain. This qualification uses that new
candidate for fresh **R8R2-A (no draft), R8R2-W3 (max3/min0), R8R2-W7 (max7/min0)**.
Roles map to the old A/W3/W7 argv. Target/drafter, weights, thread count, context,
cache types, sampler, image, environment and hardware caps otherwise stay fixed.
Physical names are `bolt-lilicorr-width-sweep-r1-r8r2-{a,w3,w7}`. Old case names are
not reused, removed or treated as collisions. Only new names and a new
`qualification-r2/` are permitted.

Attempt r1 is preserved: it stopped at ANSI-colored backend-status parsing before model startup or public requests, and restoration was independently verified. Attempt r2 changes that parser check and the attempt namespace only; the runtime binaries, payloads, numerical tolerances and route requirements are unchanged. Replay the preserved raw 30-case log and rerun both CPU suites before this attempt.

The launch delta is exactly name, parent `/src:ro` -> candidate `/src:ro`, the
additional case `/router-columns8:ro` bind, and native validation shell code.
Full original/derived argv and all positional differences are retained. Actual
PID1 records go unchanged to the old role's checker; actual container mount
source and read-only status must identify the **new** candidate, not its parent.

## Predeclared process and environment (lead only)

Research is already recorded in [README.md](README.md): fresh oneMKL GEMM and
oneDNN strict-accumulation/determinism primary sources do not establish bitwise
cross-shape identity. Those conclusions require this real public gate; no new
research, infrastructure, or native equality claim is added by the adapter.

Run on **inference-host**, existing B70/xe/Intel 2026 stack and frozen model/runtime
installation, outside interactive Pi. Preconditions: successful new native build,
unchanged accepted/parent full source maps and five binaries, candidate manifest,
healthy exact original `nemotron-bolt-probe`, original as sole DRM owner, matching
hardware/caps, at least 4 GiB reserve plus inherited 1 GiB evidence allowance,
all old code/references available, and no new cell/output collisions. No install
or GPU work is performed by the CPU runner. Candidate binary hashes are taken
from the real build, never guessed.

Exact commands, submitted by the lead as separate managed remote jobs:

```bash
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-columns8
# First: build must already have completed successfully. Do NOT rebuild/reuse a failed attempt.
set -euo pipefail
(set -o noclobber; : > "$ROOT/cpu-r2.log")
python3 -B "$ROOT/test-trace.py" 2>&1 | tee -a "$ROOT/cpu-r2.log"
(set -o noclobber; : > "$ROOT/native-replay-r1.log")
python3 -B "$ROOT/analyze-trace.py" --tag router_single2 \
  "$ROOT/qualification-r1/R8-A/restored-router-single2-native.log" 2>&1 | tee -a "$ROOT/native-replay-r1.log"
bash "$ROOT/cpu-public-gate.sh"
```

Only after CPU success and authorization of the exclusive live window:

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-columns8
test ! -e "$ROOT/public-capture-r2.log"
python3 -B "$ROOT/public-gate.py" --lead-run-qualification 2>&1 | tee "$ROOT/public-capture-r2.log"
```

Only after that command exits **captured and restored**, a separate invocation:

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-columns8
test ! -e "$ROOT/public-analysis-r2.log"
python3 -B "$ROOT/public-gate.py" --lead-after-restore 2>&1 | tee "$ROOT/public-analysis-r2.log"
```

The CPU runner records exact commands/environment and isolated attempt logs under
`worker-scratch/public-cpu-*/`; requires successful build exit/precheck markers;
checks all actual launch shells with `bash -n`; runs CPU fixtures with a
no-process/no-network/no-DRM audit hook; and freezes code/references/full candidate
source/five binaries before and after. `unittest.main(exit=False)` deliberately
returns so `preparation/public-tested-code.json` is written **only on success**.
The freeze includes all four new files, native helpers/tests/patch/source pins,
build proof files, inherited code/references and consumed prior qualification files.
Any subsequent edit or source/binary drift blocks live use; rerun the CPU gate.

## Native and public boundaries

Every new cell runs all inherited numerical/route gates. The old 30-case router
invocation/count/marker stay intact, but its Python validator is replaced by the
new analyzer requiring three singleton pairs for old n=3. After that gate, the
62-case `^router_columns8=1,` test runs under native-only `GGML_SYCL_DEBUG=1` and
the existing temporary capture flag. Its stdout/stderr go only to its own raw
file (not the legacy aggregate parser); the new PASS marker follows successful
analysis and precedes the model marker. The original cleanup still unsets capture
and LLAMA_TRACE and sets debug0 before exec; tensor tracing remains off.

Startup is collected directly both with inherited Docker timestamps and separately
**without** Docker timestamps. No timestamp/ANSI/PID rewriting is performed. The
new analyzer checks the unaltered old-30 startup segment and both actual native
files, including all expected routes. Native files are copied again from stopped
containers, including failed starts when available; copy failures cannot prevent
original restoration. Synthetic CPU startup/container inputs are never live proof.

The real user journey is **75 generation requests**, sequential 25 per arm:
20 frozen natural-stop quality prompts (max256), one natural-stop retrieval
anchor (historical prompt8238), short55 and long8270 ordinary streaming requests,
and two corresponding untimed actual-emission-ID diagnostics (128, ignore EOS).
All payloads remain frozen: temperature0, seed42, cache_prompt=false. Each request
also exercises real `/apply-template` and `/tokenize`, then `/v1/chat/completions`.
Expected: exact prompt IDs, text/finish/usage against frozen **and fresh** A;
diagnostics additionally require exact actual emission IDs and pieces, never
retokenized text or probabilities. Natural functional passes must be 21, fixed
length passes 2, actual diagnostics 2, with real rejected proposals for each active
arm. Native tolerances/route identity do not imply raw F32 or probability equality.

A valid candidate fidelity/functional mismatch rejects the width but continues
all 25 without retry. Valid diagnostic/rejection behavior is recorded even on
mismatched output. Fresh A failure, invalid/corrupt public input, malformed HTTP,
native/isolation/source failure aborts and restores. Independent analysis rereads
raw request bytes, HTTP framing/responses, public prompt responses, PID/startup,
actual read-only mounts, native logs and all 25 cases, retaining every mismatch
and first differing actual-ID ordinal. It verifies fresh live restoration and the
CPU-frozen candidate proof before classification. Passing widths are merely listed
as qualified for a separately declared future experiment: timing remains false.

## Evidence and cleanup

Retain build manifests/logs, CPU attempt files, both top-level CLI logs, and the
whole `qualification-r2/`: declaration/tested freeze; environment/hardware/source
proofs; full argv/diffs/container/PID/library/startup records; raw HTTP request and
response files; public template/tokenize evidence; per-cell qualification reports;
ready/restored raw 30/62 native files; cumulative proposal/acceptance statistics;
restoration checks/smoke, independent-restoration.json, and analysis.json. Failures
and stopped candidate containers remain preserved, with no automatic deletion or
reuse. The native raw files are `ready-` and `restored-` plus
`router-single2-native.log` / `router-columns8-native.log` in each cell directory.

The inherited finally/signal handling stops new cells and restores the original
exact ID, Config, HostConfig and complete canonical Mounts; health, natural-stop
READY smoke, sole live service/DRM owner, hardware and accepted/parent sources/five
binaries are checked. The adapter additionally rechecks the full new candidate at
preflight, every end and independent restoration and checks preserved cells are
stopped. No changes to the original service/source/runtime are intended.

## Preparation verification and pending work

Observed static close gate in the linked preparation worktree:

- `bash -n results/20261018-nemotron35-bolt-lilicorr-router-columns8/cpu-public-gate.sh`: passed.
- Python `ast.parse` plus compile-only checks for both new Python files and the
  CPU runner's inline Python: passed. No helper module was imported or executed.
- `bash -n` on the AST-extracted `NEW_NATIVE` string: passed (shell not executed).
- `git diff --cached --check`: passed; staged scope is only these four new files.

The 12 remote CPU fixture tests are **prepared, not executed**. In particular,
`test_valid_mismatches_keep_25_and_diagnostic_rejection_behavior` covers the
regression contract: valid actual-ID mismatches must not erase observed diagnostic
or rejected-proposal behavior; baseline/transport failures still abort.

No local CPU tests, builds, SSH, Docker, GPU or API execution occurred. Remote CPU,
native/public capture, independent analysis and their observed results are
**pending lead execution**. The publication checklist is incomplete: this is
preparation, not standard-compliant performance evidence, a proven router fix,
or permission to time/promote a failed width.
