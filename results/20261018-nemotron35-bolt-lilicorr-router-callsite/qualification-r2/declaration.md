# Callsite candidate public qualification r1 — preparation, not a result

**Development, quality-sensitive; no timing, promotion, or persistent launcher change.**
This is the existing phase public adapter retargeted to the actual new candidate, not a
new harness. Frozen width-sweep controller/contracts remain read-only, including the
same `FidelityFailure` / `InfraFailure` class identities.

## Baseline and intentional differences

- Patch parent: `llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-verify-phase-r1`.
- Candidate: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-callsite-r1`.
- Output root: `/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-callsite`.
- Source delta is **only `src/models/nemotron-h.cpp`**; binary delta is **only
  `libllama.so.0.7.0`** of the six pinned artifacts. Backend and native executable
  are unchanged. `check-source.py` verifies the full inherited predicate and exact
  insertion against the pinned parent; no new protected/source-hashes manifests exist.
- Arms: **RCA2-A**, **RCA2-W3**, **RCA2-W7**; physical names are `C.PREFIX + cell.lower()`.
  Output is `qualification-r2/`. No old cell/output name is reused.

Public launch argv is derived from frozen A/W3/W7. Only the name, `/src:ro` candidate
mount, inherited native validation shell, and read-only **old phase analyzer** bind
(`/router-verify-phase:ro`) differ. PID argv, target/drafter, model, sampler, context,
cache types, threads, image, hardware caps, and diagnostic-off environment stay fixed.
Full inspected Config/HostConfig/mounts are checked without rewriting raw records.
Only `OomKillDisable=None` versus `False` is allowed; `True` is rejected. Docker's
inherited atomic published-port reservation is used, not a standalone socket bind.

Fresh official research supplied by the lead on 2026-10-18:
- <https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/nemotron-h.cpp>
  confirms the direct precomputed router path bypassing the generic guard. The
  private pinned parent, not changing upstream code, governs the insertion.
- <https://docs.docker.com/engine/containers/resource_constraints/> and
  <https://raw.githubusercontent.com/moby/moby/master/daemon/daemon_unix.go> explain
  enabled-by-default OOM killing and unsupported unset/false handling. They do not
  justify relaxing any other resource field or modifying inspected evidence.

Native success does **not** prove effective marking at this callsite. Hadamard/LoRA
membership is not inferred from source: retain the strict tensor identity predicate
and require the real route checks in [TRACE.md](TRACE.md).

## Preconditions and ordered E2E (lead only)

Run on **inference-host**, existing B70/xe/Intel 2026 stack, outside interactive Pi.
The login shell is Fish: submit each Bash block using explicit `bash -lc`, e.g.
`ssh -o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no inference-host 'bash -lc "bash -s"'` with the block on stdin.
Use managed background jobs for live/long commands; do not run them in the Pi shell.

Require the successful current build, exact source/binary protection, fresh inherited
native run against **NEW `/src:ro`**, and fresh N9 control before CPU freezing.
Build evidence is `source-diff.json`, `{parent,candidate}-binaries.json`,
`build-attempt-r1/{build.log,image.json,exit-code.txt}`, `runtime-precheck.log` marker
`BOLT_VERIFY_CALLSITE_DEPENDENCIES_PASSED`, and `post-build-verification.log` marker
`BOLT_VERIFY_CALLSITE_BUILD_PASSED`. Do not rebuild or overwrite attempts just to run
this adapter. Native command/controls remain the inherited phase process described in
[README.md](README.md): 30/70/5, 61/402/35, and 2/4/0 cases/route-records/splits.
Every public arm reruns all three native gates before model startup as well.

The independent N9 slice supplies `n9/run-control.py`, `n9/test-control.py`,
`n9/run-r1/`, and cells `RCSN9-PARENT` / `RCSN9-CANDIDATE`. It reuses the already-built
`router-n9-control-r2` seeded executable and unchanged old `n9-equivalence.py` schema.
Public `check_n9_equivalence()` imports its pure **`check_current_equivalence()`**.
That function must reread fresh raw **17,856-byte** equality, live source/backend/
runner hashes, and independent restoration, returning the existing proof dictionary.
Missing dependency/proof means **pending**, never a stub or old `run-r3` hash binding.
The inherited N9 CPU error (about 1.40e-7 versus 1e-7) remains recorded data; it is not
relabeled a numerical pass or treated as the sole blocker.

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-callsite
# First follow n9's own CPU preparation instructions, then its separate live window:
python3 -B "$ROOT/n9/run-control.py" --lead-run-control
python3 -B "$ROOT/n9/run-control.py" --lead-after-restore
bash "$ROOT/cpu-public-gate.sh"
```

CPU preparation uses fresh `worker-scratch/public-cpu-*/`, validates generated shells,
copies only inherited native fixture code into NEW scratch, and runs native, public,
and first-frame fixtures. No process/network/DRM access is permitted during public/
frame fixtures. It freezes actual consumed code, references, old protection helpers,
new build artifacts, fresh N9 raw evidence and runner bindings before/after. Only then
is `preparation/public-tested-code.json` written. Any consumed-code/source/binary/
evidence drift requires rerunning this gate. Trace fixtures run later against the
completed new public controls; they cannot use old phase output as a substitute.

For each live window require the exact healthy `nemotron-bolt-probe` original,
original as sole service/DRM owner, unchanged hardware/caps, no name/output collisions,
and inherited **4 GiB reserve plus 1 GiB evidence allowance**. Do not evict another
workload or remove evidence to manufacture capacity. Run capture and independent
analysis as **separate invocations**:

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-callsite
test ! -e "$ROOT/qualification-r2"
test ! -e "$ROOT/public-capture-r1.log"
python3 -B "$ROOT/public-gate.py" --lead-run-qualification 2>&1 | tee "$ROOT/public-capture-r1.log"
```

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-callsite
test ! -e "$ROOT/public-analysis-r1.log"
python3 -B "$ROOT/public-gate.py" --lead-after-restore 2>&1 | tee "$ROOT/public-analysis-r1.log"
```

## Real journey and expected results

Exactly **75 real generation requests**, 25 per role: 20 QA, retrieval anchor, short
and long ordinary requests (55-/8270-token prompts), plus short/long 128-token actual-ID
diagnostics. Frozen payloads use the real `/apply-template`, `/tokenize`, and
`/v1/chat/completions` boundaries, temperature 0, seed 42, `cache_prompt=false`; natural
QA/anchor stops and fixed-length ordinary/diagnostic behavior remain unchanged.

Fresh A must strictly equal frozen A. W3/W7 must meet 21 functional natural-stop cases,
2 fixed-length cases, 2 diagnostics, and real rejected-draft evidence. Compare prompt
IDs, text, finish and usage to fresh/frozen A, and actual diagnostic emission IDs and
pieces, never retokenized output or probabilities. A valid wider mismatch is recorded
and the full 25 requests continue without retry; it rejects that width. Fresh-A,
transport, corrupt-input, native, source/isolation, or restoration failures abort.
Independent analysis rereads raw public bytes and native/runtime evidence after live
restoration. `qualified_widths=[]` is a result to investigate, not a reason to stop
work or claim the bug fixed. Continue the bounded trace journey even if widths fail.

## Evidence and cleanup

Retain all build/CPU logs, N9 proof/raw/inputs/restoration, all `qualification-r2/`
requests/responses and template/tokenize evidence, actual PID/library/startup records,
argv deltas, three ready/restored native files, proposal/rejection statistics, and
`analysis.json` / `independent-restoration.json`. No actual records are normalized.
Inherited finally/signal cleanup stops and preserves experimental containers and
restores the exact original ID, Config, HostConfig, complete canonical mounts,
source/binaries, hardware, sole DRM ownership, health and natural-stop READY smoke.
Keep failures/stopped containers; no deletion, retries, timing or promotion.

This package prepares commands and CPU regressions. Fresh native/N9, CPU freeze,
75-request public results and 46-request trace results remain lead-owned pending work
until those commands actually succeed; no wide-fidelity fix is claimed here.
