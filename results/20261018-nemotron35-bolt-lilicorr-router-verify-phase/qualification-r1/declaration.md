# Router verify-phase public qualification r1 — preparation, not a result

**Development tier, quality-sensitive. No timing, promotion, or persistent launcher change.**

This adapter reuses the frozen width-sweep Runtime/capture and contracts read-only. It
prepares the public boundary for the router verify-phase hypothesis; it does not claim
that the hypothesis fixes the no-draft fidelity regression. CPU fixtures, native
execution, and public qualification are lead-owned remote work.

## Hypothesis, baseline, and intentional differences

The parent W3/W7 history retains the no-draft fidelity regression: quality and anchor
checks passed, while short/long ordinary and actual-emission-ID checks diverged. Native
router success alone cannot demonstrate a public fix.

The candidate applies the bounded per-operation F32 accumulator hint only to the target
Nemotron full-output batch (n_outputs == n_tokens) for widths 3..8. N=2 retains the
old behavior; no global flags, configuration API, output-selection behavior, or
request payload changes are allowed. The public arms are **RVP-A** (no draft),
**RVP-W3** (max3/min0), and **RVP-W7** (max7/min0), mapped to the frozen A/W3/W7
roles. Physical names are `C.PREFIX + cell.lower()` and must be unique. Previous case names
are not reused or treated as valid cells.

The launch delta is exactly the fresh cell name, parent `/src:ro` to candidate
`/src:ro`, the additional case `/router-verify-phase:ro` bind, and native validation
shell code. All original/derived argv positions, actual PID arguments, target/drafter,
weights, thread count, context, cache types, sampler, image, environment, and hardware
caps remain unchanged. Actual PID records are passed unchanged to the inherited role
checker; the actual `/src` mount and read-only status must identify the new candidate.

The source scope is exactly:

- `ggml/src/ggml-sycl/ggml-sycl.cpp`
- `tests/test-backend-ops.cpp`
- `src/llama-graph.cpp`

The candidate proof pins all six candidate artifacts, including `libllama.so.0.7.0`.
Accepted and parent full source maps and the inherited five-artifact C contracts remain
read-only controls.

## Predeclared end-to-end process and environment

Run on **inference-host**, using the existing B70/xe/Intel 2026 stack and frozen
model/runtime installation, outside interactive Pi. Preconditions are a successful
candidate native build with build/dependency markers, the paired N=9 control proof,
unchanged accepted/parent sources, all six candidate binaries, a healthy exact
`nemotron-bolt-probe` with the original as sole DRM owner, matching hardware/caps,
at least 4 GiB reserve plus the inherited 1 GiB evidence allowance, all frozen
references, and no cell/output collisions. The CPU runner never installs software or
executes GPU/container/API work.

The lead submits these managed remote jobs in order:

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-verify-phase
N9=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-n9-control
# Build/core preparation is completed before this adapter's live window.
bash "$ROOT/build-candidate.sh"
python3 -B "$N9/test-control.py"
python3 -B "$N9/run-control.py" --lead-run-control
python3 -B "$N9/run-control.py" --lead-after-restore
bash "$ROOT/cpu-public-gate.sh"
```

The N=9 controller is an independent control run under the exclusive window. Its new `run-r3/` proof uses
`R8N9R3-PARENT` and `R8N9R3-CANDIDATE`, the previously built private
`router-n9-control-r2` test tree, the old columns8 test source plus the seeded
`expected_test_source` transformation, and the current six-artifact parent/candidate
backend hashes. It must contain `baseline_equivalent=true`, one exact
`BOLT_N9_OUTPUT_HEX=` marker per arm, 17,856 equal bytes, two generic-DNN route records
with default `accum_prec=0`, unchanged seed inputs, and an independent restoration with
`errors=[]`. The N=9 CPU accuracy error is retained as data, not used as the gate; only
the exact paired SYCL bytes establish equivalence. This adapter validates that proof without
importing `run-control.py`.

The CPU preparation records exact commands and environment under
`worker-scratch/public-cpu-*/`, runs `bash -n` on every generated native/server shell,
runs the native parser tests separately before the public fixture suite, installs a
no-process/no-network/no-DRM audit hook for public fixtures, and freezes code,
references, full source maps, and all six candidate binaries before and after. It
writes `preparation/public-tested-code.json` only after every test and freeze check
passes. Any later edit or source/binary drift invalidates the preparation and requires
rerunning the CPU gate.

After CPU success and authorization of the exclusive live window:

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-verify-phase
test ! -e "$ROOT/public-capture-r1.log"
python3 -B "$ROOT/public-gate.py" --lead-run-qualification 2>&1 | tee "$ROOT/public-capture-r1.log"
```

Only after that command reports captured and restored, use a separate invocation:

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-verify-phase
test ! -e "$ROOT/public-analysis-r1.log"
python3 -B "$ROOT/public-gate.py" --lead-after-restore 2>&1 | tee "$ROOT/public-analysis-r1.log"
```

## Native and public boundaries

Each new cell runs the inherited numerical gates and all three native contracts. The
original `router_single2` selector remains N=2-only: its current exact colored-status
parser is used, N=3 remains unmarked/unsplit like the parent, and every route has
`accum_prec=0`. The adapter does not assume a historical route-record total.

The verify-phase selector is exactly
`^router_columns8=1,(?!router_single2=1,weight=blk.48.ffn_gate_inp.weight,layout=0,type_a=f32,type_b=f32,m=496,n=9,)`.
Its analyzer requires 61 cases, 402 route records, and 35 split cases. Marked n=3..8
positive fixtures carry `accum_prec=10`; n=1, n=2, n=9, negative view/name, and
unsupported-width fixtures stay unsplit/default. The exact M=496,n=9 control is excluded
only from this selector and is covered by the paired byte proof, never silently omitted.

The prefill selector is exactly `^router_prefill=1,`; its analyzer requires two
M=496,n=4/5 cases, four route records, zero split cases, and default `accum_prec=0`.
Prefill negative controls are executed and checked. All changed-path native accuracy
checks remain 1e-7. Each native output is retained as its own raw file:
`originalrouter-single2-native.log` (30), `router-verify-phase-native.log` (61), and
`router-prefill-native.log` (2). The new marker
`BOLT_VERIFY_ROUTER_VERIFY_PHASE_NATIVE_PASSED` follows successful analysis and occurs
before the model marker. Native logs are copied from ready and stopped/restored
containers, including failed starts when available; copy errors are recorded and never
skip original restoration. No ANSI, timestamp, PID, or byte rewriting is permitted.

The public journey is **75 real generation requests**, sequentially 25 per arm, on the
same prompt/template/tokenization path: 20 frozen natural-stop quality prompts, one
natural-stop retrieval anchor, short ordinary streaming length 55, long ordinary
streaming length 8270, and two untimed actual-emission-ID diagnostics at 128 tokens
with EOS ignored. Every request uses the frozen template and `/apply-template`,
`/tokenize`, and `/v1/chat/completions` path, with temperature 0, seed 42, and
`cache_prompt=false`. Natural functional passes must total 21, fixed-length passes 2,
and both diagnostics must exercise actual rejection behavior. Results compare exact
prompt IDs, text, finish, and usage against frozen and fresh A; diagnostics additionally
compare exact actual emission IDs/pieces, never retokenized text or probabilities.

A valid W3/W7 fidelity or functional mismatch is recorded, rejects that width, and
continues all 25 requests without retry. Diagnostic/rejection counts remain evidence
on mismatch. Fresh A failure, corrupt public input, malformed HTTP, native/isolation/
source failure, or failed restoration aborts classification and restores. Independent
analysis rereads raw request/response bytes, HTTP framing, template/tokenize evidence,
actual PID/startup/mounts, native logs, and all 25 cases. It verifies accepted ID,
Config, HostConfig, complete canonical Mounts, health/READY smoke, stopped candidate
cells, sole service/DRM ownership, hardware, and source/binary restoration. Timing and
promotion remain false even when a width is listed as qualified.

## Evidence and cleanup

Retain build manifests and logs, CPU attempts, both top-level CLI logs, the entire
`qualification-r1/` directory, N=9 `run-r3/` proof/raw/container/snapshot/restoration
artifacts, the existing N=9 r2 build proof, source/binary hashes, full argv diffs,
PID/library/startup records, raw HTTP and template/tokenize evidence, per-cell reports,
all three ready/restored native files, cumulative proposal/acceptance statistics,
restoration checks, `independent-restoration.json`, and `analysis.json`. Preserve
failures and stopped candidate containers; do not delete, reuse, or shelve them.

The inherited finally/signal handling restores the original exact ID, Config,
HostConfig, complete canonical Mounts, health, natural-stop READY state, sole live
service/DRM owner, hardware, and accepted/parent/candidate source and six-binary
contracts. No persistent launcher, deployment, timing, or promotion change is made.

## Static preparation and pending result

Static preparation covers the seeded N=9 proof contract, exact colored native-status
parsers and negative controls for all three tags, the negative-lookahead selector, the
six-artifact candidate proof, public launch-delta fixtures, valid-mismatch continuation,
restoration failure handling, and the frozen-input revalidation. These checks are
static/CPU-fixture checks only; they do not execute local helpers, builds, native tests,
GPU code, containers, HTTP, or timing.

The lead must continue diagnosis if the real experiment fails; a failed experiment is
not a completed task and does not authorize a draft-fidelity regression claim. No
additional task-shelve or parallel process is introduced. All remaining native and
75-request public acceptance criteria are pending until the ordered remote commands
above pass.
