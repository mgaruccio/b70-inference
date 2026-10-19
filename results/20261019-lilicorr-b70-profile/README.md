# LiliCorr B70 timing-only profile (preparation, no GPU results)

**Tier: development profiling only. No speedup, standard-compliant benchmark, or
promotion is claimed.** This package prepares the requested drafter work:
“ok go ahead and get going on the drafter optimizations. the goal is to get the
drafter performant enough that we start seeing the speed benefits we should be.
profile performance and think about how to opimize the drafter for intels
architecture”. The lead owns remote build/execution, GPU access, and subsequent
optimization. Nothing here changes a persistent launcher.

## Baseline and intentional differences

Reuse `../20261019-lilicorr-q8-drafter/run-performance.py` and its exact W3 ABBA
controller/contracts, public/native gates, hardware/ownership checks, capacity
policy, signal/finally cleanup, and independent restoration. Do not change the
qualified P namespace or its immutable source/binary/code ancestry.

- Order: **VSPF-A1, VSPF-W3a, VSPF-W3b, VSPF-A2**; W7 remains excluded.
- A: unchanged qualified no-draft source, argv, mounts, environment, and target.
- W3: unchanged Q8 model, server argv, sampling, native tests, IQ2/singleton
  arithmetic. Only `/src` changes to the fresh read-only profile candidate, and
  Docker adds `LLAMA_LILICORR_PROFILE=1`. Both arms receive fresh container names.
- Parent: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2`.
- Candidate: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-b70-profile-r1`.
  Both are exported trees (no `.git`); actual `UPSTREAM_COMMIT` is
  `b6334e37e6ef286fc5091a41b6561b2113d915fb`.
- Q8 file: `/home/mike/b70-evals/20261019-lilicorr-q8-drafter/lilicorr-q8-25.gguf`,
  1,437,595,904 bytes, SHA256
  `d420b2bdcfbbe937795b9dc94d2659305eb01244309172019976dabefad363d7`.
  The exact read-only file bind remains the existing Q8 adapter's responsibility.
- Frozen SDK: `intel/deep-learning-essentials@sha256:aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0`.
  Keep the existing `/src` CMake cache. No reconfiguration, host installs, GPU
  device mounts in the CPU build, or changes to the NVFP4 backend/DFlash settings.
  SYCL GRAPH remains disabled at runtime; the full draft's CONCAT is unsupported
  by the current graph path. Merely enabling it is not an optimization.

## Instrumentation and interpretation

The patch contains **insertions only** in `common/speculative.cpp` and
`src/llama-context.cpp`. Every inserted block is delimited; removing those blocks
reproduces both downloaded originals byte for byte. The opt-in accepts exactly
`1`; absent, `0`, and other strings emit no timers or profile logs. It is also
gated on LiliCorr, not ordinary DFlash/DFlash2 or the target model. No extra
synchronization/getter calls, reordered operations, row trimming, floating-point
arithmetic, model state, or rollback changes are introduced.

`LLAMA_LILICORR_PROFILE v=1` records carry integer monotonic microseconds:

- `process_batch`: total existing feature-process host interval, input rows.
- `process_chunk`: sequence, total/sequence/chunk rows, chunk offset and first/last
  token positions; target-feature getter **wait + gather** (including resize),
  injection batch preparation, `llama_process` host submit, existing projected
  getter wait, and embedding-ring copy. All target rows are still injected.
- `draft_batch`: batch preparation and `llama_process` host submit for the full
  eight-row block; return status.
- `draft_select`: sequence/token position, existing lattice getter wait, host
  lattice walk, raw selected-token count, and the existing post-draft limit.
- `ubatch`: LiliCorr-only memory apply + graph build/reuse/allocation interval,
  input setting, graph submission, graph type, rows, node count, reuse flag, and
  status. The original pipeline-parallel synchronization stays where it was.

These are **host wall intervals, not device kernel timings**. Getter waits can
include earlier queued work, transfers, and host overhead; submit can itself
block inside an existing backend call. `ubatch` events are nested inside the
parent submit and MUST NOT be added again. Node counts are graph nodes, NOT
kernel counts. Logs and timestamps perturb both CPU scheduling and GPU overlap;
even the enclosing process interval includes profiling-log overhead. Disable
profiling for later performance comparisons. No server-wide sampling/acceptance,
KV cleanup, or target compute attribution is claimed by this bounded patch.

The analyzer uses the existing sequential `launch_slot_`/`release` markers and
request order: 25 public requests, then 8 timing requests per cell. It checks
unique slot/task identity, prompt lengths, cumulative speculative-counter deltas,
all expected stages/fields, nested row coverage, and raw-response counts. It
refuses missing, duplicate, truncated, overlapping, or uncorrelatable data. It
retains raw fields/line references, separates costs before/from the first draft,
and labels graph counts by injection/draft parent. Startup-only profile records
are retained separately, not attributed to a request. `draft_select` retains the
raw `selected` count and existing per-request `limit`: `common_speculative_draft`
applies that cap AFTER the lattice walk (the final long-request proposal can be
one token although three were selected). The analyzer derives that existing cap
without changing the raw observation or any source operation. This does not
retrofit request IDs or change server scheduling. A failed profile analysis is
not a successful experiment even if inherited HTTP analysis passes.

**Do not revive the rejected accepted-row-only injection idea.** Rejected KV
rows are noncausal and observable during the NEXT draft; eviction happens AFTER
that draft. Trimming rows or moving rollback breaks semantics.

## Research gate (completed by lead before implementation)

Fresh official Intel sources supplied by the lead on 2026-10-19:

- https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/kernel-launch.html
  distinguishes host submission from GPU completion, and recommends device event
  timing / VTune / unitrace for device attribution. Therefore these logs explicitly
  measure existing host boundaries without inserting synchronizations.
- https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-offload-many-kernels-sycl-graph.html
  discusses launch overhead from many small kernels. Therefore include graph
  build/inputs/submit and node/reuse counts before proposing an Intel optimization;
  do not enable graph replay without checking operator support.

No claims of fresh device evidence or worker-conducted web research are made.

## Defined end-to-end process (lead execution pending)

Environment/preconditions: `inference-host`, exclusive B70, original healthy
`nemotron-bolt-probe` is the sole owner, original identity/source protected,
>=5 GiB free (4 GiB reserve + 1 GiB evidence allowance), frozen SDK already local,
qualified sibling packages/data unchanged, fresh candidate/build/output paths.
Benchmark runtime/dependencies remain on the inference host, outside Pi.

Public journey: existing `/v1/chat/completions` HTTP/SSE requests; exact existing
25-job prevalidation per cell including rejection behavior, followed by the
existing short (55 rendered tokens) and long (8270) greedy requests: temperature
0, seed 42, cache_prompt=false, ignore_eos=true, max_tokens=128. One warmup and
three measured requests per point/cell. No new workload or proxy benchmark.

Expected results: all **100** public jobs pass, all **24** measured outputs match
no-draft references, and W3 short/long proposal/acceptance counts remain
**129/84** and **133/82**, respectively (also checked for the 4 W3 warmups).
Full profile rows must correlate with the existing request markers and cumulative
draft-call counters. Every native gate, including IQ2 and Q8 singleton gates,
must pass against the ACTUAL `/src` binaries. Docker inspect and actual PID1
argv/diagnostics remain validated; an additional actual `/proc/1/environ` probe
checks the opt-in. Nothing sanitizes observations to make an ancestor gate pass.
Candidate source maps, all six binaries, Q8 bytes, and adapter/build pins are
rehash-validated before/after; no memoized candidate proof is allowed.

Evidence: retain the complete fresh `profile-r1/` inherited HTTP/SSE bodies,
event clocks, source/binary/model hashes, create argv/diffs, PID environments,
native/startup/final logs, telemetry, hardware/environment, errors, and restoration
reports. Retain `build-attempt-r1/`, build manifests and dependency logs. Derived
costs/raw event references land in `profile-r1/analysis-r1/profile.json`; inherited
fidelity/transport analysis is `profile-r1/analysis-r1/analysis.json`. Preserve
failures; never reuse, overwrite, or delete an earlier attempt. Output folders
are ignored locally until the lead decides how to retain completed raw results.

Cleanup: inherited finally/signal handling stops only fresh candidate containers,
restarts the original, and verifies health/identity/ownership/source hashes. The
independent second command repeats live restoration before rereading raw data.
Leave stopped containers and raw artifacts for review; no promotion or automatic
persistent service changes. Stop on any mismatch; do not loosen pins.

### Exact lead commands

Stage this package in `/home/mike/b70-evals/20261019-lilicorr-b70-profile` beside
its existing qualified siblings, after integration. Do not stage `.scratch`.
The build refuses existing candidate/build paths and expects only
`libllama-common.so` and `libllama.so.0.7.0` to change among the six artifacts.
Unexpected build deltas require investigation, not edited expected hashes.

Run these remote scripts **separately** via the lead's background-job mechanism
(build/capture are long-running). Use `bash -s` because the login shell is fish:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no inference-host bash -s <<'REMOTE'
set -euo pipefail
cd /home/mike/b70-evals/20261019-lilicorr-b70-profile
bash build-candidate.sh
python3 -B source-proof.py --built
REMOTE
```

After successful build and the stated GPU preconditions:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no inference-host bash -s <<'REMOTE'
set -euo pipefail
cd /home/mike/b70-evals/20261019-lilicorr-b70-profile
python3 -B run-profile.py --lead-run-profile
REMOTE
```

Only once capture/restoration finishes, in a new process:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no inference-host bash -s <<'REMOTE'
set -euo pipefail
cd /home/mike/b70-evals/20261019-lilicorr-b70-profile
python3 -B run-profile.py --lead-after-profile-restore
REMOTE
```

## Local preparation verification

```bash
python3 -B results/20261019-lilicorr-b70-profile/test-profile.py
bash -n results/20261019-lilicorr-b70-profile/build-candidate.sh
```

Fixtures `fixtures/{common/speculative.cpp,src/llama-context.cpp}.gz` are exact
read-only SSH downloads of the two qualified parent files; decompressed SHA256
pins and patch/candidate hashes are in `source-hashes.json`. Local observed result:
**12/12 tests passed**, Bash syntax check and adapter CLI help passed; exact
commands, host/tool versions, and raw output are retained in `cpu-checks.log`.
Tests use temporary subdirectories inside `.scratch`, not an external checkout. They exercise
zero-fuzz apply/reverse, unchanged original operation order, strict opt-in
structure, real saved Q8 log boundaries, synthetic timing parsing, actual-inspect
validation, private namespace binding, and same-size source/binary/model drift
after earlier code-proof capture. Synthetic timings are tests, not GPU evidence.
C++ compilation and the defined live end-to-end journey are intentionally left
to the lead; CPU fixtures cannot establish those results. The repository's
publishable benchmark checklist remains incomplete.
