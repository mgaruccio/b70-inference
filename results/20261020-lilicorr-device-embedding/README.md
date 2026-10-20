# LiliCorr device-local shared embedding — CPU-prepared experiment

**Status: CPU preparation only. No remote build, GPU execution, live activation,
performance result, promotion, or persistent launcher/model change.** Development
tier; not a standard-compliant benchmark publication. The lead owns deployment,
build, exclusive B70 execution, independent restoration checks, and comparison.

## Request and bounded hypothesis

Original request: “get going on the drafter optimizations … get the drafter
performant enough that we start seeing the speed benefits … profile performance …
optimize … for Intel’s architecture.” This package changes **only LiliCorr input
embedding placement**, not NVFP4, DFlash configuration, IQ2/Q8 singleton arithmetic,
full-block-8/noncausal KV, lattice math, precision, kernels, W7, or graph enablement.
No public API, inference knob, disk/model copy, cache subsystem, service, or store.

The completed `20261020-lilicorr-context-profile` supplies an intervention
hypothesis, not proof of a bottleneck or speedup. It observed 826 scheduler / 824
backend nodes per round (657 decoder + 167 head; two uninstrumented). Per-response
median host wall: short 43 rounds, outer 311.504 ms / decoder 102.073 / head 59.865 /
log 29.896 / residual 119.739; long 45 rounds, 341.415 / 107.356 / 63.151 / 31.367 /
139.545 ms. The approximately 97.967/115.361 ms last-decoder-to-first-head intervals
(net decoder logging) include waits for queued GPU work: **not established pure
CPU cost or device event times**. TOP_K already has an observed GPU path; this is
not a fictitious batch-8 MMQ switch.

The original DFlash builder borrows target `tok_embd` when absent (this checkpoint
has none), then uses it for both noise and candidate GET_ROWS. Original model
placement deliberately keeps the target input table on CPU. Actual startup:
target SYCL0 model 20446.77 MiB + SYCL_Host 544 MiB; drafter SYCL0 model 1363.48 MiB.
This experiment adds a private device copy of that immutable 544 MiB Q8 table.

Fresh research was completed by the lead/researcher before implementation:
- https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/asynchronous-and-overlapping-data-transfers.html
  — keep intermediates/kernels on device to avoid host/device round trips; an
  isolated kernel measurement does not establish application benefit.
- https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/optimizing-data-transfers.html
  — copy/USM storage must survive all asynchronous users.
No B580-derived launch constants or new performance attribution are assumed.

## Patch and lifetime/layout contract

`device-embedding.patch` touches exactly four private source files:
`src/llama-context.cpp`, `src/llama-context.h`, `src/llama-cparams.h`, and
`src/models/dflash.cpp`. The model remains const; target tensor pointer, metadata,
buffer, placement, and bytes are never changed. A default-null private cparams
pointer selects the same context-owned table for **both** GET_ROWS consumers.

Activation requires DFlash + LiliCorr head/top-k, missing own table, `ctx_other`,
existing op/GPU offload, one selected GPU, an initialized SYCL backend, allocated
non-view contiguous 2-D standard host Q8_0 table with no backend extras, and native
GET_ROWS support. Other paths retain original behavior. Unsupported LiliCorr
paths log `inactive reason=...`; allocation/copy failure is fatal rather than a
silent optimized-looking fallback. The lead runner rejects missing/inactive or
wrong placement before any public/timed request.

Native `ggml_dup_tensor` + allocation gives fresh metadata and fresh zero-initialized
SYCL extras. Native host-to-device `ggml_backend_tensor_copy` transfers standard
Q8 bytes without copying backend/reorder state. The pinned GET_ROWS implementation
uses ordinary `dequantize_q8_0`; lazy reorder requires MUL_MAT, which never consumes
this table. No conversion or reorder/kernel change is introduced.

Important pinned-backend detail: staged `set_tensor` uploads use the default
buffer queue; `ggml_backend_synchronize` waits the compute queue, which need not be
the same queue. One native blocking one-byte `get_tensor` on the **same in-order
default queue** fences the entire upload before publishing the override. This is
initialization only, not a per-round transfer. Buffer destruction drains pending
staging on exceptions. The owners use existing ggml unique-pointer aliases and
are declared before memory/scheduler/graph members, so those users die first after
the existing destructor's `synchronize()`. The buffer dies before its metadata.

Allocations are conservatively bounded to <=1 GiB including native tensor padding;
this avoids the pinned SYCL allocator's >=4 GiB (2 MiB-rounded) host-USM path.
Expected extra device storage is **544 MiB** plus native padding, with additional
host upload-staging storage owned by the backend buffer.
The one-time active log records actual backend, buffer, tensor type, tensor bytes,
and buffer bytes. No GPU memory query was performed by the worker. The lead must
verify capacity and actual startup selection; a placement log is not a speedup.

## CPU checks and source/build preparation

From this repository, with Python 3, `patch`, Bash, and a working C++17 compiler:

```sh
CXX=clang++ python3 -B results/20261020-lilicorr-device-embedding/test-device-embedding.py
```

`cpu-checks.log` retains the final command/environment/output and preparation
failures. Tests cover exact original→candidate and reverse bytes (zero fuzz or
offsets), full-tree/private-header/unlisted-file drift, actual complete patched
translation-unit syntax against pinned original headers, extracted implementation
and native copy/dup code (24 guards, two owners, upload fence, six failure paths
including constructor unwinding), pinned layout/queue assumptions, all six binary
proofs and immutable ancestry, actual private function-global bindings, original
argv/PID contract reuse, strict container env/mount/identity checks, activation,
and exact proposal/acceptance. Fake backend operations test ownership/control
flow only; they do **not** establish GPU numerical fidelity or live activation.

Fixtures were obtained read-only from the immutable exported tree (no `.git`):
`/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2`,
`UPSTREAM_COMMIT=b6334e37e6ef286fc5091a41b6561b2113d915fb`. All fixture, original,
candidate, support-header/backend, and patch hashes are in `source-hashes.json`.
SSH inspection used `ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no
-o ControlPath=none -o ForwardAgent=no inference-host 'bash -s'` with read-only
`tar`, `sed`, `grep`, `test`, and `ls` on stdin (fish login).

**Simpler qualified seed selected:** read-only inspection found the original has
`build-sycl/build.ninja`, its original context/SYCL objects, and a complete cache
(`CMAKE_HOME_DIRECTORY=/src`, `Release`, `icpx`, `Ninja`). Thus the build copies
**the qualified original**, not any diagnostic variant. No diagnostic reversal or
profiling wrapper is needed. Original `GGML_SYCL_GRAPH:BOOL=ON` is an inherited
compile-time cache setting; runtime policy is unchanged and graphs are not enabled.

Lead-only, after staging this package and its existing sibling dependencies at
`/home/mike/b70-evals/20261020-lilicorr-device-embedding`:

```sh
cd /home/mike/b70-evals/20261020-lilicorr-device-embedding
bash build-candidate.sh
python3 -B source-proof.py --built
```

Run long jobs with the lead's background-job mechanism, not in a foreground agent
shell. Build target:
`/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-device-embedding-r1`.
Frozen SDK:
`intel/deep-learning-essentials@sha256:aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0`.
No GPU mounts, host installs, configuration regeneration, or unrelated kernel
rebuilds. SDK-container-only build dependencies match the prior build process.
A dry build plan rejects compilation outside `src/CMakeFiles/llama.dir/`; retain
and investigate a failure rather than relaxing it. The build preserves original
source/binary proof before/after, all-source candidate scope before/after, six
artifact hashes, dependency output, image inspection, raw build log and exit code.
Expected original→candidate binary delta: **only `libllama.so.0.7.0`**. All of
`llama-server`, `libllama-server-impl.so`, `libggml-sycl.so.0`, `test-backend-ops`,
`libllama-common.so`, and `libllama.so.0.7.0` are rehashed. Unexpected deltas fail.

`source-proof.py` directly reuses the first common proof in a private runpy
namespace, extending scope to both private headers; it does not memoize or rebind
the qualified gate. Exact historical `LILICORR_PROFILE_BUILD_PASSED` and
`LILICORR_PROFILE_DEPENDENCIES_PASSED` marker names are intentionally preserved to
satisfy that helper. They do not imply profiling code is present.

## Defined lead E2E process (not executed by worker)

Preconditions: qualified original service healthy; exclusive B70 ownership;
unchanged pinned Q8 drafter and target; completed candidate six-artifact proof;
fresh output/container names; sufficient device/host/disk capacity. Candidate is
original qualified source + **only** this placement patch. No profiling code,
profiling environment, graph enablement, or new runtime diagnostic is used.

`run-performance.py` directly reuses `20261019-lilicorr-q8-drafter/run-performance.py`
and the proven singleton controller/helpers. Its actual function globals share
one private contract map; qualified ancestry stays original. Baseline and A cells
always mount original source. Only candidate W3 mounts the new source. Both
captures retain the same pinned Q8 drafter bind, native IQ2/Q8 gates, public
payloads, real PID1 argv/environment proof, mounts, ownership, and restoration.

Execute **each line in a fresh process**, serially, stopping on any failure:

```sh
python3 -B run-performance.py --mode baseline --lead-run-timing
python3 -B run-performance.py --mode baseline --lead-after-timing-restore
python3 -B run-performance.py --mode candidate --lead-run-timing
python3 -B run-performance.py --mode candidate --lead-after-timing-restore
```

Each capture is the same ABBA `A1 W3a W3b A2`: 25 public checks/cell (100 total),
55/8270-token prompts, one warmup plus three measured responses/point/cell (24
measured), temperature 0, seed 42, ignore EOS, max 128, cache_prompt false. W7 is
excluded. Expected: all native/public checks and all 24 exact outputs match;
W3 short proposed/accepted **129/84**, long **133/82**; active startup on candidate
W3 exactly once with `SYCL0`, Q8_0 and `570425344` tensor bytes. All other cells
must have no candidate marker. Actual startup logs, not synthetic records, drive
this check. Offline analysis rereads raw public data and verifies the same gates.

Capture cleanup is the existing controller's automatic restoration of the original
service, including failure paths. **After EACH capture**, the independent command
verifies original identity/source/health/sole ownership again, then analyzes.
Do not start the next capture if this fails. Do not modify persistent launchers.
Keep failed attempts and artifacts, rather than overwriting/retrying the same path.

Retain entire `baseline-r1/` and `candidate-r1/` (distinct prefixes), including
native/raw startup/logs, PID/container/mount/source proofs, capacity/ownership,
HTTP/SSE outputs/events/timings, telemetry, failures, restoration records,
`startup-placement.json`, and `analysis-r1/analysis.json`. Build evidence lives at
`build-attempt-r1/` plus the root proof files. Existing sibling runner/qualified
packages must be staged unchanged; no recursive profiling runner dependency.

Compare candidate **real W3 end-to-end** to fresh original-Q8 W3 and the fresh
NoDraft A controls, short and long separately, with raw samples/medians/dispersion,
proposal/acceptance, and memory/capacity. Report server timings separately from
client wall and device timing; do not infer pure CPU savings. No promotion if
fidelity/acceptance fails, placement is inactive, capacity regresses materially,
performance regresses, or no genuine end-to-end benefit is demonstrated.
The publication checklist remains incomplete (no BetterBench/standard sweep,
community-comparable result, or claim of benefit).
