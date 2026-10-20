# LiliCorr device-local shared embedding — CPU-prepared experiment

**Status: CPU-prepared F32 R3 correction.** R2 built successfully, but the Q8
candidate failed exact acceptance (below). No F32 SDK build/GPU capture, promotion,
or persistent launcher/model change. Development tier, not a standard-compliant
benchmark publication. The lead owns build, B70 execution, restoration, and comparison.

## Request and bounded hypothesis

Original request: “get going on the drafter optimizations … get the drafter
performant enough that we start seeing the speed benefits … profile performance …
optimize … for Intel’s architecture.” This package changes **only LiliCorr input
embedding placement/materialization**, not NVFP4, DFlash configuration, IQ2/Q8
singleton arithmetic, full-block-8/noncausal KV, lattice math, model quantization,
kernels, W7, or graph enablement.
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
R3 materializes that immutable 544 MiB Q8 table's native CPU GET_ROWS values in a
private **2 GiB F32 device table**; it does not change either model's stored weights.

Fresh research was completed by the lead/researcher before implementation:
- https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/asynchronous-and-overlapping-data-transfers.html
  — keep intermediates/kernels on device to avoid host/device round trips; an
  isolated kernel measurement does not establish application benefit.
- https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/optimizing-data-transfers.html
  — copy/USM storage must survive all asynchronous users.
- https://github.khronos.org/SYCL_Reference/iface/scalar-types.html
  — SYCL half is IEEE 16-bit; half intermediate products need not equal native
  CPU F32 dequantization. This motivated exact CPU-value materialization below.
No B580-derived launch constants or new performance attribution are assumed.

## Preserved Q8 failure and bounded correction

Lead-observed R2 evidence: frozen-SDK build passed all 44 objects (11 Unity),
only `libllama.so.0.7.0` changed, the other five artifacts were unchanged, and live
source proof passed. `candidate-r1/` completed and independent restoration passed;
both W3 startups showed the 544 MiB Q8 table on SYCL0. **Analysis failed all six
short measured requests: 126 proposed / 85 accepted, required 129/84.** Long stayed
133/82. No acceptance gate is relaxed and no speedup is claimed.

Pinned cause: `GGML_SYCL_F16=ON` makes SYCL `dfloat/dfloat2` half types
(`common.hpp:216–218`); `dequantize_q8_0` multiplies in half (`dequantize.hpp:771`)
before GET_ROWS writes float. CPU GET_ROWS (`ggml-cpu/ops.cpp:5288`) instead calls
`ggml_get_type_traits(type)->to_float`, whose Q8 routine (`ggml-quants.c:553–566`)
multiplies the float scale and int8 directly into F32. Raw Q8 device placement
therefore changed rounding. R3 invokes that **same native CPU converter**, not a
replacement formula; GPU F32 GET_ROWS is already a direct assignment.

Keep `candidate-r1/`, all R1/R2 source/build/log artifacts, `device-embedding.patch`,
`source-hashes.json`, original fixtures, and R1/R2 build scripts intact. New active
files are `device-embedding-f32.patch` and `source-hashes-f32.json`; fresh source is
`...-lilicorr-device-embedding-r3`, build evidence `build-attempt-r3/`, and capture
`candidate-r2/` with prefix `bolt-lilicorr-device-embedding-candidate-r2-`.

## Patch and lifetime/layout contract

`device-embedding-f32.patch` touches exactly four private source files:
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

Only cache initialization/activation logging changed versus the Q8 patch;
DFlash selection, cparams, and owner declarations remain byte-identical. Native
`ggml_new_tensor_2d(..., GGML_TYPE_F32, ...)` and backend allocation create a fresh
table with no Q8 reorder extras. Complete Q8 rows are converted with native
`ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float` into <=16 MiB F32 scratch. Native
`ggml_backend_tensor_set` uploads each chunk at its F32 byte offset. The original
CPU table and both model pointers/metadata/bytes remain untouched.

Important pinned-backend detail: staged uploads use the default buffer queue;
`ggml_backend_synchronize` waits a different compute queue. A native blocking
four-byte `get_tensor` on the **same in-order default queue fences every chunk**
before scratch reuse/free. The override is published only after the final fence.
This is initialization only, not a per-round transfer. Scratch is declared before
the buffer so exception cleanup drains outstanding uploads before freeing scratch;
the buffer also dies before its metadata. Existing context owners precede
memory/scheduler/graph members, so those users die first after the existing
destructor's `synchronize()`.

F32 allocation is bounded to **<=3 GiB including native padding**, below the pinned
>=4 GiB (2 MiB-rounded) host-USM fallback. Expected device table: **2147483648 bytes
(2 GiB)**, versus the old 570425344 bytes. This adds 1504 MiB versus the Q8 candidate;
backend-owned upload staging is additional. No global F16 flag or kernel changes.
The active log requires SYCL0, Q8 source type/bytes and F32 destination type/bytes.
No GPU memory query was performed by the worker. The lead must
verify capacity and actual startup selection; a placement log is not a speedup.

## CPU checks and source/build preparation

From this repository, with Python 3, `patch`, Bash, and Clang C11/C++17 with sanitizers:

```sh
CXX=clang++ python3 -B results/20261020-lilicorr-device-embedding/test-device-embedding.py
```

`cpu-checks-f32.log` retains current commands/environment/results; old
`cpu-checks.log` and `cpu-build-r2-checks.log` remain untouched. Existing nine
placement/runner and seven build tests are retained, with one new native numerical
test (10 + 7). Tests verify exact forward/reverse scope, drift rejection, full
patched translation-unit syntax, unchanged cparams/DFlash/owners, initializer-only
delta versus R2, six binaries/ancestry, private runtime bindings, container
contracts, strict F32/source activation, and unchanged exact acceptance.

Numerical proof compiles the pinned native Q8 converter/type-trait entries and
CPU GET_ROWS function verbatim, not a mock formula. Complete uploaded bytes match
native whole-table conversion; selected rows also match actual native CPU
GET_ROWS. Inputs include all 65536 half bit patterns, signed zeros, small scales,
int8 extremes, 32- and 4096-wide rows, and below/exact/above 16 MiB boundaries plus
multi-chunk tails. Mock backend queues retain scratch pointers until fences,
poison consumed scratch, and exercise 30 guards/eight failure paths including
late-chunk and constructor unwinding. ASan and UBSan failures are fatal. Only
function-type metadata in the extracted native C unit is disabled: upstream's
explicit `ggml_to_float_t` cast erases its typed Q8 pointer signature; the original
trait/function is deliberately not rewritten. Other sanitizer checks remain on.
Mock GPU operations establish control/lifetime only: **live GPU fidelity,
activation, capacity, and end-to-end benefit remain lead verification gates**.

Fixtures were obtained read-only from the immutable exported tree (no `.git`):
`/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2`,
`UPSTREAM_COMMIT=b6334e37e6ef286fc5091a41b6561b2113d915fb`. All fixture, original,
candidate, support-header/backend, and patch hashes are in `source-hashes-f32.json`.
The two additional upstream fixtures are `ggml-quants.c` and CPU `ops.cpp` under `fixtures-f32/`;
the original fixtures/support archive and Q8 pins remain unchanged.
SSH inspection used `ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no
-o ControlPath=none -o ForwardAgent=no inference-host 'bash -s'` with read-only
`tar`, `sed`, `grep`, `test`, and `ls` on stdin (fish login).

## Validated R2 build recipe, unchanged R3 reuse

The lead's R1 build (`bf97b689c`) stopped at its dry-plan guard **before any
compilation**: the original cache requested 331 whole-server actions, including
unrelated SYCL kernels and always-dirty UI assets. Read-only SDK diagnosis found
missing Ninja command-log lineage, not compiler/flag/source drift. Keep
`build-candidate.sh`, failed source `...-lilicorr-device-embedding-r1`,
`build-attempt-r1/`, and `build-diagnosis-r2/` through `build-diagnosis-r4/` untouched.
Do not repair/import Ninja logs, touch timestamps, reconfigure, or widen the build.

R2 successfully built the qualified original + Q8 placement patch. R3 instead
clones the **qualified original** afresh to `...-lilicorr-device-embedding-r3` and
applies only `device-embedding-f32.patch`. The unchanged `build-module-r2.py` reads the actual
final generated linker via `ninja -t commands`, tokenizes it with `shlex`, and
cross-checks every link object against `ninja -t targets all` and `ninja -t query`.
The complete closure is **44 objects: 33 `.cpp.o` + 11 Unity `.cxx.o`**. The diagnosis-R3
33-object selection was incomplete and is not used. All 44 must appear in both the
read-only plan and completed build; none of the modified module is silently reused.

`build-r2-inputs.json.gz` preserves the established R4 linker, module commands,
object list and plan. Target/query fixtures were derived locally, without execution,
from read-only copies of the original generated Ninja graph; they match the R4
commands. The driver pins graph/cache/compiler definitions and exact compile/link
arguments; checks SDK icpx `2026.0.0.20260331`, CMake `3.28.3`, Ninja `1.11.1`;
checks all 11 Unity wrappers against the original; and rehashes original GGML link
libraries plus the five protected artifacts before/after. It invokes only existing
module object rules, then the existing generated linker as a validated argv list
(no shell, response file, flags/target substitution, or additional link action).
No kernel, common, server, test, UI, or CMake-generation action is permitted.
Ninja may write its normal build log/dependency records; no metadata is fabricated.

Lead-only, stage the updated package and existing sibling dependencies, then:

```sh
cd /home/mike/b70-evals/20261020-lilicorr-device-embedding
bash build-candidate-r3.sh
python3 -B source-proof.py --built
```

Use the lead's background-job mechanism for the build. All new build evidence,
including exact commands, raw compiler/linker output, failure logs, source checks,
image, six binary hashes and exit status, stays under **`build-attempt-r3/`**.
Frozen SDK:
`intel/deep-learning-essentials@sha256:aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0`.
Vendor setvars is sourced with `set -eo pipefail`; nounset is enabled only afterward.
No host install or GPU mount. Runtime/profile/graph policy, helper, link recipe,
and R4 build-input pins are unchanged. Original `GGML_SYCL_GRAPH:BOOL=ON` and
`GGML_SYCL_F16=ON` remain inherited cache settings, not newly enabled policies.

The whole-tree/source-ancestry and six-artifact proof is unchanged in strength;
its build files now resolve under R3, without rewriting R1/R2 evidence. Historical
`LILICORR_PROFILE_BUILD_PASSED` / `LILICORR_PROFILE_DEPENDENCIES_PASSED` markers
remain, as does the unchanged helper's `LILICORR_DEVICE_EMBEDDING_MODULE_R2_PASSED`
marker (recipe version, not an R2 source selection). Expected binary delta is
still **only `libllama.so.0.7.0`**. The original `llama-server`,
`libllama-server-impl.so`, `libggml-sycl.so.0`, `test-backend-ops`, and
`libllama-common.so` must remain byte-identical. Stop on any mismatch; do not relax
that gate.

Local recovery verification (no SDK build, GPU job, or service control):

```sh
python3 -B results/20261020-lilicorr-device-embedding/test-build-r2.py
CXX=clang++ python3 -B results/20261020-lilicorr-device-embedding/test-device-embedding.py
```

Seven build tests validate the actual 44/Unity closure and reject missing/foreign
objects, unsafe linker/target drift, changed compile flags, graph/input mismatches,
nonmodule actions, and protected-library drift. Their R3 check also requires an
otherwise identical R2 shell recipe. All placement/proof tests now use R3/F32
paths and retain rejection of refreshed records for any protected artifact.
`cpu-build-r2-checks.log` remains historical; simulated driver tests are not an SDK build.

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

The lead already completed the fresh original-Q8 baseline at
`/home/mike/b70-evals/20261019-lilicorr-q8-drafter/timing-20261020-baseline-r1`:
100 public checks, 24 measured outputs, native/acceptance gates and independent
restoration passed. **Do not run another baseline.** Once the lead approves GPU
execution after the R3 build proof, execute each candidate line in a fresh process:

```sh
python3 -B run-performance.py --mode candidate --lead-run-timing
python3 -B run-performance.py --mode candidate --lead-after-timing-restore
```

Each capture is the same ABBA `A1 W3a W3b A2`: 25 public checks/cell (100 total),
55/8270-token prompts, one warmup plus three measured responses/point/cell (24
measured), temperature 0, seed 42, ignore EOS, max 128, cache_prompt false. W7 is
excluded. Expected: all native/public checks and all 24 exact outputs match;
W3 short proposed/accepted **129/84**, long **133/82**; active startup on candidate
W3 exactly once with `SYCL0`, `source_type=q8_0 source_bytes=570425344`,
`type=f32 tensor_bytes=2147483648`, and buffer bytes in [2 GiB, 3 GiB]. All other cells
must have no candidate marker. Actual startup logs, not synthetic records, drive
this check. Offline analysis rereads raw public data and verifies the same gates.

Capture cleanup is the existing controller's automatic restoration of the original
service, including failure paths. **After EACH capture**, the independent command
verifies original identity/source/health/sole ownership again, then analyzes.
Do not start the next capture if this fails. Do not modify persistent launchers.
Keep failed attempts and artifacts, rather than overwriting/retrying the same path.

Retain the entire completed baseline directory and both `candidate-r1/` / `candidate-r2/`, including
native/raw startup/logs, PID/container/mount/source proofs, capacity/ownership,
HTTP/SSE outputs/events/timings, telemetry, failures, restoration records,
`startup-placement.json`, and `analysis-r1/analysis.json`. Build evidence lives at
`build-attempt-r3/`; preserve R1/R2 separately. Existing sibling runner/qualified
packages must be staged unchanged; no recursive profiling runner dependency.

Compare candidate **real W3 end-to-end** to fresh original-Q8 W3 and the fresh
NoDraft A controls, short and long separately, with raw samples/medians/dispersion,
proposal/acceptance, and memory/capacity. Report server timings separately from
client wall and device timing; do not infer pure CPU savings. No promotion if
fidelity/acceptance fails, placement is inactive, capacity regresses materially,
performance regresses, or no genuine end-to-end benefit is demonstrated.
The publication checklist remains incomplete (no BetterBench/standard sweep,
community-comparable result, or claim of benefit).
