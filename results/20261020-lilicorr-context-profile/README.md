# LiliCorr explicit context-scoped SYCL diagnostic

**Development preparation only; live build/coverage proof pending. No optimization,
GPU-event-time result, clean-speed comparison, or promotion.** The previous two
marker-based runs covered only the 167-node head of the 826-node draft graph
(about 58/61 ms of 310/340 ms diagnostic outer-call wall). Adding the exact
`inp_noise_embd` name did not expand coverage. This package stops name heuristics.

User request (verbatim):
> ok go ahead and get going on the drafter optimizations. the goal is to get the drafter performant enough that we start seeing the speed benefits we should be. profile performance and think about how to opimize the drafter for intels architecture

## Bounded implementation

The full patch changes only `common/speculative.cpp`, `src/llama-context.cpp`,
and `ggml/src/ggml-sycl/ggml-sycl.cpp` relative to the immutable qualified export:
`/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2`,
upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb` (no `.git`).
The seed is the completed
`/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-b70-decoder-profile-r1`.
The new candidate is
`/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-context-profile-r1`.

Only `seed-incremental.patch` is applied to a reflink copy of the seed. Full and
incremental patches, exact parent/seed/candidate hashes, and parent fixtures are
retained. Expected deltas from the qualified original remain exactly
`libggml-sycl.so.0`, `libllama-common.so`, `libllama.so.0.7.0`; the complete source
maps and all six binaries are freshly hashed. Build/dependency success markers
remain `LILICORR_PROFILE_BUILD_PASSED` and `LILICORR_PROFILE_DEPENDENCIES_PASSED`.
Seed and candidate-source proofs are retained before/after the incremental build.
The build uses the inherited frozen SDK digest, no GPU mounts or host installs,
and preserves 4 GiB reserve plus 1 GiB evidence space.

`llama_context::graph_compute` constructs a private RAII guard immediately before
its **existing** scheduler call and restores immediately after it (also on
exceptions). Only `LLAMA_LILICORR_PROFILE=1` and `lilicorr_candidate_topk > 0`
activate a new scope. It resolves `set_lilicorr_profile_scope` through the existing
backend procedure-address facility. No public header/API was added. The SYCL
procedure changes a default-false thread-local flag and returns its prior value.
All host storage and procedure lookup complete before any flag changes; setters
are `noexcept`. Saved values are restored in reverse, including duplicate
registries. Nested non-drafter contexts temporarily **suspend** the outer flag,
then restore it. Ordinary target calls do no scope lookup/allocation/clock work;
standalone `test-backend-ops` remains unscoped even with the environment enabled.

Read-only verification against the exact qualified source established:
- `ggml-backend.cpp:2118–2130`: async scheduler entry directly calls
  `ggml_backend_sched_compute_splits`.
- `:1865–1935`: normal and callback-eval split loops directly call
  `ggml_backend_graph_compute_async`, including their error returns.
- `:601–604`: backend entry directly calls `backend->iface.graph_compute`.
- SYCL's backend wrapper directly calls `ggml_backend_sycl_graph_compute_impl`.
  Thus dispatch-loop execution uses the calling host thread; GPU work may remain
  asynchronous. Existing internal waits remain included in host-wall intervals.
- `ggml-backend.h:120,188,205` declares the existing lookup facilities;
  `llama-context.h:396` owns `std::vector<ggml_backend_ptr>`.

The backend gate is explicit TLS scope **and** environment opt-in, not any tensor
name. Every scoped SYCL graph, including injections and renamed copies, uses
v1 `marker_name=lilicorr.context`: a context label, not a tensor identity.
The existing shared `ggml_time_us()` clock, every fusion branch, operation order,
and buffered post-loop logging remain unchanged. No new synchronization, queue
command, arithmetic, precision, acceptance, lattice, KV, or kernel change.
Core wall and logging wall remain separate; neither is GPU event time.

Runtime and source proof directly import the first common profile implementation,
not a chain of continuation runners. Only private case globals change; qualified
P/C ancestry and the Q8 adapter/model constants remain unchanged. Imported code
paths are included in `OWN_FILES`. The analyzer wraps the current 18-tested SYCL
validator, preserving strict schemas, overlap/ownership/per-round checks, public
W3 events, and rejection of **any** A-cell SYCL events. Raw/gzip logs are read
without staging or rewriting; if both exist, bytes must match.

The new measured gate requires every injection owner (59-node ubatches), every
full draft owner (826 nodes), exactly one `blk.0.attn_q.weight` **bf16 MUL_MAT** per
round regardless of fusion-group label (43 short / 45 long), one original
167-node head plus a distinct decoder segment, and observed backend nodes greater
than 167 **per round**. First/body coverage, metadata nodes, and CPU/uninstrumented
scheduler nodes are reported separately, not relabeled as GPU work. Missing or
metadata-only decoder evidence fails closed. Unexpected empty metadata-only
backend segments retain the current strict analyzer rejection and require review,
not silently dropping records or loosening the gate.

## Research and CPU preparation

Task research retained from the lead's fresh official-source review:
- https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/kernel-launch.html
  — submission wall and device completion are distinct.
- https://intel.github.io/llvm/syclgraph/SYCLGraphUsageGuide.html
  — lifetime/update synchronization constrain replay; no graph replay is enabled.
- https://uxlfoundation.github.io/oneDNN/dev_guide_primitive_cache.html
  — global primitive cache defaults to 1024 and avoids repeated JIT; do not infer
  repeated JIT from host wall alone.

CPU-only checks (no Docker/GPU/network/install):
```bash
python3 -B results/20261020-lilicorr-context-profile/test-profile.py
bash -n results/20261020-lilicorr-context-profile/build-candidate.sh
```
They include the original 18-test SYCL suite unchanged (and its 13-test ancestor),
zero-fuzz forward/reverse patches, exact pins and ordinary-source identity,
extracted actual C++ helpers **and graph_compute** with mocked registry/scheduler
callbacks (off/target/drafter, duplicate/nested/thread isolation, errors,
exceptions, reserve/lookup failure), new runtime bindings/repins, and strict
coverage negatives. The combined synthetic fixture inserts backend records
**inside real launch/release and ubatch boundaries**, not after release. It is a
parser/contract fixture, **not live producer proof**. CPU evidence: `cpu-checks.log`.

## Lead-only build and real end-to-end journey

Preconditions: inference-host, exclusive B70, healthy sole original
`nemotron-bolt-probe`, frozen SDK available, at least 5 GiB free **plus seed-copy
space**, new candidate/output paths absent. Deploy only this new package and its
already-pinned dependencies to the corresponding `/home/mike/b70-evals/` paths.
No completed package, model, persistent launcher or launch flag is edited.

Exact commands on inference-host (separate processes):
```bash
ROOT=/home/mike/b70-evals/20261020-lilicorr-context-profile
bash "$ROOT/build-candidate.sh"
python3 -B "$ROOT/source-proof.py" --built
python3 -B "$ROOT/run-profile.py" --lead-run-profile
# Capture's finally/signal cleanup restores original. Independently verify it:
python3 -B "$ROOT/run-profile.py" --lead-after-profile-restore
```
The last command independently verifies live original identity/health/source/
ownership **before** analysis. Do not analyze or claim coverage without restoration.
Baseline: qualified no-draft A; diagnostic W3 uses the unchanged Q8 drafter,
private source bind, and development-only profiling opt-in. Run the unchanged
ABBA sequence `VSPF-A1,VSPF-W3a,VSPF-W3b,VSPF-A2`, all native IQ2/Q8 gates,
25 public jobs/cell and 55/8270-token short/long prompts at greedy temperature 0,
seed 42, `cache_prompt=false`, ignore EOS, `max_tokens=128`, one warmup and three
measured requests/point/cell. Expected: all 100 public and 24 measured outputs
identical; proposals/accepted short 129/84 and long 133/82; context coverage gate
passes every measured W3 request. Full eight blocks, 7x8 noncausal lattice and
rejected-KV behavior remain intact. No NVFP4 or DFlash tuning.

After the first unchanged public job in each W3 cell, the lead runner reads that
request's actual `docker logs` stderr and requires both a context-scoped BF16
decoder projection and lattice operations. A missing scope aborts early through
the existing restoration lifecycle; no additional request is introduced. Raw
probe stdout/stderr and `scope-probe-first-job.json` are retained. This is an early
presence check, not a substitute for full per-round coverage or any fidelity gate.

Retain exact commands/environment, build-attempt logs/exit status, seed/source
proofs, six-binary/dependency proof, native/public artifacts, raw server logs,
actual PID/mount/environment/source/model/ownership records, restoration records,
and `profile-r1/analysis-r1/profile.json`. Keep failures. Cleanup is the inherited
finally/signal original restoration; never remove/alter the qualified or seed
source. CPU tests clean only their temporary trees. Live results remain pending;
the user's performance goal is not yet achieved and this is not a
BENCHMARKING_STANDARDS.md publishable performance comparison.
