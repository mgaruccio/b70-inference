# LiliCorr B70 SYCL operation timing diagnostic (preparation, no GPU results)

**Tier: development diagnostic only.** This package does not claim a speedup,
device time, benchmark publication, or optimization. It implements the requested
second, opt-in host-wall diagnostic for attributing the actual LiliCorr SYCL
backend call; the lead owns the frozen-SDK build, GPU run, and interpretation.
No persistent launcher is changed.

User request:

> ok go ahead and get going on the drafter optimizations. the goal is to get the
drafter performant enough that we start seeing the speed benefits we should be.
profile performance and think about how to opimize the drafter for intels
architecture

## Scope and baseline

The baseline is the already qualified immutable
`20261019-nemotron35-bolt-lilicorr-verify-singletons` result and its exact
Q8 ABBA controller. The immutable qualified parent is
`/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2`;
the lead-only candidate sibling is
`/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-b70-sycl-profile-r1`.
The only candidate source changes are:

- `common/speculative.cpp` (first-phase timing retained unchanged),
- `src/llama-context.cpp` (first-phase timing retained unchanged), and
- `ggml/src/ggml-sycl/ggml-sycl.cpp` (this second diagnostic).

The inherited Q8 adapter supplies the exact unchanged drafter file
`/home/mike/b70-evals/20261019-lilicorr-q8-drafter/lilicorr-q8-25.gguf`,
1,437,595,904 bytes, SHA256
`d420b2bdcfbbe937795b9dc94d2659305eb01244309172019976dabefad363d7`; this
package does not restate or mutate that controller/model contract.

The new backend code is gated by **both** `LLAMA_LILICORR_PROFILE=1` and a
tensor named exactly `lilicorr_lattice` or beginning with `lilicorr.`. It records no SYCL events, waits,
queue commands, graph replay, kernel/math/precision changes, or synchronization.
It wraps the existing dispatch/fusion branches without reordering them. Host
wall intervals are not device timings and include backend submission/wait
behavior as observed by the host.

The prefix `LLAMA_LILICORR_SYCL_PROFILE` is deliberately distinct from the
existing `LLAMA_LILICORR_PROFILE` phase marker. Records are accumulated in
preallocated host vectors and emitted only after the graph loop. Each graph
reports its gate, exact covered/uncovered node ranges, core wall, summed event
wall, uncounted wall, and logging wall. Each operation span reports its group,
first/last consumed node, source/weight labels, shapes/types, and host elapsed
microseconds. A missing or partial range is not treated as full graph coverage.

## Fresh research affecting the design

The lead's current official Intel research is retained as part of the task
record:

- [Intel Optimization Guide: Kernel Launch](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/kernel-launch.html): host submission and GPU completion are distinct, so this diagnostic makes no device-time claim.
- [Intel SYCL Graph Usage Guide](https://intel.github.io/llvm/syclgraph/SYCLGraphUsageGuide.html): graph pointer lifetimes/update synchronization constrain graph replay; this package does not enable or replay graphs.

## Files and lead-only build

`lilicorr-profile.patch` is a zero-fuzz combined patch. `fixtures/` contains
exact gzip downloads of all three parent source files, including the SYCL backend. The combined fixture uses real first-profile JSON/request logs but **synthetic backend records**. It tests parser logic, not live emission; the full-pipeline regression puts those records inside the actual request boundaries.
`source-proof.py` reuses the first package's qualified ancestry, complete
source-map, and six-binary proof, extended to the three changed source files.
The expected changed libraries are exactly `libllama-common.so`,
`libllama.so.0.7.0`, and `libggml-sycl.so.0`; any other binary delta fails
closed.

The build is lead-only on `inference-host` in the frozen SDK; it performs no
host install, GPU mount, protected-source modification, or launcher change:

```bash
bash results/20261019-lilicorr-b70-sycl-profile/build-candidate.sh
```

## Defined end-to-end journey (lead only)

After CPU checks, review, and a successful build, run the same exclusive B70
ABBA sequence `VSPF-A1,VSPF-W3a,VSPF-W3b,VSPF-A2`, with native/IQ2/Q8 gates and
25 public jobs/cell before the timed points. Use the same short prompt 55 and
long prompt 8270, `temperature=0`, seed 42, `cache_prompt=false`,
`ignoreEOS=true`, `max_tokens=128`, and 1 warmup plus 3 measured requests per
point/cell. Keep 100 public and 24 timed outputs, and require the established
short/long proposal and acceptance counts unchanged. Capture fresh outputs,
logs, source/model/binary/environment/ownership checks, and failures; verify
restoration before and after, then independently verify live restoration in a
second process. Run the full phase analyzer and this operation analyzer after
restoration. No-draft A cells must contain no new SYCL diagnostic events.

Expected analysis artifacts are `analysis-r1/profile.json` (inherited phase
records plus per-request SYCL records) and the retained raw server logs.
Startup, public-prevalidation and warmup diagnostics are retained separately from
measured `graphs`; the opt-in is active during public validation too. No-draft A
must emit no diagnostic records. Every measured eight-row draft ubatch must have
backend records; missing an entire measured request or draft round fails closed.
Injection graphs and scheduler segments without the exact marker may be absent:
their node counts remain explicitly uninstrumented, never claimed as covered.
Multiple nonoverlapping backend segments can belong to one scheduler ubatch;
local segment node indices are not global scheduler indices. Parsing rejects
negative/overlapping/duplicate or inconsistent spans and changed proposal/acceptance
counts. Nested spans are not added to graph core wall. Post-loop log formatting
is inside the outer phase timer; use core wall for attribution, never a clean
speed comparison. All results remain development-only.

## Local preparation verification

```bash
python3 -B results/20261019-lilicorr-b70-sycl-profile/test-profile.py
bash -n results/20261019-lilicorr-b70-sycl-profile/build-candidate.sh
```

The CPU suite uses only local fixtures (raw or gzip logs where available),
checks zero-fuzz forward/reverse application, shared-clock and exact-marker
behavior, strict nested event/scheduler coverage, startup and warmup separation,
private namespace/container rebinding, and same-size source/binary/model drift
rejection through the actual adapter wrappers. It also runs the unchanged first
package's 13-test CPU suite as inherited coverage. It does not build, access a
GPU, mutate a remote host, or establish a performance result. The benchmark
publication checklist remains incomplete until the lead executes and reviews
the defined journey.
