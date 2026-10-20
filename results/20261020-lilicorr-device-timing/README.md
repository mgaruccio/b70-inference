# LiliCorr selected Q8 device timing — development diagnostic only

**Source/CPU slice complete; frozen-SDK build and GPU journey not run here. No
speed result, numerical/device proof, qualification, or promotion.** No launcher,
model, NVFP4 arithmetic, DFlash configuration, acceptance/KV behavior, or existing
package was changed. This does not achieve the requested dramatically faster
drafter; it makes the next profiling step more informative.

## Exact composition

Use a new private copy, never an original inference tree. Apply in this order:

1. Qualified r2 source, upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb`.
   Inspected read-only snapshot:
   `/tmp/lilicorr-verify-r2-source-inspect-20261020`.
   Missing `common/speculative.cpp` and `src/llama-context.cpp` come from the exact
   existing `../20261020-lilicorr-context-profile/fixtures/*.cpp.gz` hierarchy.
2. `../20261020-lilicorr-context-profile/lilicorr-profile.patch`
   SHA-256 `6691e17bc02ced6af489aad4dff3795eb1341c846acc8cbf0f292a02dee18056`.
   This is the passed explicit proc-address/TLS context-profile ancestry.
3. `../20261020-lilicorr-iq2-batch4/iq2-batch4.patch`
   SHA-256 `2688d72ca13d1c7b03683ce399ec0af38e6aaf554d43aaafbeac06a66e5324e1`.
   This is the already-qualified n4-only IQ2 delta supplied by the lead, unchanged.
4. **`device-timing.patch` is incremental to that composed seed**, not to upstream,
   r2, or either uncomposed candidate alone. SHA-256:
   `73cc432a754b75aa3483fe83261896f06b14b679106d782f90b98cb224545523`.

`source-pins.json` pins the ancestor artifacts/fixtures and all 13 inspected source
inputs at parent, context-seed, and composed-seed stages, plus the diagnostic's new
header. It is a bounded source check, **not** the lead's complete 177-file/source
or six-binary proof. Tests require the pinned snapshot (or the read-only exact
parent via `LILICORR_SOURCE_ROOT`), apply both ancestries and this delta, and check
exact hashes and reverse application. The ancestors also commute in the test.

The incremental patch changes five existing private SYCL files and adds one
private header: `common.hpp`, `dpct/helper.hpp`, `ggml-sycl.cpp`, `mmvq.cpp`,
`quantize.hpp`, `lilicorr-device-timing.hpp`. No public API or new translation unit.
Neither common/speculative nor llama-context changes in this increment.

## Small measurement boundary

- Reuses **only** `LLAMA_LILICORR_PROFILE=1`, the existing
  `set_lilicorr_profile_scope` TLS gate, and the existing host graph sequence ID.
  Keep `GGML_SYCL_ENABLE_GRAPH=0`. Graph-enabled dispatch reports unsupported and
  collects no events; graph replay/cache/lifetime is untouched.
- Adds `enable_profiling` to the **existing default in-order queue's creation**
  only when opted in and `queue_profiling` is supported. No new queue or
  device-global state. Out-of-order/other creation paths are unchanged. The
  existing `DPCT_PROFILING_ENABLED` compile-time behavior is preserved, if present.
  Set environment before startup: an already-created non-profiling queue is not
  replaced or silently upgraded.
- Exactly two launch sites: the returned `parallel_for` event for reordered Q8_1
  quantization (`ky=8`), and the returned `submit` event for the existing Q8_0
  reordered MMVQ `ncols_dst=8` kernel. Both require scoped Q8 `MUL_MAT` metadata
  with eight activation columns. This does **not** add batching: the actual route
  already has one quantizer and one ncols8 launch. No barrier intervals.
- Kernel bodies, geometry, symbolic `WARP_SIZE` (frozen build: 16), K loops,
  reductions, XOR order, arithmetic, precision tags, dispatch/fusion branches,
  allocations/pools, and waits are unchanged. Inline diagnostic markers around
  the original returned-event expressions make full-file byte restoration
  testable, not just selected arithmetic snippets.
- Each backend owns at most **2048 pending records/events** between its existing
  synchronizations. Metadata names are copied into fixed arrays; no tensor,
  graph, queue, or stack pointer is retained in a record. TLS scopes borrow
  context/queue references only during synchronous dispatch and restore nesting.
  Contexts retain separate pending batches even when sharing the same queue.
  Sequential host-thread handoff is supported; this does not make the existing
  backend safe for concurrent calls on the *same* context.
- Queries occur only after **that backend's original `stream->wait()` returns**.
  A nonblocking completion-status check rejects incomplete events before any
  profiling query. No query at submit/graph end, no event wait, no additional
  queue wait. Drain releases the entire batch; teardown without that existing
  synchronization discards records explicitly, without querying or waiting.

### Separate records and honest coverage

Existing `LLAMA_LILICORR_PROFILE` and `LLAMA_LILICORR_SYCL_PROFILE` schemas are
byte-for-byte preserved. New records use **`LLAMA_LILICORR_DEVICE_TIMING v=1`**:

- `stage=coverage`: original graph ID, node count, selected submissions, retained
  records, dropped records, submission exceptions, and explicit
  `coverage=q8_reordered_ncols8_only other_device_work=unmeasured`.
- `stage=event`: graph/node/ordinal/device, copied output/weight names, dimensions,
  actual kernel kind, host submission begin/end/elapsed in **microseconds**,
  device command start/end/elapsed in **nanoseconds**, and status. Missing,
  unsupported, foreign-queue, incomplete, invalid, or failed profiling produces
  **`unavailable`**, never a fabricated zero. Storage/copy/property failures and
  capacity overflow increment dropped coverage; original submit exceptions still
  propagate. Names are whitespace/`=` sanitized for single-line fields.
- `stage=drain`: existing wait's host interval, separately measured query/log host
  cost, record count, `attribution=shared_queue_not_pure_drafter`. Kernel records
  carry the owning graph IDs; this shared-queue wait is not a drafter fraction.
- `stage=discard`: pending count and unavailable time when no existing backend
  synchronization drained the records before teardown.

Device and host clocks have different domains; do not subtract their timestamps
or treat kernel sums as total drafter time. No BF16 projection, TOP_K, copies,
other dtypes/kernels, or target work is covered. A graph with no selected events
is **unmeasured**, not free. Other existing waits may complete work earlier; the
record remains owned until the backend synchronization, rather than adding a
flush hook at every wait. Original ubatch `submit_us` still includes host runtime
stalls/internal waits. Original process feature-read/injection clocks remain;
feature reads can drain target work. `t_draft_us` still excludes `impl->process`.
No total drafter fraction can be inferred from this slice.

Queue profiling affects all commands on that shared queue in an opted-in process,
but only scoped selected kernels are collected. Diagnostic CPU/logging/property
and event-retention costs perturb timings. With the opt-in off, the CPU checks
observe identical submissions/properties and no diagnostic clock, allocation,
query, retention, or logging; this is **not** a claim of identical binaries or
zero extra host instructions. Timings with diagnostics on are never clean speed
evidence.

## Research and CPU/source close gate

Fresh official research was supplied by the lead before implementation:

- https://github.khronos.org/SYCL_Reference/iface/event.html — command start/end are
  device timestamps; profiling queries can block. Query only after the existing
  wait and verify completion first.
- https://github.khronos.org/SYCL_Reference/iface/usm_allocations.html — freeing USM
  does not wait. No USM lifetime/free/synchronization changes are made.
- https://intel.github.io/llvm/syclgraph/SYCLGraphUsageGuide.html — recording and
  profiling/wait constraints: graphs remain off; no rolling-header graph APIs.

Run tier: **development CPU/source preparation**, not a performance comparison.
Source baseline: composed context-profile + n4 IQ2 seed. Intentional difference:
only the opt-in diagnostic above. Exact command from this worktree:

```sh
python3 -B results/20261020-lilicorr-device-timing/test-device-timing.py
```

Observed: **5 tests passed**. Raw command/environment/output:
`cpu-checks.txt` (Python 3.14.7, host GCC 16.2.1, Linux x86_64). The actual private
header, scope gate, queue-init body, two launch functions and backend synchronize
function are used by a CPU C++ mock driver, compiled with warnings-as-errors both
with and without `DPCT_PROFILING_ENABLED`. Tests cover opt-out/target isolation,
nested and separate-thread scopes, cross-thread drain, deferred queries, exact
original wait count, queue identity, copied metadata after tensor destruction,
unsupported/error/incomplete events, submit/wait failures, bounded overflow and
no-wait discard. The deliberate original-wait-error invocation returns **1** and
confirms no profiling query; it is an expected negative gate, not a failing suite.
Full-file stripping restores the composed source exactly, including original
IQ2/Q8 arithmetic, geometry, all old schemas, and untouched phase clocks.
Temporary test sources/executables live only in ignored `.scratch/` and are
removed on test exit. No SYCL SDK/application build, GPU access, installation,
container, remote command, service, or device-global profiler was used.

## Lead-only integration / E2E remains mandatory

The lead owns build/runtime bindings and must profile **this composed source**,
not invoke an unchanged runner still bound to its old seed. No adapters, build
framework, or analyzer are added here. The frozen oneAPI compiler/runtime
**2026.0.0** must verify these standard event/property APIs and actual lowering;
CPU mocks do not establish their availability or device completion behavior.

Defined journey supplied before implementation:

1. Frozen SDK build, no GPU mounts or host installs; complete source and six-
   artifact proof. Relative to the composed seed only the SYCL library should
   change; account for header dependencies in the actual build. Verify old strict
   context coverage records without weakening schemas. Missing event records,
   unsupported/error status, or dropped coverage are not a timing success.
2. Healthy sole original owner and exclusive B70. Run native IQ2/Q8 byte and
   negative-route gates first using the already-qualified batch4 route contract.
   Preserve full eight positions, 7x8 lattice, rejected-KV visibility/cleanup,
   max3, original launch arguments and all arithmetic/precision constraints.
3. Existing public HTTP/SSE **A1/W3a/W3b/A2** journey, only W3 privately rebound to
   the composed diagnostic and opt-in: **100 public + 24 measured** requests,
   short 55 / long 8270 input tokens, temperature 0, seed 42, `max_tokens=128`,
   `cache_prompt=false`, ignore EOS, unchanged warmups. Require exact output
   identity and proposals/accepted **short 129/84, long 133/82**. Retain original
   phase/coverage records plus raw new event records with matching graph IDs.
4. Independent original identity/source/health/sole-GPU-owner restoration in a
   **separate invocation**, including after failure. Use the existing cleanup
   lifecycle, not a new service/launcher. Retain exact executable commands,
   environment, build and native/public failures, all raw outputs/server logs,
   complete source/six-artifact proof and independent restoration evidence under
   the lead's new private run directory. Do not replace an existing result.

Those exact runtime commands/artifact paths are intentionally left to the lead's
existing adapter bindings; this source-only slice cannot truthfully report an
executed GPU E2E. No benchmark-standard publication checklist is complete and no
promotion is authorized by diagnostic timing.
