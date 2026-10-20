# LiliCorr process-boundary / selected Q8 timing — development only

**Source/CPU slice complete; SDK/GPU validation remains lead-owned. No speed,
numerical/device, full critical-path, or promotion claim.** This follow-up keeps
`8b560b9db5fbcd5c6fee7c4aeee33407d22b9051` as its committed base and extends the
same diagnostic, not the optimization scope. The lead explicitly authorized
partial device coverage: width-46 library execution is **host-only/unavailable
on device**, not a fabricated kernel duration.

## Exact composition

Apply to a new private copy, never an original inference tree:

1. Qualified r2 source, upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb`,
   inspected at `/tmp/lilicorr-verify-r2-source-inspect-20261020`. Missing common/
   context files use the existing exact context-profile gzip fixture hierarchy.
2. `../20261020-lilicorr-context-profile/lilicorr-profile.patch`, SHA-256
   `6691e17bc02ced6af489aad4dff3795eb1341c846acc8cbf0f292a02dee18056`.
3. `../20261020-lilicorr-iq2-batch4/iq2-batch4.patch`, SHA-256
   `2688d72ca13d1c7b03683ce399ec0af38e6aaf554d43aaafbeac06a66e5324e1`.
4. The **current** `device-timing.patch`, SHA-256
   `84af63e37259303c3758cf765d935f6fd08fa2e5aec135b6ddd5b7fc932ad0b2`.

This file remains a cumulative increment relative to **context-profile + IQ2**.
Do not apply it on top of the old device-timing patch. The old patch remains in
commit `8b560b9d`, with its pin preserved in `source-pins.json`. That JSON pins all
13 inspected inputs at parent/context/composed stages and the diagnostic's new
header; this is not the lead's complete source or six-binary proof.

The current patch touches `common/speculative.cpp` and five existing private
SYCL files (`common.hpp`, `dpct/helper.hpp`, `ggml-sycl.cpp`, `mmvq.cpp`,
`quantize.hpp`), plus private `lilicorr-device-timing.hpp`. No public API, new TU,
flag, queue, pool, service, persistent store, graph change, or adapter/framework.

## Corrected route evidence and bounded device coverage

The supplied read-only fulltrace is:
`/home/mike/code/b70-inference/results/20261020-lilicorr-verification-routes/capture-r2/VSPF-W3a/server.stderr.log`
(SHA-256 `70db71ff814d79aeda75e6965c92578ef4b83fdfc1c32444354fef28410c60c9`).
A streaming scan of its `BOLT_ROUTE` key/value records found **62,823 records,
zero `fc.weight` or K=24576/M=4096 records**. Its sole literal `fc.weight` hit is
model loading at line 1130. The immutable backend's
`bolt_route_selected_weight` explicitly excludes that weight. Therefore earlier
claims of observed fc route coverage in this trace are not evidence.

Source plus the supplied frozen build closure establish the width-46 outer
route: MMVQ is capped at 8; `SYCL_USE_XMX` is enabled without FORCE_MMQ and caps
MMQ at 32; `fg_plain_ok<block_q8_0>` rejects dense fused GEMM. The remaining
library route's actual precision/oneDNN-versus-MKL choice is **not** established
by that trace. `gemm.hpp` uses `matmul_prim.execute(stream, matmul_args)` without
an event return. No interop/API swap or dependency was introduced here.

The source-only extension therefore:

- Removes the unjustified n8 restriction at the **same two actual launch sites**:
  reordered Q8_1 quantizer and reordered Q8_0 multi-column MMVQ, now widths 2..8
  (including injection 4/5 and generation 8). An unreordered/other fallback is
  still unmeasured. Live records must establish which hooks actually execute.
- Copies backend-context identity as an integer, weight/output names, logical
  weight K/M and input columns, actual launch K/rows/columns, and route family.
  Graph ID still comes from the existing scoped context-profile sequence.
- Adds four tiny, fc.weight/n46-only host markers **inside the existing terminal
  branches**: `LIBRARY_DNNL_F16`, `LIBRARY_MKL_F16`, `LIBRARY_DNNL_F32`,
  `LIBRARY_MKL_F32`. Their native calls and arguments remain unchanged. The
  executed branch produces `stage=library_submit coverage=host_only_library_call
  status=device_unavailable_library`, with device fields `unavailable`.
  This interval excludes preceding conversion/dequantization and is not GPU time.

`LLAMA_LILICORR_DEVICE_TIMING v=1` remains separate from old strict schemas.
`stage=event` reports actual kernel events: host submit begin/end/elapsed in **us**
and device command start/end/elapsed in **ns**. `stage=coverage` counts observed
submission sites/records (including library calls, **not** physical kernel count),
drops/errors, and explicitly unmeasured other work. `stage=drain` separates the
original wait's host interval from query/log host cost and labels shared-queue
attribution. `stage=discard` reports pending data discarded without an existing
synchronization. Unsupported/error/incomplete data never becomes zero time.

The opt-in remains only `LLAMA_LILICORR_PROFILE=1`, set before queue creation;
keep `GGML_SYCL_ENABLE_GRAPH=0`. Queue-local profiling is enabled only on the
existing default in-order queue when supported. Existing DPCT compile-time
profiling behavior is preserved. TLS scopes restore nested target isolation.
Per-backend ownership stays bounded at 2048 pending records/events; names and
metadata are values, not retained tensor/graph/stack pointers. Collection queries
occur only after the backend's **existing** wait, with a nonblocking completion
check first. No new wait/barrier/free or buffer-lifetime changes. Sequential
thread handoff is supported, not concurrent use of one backend context.

## Host target-to-drafter boundary

New `LLAMA_LILICORR_PROCESS_COMPONENTS v=1` records use the existing profile gate
and a temporary per-chunk stack accumulator. They separate:

- Sum of the existing target layer getter intervals. This can include **target
  queue drain and output reorder**, not pure copy or drafter compute.
- Existing feature CPU memcpy loops, with actual calls/bytes. Loop time includes
  original pointer/index work. No new memcpy or getter calls.
- Existing injection `llama_process` interval, reusing the adjacent original
  submit timestamps. It includes host submission and any internal waits.
- The **existing** `llama_get_embeddings_nextn(ctx_dft)` interval. It can include
  injection/shared-queue drain; no extra getter or synchronization is inserted.
- Existing projected-ring copy loop, including original position/slot work,
  with actual copied rows/bytes. Not-called/missing projection remains explicit.

Copied target/draft context IDs, sequence, batch/sequence/chunk rows, offset,
original batch/chunk begin timestamps and process/projection intervals permit
association with existing owners. Failed process return codes remain unchanged;
projection/ring data not executed is `unavailable`. One summary plus one separate
format/log host-cost record is emitted per chunk, outside the timed components.
The logging-cost record excludes its own reporting call. No persistent registry.

Original `LLAMA_LILICORR_PROFILE` and `LLAMA_LILICORR_SYCL_PROFILE` schemas are
unchanged. `t_draft_us` still excludes process; it cannot establish the total
drafter fraction. BF16, TOP_K, other kernels/copies and target GPU work remain
unmeasured. Do not mix host/device clock domains or sum these into total GPU time.
Profiling properties, metadata, clocks and logs perturb costs: this is diagnostic
measurement, **not clean performance or pure critical-path evidence**.

## Research and executed close gate

Fresh official research supplied before implementation:
- https://github.khronos.org/SYCL_Reference/iface/event.html — profiling queries can
  block; device timestamps are distinct from host time. Defer until existing wait.
- https://github.khronos.org/SYCL_Reference/iface/usm_allocations.html — frees do not
  wait; preserve all existing USM lifetimes and synchronization.
- https://intel.github.io/llvm/syclgraph/SYCLGraphUsageGuide.html — no profiling or
  waits while recording; graph behavior is untouched and remains off.

Tier: **development CPU/source preparation**. Baseline: exact composed source.
Intentional difference: only this opt-in measurement. Command:

```sh
python3 -B results/20261020-lilicorr-device-timing/test-device-timing.py
```

**7 tests passed**, retaining the original five-test core. `cpu-checks.txt` keeps
raw commands/environment/results, including the earlier five-test checks and
expected wait-error exits (1, with no query). Host GCC 16.2.1 and Python 3.14.7;
no SDK/application build or device access.

Tests strip all additions to restore whole source files byte-for-byte, verify
pins/composition/reversal and old schemas, and compile actual extracted C++ with
warnings-as-errors. SYCL mocks cover off/target/nested/thread isolation, original
wait count, deferred queries, copied metadata, capacity/errors/discard, widths
4/5/8 and the four actual fc46 library call blocks without event capture. The
host fixture executes the actual process chunk with mocked existing APIs and
real CPU gather/ring loops at widths 4/5/46: call order/counts, data/bytes, opt-out,
non-LiliCorr, return-error and missing-projection guards, honest attribution and
separate logging cost. These are **not** numerical/GPU or oneAPI compatibility
proofs. Test sources/executables stay in ignored worktree-local `.scratch/` and
are removed on exit.

## Lead-only SDK / native / public E2E

Lead must bind the **current composed source** in existing adapters, build with
frozen oneAPI **2026.0.0** (no GPU mounts/host installs), and retain complete source
and six-artifact proof. Since common now changes, relative to the prior diagnostic
both SYCL and common libraries can differ; llama-context source stays unchanged.
Verify real routes/coverage without weakening the old full-phase validator.

Then unchanged exclusive-B70 **A1/W3a/W3b/A2**: native IQ2/Q8 byte and negative
route gates first, **100 public + 24 measured** HTTP/SSE requests, short 55 / long
8270 tokens, temperature 0, seed 42, max128, cache false, ignore EOS and original
warmups. Exact outputs and proposals/accepted short **129/84**, long **133/82**;
all eight positions, 7x8 lattice, rejected-KV visibility/cleanup and max3 retained.
No NVFP4 or DFlash tuning. Independently restore/verify original identity, source,
health and sole GPU ownership in a **separate invocation**, including failures.
Retain exact commands/environment, failures, raw native/public/server logs and
restoration evidence under the lead's new private run directory. No promotion.
Partial timings can select the next probe, not establish whole-injection/drafter
GPU time or the requested dramatically faster drafter.
