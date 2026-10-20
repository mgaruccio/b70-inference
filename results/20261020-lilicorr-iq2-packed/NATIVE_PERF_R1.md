# Native n4 comparison: packing regresses the actual operator

**Development only. No serving-speedup claim or promotion.**

The unchanged frozen `test-backend-ops --test-file` exercised the actual target
up/down shapes (K1024/M2048 shared input; K2048/M1024 expert input), n4, 9/240
experts and 960-byte ID stride. See `NATIVE_PERF_PLAN.md` for the complete process,
fixed-input limitation, exact commands and fresh primary research. No application
source, library, SDK, driver, math, graph or queue setting was modified.

## Observed timing

ABBA x3, six processes per variant and two operations per process: **24 measured
rows**, each from the native tool's warmup and >=1-second steady measurement.
Aggregate statistic below is median of the six per-process means.

| Operator | Passed batch4 R1 | Rejected packed R2 | Packed throughput ratio | Packed wall-latency delta |
|---|---:|---:|---:|---:|
| IQ2 up n4 | 144.865 us | 188.130 us | 0.7700x | +29.87% |
| IQ2 down n4 | 140.635 us | 190.020 us | 0.7401x | +35.12% |

Passed ranges were 143.38–145.21 / 139.25–141.07 us; packed ranges were
186.69–189.19 / 188.90–190.36 us. Every ABBA block regressed in both directions.
All samples, inclusive quartiles and block ratios are in
`native-perf-summary-r1.json`; no outlier was dropped.

These are **synchronized operator-graph wall times**, including native Q8_1
quantization/dispatch/math and amortized synchronization, not pure GEMV or GPU
profiling-event durations. The generic unchanged initializer uses shuffled IDs
0..8 and independently randomized values: this cache-hot comparison is not a
serving replay or an estimate of total decode time.

## Real path and fidelity

Both variants independently passed strict unchanged Q8 **82 cases / 54 byte-exact**
and IQ2 **34 / 16** gates, followed by two representative CPU comparisons. The
actual loaded SYCL DLL paths were `/baseline/build-sycl/bin/libggml-sycl.so.0`
and `/src/build-sycl/bin/libggml-sycl.so.0`. Raw UR traces actually include the
full n4 kernel name in `urKernelCreate`; each returned n4 handle was matched to
**two `urEnqueueKernelLaunchWithArgsExp` records**. No profiling queue flag was
present. Loader/UR/debug/capture diagnostics were disabled during timing.

This supports the existing serving rejection: real hardware-dot lowering and a
smaller offline ISA did **not** make packed n4 faster. The serial integer chain
remains a possible cause, not a demonstrated causal timing breakdown. Keep passed
batch4 as the performance baseline, not packed R2.

## Restoration and retained failure

All native and timing work completed, but the initial capture and independent
recheck reported `stopped_cells` failure. Docker changed only the stopped native
container's `HostConfig.OomKillDisable` from `false` to `null`; all other canonical
container fields matched. Original identity/source/health/sole-owner checks passed.

A **new process** ran `verify-native-perf-restored.py`, repeating every original
and qualified check, and all stopped-container checks. Only that known false/null
field was normalized, and only for the exact captured native container Id. The
original container comparison is not normalized. Ten positive/negative CPU checks
also confirmed other field/mount/Id changes, true/0/string OOM values and original
container drift are rejected.

The new `independent-original-restoration-oom-normalized.json` passed with the
original Id `7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`,
health `ok`, and sole observed GPU-owner PID **1103600**. Earlier failed summaries
and independent verification files remain unchanged. No performance rerun or
fidelity waiver was used to resolve the metadata-only failure.

Raw outputs remain under `native-perf-r1/`, locally and at the corresponding
remote packed-R2 package. `retention-native-perf-r1.json` records their cross-host
SHA256 matches and any additional lossless gzip sidecars; raw originals are not
removed. Commit history retains the exact capture adapter at `2247b17e`.

Retention verified **92 files / 815,919,794 raw bytes**, with ten additional gzip
sidecars. The initial retention attempt missed the ownership snapshot created by
the new independent process; that file was copied and the full SHA check repeated.
Its failure log is retained. The 795 MB cumulative original-service stdout (47 MB
gzip) stays local/remotely with its manifest checksum rather than duplicating that
large historical log in Git; native operator logs and restoration evidence are
published.
