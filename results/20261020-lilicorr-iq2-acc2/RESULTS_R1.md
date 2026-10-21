# Exact IQ2 acc2 — native fidelity passed, operator performance rejected

**Development only. NOT promoted. No new public serving run.**
The requested dramatic end-to-end gain has not been achieved.

## Actual result

The frozen existing operator tool compared the passed IQ2 batch4 baseline against
acc2 in ABBA x3: six processes/variant, two operations/process, **24 measured rows**.
Median of the six per-process graph means:

| Operator | Passed batch4 | Acc2 | Acc2 throughput ratio | Acc2 latency delta |
|---|---:|---:|---:|---:|
| up n4 | 143.565 us | 182.965 us | 0.78466x | +27.44% |
| down n4 | 140.450 us | 187.485 us | 0.74913x | +33.49% |

Every ABBA block regressed in both directions. Baseline ranges:143.32–144.92us
and140.10–140.84us; acc2:182.32–184.44us and186.23–188.37us. All samples/block
ratios remain in `native-perf-summary-r1.json`. No outlier removal or significance
claim. These synchronous whole operator-graph times include quantization,
dispatch and amortized synchronization, not pure GPU event/kernel durations.
Generic random/cache-hot selected IDs0..8/240 are not a complete serving replay.

Two integer accumulators do not rescue packing here. Any small cross-run movement
relative to the older serial-packed result is not a paired win and does not change
the rejection versus the passed baseline. Do not promote acc2 or spend a full
public timing run on this slower candidate. The original scalar batch4 remains
the performance reference. No claim that dependency latency was the cause.

## Qualification

- Source change: only private n4 helper/callee in `mmvq.cpp`, cumulative from
  passed batch4. Signed integer partials combine before unchanged FP32 scale,
  K accumulation and XOR; all geometry/strides/other-width guards unchanged.
- Worker and lead CPU tests passed **1,684,480 exact comparisons**, source pins,
  exact GNU/Git composition/reversal, native K/XOR checks and UBSan. The malformed
  development `quants.hpp` pin was corrected before these successful checks.
- Lead12 source/build/runtime binding tests and11 native-perf contract tests pass.
- Frozen SDK rebuilt the full **105-object** SYCL closure. Whole original/seed/
  candidate source and graph checks passed; only `libggml-sycl.so.0` changed.
- Both baseline and acc2 passed unchanged native Q8 **82/54-byte** and IQ2
  **34/16-byte** gates, then both representative CPU comparisons. Exact DLL load
  and real UR calls were retained separately. Diagnostics were off during timing;
  no profiling queue, math/SDK/driver/adapter change or persistent launcher edit.

## Restoration and evidence

The capture and independent **new SSH/Python process** both exited0. Original
identity/source/health/sole-GPU/stopped-experiment checks passed. Original Id:
`7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`, health `ok`,
sole observed GPU owner PID **1136625**. See
`native-perf-r1/independent-original-restoration.json` (`passed=true`, `errors=[]`).
Only during stopped-native comparison, the existing exact-Id false/null
OomKillDisable spelling comparator was used; original comparisons, every other
field and all captured data remain untouched. The inherited comparison is restored
in `finally`, with positive/negative and exception-path CPU tests.

Commands and exact preconditions/data/cleanup are in `NATIVE_PERF_PLAN.md`,
`lead-build.sh` and `lead-native-perf.sh`. Build outputs and all raw native/UR/perf,
container/ownership/restoration data remain locally and under
`/home/mike/b70-evals/20261020-lilicorr-iq2-acc2/`. `retention-r1.json` records the
cross-host SHA matches and lossless large-file sidecars. Raw originals remain
unchanged; cumulative original-service history may stay outside Git, checksummed.
