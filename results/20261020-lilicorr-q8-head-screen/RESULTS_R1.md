# Q8 head screen — do not implement the proposed weight-reuse change

**Development diagnostic only. No optimization, serving-speedup or promotion.**
The original request for a dramatically faster drafter remains unmet.

## Observed native operator cost

The unchanged frozen executable/library ran the existing `--test-file` tool with
Q8_0 `[4096,131072]` weights and F32 inputs, neutral parameters, in n1/n4/n8 file
order. Three independent `perf` processes produced nine rows after the tool's
ordinary warmup and >=1-second steady graph measurements. Median of the three
per-process means:

| Generic case | Operator graph wall time | Range |
|---|---:|---:|
| n1 | 952.18 us | 952.13–952.19 us |
| n4 | 1162.88 us | 1162.87–1162.92 us |
| n8 | 1258.94 us | 1257.11–1261.39 us |

These are random cache-hot whole operator graph wall times, including Q8_1
quantization/dispatch/amortized synchronization. They are **not device-event,
individual-kernel, serving, or aggregate drafter durations**. No output/head share
of real serving latency is asserted. Even an illustrative 43 × 1.163ms is only
~50ms; actual call frequency and cache/context effects would be needed to turn
that into serving attribution. This screen provides no credible dramatic-payoff
case for a head-only rewrite, so the proposed shared-weight change is dropped.
A false source `shared_weights` trait alone never proved physical repeated loads.

## Actual route, fidelity and limitations

The separate diagnostic passed all three CPU comparisons and loaded the exact
`/src/build-sycl/bin/libggml-sycl.so.0`. Raw UR logs include real n4/n8
`reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<4/8>` kernel names; each returned handle
maps to **one successful launch** in the diagnostic. Timing processes had
loader/UR/debug/capture diagnostics off. No profiling queue or workaround was used.

The generic n1 case instead launches `q8_0_esimd_launch<8>` (the template argument
is its row geometry), unlike served `output.weight` n1's MMVQ_REORDERED route.
Therefore the n1/n4 numbers must **not** become a NoDraft or batching speed ratio.
The n4/n8 route observations are kept separately in `native-perf-summary-r1.json`.
Native Q8 82/54-byte and IQ2 34/16-byte gates also passed unchanged. These operator
checks do not substitute for a new optimization's full public journey.

An initial offline analysis assertion expected two successful n4 launches by
analogy with an older diagnostic. This was an analyst assumption, not a frozen
native predicate: the actual trace has one successful return for the created
handle. The corrected analysis requires a real handle-to-successful-launch match
and records the observed count. No captured record or native gate was changed.

## Restoration and retained failure

Native work exited0 and all nine timing rows completed. The initial capture and
independent restoration summaries remain **failed**, because Docker changed only
the stopped native container's `HostConfig.OomKillDisable` false→null. Same Id,
Config and canonical mounts matched; original identity/source/health/sole-GPU
checks passed throughout.

A separate SSH/Python invocation of `verify-native-perf-restored.py` then repeated
all original, source/build, hash, ownership and stopped-container checks. It reuses
the existing exact-Id false/null comparator from the earlier IQ2 diagnostic. Only
that exact stopped experimental Id is projected for comparison; the original
container is not normalized, and every captured record remains unchanged. Current
before/after and negative field/mount/Id/true/0/string controls passed locally.

The new `native-perf-r1/independent-original-restoration-oom-spelling.json` has
`passed=true`, `errors=[]`; task `b88391ea0` exited0 and emitted
`Q8_HEAD_SCREEN_STRICT_ORIGINAL_AND_EXACT_ID_OOM_SPELLING_RECHECK_PASSED`.
Original Id remains `7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`,
health `ok`, sole GPU owner PID **1127859**. No performance rerun or promotion.

The first two worker CPU issues (ABI field count and capture mkdir) were fixed
before GPU execution. Lead review additionally fixed copied-runpy-dictionary
binding; all eight CPU tests passed locally and remotely before capture. Real
remote baseline/hash guards remained unmocked and passed. CPU-only tests mock
remote-only input/path reads and are not presented as remote proof.

Exact invocation is retained in `lead-native-perf.sh`; plan, primary research and
limitations are in `NATIVE_PERF_PLAN.md`. Raw outputs and failed/successful
restoration records remain under `native-perf-r1/`, locally and at
`/home/mike/b70-evals/20261020-lilicorr-q8-head-screen/`. Retention records preserve
cross-host checksums and lossless gzip sidecars; raw originals are not deleted.
