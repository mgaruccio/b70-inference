# Bolt warmed decode attribution before MTP

Development-only experiment. No kernel candidate or persistent deployment change.
Baseline: accepted staged-SSM archive
`/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1`,
upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb`.

## Declared process (before execution)

All workloads and analysis run on `mike@inference-host`, explicit Bash, SSH
`-o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no`.
Remote artifacts: `/home/mike/b70-evals/20261017-nemotron35-bolt-decode-profile/`.
Use original weights, q8 KV, F32 recurrent state, context65536, batch/ubatch1024,
single sequence, ESIMD1/FUSION1/graphs0; boost0, 16 CPU caps3801000kHz, 275W.
No unrelated GPU work, weight/source writes, hardware changes or persistent launcher edits.

1. Capture original container, accepted binary hashes, hardware, model verification
   record, tool revision and exact experimental launch argv. Retain the original
   container (stop/start, never recreate it).
2. Fresh **uninstrumented non-debug** baseline runs the existing native385 gates;
   run existing `probe-api.py --mode smoke`, `--mode quality` (gate-v2 20/20),
   then `--mode bench` and `--mode long` with three repetitions each. Stop on failure.
   Validate frozen measured prompt counts55/8270, output128, cache_n=0 and
   exact8238-token retrieval `BOLT-7319-CEDAR`. Keep inherited warmups unchanged.
3. Fresh profile launch uses identical accepted binaries and runtime, with only
   unitrace and its two isolated mounts added. Native gates run before unitrace.
   Unitrace collection starts paused. Run fresh API gate-v2 while paused, then
   an actual8270/128 forced-decode warmup. Resume immediately before one identical
   8270/128 streamed `/v1/chat/completions` request; stop collection after response.
   This warmup intentionally uses the full measured prompt, not output1.
4. Collect complete request timeline, server log, requests, responses and counters.
   Decode attribution requires exactly128 device-to-host logits copies of
   131072 F32 values (524288 bytes), each immediately following the output Q8
   projection. Accepted `src/llama-context.cpp:2054-2059` copies
   `n_outputs*n_vocab*sizeof(float)`; prior output1 trace has exactly one such
   projection/copy pair, and server metadata confirms n_vocab131072.
   First pair completes prefill; subsequent127 pairs complete autoregressive steps.
   Report all127 and a steady subset after copy16 through copy128 (112 steps),
   checking full-window coverage, absence of prefill-only kernels and per-step
   repeated graph structure. If boundaries do not validate, do not claim decode attribution.
5. In nested `finally`, stop each owned experiment container, retain logs and trace,
   restart original accepted container, wait for real API readiness, check READY,
   unchanged accepted hashes and hardware. No retry-until-positive profiling.
6. Analyze on inference-host; retain all raw logs losslessly (`gzip -n` for large
   local files), exact commands and failures. Decide at most one straightforward
   material optimization, otherwise move to MTP/drafting without another rewrite.

Profiler durations are diagnostic only. Report device sum, interval union, span,
gaps and host API durations separately; synchronization API times include waiting.
No bandwidth saturation, occupancy or transferable GPU-speed claims. This is not
publication-standard BetterBench/concurrency/maximum-context/peak-memory coverage.

## Fresh external research (2026-10-17)

- https://github.com/intel/pti-gpu/blob/master/tools/unitrace/README.md
  External Linux session control supports `--start-paused`, `--resume`, `--pause`
  and `--stop`; no application instrumentation is required. Deprecated conditional
  collection, new ITT hooks, guessed kernel filters and stall sampling are unnecessary.
- https://github.com/intel/pti-gpu/blob/master/chapters/device_activity_tracing/LevelZero.md
  GPU ticks require host/device correlation; Chrome trace timestamps and durations
  use microseconds. Do not compare raw device ticks directly to API wall clocks.
- https://oneapi-src.github.io/level-zero-spec/level-zero/1.14.33/core/api.html#zedevicegetglobaltimestamps
  Timer resolution and valid bits govern Level Zero timestamp interpretation.
- https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
  Prompt-cache and forced-output controls must be explicit. Preserve the existing
  proven chat payload rather than switch API or template for the profile.

Local unitrace source revision: `887bba6e28ce84cc0d3813ef876e24add107c318`.
Its `unitimer.h` and `chromelogger.h` use microseconds with an intended epoch offset.
**Observed correction:** this build derives the offset from CLOCK_BOOTTIME but adds
CLOCK_MONOTONIC_RAW timestamps, so its absolute trace clock is not wall time (see
failure2 below). Semantic boundaries use ordered logits copies and relative durations.

## Preflight notes

Read-only orientation initially guessed `build/bin` instead of `build-sycl/bin`,
`power1_max` instead of `power1_cap`, and an uncompressed historical trace path
instead of its `.gz` file. Those probes failed; no workload or runtime was changed.
Corrected checks verified the accepted hashes, health and hardware. The native Lab
preview tool returned ENOENT for its unavailable socket; no repair was attempted.


## Observed gates and fresh uninstrumented baseline

Both fresh launches passed native385 and API gate-v2 **20/20**. Baseline also passed
smoke and exact8238-token retrieval. Measured short55/long8270 prompts each produced
128 forced tokens with cache_n0. Warmups retained54/8269 prompt tokens respectively.

- Short prefill:468.099,469.816,469.258ms; median**469.258ms**.
- Long prefill:8221.605,8234.157,8228.107ms; median**8228.107ms** (1005.091tok/s).
- Short decode:40.426689,40.408180,40.355407tok/s; median**40.408180tok/s**.
- Long decode:38.878392,39.057214,39.029847tok/s; median**39.029847tok/s**.

These freeze the current development baseline, not a candidate speedup. All samples
remain included. `run-r1/baseline/{bench,long}/` holds requests, streamed events,
results and summaries; `gates-verified-final.log` independently checks the counts.

## Retained failures and corrections

1. **Controller final restoration assertion:** both experiment phases passed, original
   container restarted and READY passed, but Docker returned the identical mount
   records in a different list order. Raw `run-r1/summary.json` remains **failed**;
   `run-r1/controller.py` preserves the executed controller. A separate
   `check-restoration-r1.py` verified same container ID, exact Config, every mount
   field after sorting by destination, original hashes, hardware, health and READY.
   `run-r1/restoration-check.json` records that independent success. Top-level
   controller now compares mount records order-independently; it was **not rerun**.
2. **First analysis failed its wall-clock assertion:** trace starts9.161s before the
   API wall timestamp and ends9.179s before response. Both CPU and GPU trace clocks
   share this offset. `clock-offset-check.json` (measured after capture) finds
   BOOTTIME−MONOTONIC_RAW≈9.178755s; pinned unitimer source explains the mixed clock
   bases. No driver/tool/network clock changes were made. Use relative durations:
   trace prefill8883.103ms vs API8888.743ms, decode6997.698ms vs API7003.434ms.
   Final analysis requires both residuals<50ms and<1%; no inferred epoch correction.
   `analysis-r1.log`, trace diagnostics and raw trace remain unchanged.
3. Real data also showed Q8 kernel names alone are not unique to the output head:
   there are9272 `q8_0_esimd_launch<8>` calls, but exactly128 use the output geometry
   `{65536;1;1}/{8;1;1}` and immediately precede the128 full-logits copies. Smaller
   projections are now excluded only from **boundary detection**, not attribution.
   Epoch-sized float addition also rounded sub-us interval ends; union and window
   calculations now center timestamps before adding durations. Regression tests
   cover non-output Q8 calls, epoch-sized union and shift-invariant window accounting.
4. Independent native-log inspection initially expected each success line once;
   shell `tee` plus validating `grep` prints each twice. The corrected check verifies
   the exact duplicated sequence for the nine suites totaling385, not770 tests.
5. First compression launch failed shell parsing before compression. Its raw output
   is retained in `compression-launch-failure.log`. Literal `compress-evidence.sh`
   succeeded with `gzip -n` and `gzip -t`; remote raw originals remain intact.

No inference, quality gate, timing request or profile capture was retried. Static
review agreed on127/112 window indexing and the diagnostic contract; subsequent
real-trace failures above took precedence over its preliminary pass.


## Final trace validation

`analysis-r2.log` ends `BOLT_WARMED_DECODE_ANALYSIS_PASSED`; final analyzer tests
pass **10/10** on inference-host. `analysis.json` includes full machine-readable
kernel attribution, all127 copy-to-copy intervals and steady112-step intervals.
All302470 device events reconcile with the unitrace table's counts and summed
nanoseconds; the128 geometry-qualified projection/copy pairs delimit prefill and
decode, with no grouped GEMM or staged-prefill SSM kernels in decode.

- All127 decode steps: summed device3095.316ms, union3094.791ms,
  copy-end window6997.698ms, uncovered3902.908ms.
- Steady112 steps: summed device2729.749ms, union2729.285ms,
  copy-end window6173.375ms, uncovered3444.089ms; median interval55.048ms.

These are **instrumented** intervals; uncovered time includes host sampling and
profiler costs. They must not replace the uninstrumented baseline above.
The union is computed from centered timestamps; tiny apparent overlaps may reflect
trace timestamp quantization and do not establish concurrent execution.
Clock implementation examined:
https://github.com/intel/pti-gpu/blob/887bba6e28ce84cc0d3813ef876e24add107c318/tools/unitrace/src/unitimer.h

## Decision

**Stop pre-MTP kernel tuning; keep the accepted SSM winner and move to MTP/drafting.**

The warmed trace identifies heavy matrix-vector work, not a clear small wasted-work
fix. Reordered NVFP4≈23.797%, ordinary IQ2 matvec≈22.473%, and routed IQ2 matvec≈20.850%
of summed steady-window device durations dominate (≈67.12% combined). Each executes
80 calls per autoregressive step. Router≈1.874%, recurrent SSM≈2.283%, and flash
attention≈2.523% are smaller; the vocabulary Q8 projection alone is≈3.91%.

The reordered NVFP4 path is active; the output1-only cold-layout prefill attribution
must not be reused to motivate another decode dequant rewrite. Kernel names/shapes
establish implementations, not bandwidth saturation or a proven bottleneck cause.
Improving these matvec paths would need further kernel work and correctness/performance
experiments rather than a straightforward cleanup, so no new candidate is selected.

Instrumentation increased reported decode time from baseline median3253.920ms to
7003.434ms (about2.15×). Consequently trace gaps are **not** evidence that production
is launch-bound, and device shares cannot be treated as guaranteed speedup ceilings.
MTP/drafting is the next experiment, not an improvement claimed by this profile.

## Reproduction and evidence

Run on inference-host in the remote root with the SSH options above:

```bash
set -euo pipefail
python3 -B run-decode-profile.py 2>&1 | tee capture.log
# Actual run-r1 controller is retained; top-level differs only in mount-order validation.
python3 -B check-restoration-r1.py 2>&1 | tee restoration-check.log
python3 -B test-analyze-decode-profile.py 2>&1 | tee analysis-tests-r2.log
python3 -B analyze-decode-profile.py 2>&1 | tee analysis-r2.log
bash compress-evidence.sh
```

Fresh capture refuses existing run/container names. Do not rerun against this
artifact directory. CPU-only analysis reads the saved data; local large evidence is
losslessly compressed, while remote raw files remain. The compressed Chrome trace
is about19MiB; no trace events or benchmark samples were removed. All failed logs
are retained separately from corrected analysis and restoration results.
Log names above refer to remote raw paths; repository log copies use `.gz`.
`final-residency.log.gz` verifies the accepted container, three hashes, hardware
and health after analysis; `trace-compressed.sha256` matches the retrieved trace.
No autonomous workers or GPU experiment processes remain; experimental containers
are stopped and retained. Only the original accepted Bolt container is running.
