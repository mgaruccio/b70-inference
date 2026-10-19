# LiliCorr B70 phase profile — development results

**Profiling only: no optimization or speedup is claimed, and nothing is promoted.**
The original service was restored and independently checked. Both analyzers report
`status=analyzed`, `errors=[]`; inherited analysis reports
`independently_restored=true`. All 100 public prevalidation jobs and 24 measured
responses passed. All existing native gates, including IQ2/Q8 singleton gates,
passed on the actual mounted binaries. Q8 proposal/acceptance counts remained
129/84 (short) and 133/82 (long). Long requests selected 135 raw lattice tokens;
the existing post-walk cap yields 133 proposals, without changing observations.

## Measured attribution

Median of six measured W3 requests per point; milliseconds per response. These
are host wall intervals, including any existing backend waits, **not GPU event
times**. Logging is enabled only in W3, so this run cannot establish a performance
comparison or a candidate speedup.

| Decode-region work | Short (43 rounds) | Long (45 rounds) |
|---|---:|---:|
| Draft batch preparation | 0.126 | 0.132 |
| Draft `llama_process` call | 301.112 | 330.523 |
| Existing lattice getter / completion wait | 0.405 | 0.420 |
| CPU lattice selection | 0.024 | 0.021 |
| Full feature-processing callback | 35.931 | 36.694 |
| ↳ target-feature getter waits + host gather | 5.819 | 5.770 |
| ↳ injection `llama_process` call | 29.413 | 30.543 |
| ↳ projected-output getter / completion wait | 0.309 | 0.342 |
| ↳ embedding ring copy | 0.142 | 0.151 |
| Graph build/allocation, injection + draft (**nested**) | 17.878 | 18.435 |
| Graph input setup, injection + draft (**nested**) | 1.069 | 2.235 |

Indented components are inside the feature callback. The last two rows are
inside the process/draft calls and must **not** be added again. Component medians
need not sum to the median parent duration. Every measured draft has eight rows,
826 graph nodes, and `reused=0`. Injection graphs have 59 nodes; consecutive
1024-row long-prefill injections can reuse their graph.

Before the first draft, feature callbacks total 10.413 ms short / 672.002 ms long.
For long requests this includes 309.661 ms in feature getters/gather,
324.164 ms in injection calls, and 37.795 ms in projected-output getters.
The gather timer includes existing target completion waits; it is not a pure
memory-copy timer or a causal estimate of removable prefill overhead.

**Decision:** CPU lattice walking and ring copies are negligible. Graph rebuilding
is real but accounts for only about 18 ms across injection and draft calls per response.
The substantive drafting cost is inside backend execution/submission, which can
block internally. Operator-level attribution is the next experiment; do not call
301/331 ms “CPU launch overhead” or assume device-resident ring storage removes it.

A separate source review rejected simply enabling SYCL graph replay: the current
path records/updates every call; alternating injection/draft graph allocations,
dynamic KV shapes, and temporary-buffer lifetimes make cached replay unsafe without
further work. The blanket CONCAT rejection is conservative/stale, not evidence
that the current LiliCorr concat kernels themselves wait on the host.

## Execution and retained evidence

On `inference-host`, package `/home/mike/b70-evals/20261019-lilicorr-b70-profile`:

```bash
bash build-candidate.sh
python3 -B source-proof.py --built
python3 -B test-profile.py -v
python3 -B run-profile.py --lead-run-profile
# Separate process, after capture/restoration:
python3 -B run-profile.py --lead-after-profile-restore
```

The isolated frozen-SDK CPU build exited 0; only `libllama-common.so` and
`libllama.so.0.7.0` changed among the six protected artifacts. Parent source,
backend library, target arithmetic, model bytes, requests, acceptance policy,
and persistent launchers were unchanged. Full environment, create/PID arguments,
actual opt-in environment, mounts, telemetry, source/binary/model hashes, native
logs, HTTP/SSE data, and restoration records are under `profile-r1/`.

- `profile-r1/analysis-r1/profile.json`: per-request events, phase totals and graphs.
- `profile-r1/analysis-r1/analysis.json`: inherited HTTP/fidelity analysis.
- `profile-r1/analysis-live-restoration.json`: independent live restoration proof.
- `build-attempt-r1/`, `post-build-verification.log`, `runtime-precheck.log`: build.
- `cpu-checks-host.log`: retained initial host test failure: two fixtures assumed
  gzip logs, but the host retains the same original evidence uncompressed.
- `cpu-checks-host-r2.log`: all 13 tests passed after reading raw or gzip forms;
  disagreement between both forms is rejected. Local 13-test run also passed.

The first foreground evidence copy hit its 60-second command timeout, not a GPU
or validation failure. Transfer was continued as a background copy from immutable
remote originals; no GPU rerun or observation rewrite was required.
All **2,481** transferred profile files were SHA256-equal to remote originals.
For Git retention, **80** log files (879,143,677 raw bytes) have verified lossless
gzip copies totaling 52,086,770 bytes. Local and remote raw originals remain;
Git stores large logs as `.log.gz` (decompress to restore the original filenames).

For reproduction of the table, select `phase == 'measured'` and each `point` in
`profile.json.requests`, take `statistics.median` of each corresponding
`from_first_draft[stage][field]`, and divide microseconds by 1000. Prefill values
use `before_first_draft`. Parent and nested fields remain separate in the artifact.

Intel reference informing the timing boundaries:
https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/kernel-launch.html
