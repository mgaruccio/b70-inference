# LiliCorr SYCL operator profile — partial coverage, development only

**No speedup or promotion.** Capture and independent analysis completed with
`status=analyzed`, `errors=[]`, and `independently_restored=true`. All 100 public
prevalidation jobs, 24 measured responses, and inherited native gates passed.
Q8 proposals/acceptance remained short 129/84 and long 133/82. The original
service was independently restored with identity, health, source and ownership
checks. No model, precision, acceptance, kernel or persistent launcher changed.

## What this run actually covered

The exact marker gate captured the **last 167 nodes of each 826-node draft**:
the LiliCorr lattice head, not the preceding decoder segment. All 43 short / 45
long measured draft rounds have head records, but this is **not full drafter
operator coverage**. The analyzer explicitly retains uninstrumented scheduler
nodes and injection ubatches; they are not silently counted as profiled.

Median of six measured requests per point, milliseconds per response:

| Host wall interval | Short | Long |
|---|---:|---:|
| Outer draft `llama_process` submission, including logging | 310.188 | 340.263 |
| Instrumented head core | 58.169 | 61.355 |
| ↳ timed operation/fusion spans (nested) | 57.921 | 61.090 |
| ↳ loop overhead outside spans (nested) | 0.243 | 0.265 |
| Post-loop logging, excluding final summary write | 8.376 | 8.777 |

Core/span clocks are the same `ggml_time_us()` clock used by the phase profile.
These are **host wall times**, including internal backend waits when present,
not GPU event durations. Post-loop logging is inside the outer phase timer;
this run cannot be used as a clean A/B performance comparison. Component medians
need not sum to the median parent interval.

The short-request coverage record contains 89 scheduler ubatches, 43 observed
backend graphs, all 43 expected draft owners, 38,232 scheduler nodes, and 7,181
observed backend nodes (`43 × 167`). Of those, 5,117 are consumed operation/fusion
span nodes; metadata/view/non-COMPUTE nodes remain explicitly uncovered. The long
record has 101 scheduler ubatches, 45 observed backend graphs/all 45 draft owners,
40,474 scheduler nodes and 7,515 observed backend nodes (`45 × 167`).

Largest measured head operation groups (nested host wall ms, not device time):

| Group | Short | Long |
|---|---:|---:|
| F32 matrix products, combined | 20.663 | 21.780 |
| Q8 matrix products, combined | 11.069 | 11.966 |
| F32 outer products | 5.979 | 6.221 |
| 56×56×8 attention-score product (inside F32 total) | 7.831 | 8.118 |

**Decision:** do not optimize this head as if it explains the full draft cost.
The remaining outer interval includes earlier decoder work, backend/scheduler
synchronization and profiling setup; it is not uniquely attributed by this run.
A continuation adds only the exact existing drafter marker `inp_noise_embd` to
cover the earlier decoder segment. Its source callback is in
`src/models/dflash.cpp`; the target Nemotron graph does not use that input marker.

## Execution and evidence

On `inference-host`, in
`/home/mike/b70-evals/20261019-lilicorr-b70-sycl-profile`:

```bash
bash build-candidate.sh
python3 -B source-proof.py --built
python3 -B test-profile.py
python3 -B run-profile.py --lead-run-profile
# Separate process, after capture/restoration:
python3 -B run-profile.py --lead-after-profile-restore
```

The frozen-SDK CPU build exited 0. Exactly `libggml-sycl.so.0`,
`libllama-common.so`, and `libllama.so.0.7.0` changed among the six pinned artifacts;
complete qualified ancestor source/binary proofs remained unchanged.

- `profile-r1/analysis-r1/analysis.json`: native/public/runtime evidence analysis.
- `profile-r1/analysis-r1/profile.json`: phase events plus operator records,
  coverage, startup/public/warmup separation and raw event references.
- `profile-r1/analysis-live-restoration.json`: independent live restoration.
- `build-attempt-r1/`, build manifests and dependency logs: actual build.
- `cpu-checks-host-r2.log`: all 18 diagnostic tests passed on host, including the
  unchanged inherited 13-test suite. Local tests also passed.

Lead fixes before GPU execution covered real ggml names containing spaces,
public-request diagnostics, complete measured-request/draft-round requirements,
partial scheduler segments and portable raw/gzip fixtures. The synthetic
backend fixture is CPU parser evidence only; it does not establish live emission.
The successful GPU capture above supplies the actual records.

All **2,481** raw profile files were SHA256-equal to the remote originals.
Git retains **82** verified lossless log copies (1,015,495,968 raw bytes →
64,737,472 gzip bytes). The derived `profile.json` is retained in Git as
`profile.json.gz` (128,778,667 → 6,584,430 bytes); its decompressed SHA256 is
`90c0d90fa99dd7dc2049c6a58b2c3f24cc457ed6d8ca7aaf0299de595e0248d0`.
All raw local and remote originals remain. Decompress for original filenames.

For reproducible aggregates, filter `profile.json.requests` by measured phase and
point, then take medians of `sycl.graph_core_wall_us`, `operation_wall_us`,
`uncounted_wall_us`, and `logging_wall_us`, divided by 1000. For operator groups,
sum each request's `sycl.operator_totals[].host_wall_us` by operation/type/shape,
then take the median across requests. Never add nested spans to their parent.
