# B70 explicit-context LiliCorr profile — development only

The explicit TLS/RAII scope worked on the real path. Both first-public-job probes
passed; every measured draft round has the BF16 decoder projection and original
lattice head. This is profiling progress, **not an optimization or speedup claim**.

## Validation and retained evidence

- Frozen SDK CPU build passed; exactly the expected three libraries changed.
  Qualified source, completed seed, model bytes and persistent launchers stayed
  unchanged. Actual host CPU gate: 14 context tests plus inherited 18/13 tests.
- The existing exclusive ABBA controller ran unchanged native qualification,
  25 public jobs/cell and short/long warmup + three measured requests/point/cell.
  All 100 public and 24 measured outputs passed; proposals/accepted remained
  short 129/84 and long 133/82. Full block eight, 7x8 lattice, noncausal KV rows
  and cleanup order stayed intact; W7 remained excluded.
- Automatic restoration passed. A separate process then independently verified
  original identity, health, source and sole ownership before analysis passed.
- Evidence: `profile-r1/analysis-r1/{analysis,profile}.json`,
  `profile-r1/analysis-live-restoration.json`, per-cell native/public/timing
  files and `scope-probe-first-job.{json,stdout.log,stderr.log}`.
  Raw originals remain on inference-host and locally. Large Git-retained JSON
  and logs use verified lossless gzip copies, not substituted observations.

Exact inference-host commands, in the package directory:

```bash
python3 -B test-profile.py
bash build-candidate.sh
python3 -B source-proof.py --built > lead-built-proof.json
python3 -B run-profile.py --lead-run-profile
# New independent process, after capture/restoration:
python3 -B run-profile.py --lead-after-profile-restore
```

Source/runtime pins, preconditions, tools and cleanup are in `README.md` and the
unchanged inherited controller. This was the actual public HTTP journey, not a
mock or standalone-kernel substitute.

## Coverage and host-wall findings

Each of the 43 short / 45 long draft rounds observes **824 of 826 scheduler
nodes**: 657 decoder plus 167 head nodes. Of those, 448 are covered compute nodes
and 376 backend metadata nodes; two scheduler nodes remain uninstrumented and
are **not** called GPU work. Every injection owner is covered as well.

Medians of six measured responses per point, milliseconds:

| Quantity | Short | Long |
|---|---:|---:|
| Outer draft submission | 311.504 | 341.415 |
| Decoder backend core | 102.073 | 107.356 |
| Head backend core | 59.865 | 63.151 |
| Decoder logging | 21.552 | 22.631 |
| Head logging | 8.344 | 8.736 |
| Outer minus backend core/logging | 119.739 | 139.545 |

These are **host wall**, including any internal waits, not GPU-event durations.
Logging perturbs the workload; no clean-speed comparison is permitted. Medians
of components need not add to a median total.

Using actual operation begin/end references, the approximate gap from the last
timed decoder operation to the first timed head operation, minus decoder logging,
is **97.967 / 115.361 ms** per short/long response. The pre-decoder portion is
21.277 / 23.718 ms, including draft graph build/allocation 15.778 / 16.644 ms and
inputs 0.414 / 1.474 ms; the post-head portion is only 0.535 / 0.553 ms. This
localizes most residual wall to the decoder-to-head boundary, but does not by
itself distinguish CPU work, copies and waiting for queued GPU execution.

`TOP_K` is already GPU-supported and observed in the decoder segment. It is not
a missing GPU implementation. Before selecting a change, trace the actual
boundary and tensor placement; do not infer a bottleneck from asynchronous
operator submission time alone. B580 architecture tables are not proof of B70
launch limits. Any geometry test must use actual B70 capabilities and timing.

No configuration, model or kernel has been promoted. The user's net-speed goal
remains open.

## Next measured intervention

The captured startup shows 544 MiB of target weights in `SYCL_Host`.
Qualified `llama-model.cpp:1670–1672` places input embeddings on CPU;
`models/dflash.cpp:918–925,947,702` borrows that table for noise-token and
candidate-token gathers. This is source/placement evidence, not recovery of the
two absent node names from backend logs. The next isolated experiment tests a
drafter-context-owned device copy of the same table, leaving the target table,
Q8 bytes, arithmetic and complete lattice untouched. Its speed effect is unknown.

Fresh [Intel data-movement guidance](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/asynchronous-and-overlapping-data-transfers.html)
recommends keeping intermediates on the device to avoid host/device round trips,
even if an individual small kernel alone is not optimal. Allocation/copy lifetime
obligations are covered by [Intel transfer guidance](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/optimizing-data-transfers.html).
Clean baseline/candidate public runs, with diagnostic profiling disabled, decide
whether this intervention helps; the current host-wall profile cannot decide it.

Do not assume every multi-column Q8 operation uses MMQ: qualified
`ggml-sycl.cpp:5613–5625,5819–5844` restricts DMMV to one column, permits MMVQ
through eight columns, and dispatches eligible MMVQ before MMQ. No dispatch change
is justified by the contrary inference alone.

## Lossless retention

All 2,487 transferred raw profile files matched remote SHA256. All 528 measured
draft rounds have the coverage counts above; the 12 measured W3 requests have
exactly the stated proposal/acceptance counters. Common analysis has 24 measured
rows and no errors.

Git retention uses 83 verified lossless log copies (1,437,987,383 raw bytes →
97,562,846 gzip bytes) and `profile.json.gz` (515,148,092 → 30,552,797 bytes).
The decompressed JSON SHA256 is
`1ce43b1af50bbef4251ede7a68ca5549a2f3d4ff4d5abd7e5a31dd7b37ce98ef`.
All local and remote raw originals remain; decompress for original filenames.
