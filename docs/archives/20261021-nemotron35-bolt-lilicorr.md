# Nemotron 3.5 Bolt / LiLiCorr — investigation archived

**Paused at the user's request on 2026-10-21.** No further experiments or retries
are queued. This archives the model investigation, not the repository or live
service. Model files, private source/build trees and raw evidence are preserved.

## Outcome

The objective was a dramatically faster, exact full-eight-position drafter on
one Intel Arc Pro B70. **That objective was not achieved.** Qualified speculative
configurations still lost to NoDraft. No experimental configuration was promoted,
and no persistent inference launcher was changed.

**A qualified full-serving GPU cost breakdown was not obtained.** Earlier host
timings, isolated operator timings and static ISA inspections were partial
measurements, not substitutes. Previous assurances implying complete profiling
were unsupported; do not carry those assurances into a future handoff.

## Operating point to preserve

- Target: NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB GGUF, llama.cpp/SYCL private
  upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb`, frozen oneAPI2026 image.
- Drafter: LiLiCorr/DFlash `Li5_SWA`, 1.1B, five decoder blocks; full eight positions,
  7 × 8 candidate lattice, max3, unchanged noncausal rejected-KV behavior.
- Qualified Q8 drafter:
  `/home/mike/b70-evals/20261019-lilicorr-q8-drafter/lilicorr-q8-25.gguf`,
  SHA256 `d420b2bdcfbbe937795b9dc94d2659305eb01244309172019976dabefad363d7`.
- Native exact arithmetic/precision and all native/public/W7 eligibility checks
  remain required. Core gates: Q8 **82 cases/54 byte-exact**; IQ2 **34/16**.
- Serving qualification: A1/W3a/W3b/A2, 100 public jobs, 8 warmups, 24 measured
  responses; short55/long8270, temperature0/seed42, no prompt cache, ignore EOS,
  output128. Expected W3 proposed/accepted: short129/84, long133/82.
- No SDK/driver/adapter workaround, math flag, queue/graph-policy change,
  NVFP4/DFlash configuration detour or persistent launcher promotion.

## Qualified results and rejected branches

All numbers below are development results, not community-standard benchmarks.
Use each linked result's own paired baseline; do not compare different runs as
if they were a controlled experiment.

| Work | Observed outcome | Evidence |
|---|---|---|
| IQ2 batch4 | Exact native/public qualification passed; W3 still slower than NoDraft | [Result](../../results/20261020-lilicorr-iq2-batch4/RESULTS_R1.md) |
| Target-Q8 n4 batching on IQ2batch4 | Native/public passed; complete 105-object frozen SYCL build; no net serving win | [Result](../../results/20261020-lilicorr-q8-target-batch4/RESULTS_R1.md) |
| Original-queue host-boundary diagnostic | Public/native passed; selected GPU timestamps unavailable; host intervals only | [Result](../../results/20261020-lilicorr-host-boundary/RESULTS_R1.md) |
| Serial packed IQ2 | Real native operator regression; rejected | [Result](../../results/20261020-lilicorr-iq2-packed/NATIVE_PERF_R1.md) |
| Exact IQ2 two-accumulator | 1,684,480 CPU exact comparisons and native gates passed; up/down operators regressed 27.44%/33.49%; no serving rerun | [Result](../../results/20261020-lilicorr-iq2-acc2/RESULTS_R1.md) |
| Q8 head screen | Isolated timing did not establish full-serving cost share or dramatic ROI; rewrite dropped | [Result](../../results/20261020-lilicorr-q8-head-screen/RESULTS_R1.md) |

The most recent complete serving qualification, target-Q8 n4 batching, measured:

| Workload | NoDraft decode | W3 decode | NoDraft end-to-end | W3 end-to-end |
|---|---:|---:|---:|---:|
| short55 |44.8151 tok/s|41.2037 tok/s|38.8080 tok/s|35.8938 tok/s|
| long8270 |43.1532 tok/s|38.8172 tok/s|11.1071 tok/s|10.1757 tok/s|

This candidate remains an experimental reference, **not a production choice**.
Its serving source tree is
`/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-q8-target-batch4-r1`.
The passed scalar IQ2batch4 tree remains the operator reference; packed/acc2
variants must not replace it.

## Final profiling attempt: actual failure, not completed profiling

Package: [20261021 serving-profile results](../../results/20261021-lilicorr-serving-profile/RESULTS_R1.md).

1. Reused the installed external unitrace 2.3 tool and existing exclusive serving
   lifecycle. No application build, queue property or runtime-stack edit.
2. Clean A1/W3a finished: 50 public qualification jobs and 12 measured responses
   were independently reread successfully. These are clean controls, **not a
   completed clean/traced comparison**.
3. Traced W3b never reached server startup. Its first 56-case native NVFP4 gate
   stopped progressing after the n9 `LIBRARY_MKL_F32` marker. Readiness timed out;
   normal stop then forced termination was retained (137, not OOM). Neither traced
   Q8/IQ2 gates nor traced public serving qualified.
4. A bounded native-only follow-up used the already-installed unitrace 2.4,
   unchanged flags `-d -h -s -v --chrome-kernel-logging`, and the same gate.
   **Clean passed 56/56; traced timed out after 180 seconds at the same marker.**
   Both tools remain unqualified for this frozen native path. The last log marker
   identifies the observed boundary, not a proven oneMKL/driver root cause.
5. Each GPU attempt restored the original in finally and passed a new independent
   SSH/Python restoration check. Archive preparation repeats that check separately.

No trustworthy kernel-cost attribution or tracing-overhead ratio can be inferred
from these failures. Do not use their partial native traces as serving evidence.

Two additional pitfalls are retained for a future investigation:
- An older [20261019 trace package](../../results/20261019-nemotron35-bolt-lilicorr-w3-kernel-profile/README.md)
  was left `analysis_pending=true`; its native gates were untraced and its drafter
  predates the current Q8 export. It is not current, fully qualified attribution.
- The pinned older profiler's Chrome clock uses
  `(REALTIME - BOOTTIME at initialization) + MONOTONIC_RAW`, not ordinary
  REALTIME. Request matching needs the retained clock calibration. Generic
  kernel names also do not uniquely identify drafter/target phases.

## Restored service and archive locations

The original `nemotron-bolt-probe` service is left running, not uninstalled.
Container Id:
`7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`.
It remains the original NoDraft launcher/source, not an experimental W3 service.

The final independent archive check, its health and sole observed GPU-owner PID
are recorded in [archive-status.json](../../results/20261021-lilicorr-serving-profile/archive-status.json).
The complete check is retained as `archive-live-state.json` and a lossless gzip
sidecar; ownership is a point-in-time observation, not continuous surveillance.

Latest raw evidence is retained on **both** hosts:
- Local: `/home/mike/code/b70-inference/results/20261021-lilicorr-serving-profile/`
- Remote: `/home/mike/b70-evals/20261021-lilicorr-serving-profile/`
- Raw subdirectories: `capture-r1/`, `native-pilot-240-r1/`, `preflight-r1/`.
- [Retention record](../../results/20261021-lilicorr-serving-profile/archive-retention.json)
  and [per-file SHA256 manifest](../../results/20261021-lilicorr-serving-profile/archive-raw-remote-manifest.json)
  bind the cross-host copies. Selected evidence is committed compressed; large
  originals remain outside Git and must not be blanket-cleaned.

Earlier packages retain their own raw roots, sidecars and retention records.
No model weights, exported sources, build directories or negative results were
removed as part of this archive.

## If explicitly resumed later

First verify the live state and read the final profiling failure evidence.
Resolve a concrete fidelity-safe measurement path before another optimization.
Do not rerun rejected packed/accumulator/head-reuse variants without new evidence.
Preserve the full native/public gates and independent restoration; never turn a
tracer failure into a profiling success by skipping gates or changing the stack.
No GPU-cost share or next high-ROI exact optimization is currently established.
