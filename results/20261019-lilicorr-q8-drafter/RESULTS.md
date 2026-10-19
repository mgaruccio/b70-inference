# Q8 LiLiCorr drafter: fidelity passes, still below no-draft throughput

Development-tier result; no promotion. Original service independently verified restored.

Direct conversion completed with existing llama-quantize: 25 NVFP4 tensors → Q8_0, preserving all 128 F32/BF16 payloads byte-for-byte. Metadata differs only in general.file_type (39 → 7). Candidate: 1,437,595,904 bytes, SHA256 d420b2bdcfbbe937795b9dc94d2659305eb01244309172019976dabefad363d7. Quantization is lossy and requires public fidelity testing. No FP16 intermediate or backend kernel change.

All 8 CPU adapter checks passed on inference-host. `python3 -B run-performance.py --lead-run-timing` failed during initial capacity preflight: 5,247,606,784 bytes available, 5,368,709,120 required (4 GiB reserve plus 1 GiB evidence allowance). No cells completed or measured attempts occurred. The gate explicitly forbids automatic cleanup/retry; it was not weakened.

The failed run's restoration report is retained unchanged: it cannot compare container/hardware/source snapshots because capacity failed before those snapshots were created. A separate check against the completed FP16 run's existing baseline succeeded, including original READY/health/identity/source and sole ownership: `blocked-original-vs-fp16-restored.json`. This does not rewrite the failed run as restored or successful.

Conversion launch failures retained in session task logs: initial shell-quoting failure before remote conversion; subsequent successful dry run stopped by an overbroad arrow-line count that included preservation overrides. Narrowing the count to actual tensor-conversion lines confirmed exactly 25 and conversion succeeded. Neither failed attempt produced a model.

## Approved cleanup and successful timing-r2

The user authorized cleanup. Verified lossless archives replaced 163 remote raw logs from completed FP16 and earlier Q8-backend experiments; local raw copies remain. Btrfs physical savings were small. Removing 87 unused, registry-backed `alexgshaw/*` Docker images (no running or stopped container references, no forced removal) increased free space from 5,291,917,312 to 36,385,333,248 bytes. All containers, Intel SDK images, models, source and benchmark evidence were preserved. The earlier blocked run remains unchanged.

Fresh `run-performance-r2.py` changes only the output directory/container prefix and includes itself in code pins. On inference-host:

```bash
python3 -B test-performance.py -v
python3 -B run-performance-r2.py --lead-run-timing
python3 -B run-performance-r2.py --lead-after-timing-restore
```

All 8 host CPU tests passed. All inherited native gates and 100 public prevalidation jobs passed. Independent analysis accepted 24 measured responses (six per arm/prompt point), with matching outputs, `status=analyzed`, `errors=[]`, and `independently_restored=true`.

| Request | No-draft A tok/s | Q8 drafter W3 tok/s | Change |
|---|---:|---:|---:|
| short55 | 44.863367 | 39.788933 | -11.31% |
| long8270 | 43.149967 | 37.332535 | -13.48% |

These are median server `predicted_per_second` values. Full-request output throughput was also lower: 34.824564 vs 38.904510 short, 10.079007 vs 11.110925 long. W3 proposed/accepted counts were 129/84 short and 133/82 long in each measured response. No statistical-significance, raw-F32 equality or controlled cross-run FP16/Q8 improvement is claimed. This candidate does not yet achieve the requested speedup.

Evidence is under `timing-r2/`, including `analysis-r1/analysis.json`, raw HTTP/SSE, identity/mount/source/native checks, telemetry and restoration. Log archives are lossless verified `.log.gz`; raw local copies and remote model remain outside Git. Next investigation is restricted to avoidable work in the actual LiLiCorr draft hot path, not generic backend kernels.
