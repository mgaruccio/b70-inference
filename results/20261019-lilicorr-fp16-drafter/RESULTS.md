# FP16 LiLiCorr drafter: fidelity gates pass, speedup not yet achieved

Development-tier experiment on inference-host / Intel B70. Do not promote.

Only the drafter changed: 25 NVFP4 tensors converted offline to F16 using the existing quantizer. The remaining 128 F32/BF16 tensor payloads were checked byte-for-byte unchanged. All metadata remained identical except `general.file_type` (39 → 1). Candidate size: 2,261,383,424 bytes; SHA256 `ebb53a8e11d86fbd4036512c51339e776f90ecc325263b266f4fb62e672623ea`. This conversion rounds values; it is not claimed lossless. Target weights, qualified r2 source/binaries, server arguments, acceptance rules and requests were unchanged.

## Observed results

All 100 public prevalidation jobs passed (25 per cell); the inherited native gates passed. Independent analysis accepted 24 measured responses, six per arm/prompt point, with matching outputs and no errors. Original service restoration was independently verified. No persistent launcher was changed.

Median server `predicted_per_second`:

| Request | No-draft A | FP16 drafter W3 | Change |
|---|---:|---:|---:|
| short55 | 44.859935 | 37.409200 | -16.61% |
| long8270 | 43.151984 | 33.799921 | -21.67% |

These are within-run ABBA comparisons, not statistical-significance or community-publishable claims. Historical W3 runs used different experiments; no controlled old-drafter/new-drafter speedup is claimed. FP16 increases drafter storage by 1,204.625 MiB; usable maximum context has not been established for this candidate.

## Execution and evidence

On inference-host in `/home/mike/b70-evals/20261019-lilicorr-fp16-drafter`:

```bash
python3 -B test-performance.py -v
python3 -B run-performance.py --lead-run-timing
python3 -B run-performance.py --lead-after-timing-restore
```

All 8 host CPU tests passed. Both real-run commands exited successfully. `timing-r1/analysis-r1/analysis.json` records `status=analyzed`, `independently_restored=true`, `errors=[]`. Raw requests, HTTP/SSE responses, mount/PID records, native logs and telemetry are retained under `timing-r1/`. Large logs are retained as lossless `.log.gz` copies; raw originals remain local and remote and are ignored by Git. The GGUF remains on inference-host, not in Git.

Next work remains drafter-specific: assess selective Q8_0 weights using the existing Intel execution path to reduce bandwidth relative to FP16. That candidate requires its own unchanged fidelity gates and measured comparison; it is not a demonstrated improvement.
