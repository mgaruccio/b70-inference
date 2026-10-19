# Q8 LiLiCorr drafter: converted, GPU comparison blocked by disk reserve

No performance result and no promotion. Original service remains healthy.

Direct conversion completed with existing llama-quantize: 25 NVFP4 tensors → Q8_0, preserving all 128 F32/BF16 payloads byte-for-byte. Metadata differs only in general.file_type (39 → 7). Candidate: 1,437,595,904 bytes, SHA256 d420b2bdcfbbe937795b9dc94d2659305eb01244309172019976dabefad363d7. Quantization is lossy and requires public fidelity testing. No FP16 intermediate or backend kernel change.

All 8 CPU adapter checks passed on inference-host. `python3 -B run-performance.py --lead-run-timing` failed during initial capacity preflight: 5,247,606,784 bytes available, 5,368,709,120 required (4 GiB reserve plus 1 GiB evidence allowance). No cells completed or measured attempts occurred. The gate explicitly forbids automatic cleanup/retry; it was not weakened.

The failed run's restoration report is retained unchanged: it cannot compare container/hardware/source snapshots because capacity failed before those snapshots were created. A separate check against the completed FP16 run's existing baseline succeeded, including original READY/health/identity/source and sole ownership: `blocked-original-vs-fp16-restored.json`. This does not rewrite the failed run as restored or successful.

Conversion launch failures retained in session task logs: initial shell-quoting failure before remote conversion; subsequent successful dry run stopped by an overbroad arrow-line count that included preservation overrides. Narrowing the count to actual tensor-conversion lines confirmed exactly 25 and conversion succeeded. Neither failed attempt produced a model.

Next requirement: authorize evidence-preserving disk cleanup or provide more disk space, then use a fresh run namespace rather than overwriting timing-r1. No cleanup, deletion, new GPU run, or persistent launcher change has been performed.
