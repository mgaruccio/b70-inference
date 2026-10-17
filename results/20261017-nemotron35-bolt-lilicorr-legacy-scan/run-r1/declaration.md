# Bolt LiLiCorr legacy-only SSM scan control

Development-only single-factor follow-up to `../20261017-nemotron35-bolt-lilicorr-verify-trace` (39ad93cb). That paired capture found full target-score divergence at output2, before any rollback, for one-row versus two-row target evaluation. Both selected their own raw argmax at output116. The source audit identifies several batch-dependent dispatch differences, including ordinary/MoE NVFP4 kernels/layout and the custom SSM prefill route; it does not identify a culprit.

## Research and exact intervention

Fresh primary source fetched on inference-host before implementation: https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/ssm_scan.cpp . The original scan supports serial recurrence and rollback snapshot slots (lines69-75), with its existing dispatcher at157. The local accepted archive additionally has a custom two-stage prefill path. Its selector at209-267 allows nt>=2 and K<=4; one-token decode uses the original scan.

Clone ONLY the existing verify-trace candidate to `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-legacy-scan-r1`. Change exactly one dispatch condition in `ggml/src/ggml-sycl/ssm_scan.cpp` to `if (false && ssm_scan_prefill_supported(ctx, dst))`, with an explanatory diagnostic comment if needed. Do not change either scan implementation, any NVFP4 route, or global Fusion/VMM/ESIMD controls. This forces the original scan for both prompt and verification. All capture instrumentation remains byte-identical to the prior candidate.

Build using Intel2026/oneDNN3.14, six jobs. Protect original and parent FIVE artifact hashes including libllama-common.so. Only candidate libggml-sycl.so.0 should change; wrapper, backend test, common library and server implementation must match the trace parent. Source diff must be exactly the one scan file. Preserve build/dependency logs. No host/runtime upgrades, accepted-source changes or promotion.

## Fixed paired E2E

Fresh **S0 then C0**, same new candidate in both:
- S0: LiLiCorr n_max1/n_min2, proposals suppressed.
- C0: LiLiCorr n_max1/n_min0, active proposals.

Both n_rs_seq1/K2, block8, capture enabled at fresh /capture mounts; p_min0, greedy draft sampling, no draft backend sampling, drafter GPU layers999. Keep tensor callback0, oneDNN verbosity0, DNN1, FUSION1/ESIMD1/OPT1/VMM1,graphs0,q8KV,F32state,ctx65536,batch/ubatch1024,parallel1. Hardware unchanged: boost0, sixteen3801000kHz CPU caps,275W. Four GiB free capture reserve per launch is required; the preceding identical bounded request set captured about0.5GiB total across both arms, and no new request class is added.

All builds/tests/analysis/GPU/API work runs on mike@inference-host via explicit local AND remote Bash, SSH BatchMode=yes,ConnectTimeout=8,ControlMaster=no,ControlPath=none,ForwardAgent=no. Local editing/git only. Remote root `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-legacy-scan`; commands `bash build-candidate.sh`, `python3 -B test-trace.py`, `python3 -B run-trace.py`, `python3 -B analyze-trace.py`.

Each arm: native385 plus190 numerical/9 support checks; real-model startup and instrumented API20/20 on http://127.0.0.1:18098; unchanged55/128 warmup, one ordinary streamed55/128 response and the exact prior nonstream diagnostic payload (backend_sampling=false). Preserve all frozen request/count/cache/finish/activation/counter checks. S0 zero proposals; C0 nonzero proposals with exact log/API agreement. Stop the container before final log collection to flush records.

The inherited native launch requires23 custom-prefill calls with Fusion1. For this deliberately legacy-only candidate, replace exactly that one dispatch-count expectation with0 (fail closed if the original guard is missing or duplicated). Keep every numerical test, CPU comparison, state/canary check, and other dispatch guard unchanged. Startup validation additionally rejects any native custom-prefill call. Saved-log unit fixtures adapt dispatch lines explicitly; only the subsequent real candidate run establishes native results.

Output interpretation intentionally differs from the instrumentation-only experiment: this control changes model computation. Require each arm's ordinary and diagnostic text to agree, but REPORT rather than assert its equality to the previous reference text/IDs. Do not label this as proof of equivalence to an uninstrumented version of the changed backend. Instrumentation/common/server hashes are unchanged, and this is a diagnostic control only. Any change in serial S0's output relative to the accepted baseline must remain explicit; do not redefine the accepted fidelity baseline.

Reuse the corrected strict prefill-boundary parser, exact128 CPU-row/position/history alignment and checked full-F32 byte comparisons. Compare S0/C0 only under identical prior emitted history, including the first differing output. Record earliest score difference and prior rollbacks, raw argmax/primary-original scores near116, and old-reference match flags. If score divergence disappears, this implicates the disabled prefill route for this pair; if it remains, the route is not sufficient to explain it. Neither outcome establishes all other kernels correct.

No retries to obtain matching output; preserve every failure. Nested cleanup restores original nemotron-bolt-probe and independently verifies ID/config/HostConfig/all mounts/FIVE artifacts/hardware/health/READY and stopped experiments. Save exact declarations/source patch/launches/requests/responses/logs and analysis; raw vectors stay remote and are never added to Git. Commit/push scoped evidence only. No timing or promotion; any apparent correction still needs subsequent non-debug correctness/fidelity gates.
