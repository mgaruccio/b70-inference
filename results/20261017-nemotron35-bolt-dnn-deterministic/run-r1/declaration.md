# Bolt oneDNN deterministic candidate

Development-only follow-up to `../20261017-nemotron35-bolt-dnn-controls` (7cfc4d36). Same-binary DNN0 stabilized all six prefix score vectors in both observer modes and all five non-baseline projection outputs. DNN1 previously produced differing outputs for byte-identical public operands. No timing, promotion or persistent launcher edits.

## Research and precise candidate

Fresh primary sources consulted for this change:
- https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_attributes_deterministic.html
- https://uxlfoundation.github.io/oneDNN/v3.14/struct_dnnl_primitive_attr-2.html
- https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_verbose.html

`primitive_attr.set_deterministic(true)` before descriptor creation requests repeatable execution on the same primitive/platform/software/environment. It can choose a slower implementation or fail if unsupported. Verbose implementation names/extra flags do not prove split-K or atomics. Installed oneDNN headers confirm3.14.0 and this API; `src/common/verbose.cpp` prints `attr-deterministic:1`. Runtime support still needs testing.

Create isolated `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-dnn-deterministic-r1` from unchanged accepted ssm-prefill-r1. Apply the exact prior `projection-repeat/server-callback.patch` for optional diagnosis, plus **one semantic backend line** in `ggml/src/ggml-sycl/gemm.hpp`: `primitive_attr.set_deterministic(true);` after scratchpad mode and before matmul descriptor creation. No new flag, kernel, quantization, allocation or fallback. Only these two source files may differ from accepted.

Build using existing Intel2026 image, oneDNN and six jobs. Candidate wrapper/backend-test binary must remain byte-identical to accepted; server implementation must match the prior projection-repeat candidate; SYCL library must change. All four accepted hashes stay unchanged. Validate source scope and runtime dependencies. Preserve every failed build/run; do not silently fall back.

## Predeclared real-system E2E

All builds/tests/analysis/GPU/API work on mike@inference-host through explicit Bash/SSH (BatchMode=yes,ConnectTimeout=8,ControlMaster=no,ControlPath=none,ForwardAgent=no). Commands: `python3 -B test-trace.py`; `bash build-candidate.sh`; `python3 -B run-trace.py`; `python3 -B analyze-trace.py`.

Exactly trace-on then fresh trace-off from this candidate, **DNN1** both. Trace-on also sets `ONEDNN_VERBOSE=all,filter=matmul`; trace-off sets0. Each launch: native385; real API20/20; original frozen55/128 full-decode warmup; **three additional identical streamed55/128 full generations**, cache0; exact55-ID template/tokenization/detokenization round trip; six exact frozen94-token native `/completion` requests at http://127.0.0.1:18098, one forced output,temp0,seed42,cache_prompt=false,ignore_eos=true,n_probs5,return_tokens=true,backend_sampling=false. No drafter or retries for mismatch. Full generations precede the six prefix replays, preserving request/log alignment and the90-column comparison window.

Require native/API/count/cache/template success; same full-generation text across all three repeats within each mode; one prefix score vector per mode; exact projection weight/input/output equality in all five non-baseline trace-on samples; aligned24 first-SSM checkpoints. Trace-off must emit no comparison/checkpoint records. Trace-on must emit oneDNN matmul execution records for the4096×8192 by8192×90 shape with the deterministic attribute. Report actual implementation names without inferring internal atomic algorithms. Preserve raw verbose timings only as implementation diagnostics, not performance evidence.

Keep FUSION1/ESIMD1/OPT1/VMM1,graphs0,q8 KV,F32 recurrent,ctx65536,batch/ubatch1024,parallel1,continuous batching; hardware boost0,16 CPU caps3801000kHz,275W. Stop on functional/quality/count failures and preserve repeatability mismatches without retrying. Nested cleanup restores original accepted identity/config/all mounts/four hashes/hardware/health/READY; independent verification follows. Commit/push scoped evidence and continue toward LiLiCorr fidelity only if the candidate passes. No speed claim or promotion follows from this development gate.
