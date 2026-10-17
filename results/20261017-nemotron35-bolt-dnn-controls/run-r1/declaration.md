# Bolt oneDNN-disabled control

Development-only follow-up to `../20261017-nemotron35-bolt-projection-repeat` (7dd6ad2e), where all five non-baseline90-column projections had byte-identical public weights/activations but different output bytes. No timing, promotion or persistent launcher changes.

## Fresh research and hypothesis

Fresh official source: https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/docs/backend/SYCL.md documents `GGML_SYCL_ENABLE_DNN` (default1). Saved excerpt accompanies this result. Accepted source `ggml-sycl.cpp:3145–3193` is authoritative about its effect here: the F16 operand conversion route chooses oneDNN matmul when enabled, otherwise oneMKL `dpct::gemm`. The90-column NVFP4 operation does not use MMVQ/q8_1. The exact oneDNN implementation/atomic behavior is not yet established.

## Predeclared no-code control

Reuse exactly `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-projection-repeat-r1` and its four checked binaries from the preceding experiment. No clone, source patch, installation or rebuild. The copied source-scope/runtime/build/hash receipts are explicitly inherited from that original build, not evidence of a new build; live hashes are checked before/after execution. Accepted source and all four accepted hashes remain unchanged.

Run two fresh launches, trace-on then trace-off, both with the sole computational configuration change `GGML_SYCL_ENABLE_DNN=0`. Require startup logging to confirm0. Compare against the preceding DNN1 trace-on/off pair with identical candidate binaries, hardware and frozen payloads. This is a historical-reference diagnostic control, not a standard performance comparison or an isolated change to only one model operation. Leave every other flag unchanged, including attention configuration.

## Real API process and expected evidence

Execute on inference-host through explicit Bash/SSH (BatchMode=yes,ConnectTimeout=8,ControlMaster=no,ControlPath=none,ForwardAgent=no): `python3 -B test-trace.py`; `python3 -B run-trace.py`; `python3 -B analyze-trace.py`. No build command. Each launch: native385; real API20/20; original frozen55/128 full-decode warmup; exact55-ID template/tokenization/detokenization round trip; six frozen94-token native `/completion` requests at http://127.0.0.1:18098, one forced output,temp0,seed42,cache_prompt=false,ignore_eos=true,n_probs5,return_tokens=true,backend_sampling=false. No drafter or retries for mismatch.

Preserve all API bodies and logs. Trace-on retains24 first-SSM checkpoints and the six actual public weight/activation/result byte comparisons for the90-column output projection; trace-off must emit neither. Require samples0–5, only first seed, fixed byte counts, checkpoint name/shape/op alignment, and DNN0 in both startup logs. Report output equality or inequality without assuming the library switch fixes it. If the operation becomes repeatable, inspect full response score stability too; no timing or promotion follows automatically.

Keep FUSION1/ESIMD1/OPT1/VMM1,graphs0,q8 KV,F32 recurrent,ctx65536,batch/ubatch1024,parallel1,continuous batching; hardware boost0,16 CPU caps3801000kHz,275W. Nested cleanup restores original accepted container identity/config/all mounts/four hashes/hardware/health/READY; independently verify and stop owned experiments. Raw logs remain remote, lossless copies and exact commands/environment/API artifacts under this result root. Commit/push scoped evidence and continue bounded diagnosis.
