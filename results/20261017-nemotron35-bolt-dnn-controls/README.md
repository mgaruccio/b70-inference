# Bolt oneDNN-disabled control

Development-only follow-up to `../20261017-nemotron35-bolt-projection-repeat` (7dd6ad2e), where all five non-baseline90-column projections had byte-identical public weights/activations but different output bytes. No timing, promotion or persistent launcher changes.

## Fresh research and hypothesis

Fresh official source: https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/docs/backend/SYCL.md documents `GGML_SYCL_ENABLE_DNN` (default1). Saved excerpt accompanies this result. Correction from the subsequent real-model verbose trace: this NVFP4 graph requests BF16 accumulation (`llama-graph.cpp:1626–1627`) and takes the F32 operand route (`ggml-sycl.cpp:3198–3244`), choosing oneDNN matmul when enabled, otherwise oneMKL column-major GEMM. The earlier F16/dpct description inferred the native-test route rather than the model's precision hint. The90-column NVFP4 operation does not use MMVQ/q8_1. The exact oneDNN implementation/atomic behavior is not yet established.

## Predeclared no-code control

Reuse exactly `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-projection-repeat-r1` and its four checked binaries from the preceding experiment. No clone, source patch, installation or rebuild. The copied source-scope/runtime/build/hash receipts are explicitly inherited from that original build, not evidence of a new build; live hashes are checked before/after execution. Accepted source and all four accepted hashes remain unchanged.

Run two fresh launches, trace-on then trace-off, both with the sole computational configuration change `GGML_SYCL_ENABLE_DNN=0`. Require startup logging to confirm0. Compare against the preceding DNN1 trace-on/off pair with identical candidate binaries, hardware and frozen payloads. This is a historical-reference diagnostic control, not a standard performance comparison or an isolated change to only one model operation. Leave every other flag unchanged, including attention configuration.

## Real API process and expected evidence

Execute on inference-host through explicit Bash/SSH (BatchMode=yes,ConnectTimeout=8,ControlMaster=no,ControlPath=none,ForwardAgent=no): `python3 -B test-trace.py`; `python3 -B run-trace.py`; `python3 -B analyze-trace.py`. No build command. Each launch: native385; real API20/20; original frozen55/128 full-decode warmup; exact55-ID template/tokenization/detokenization round trip; six frozen94-token native `/completion` requests at http://127.0.0.1:18098, one forced output,temp0,seed42,cache_prompt=false,ignore_eos=true,n_probs5,return_tokens=true,backend_sampling=false. No drafter or retries for mismatch.

Preserve all API bodies and logs. Trace-on retains24 first-SSM checkpoints and the six actual public weight/activation/result byte comparisons for the90-column output projection; trace-off must emit neither. Require samples0–5, only first seed, fixed byte counts, checkpoint name/shape/op alignment, and DNN0 in both startup logs. Report output equality or inequality without assuming the library switch fixes it. If the operation becomes repeatable, inspect full response score stability too; no timing or promotion follows automatically.

Keep FUSION1/ESIMD1/OPT1/VMM1,graphs0,q8 KV,F32 recurrent,ctx65536,batch/ubatch1024,parallel1,continuous batching; hardware boost0,16 CPU caps3801000kHz,275W. Nested cleanup restores original accepted container identity/config/all mounts/four hashes/hardware/health/READY; independently verify and stop owned experiments. Raw logs remain remote, lossless copies and exact commands/environment/API artifacts under this result root. Commit/push scoped evidence and continue bounded diagnosis.

## Conditional follow-up research (not applied here)

Primary oneDNN3.14 documentation consulted by the researcher:
- https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_verbose.html — `ONEDNN_VERBOSE=all,filter=matmul` or `profile_exec,filter=matmul` identifies implementation, descriptors, attributes and dimensions, provided verbose support was built in. It does not document split-K/atomic accumulation as a field; unspecified extra flags are not proof.
- https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_attributes_deterministic.html and https://uxlfoundation.github.io/oneDNN/v3.14/struct_dnnl_primitive_attr-2.html — `primitive_attr.set_deterministic(true)` before primitive descriptor creation requests deterministic execution; defaultfalse. It can select a slower implementation or fail when no deterministic kernel exists. Guarantees are bounded to the same primitive/platform/software/environment.
- https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_attributes_fpmath_mode.html — FPMath controls permitted down-conversion, not reduction ordering. No documented global `ONEDNN_DETERMINISTIC`/`DNNL_DETERMINISTIC` switch was found.

These are conditional next-step options, not changes in this DNN0 experiment. Await the control result before attributing the variation to the oneDNN route; do not claim a deterministic attribute fixes the observed defect without exercising it.

## Observed result

Controller/parser contracts8/8; native385 per launch; API20/20 per launch (40/40 total); both DNN0 launches completed all frozen payload/count/cache/template contracts. Candidate binaries and hardware matched the DNN1 reference exactly. Startup and container environments independently confirmed DNN0.

All five non-baseline90-column projections reported byte-identical weights, activations **and outputs**. All24 selected checkpoint sums were constant across the six trace-on replays. Both trace-on and trace-off produced exactly one score vector and one emitted token sequence across six requests, versus six score vectors/two token sequences in the same-binary DNN1 reference. This strongly isolates the observed variability to the oneDNN-enabled route for this workload; it does not identify atomics/split-K or establish full-generation/speculative fidelity.

Executed the three predeclared commands (no build). Independent restoration passed at2026-10-17T14:45:59.599290+00:00: original container identity/config/all mounts/four hashes/hardware/health/READY intact, both controls stopped. Exact commands/environment/API bodies/comparisons/checkpoints and lossless logs are retained under `run-r1/` and `verification.json`, with raw originals remote. No timing or promotion.

Next bounded candidate: request the documented oneDNN deterministic primitive attribute in the existing GEMM wrapper, keep DNN enabled, verify native/API gates and exact operand replay again, and add repeated original55/128 full generations. The installed headers confirm oneDNN3.14.0 and `set_deterministic(bool)` support; successful runtime support still must be tested.
