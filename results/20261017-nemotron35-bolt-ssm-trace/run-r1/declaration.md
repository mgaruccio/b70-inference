# Bolt first-SSM operation trace

Development-only continuation of `../20261017-nemotron35-bolt-block-trace` (fe5bc0ab). The first observed varying sum was complete block0 output; embedding and initial-normalization sums were constant. Equal sums are not bitwise equality, and no kernel defect is established. No timing or promotion.

## Fresh research

Fetched pinned primary sources on inference-host before implementation; saved URLs/responses accompany this result:
- https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/src/models/mamba-base.cpp — first SSM path: input projection, convolution/bias/SiLU, scan, D-add, SwiGLU, grouped RMS norm and output projection. Existing names include `mamba2_y_add_d-0` and `mamba_out-0`.
- https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml.c — SSM_CONV weight is src1; SSM_SCAN A is src3, and the output packs y plus state. These identify layer0 without inventing unstable node-number filters.

Accepted `build_norm` also labels unweighted RMS outputs `norm-0`; initial and grouped occurrences differ by shape/order. Existing common debug helper owns readback/printing. Its float sums are lossy and observation can change scheduling.

## Predeclared change and real API process

New isolated accepted-archive clone: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-trace-r1`. Only `tools/server/server-context.cpp` changes. Keep the previous context-owned callback/default-off activation (`BOLT_TRACE_TENSORS=1`), but select exact `embd`, `norm-0`, `attn_norm-0`, `mamba2_y_add_d-0`, `mamba_out-0`, `nemotron_h_block_out-0`, `result_output`; additionally select MUL_MAT/SSM_CONV/SSM_SCAN nodes whose relevant weight operand belongs to `blk.0.` (src0/src1/src3 respectively). This is first-block-only inspection, not arithmetic changes, a new tracing API or a persistent facility. Other clones remain untouched.

Build with the same Intel2026 image, oneDNN and six jobs. Require only the server-context source to differ; only the server implementation library may differ among the four checked artifacts. Accepted wrapper/implementation/SYCL/backend-test hashes remain unchanged; candidate wrapper/SYCL/tests match accepted. Runtime dependency/version precheck before any inference.

Run remotely through explicit Bash/SSH (BatchMode=yes,ConnectTimeout=8,ControlMaster=no,ControlPath=none,ForwardAgent=no): `python3 -B test-trace.py`, `bash build-candidate.sh`, `python3 -B run-trace.py`, `python3 -B analyze-trace.py`. Exactly trace-on then fresh trace-off. Each launch: native385; real API20/20; original frozen55/128 full-decode warmup; verified template/tokenization/detokenization55-ID round trip; six native `/completion` requests to `http://127.0.0.1:18098` with the exact saved94-token prefix, one forced output,temp0,seed42,cache_prompt=false,ignore_eos=true,n_probs5,return_tokens=true,backend_sampling=false. No drafter and no retries for mismatches.

Trace-on must observe all named checkpoints and first-block MUL_MAT/SSM_CONV/SSM_SCAN operations with aligned names/shapes/ops across six requests. Trace-off must emit none. Preserve each original tensor header to identify unnamed operations by actual op/source rather than assuming a node number. Compare each processing-chunk position separately; do not treat re-prefill as incremental decoding or equal sums as tensor identity.

Unchanged FUSION1/ESIMD1/OPT1/VMM1,graphs0,q8 KV,F32 recurrent,ctx65536,batch/ubatch1024,parallel1,continuous batching; hardware boost0,16 CPU caps3801000kHz,275W. Nested cleanup stops owned containers and restores the original accepted container. Independent checks cover identity/config/all mounts/four hashes/hardware/health/READY. Preserve exact commands/API bodies/logs/failures in this result root, losslessly compress logs, commit/push, and continue bounded diagnosis from the result. No persistent launcher change or accepted archive edit.
