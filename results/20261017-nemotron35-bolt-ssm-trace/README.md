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

## Observed result

Build/source scope/runtime precheck passed. Controller/parser contracts7/7; native385 per launch; API20/20 per launch (40/40 total); all exact payload/count/cache/template checks passed. Trace-on had six distinct score vectors and two emitted token sequences; trace-off had six distinct score vectors and one emitted token sequence. Trace-on captured24 checkpoints per request (12 each for90-token and4-token chunks), with required named checkpoints and projection/convolution/scan operators; trace-off captured none. No timing or promotion.

The first changing observed sum was the90-token output projection, `node_57`, MUL_MAT of `blk.0.ssm_out.weight` [8192,4096,1,1] and the reshaped activation [8192,90,1,1], producing [4096,90,1,1]. Six sums: -4571.770996,-4571.814453,-4571.852051,-4571.822266,-4571.822754,-4571.823730. Embedding, initial RMS/weighted norm, input projection, convolution, packed scan result, D-add, and grouped unweighted RMS sums were constant within this run. **The grouped weighted-norm result immediately feeding the projection was not separately observed**, so this does not prove identical matrix inputs or a defective GEMM kernel. The4-token chunk's first-block checkpoints all had constant sums; its final logits still varied. The90-token chunk correctly had a zero-output logits tensor [131072,0,1,1].

Executed commands are the predeclared four commands above; exact environment/launch argv, API bodies, checkpoint headers and full lossless logs are saved under `run-r1/`, `trace-boundaries.json`, and `verification.json`. Independent restoration passed at2026-10-17T14:24:07.836740+00:00: original identity/config/all mounts/four binary hashes/hardware/health/READY intact, both experiments stopped. Raw logs remain remote.

Next: audit the exact output-projection weight type and SYCL dispatch for this90-column shape, distinguish dispatch hypotheses from proof, and close the unobserved weighted-input gap before claiming kernel attribution. No unsupported padding/sum-correction fix or performance experiment.
