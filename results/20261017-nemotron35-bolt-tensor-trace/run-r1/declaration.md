# Bolt isolated server tensor-boundary trace

Development diagnosis following `../20261017-nemotron35-bolt-state-controls`: reorder-off and VMM-off both retained changing scores/argmax for identical cold94/1 requests. No runtime fix, timing, promotion or change to the accepted archive. The user explicitly requested continued work instead of stopping at diagnostic checkpoints.

## Research and scope

Fresh primary source: https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/common/debug.cpp and matching `common/debug.h`. The existing `common_debug_cb_eval` prints tensor metadata, partial values and a whole-tensor float sum. Its ordinary filter only limits printing; ask always returns true and all non-host tensors are read back. A narrow server adapter must filter the scheduler's ask phase **before** calling that helper, to read only `inp_embd`, `l_out-*`, and `result_output`. Those names are confirmed in the accepted `src/llama-graph.cpp` and `src/models/nemotron-h.cpp`.

Equal printed sums do not prove bitwise equality. Readbacks/scheduler splits can change execution and mask a fault. Use the earliest changing printed boundary only as a localization hint, not root-cause proof. No generic tracing API, persistent service, extra storage protocol or source arithmetic change is requested.

The q8_1 scratch-padding candidate was closed by source inspection: actual NVFP4 consumers bound loops by logical width and do not read padded tail blocks. NVFP4 also ignores the q8_1 sum-correction component. Do not zero padding or replace that unused sum reduction as an unsupported fix.

## Predeclared implementation and verification process

Prepare one patch to `tools/server/server-context.cpp`, in an isolated clone of accepted archive `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1`. Do not modify that accepted archive. The adapter is off by default and enabled only by `BOLT_TRACE_TENSORS=1`; it uses context-owned `common_debug_cb_user_data` whose lifetime exceeds the model/context. It forwards selected completed tensors to the existing helper, without enabling abort_on_nan (which exits0). No GPU-kernel or common-debug implementation changes.

All compilation, tests, inference and analysis run on `mike@inference-host` through explicit Bash/SSH (BatchMode=yes,ConnectTimeout=8,ControlMaster=no,ControlPath=none,ForwardAgent=no). Build the isolated server with the existing Intel2026 image/toolchain and at most six jobs. Check patch application, source-difference scope, and hashes: only the server source and `libllama-server-impl.so` may change; accepted binary hashes and candidate server wrapper/SYCL/test binaries must remain identical to accepted. Keep native385 as the numerical gate.

Exactly two diagnostic server launches, **trace-on then fresh trace-off**, using the same isolated binary. For each: native385; real API gate20/20; one original frozen55/128 full-decode warmup; the previously verified template/tokenization/detokenization55-ID round-trip; six nonstream native `/completion` requests with the exact saved94-token prefix,one output,temp0,seed42,cache_prompt=false,ignore_eos=true,n_probs5,return_tokens=true,backend_sampling=false. No drafter. Trace-on must actually emit selected model boundaries; trace-off must emit none. Compare response scores/IDs within each mode and compare the six trace-on boundary sequences by shape/name/order and printed sums. Timing remains disabled even if tracing makes responses stable.

Keep FUSION1,ESIMD1,OPT1,VMM1,graphs0,q8 KV,F32 recurrent state,ctx65536,batch/ubatch1024,parallel1 and continuous batching unchanged. Hardware remains boost0,16 CPU caps3801000kHz,275W. Stop on functional/native/API/quality/shape failure; preserve every request/log and do not retry mismatches. Nested cleanup stops owned experiments and restores the exact original accepted container, checking identity/config/mounts/binaries/hardware/health/READY independently. Save commands,source patch,build output,declaration,controller,all API bodies and trace logs; losslessly compress logs. Commit/push scoped evidence, then continue diagnosis from the result rather than treating this checkpoint as completion.


## Build-check correction

`build-r1.log` and `build-candidate-r1.sh` preserve the initial failed post-build assertion. Compilation and the runtime dependency/version precheck succeeded; the erroneous assertion expected the executable wrapper hash to change. The actual changed artifact is `libllama-server-impl.so` (accepted `67ae7c8673110b52d11924677ad662141dc93f57d7c21d67a3f0d3e4489e09ad`, candidate `ac10d21cec7a519b948679c54afdb7d329483f12fcfc4ec8d929a2228842ea9b`). Corrected post-build validation is recorded separately in `post-build-verification.log`; no rebuild or inference retry is needed. Both controller and independent analysis now verify all four artifacts, including the implementation library.
