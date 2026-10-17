# Bolt output-projection operand repeatability

Development diagnosis following `../20261017-nemotron35-bolt-ssm-trace` (e962eb19). First observed changing sum was the90-column first-SSM output projection. Its actual weighted activation was not separately observed, and equal upstream sums cannot establish equality. No timing, promotion or accepted-runtime modification.

## Fresh research and bounded purpose

Fresh pinned primary source, fetched on inference-host before implementation: https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-backend.cpp . `ggml_backend_tensor_get` handles view buffers, checks allocation/bounds and delegates to the buffer getter. The prior callback already synchronizes selected completed operations. Use that boundary to compare actual tensor bytes, not infer equality from printed sums.

This adds only an ephemeral, context-owned in-memory comparison for this one matrix shape. No tensor files, hashes, persistent store, new tracing API, service or artifact lifecycle. First matching operands/result are held in RAM until context destruction; subsequent matching reads are compared with `memcmp`. Existing logs retain equality booleans and byte counts, not tensor snapshots. Backend-private reordered/quantized scratch is not captured, so equal public operands cannot establish equality of hidden backend state.

## Predeclared implementation

New isolated clone of accepted ssm-prefill-r1: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-projection-repeat-r1`. Modify only `tools/server/server-context.cpp`. Retain the preceding selected first-SSM callback checkpoints and default-off `BOLT_TRACE_TENSORS=1`. Wrap common-debug user data with context-owned baseline weight/input/output byte vectors and a sample counter. Only after completed MUL_MAT of `blk.0.ssm_out.weight`, input[8192,90,1,1] and output[4096,90,1,1], read the actual public weight, activation and result tensors. Require contiguous tensors and F32 activation/result. First sample seeds the baseline; five subsequent samples report exact byte equality against it. This does not change arithmetic or retry inference.

Before execution, verify previous full trace contains exactly six such90-column projections, all from the declared replays, avoiding accidental warmup seeding. Build with the same Intel2026 image, oneDNN and six jobs. Only server implementation library may change among checked candidate binaries; wrapper/SYCL/tests remain byte-identical. All four accepted hashes and accepted source remain unchanged. Runtime dependency/version and source-scope checks precede model work.

## Real-system E2E

Run all compilation/tests/analysis/inference on mike@inference-host using explicit Bash/SSH (BatchMode=yes,ConnectTimeout=8,ControlMaster=no,ControlPath=none,ForwardAgent=no): `python3 -B test-trace.py`; `bash build-candidate.sh`; `python3 -B run-trace.py`; `python3 -B analyze-trace.py`. Exactly trace-on then fresh trace-off, no drafter. Each passes native385 and API20/20, original frozen55/128 warmup, exact55-ID template/tokenization/detokenization round trip, then six exact frozen94-token native `/completion` requests at http://127.0.0.1:18098 with one forced output,temp0,seed42,cache_prompt=false,ignore_eos=true,n_probs5,return_tokens=true,backend_sampling=false. No retries for mismatch.

Require six comparison records on trace-on, sample0–5 with only sample0 marked baseline; fixed tensor byte counts; none on trace-off. Preserve equality outcomes as diagnostic observations, including unexpected unequal operands. Retain the preceding first-SSM checkpoint alignment/coverage checks and all API bodies. No speed claims regardless of outcome.

Unchanged FUSION1/ESIMD1/OPT1/VMM1,graphs0,q8 KV,F32 recurrent,ctx65536,batch/ubatch1024,parallel1,continuous batching; hardware boost0,16 CPU caps3801000kHz,275W. Nested cleanup and independent verification restore the exact original accepted container, identity/config/all mount fields/four hashes/hardware/health/READY, and stop experimental containers. Preserve failures and lossless logs; commit/push scoped evidence and continue diagnosis. Original source, models and persistent launchers remain untouched.

## Observed result

Build/source scope/runtime precheck passed; controller/parser contracts8/8, native385 per launch and API20/20 per launch (40/40 total). Both launches completed all frozen payload/count/cache/template checks. Trace-on and trace-off each had six distinct score vectors and two emitted token sequences. No timing or promotion.

Exactly six90-column comparisons were emitted, correlated one-to-one with the six replay requests; no warmup seeding. Sample0 seeded the baseline. Samples1–5 all reported `weight_equal=1`, `input_equal=1`, `output_equal=0`. Compared public tensors: NVFP4 weights18,874,368bytes; F32 activation2,949,120bytes; F32 result1,474,560bytes. Thus identical public operand bytes produced differing result bytes at this operation. This is stronger than equal printed sums and closes the unobserved weighted-input gap, but does not identify which backend kernel/private buffer is responsible. No persistent tensor snapshots were produced.

The90-column output projection remains the first selected changing sum. Existing24 first-SSM checkpoints aligned across replays. Trace-off emitted no comparison or checkpoint records. `verification.json` contains the exact per-sample booleans and byte counts; `trace-boundaries.json` retains selected tensor headers/sums. All API bodies/launch argv/environment and lossless logs are under `run-r1/`; original raw logs remain remote.

Executed the four predeclared commands above on inference-host. Independent restoration passed at2026-10-17T14:36:11.738580+00:00: original identity/config/all mounts/four hashes/hardware/health/READY unchanged, both experiments stopped. Next work is the bounded NVFP4 dispatch/kernel audit and a targeted reproduction/fix, not a performance run or promotion.

## Dispatch audit and next control

Read-only audit, confirmed by parent source inspection: at90 columns NVFP4 misses MMVQ (limit8), DMMV type support, disabled MMQ, and the fused-dequant GEMM. `ggml/src/ggml-sycl/ggml-sycl.cpp:3145–3193` converts public operands to F16 and selects `DnnlGemmWrapper::row_gemm` when DNN is enabled, otherwise oneMKL `dpct::gemm`. `gemm.hpp:38–83` constructs oneDNN matmul with user scratchpad and fpmath=f16, without a deterministic attribute. Actual startup reports DNN enabled. This makes the library route a supported next control; it does **not** prove split-K/atomic accumulation or rule out private conversion/scratch state. The q8_1 path is not reached by this90-column operation.

Next experiment uses the existing `GGML_SYCL_ENABLE_DNN=0` switch with these same diagnostic binaries and frozen payloads, trace-on then fresh trace-off. Compare against this DNN1 pair, preserving all other settings and no timing. This avoids an unsupported kernel edit while testing the identified dispatch boundary.
