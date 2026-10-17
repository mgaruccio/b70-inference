# Bolt matched repeatability and draft-off controls

Status: declared diagnostic development experiment; not a speed benchmark. User requested continuing after the isolated OUT_PROD fix passed numerical checks but LiLiCorr output differed after195 characters, with accepted cross-launch outputs also differing after346 characters. Prior evidence: `../20261017-nemotron35-bolt-lilicorr-outprod`, commit `852f6328`.

## Frozen controls and scope

No source edits, rebuilds, weights changes, hardware changes, persistent launch edits or promotion. Accepted source is `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1`; patched source is `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-outprod-r1`. Same upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb`, original target GGUF, pinned official LiLiCorr draft, original q8 target KV/F32 recurrent state, context65536,batch/ubatch1024,parallel1,ESIMD1/FUSION1/graphs0,boost0,16 CPU caps3801000kHz,275W. Use recorded hashes from prior evidence. Single client, no concurrent requests. Preserve original continuous-batching setting; do not silently introduce an additional runtime change.

All workloads, source inspection, tests and analysis execute on `mike@inference-host` with explicit Bash and SSH `-o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no`. Remote artifacts: `/home/mike/b70-evals/20261017-nemotron35-bolt-fidelity-controls`. Local repository receives scoped evidence only. Lab preview/patch sockets are already known unavailable; no retries or repair.

## Fresh external research and interface limits

- https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/tools/server/server-context.cpp — pre-sampling probabilities disable backend sampling when enabled; target defaults already disable it. Speculative acceptance sets probability1.0 and has `TODO: set result.probs`, so speculative logprob0/empty top-k is not confidence evidence.
- https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/tools/server/server-common.cpp — chat `logprobs` maps to `n_probs`; no tools+stream combination here.
- https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/tools/server/server-task.cpp — non-stream verbose response exposes actual generated token IDs; probability rows include IDs but speculative probabilities are unimplemented.
- https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/tools/server/server-schema.cpp — `return_tokens`, `verbose`, `backend_sampling` request fields.
- https://github.com/ggml-org/llama.cpp/issues/3014 and https://github.com/ggml-org/llama.cpp/issues/7228 — batching/backend operation order can affect greedy reproducibility; these reports are not B70-specific proof or a diagnosis of this case.

## Predeclared matrix and real API process

Exactly six fresh launches, in order: **A0,B0,C0,C1,B1,A1**. A=accepted non-speculative, B=patched non-speculative, C=patched LiLiCorr depth4/n_min0/p_min0/greedy/no draft-backend sampling. Bracketing reduces order confounding but two launches do not exhaust all sources of nondeterminism. All server launches are non-debug (`GGML_SYCL_DEBUG` unset); add identical `LLAMA_TRACE=1` logging to every arm solely for verification-event evidence. Native tests retain their inherited explicit debug settings.

For each launch:

1. Existing native385 passes before the server. Patched arms also run the unchanged prior `native-outprod.sh` (190 numerical plus nine support-contract checks).
2. Fresh real `/v1/chat/completions` gate-v2 API **20/20** must pass. A functional/native/quality failure stops the entire matrix; preserve it, no retry-until-lucky.
3. One full128-token decode warmup using the frozen short measured55-token payload. This diagnostic deliberately uses the same payload as the following probes, all cache_prompt=false. No output1-only warmup.
4. Three streamed requests with byte-equivalent JSON payload to the frozen short55/128 request: temperature0,seed42,thinking disabled,EOS ignored,cache_prompt=false. Verify exactly55 prompt tokens,128 outputs,cache_n=0 and finish reason length. Save every response and all events. **Output mismatches are diagnostic observations, not reasons to retry or truncate the fixed matrix.**
5. One separately labeled diagnostic request with the same messages/sampling/output limit, but stream=false,verbose=true,return_tokens=true,logprobs=true,top_logprobs5,backend_sampling=false. Save actual token IDs and complete body. This is a separate request variant, not silently substituted for the frozen stream. Compare token IDs only between matched diagnostic requests. A diagnostic response differing from the preceding streamed output must be reported, not equated. Speculative probability values are unimplemented and must never be interpreted as logits/confidence.
6. Stop and retain that container, verify unchanged accepted/candidate hashes and hardware before advancing. In nested finally, stop all owned experiments, restart the same original accepted container, and verify ID/config/all mount fields/hashes/hardware/health/READY.

Total planned ordinary probe responses:18, plus6 warmups,6 diagnostic responses and120 quality checks. No long-context probe, timed benchmark or fidelity promotion in this matrix. Incidental response timing fields are retained but not performance results.

## Analysis and stop boundary

Analyze within-launch and cross-launch identity before cross-arm comparisons. Record full rendered-output identity, common-prefix character length, diagnostic token-ID identity/first differing index and token pieces, prompt/output/cache counts, actual sampling settings, and draft proposals/accepts/rejection events where exposed. Preserve all outcomes. Baseline variability, patched-off differences, draft-on differences, and diagnostic-instrumentation differences are distinct findings; none alone proves recurrent rollback failure. Do not infer real logits from placeholder speculative probability rows.

The matrix always leaves timing disabled. Any first-divergence logit/replay instrumentation or source change is a separately declared next step based on these results, not an automatic patch or relaxed-fidelity path. Commit/push only scoped scripts, raw responses, logs (lossless gzip copies) and the analysis; keep original raw files remotely and accepted Bolt resident.
