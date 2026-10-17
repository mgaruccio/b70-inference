# Bolt embedding and complete-block trace

Development-only continuation of `../20261017-nemotron35-bolt-tensor-trace` (committed2cb37cde). That experiment observed variation by the first FFN output but missed the actual embedding and complete block boundaries. No timing, promotion, accepted-archive edits, or persistent launcher changes.

## Fresh primary-source research

Fetched on inference-host before this change:
- https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/src/llama-graph.cpp — token-ID input produces `embd`; `inp_embd` is the separate raw-embedding input.
- https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/src/models/nemotron-h.cpp — `attn_norm-0` follows initial normalization; `nemotron_h_block_out-*` follows each block's residual addition. `l_out-*` only labels FFN output.

Saved upstream responses and research URLs remain with this result. The local accepted source confirms those names. Existing `common_debug_cb_eval` remains the readback/printing implementation; sums are localization hints, not bitwise identity or kernel attribution.

## Predeclared change and E2E process

Use a new isolated clone `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-block-trace-r1` of the unchanged accepted ssm-prefill-r1 archive. Only `tools/server/server-context.cpp` changes. Relative to the preceding adapter, change the selected names to exact `embd`, exact `attn_norm-0`, prefix `nemotron_h_block_out-`, and exact `result_output`. Activation remains exactly `BOLT_TRACE_TENSORS=1`, default off. No arithmetic or backend changes. The original experiment/source clone stays untouched.

Reuse the preceding controller/test/analyzer with the new result/candidate/container paths and explicit coverage checks. Build with the same Intel2026 image, six jobs, unchanged oneDNN. The server implementation library may differ; the wrapper, SYCL library and backend tests must remain byte-identical, and all four accepted hashes must remain unchanged. Validate source scope and runtime dependencies before touching the running service.

On inference-host, run `python3 -B test-trace.py`, then `python3 -B run-trace.py`: exactly trace-on followed by fresh trace-off, both from this isolated candidate. Each launch must pass native385 and real API20/20, then original frozen55/128 full-decode warmup, exact template/tokenization/detokenization55-ID round trip, and six identical native `/completion` requests with the frozen94-token prefix, one forced output, temp0, seed42, cache_prompt=false, ignore_eos=true, n_probs5, return_tokens=true, backend_sampling=false. Endpoint is `http://127.0.0.1:18098`; no drafter. Preserve all API bodies and raw logs; no retries for mismatches.

Analyze with `python3 -B analyze-trace.py`. Trace-on must show actual `embd`, `attn_norm-0`, all88 complete-block names, and final logits, with matching name/shape/order across six replays. Trace-off must show none. Compare printed sums and response score vectors without timing claims. Different sums identify only an observed boundary; identical sums do not establish tensor equality. Record the existing90/4 processing chunks rather than treating re-prefill as incremental decode.

Keep FUSION1/ESIMD1/OPT1/VMM1, graphs0, q8 KV, F32 recurrent state, ctx65536, batch/ubatch1024, parallel1 and continuous batching. Hardware: boost0,16 CPU caps3801000kHz,275W. All builds/tests/analysis/inference use explicit Bash over SSH (BatchMode=yes,ConnectTimeout=8,ControlMaster=no,ControlPath=none,ForwardAgent=no). Nested cleanup restores the original `nemotron-bolt-probe` container; independently check identity/config/all mount fields/four hashes/hardware/health/READY and stopped experimental containers. Preserve failures and lossless logs under this result root, commit/push, then continue bounded diagnosis from the observations.
