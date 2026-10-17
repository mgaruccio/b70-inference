# Bolt target-only exact-prefix and kernel controls

Development diagnosis continuing `../20261017-nemotron35-bolt-fidelity-controls` (`bb0b7084`): accepted draft-off varies within launch. User explicitly requested continued investigation rather than stopping at that checkpoint. No timing or promotion authorization.

## Fresh research and source checks

Pinned upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb`:
- https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/tools/server/README.md — native `/completion` accepts integer-array prompts, `n_predict`, `n_probs`, `return_tokens`; `/apply-template` and `/tokenize` preserve the configured chat template and explicit special-token policy.
- https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/tools/server/server-context.cpp — apply-template uses the chat parser without inference; tokenize defaults add_special=false/parse_special=true. Native completion and chat feed the same task tokenizer; token-array input bypasses text tokenization.
- https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/tools/server/server-common.cpp — pure integer arrays do not insert BOS. Preserve required special tokens explicitly.
- https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/docs/backend/SYCL.md and https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/ggml-sycl.cpp — GGML_SYCL_ENABLE_FUSION and GGML_SYCL_ENABLE_ESIMD select optimized paths, default1, and log effective values. Neither guarantees determinism.

No source establishes the current B70 variability as NVFP4, MoE, recurrent-state or arithmetic noise. Those remain hypotheses. Saved non-spec probability rows are scores, not calibrated confidence; speculative probability placeholders are irrelevant here because speculation is disabled throughout.

## Predeclared end-to-end process

Remote-only on `mike@inference-host`, with explicit local/remote Bash and SSH BatchMode=yes,ConnectTimeout=8,ControlMaster=no,ControlPath=none,ForwardAgent=no. Artifact root `/home/mike/b70-evals/20261017-nemotron35-bolt-prefix-replay`.

Use only accepted archive `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1`, its previously verified binary hashes, original target GGUF, original q8 KV/F32 recurrent state,ctx65536,batch/ubatch1024,parallel1,graphs0,boost0,16 CPU caps3801000kHz,275W. Preserve continuous batching. No rebuild, model change, persistent-launch change, draft model or source patch. One client, no concurrent requests. All model servers non-debug, identical LLAMA_TRACE=1.

Exactly four fresh launches in Gray-code order: **F1E1,F0E1,F0E0,F1E0** (fusion/ESIMD). This is a coarse kernel-path control, not a claim about an individual kernel. Native385 is inherited and must pass in every launch; its inline fusion0/1 tests remain unchanged while ESIMD inherits the cell's value. Stop on functional/native/quality failure; retain output differences without retries.

Each launch:
1. Native385, then a fresh real gate-v2 API20/20.
2. One full128-token frozen streamed short55-token warmup,cache_prompt=false.
3. `/apply-template` on the exact saved diagnostic chat request; require rendered prompt equal to the saved verbose prompt. `/tokenize` that prompt twice with parse_special=true and add_special=false/true; require identical55 IDs (no implicit BOS discrepancy). Detokenize them and require byte-equivalent rendered prompt. Save every API body.
4. Three full128-token nonstream chat diagnostics, exactly the prior diagnostic request (seed42,temp0,cachefalse,n_probs5,return_tokens,backend_sampling=false).
5. Three full128-token native `/completion` requests using the explicit55 prompt IDs and matched sampling settings,cachefalse. This separate API variant must not be silently equated with chat responses.
6. Six native one-token requests using those55 IDs plus the saved common39 generated IDs (94 prompt tokens),cachefalse,n_probs5. Append generated IDs directly; never retokenize their text. The prefix is frozen from the prior six diagnostic bodies and must match all six through39 IDs.
7. Preserve all responses, exact argv, effective server settings, logs, counts, stop reasons and token IDs; stop/retain each experiment. Check hashes and hardware between cells.

Planned totals:80 quality checks,4 full-decode warmups,12 full chat diagnostics,12 full native diagnostics,24 one-token replays. No cache-enabled sequence or forced next-token sampling in this matrix. A one-token replay **re-prefills** the prefix and is not execution-equivalent to incremental generation; it cannot by itself identify the original failing forward step. Compare within each request family before comparing modes. Any stable result is limited to these repeats/prompt.

Nested cleanup stops owned experiments and restores the same original accepted container, verifying ID/config/mounts/binary hashes/hardware/health and READY. Original weights, archive, launchers and hardware remain unchanged. Keep raw remote artifacts; retrieve logs with lossless gzip, commit/push only this experiment's evidence. Continue diagnosis from observed results; do not treat completing this matrix as fidelity clearance or an automatic stopping point.
