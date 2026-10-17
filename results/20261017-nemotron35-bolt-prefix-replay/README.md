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

## Results: variation survives exact-prefix replay

All four launches completed; native385 passed in each and API gates passed **80/80** total. All planned4 warmups,12 chat128 responses,12 native128 responses and24 native94/1 replays completed. Template equality,55-token special-token round-trip, identical integer prefixes,zero cache hits,greedy settings,actual output counts and effective server flags were verified.

Distinct full token sequences, out of three requests in each family: F1E1 chat3/native3; F0E1 chat3/native3; F0E0 chat2/native3; F1E0 chat3/native3. Thus neither API framing nor disabling these two optimization switches eliminated the observed variation.

Every cell's six identical one-token replays produced **six distinct top-five score vectors**. F1E1 selected ` quickly` (10533) in all six; F0E1,F0E0,F1E0 each selected ` quickly` five times and `.` (1046) once. In F0E1 one winning top-two logprob gap was only0.0002061; this is a recorded score margin, not an explanation of the variation. A one-token request has no preceding generated-token sampling decisions, and all replays supplied identical token IDs, so this observation narrows the problem to target forward execution/state under repeated requests. It does not distinguish arithmetic noise,uninitialized state,buffer reuse or any particular kernel.

Source reconnaissance further limits interpretation: FUSION gates routing/normalization/SSM fusion, but **not all NVFP4 GEMV paths**. ESIMD excludes NVFP4 weights and may be inert for relevant tensors. In-place SSM-weight reorder is separately controlled by GGML_SYCL_ENABLE_OPT; VMM pooling by GGML_SYCL_ENABLE_VMM. A one-time reorder transition alone does not explain changing scores among repeated94-token prefill calls after identical full-decode warmup. These are next diagnostic leads, not confirmed faults.

Controller contracts passed10/10 (`controller-tests-r1.log.gz`); analyzer contracts passed5/5 (`analysis-contract-tests.log.gz`). Executed remotely:

```bash
cd /home/mike/b70-evals/20261017-nemotron35-bolt-prefix-replay
python3 -B test-replay.py 2>&1 | tee controller-tests-r1.log
python3 -B run-replay.py 2>&1 | tee controller-run-r1.log
python3 -B analyze-replay.py 2>&1 | tee analysis-r1.log
```

`run-r1/` freezes the declaration/controller, exact argv, native and model logs, all public API requests/responses and cleanup. `verification.json` records per-family token groups, shared-prefix checks, score rows, sampling equality and independent restoration. Logs use lossless `gzip -n`, with decompressed comparison to remote originals and retrieved gzip integrity checks. Original raw logs remain remote.

At2026-10-17T13:26:40Z the original accepted container `7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e` was independently verified running with identical configuration,mounts,binary hashes and hardware; health `ok`,smoke `READY`,all four experiments stopped. Timing and promotion remain blocked. Investigation continues; no source/runtime fix is claimed.
