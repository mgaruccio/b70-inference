# Qwen 27B native quant-aware trace tune — development result

**Verdict: no convincing acceptance or throughput win; keep stock.** The real native capture → offline recursive training → BF16 overlay → runtime RTN → speculative decoding pipeline works. This experiment does not establish that quant-aware feature training improves serving performance.

## Held-out result

Frozen candidate: optimizer step **60**, selected only using the post-RTN development objective before test evaluation. Stock → candidate → candidate → stock (ABBA), 157 held-out contexts per cell, 628 completed requests, no skipped test prompts.

| Pooled metric | Stock | Tuned | Relative change | Family-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| Accepted draft tokens / speculative pass | 2.65575 | 2.67289 | +0.645% | −0.284% to +1.410% |
| Decode tokens/sec | 88.9597 | 89.2065 | +0.277% | −0.377% to +0.814% |
| End-to-end tokens/sec | 76.1604 | 76.2390 | +0.103% | −0.233% to +0.275% |
| Draft acceptance fraction | 66.3938% | 66.8223% | +0.4285 percentage points | — |
| Median request latency | 3.91422 s | 3.91363 s | Essentially unchanged | — |
| Median TTFT | 0.56401 s | 0.56468 s | Essentially unchanged | — |

Unconditional acceptance by draft position, stock → tuned: **85.932→86.303%, 71.473→72.036%, 59.135→59.682%, 49.036→49.268%**. Full run-level values are in `summary.json`.

Decode throughput is `sum(generated_tokens - 1) / sum(server request_decode_time_seconds)`; end-to-end throughput is `sum(generated_tokens) / sum(client request wall seconds)`. Speculative passes are native `spec_decode_num_drafts_total`, **not every verifier forward**. Native acceptance counters can include terminal candidates discarded by the API. Counter deltas were checked against each request's exact completion-token count; prefix-cache hits were zero.

### Variability and uncertainty

- Stock repeats produced identical token sequences on only **73/157** prompts; tuned repeats on **82/157**. Cross-head exact matches were 71/157 and 80/157. These are observed differences, not proof of functional quality regression or equivalence.
- Only **56 prompts** had identical outputs across all four cells. On that post-hoc subset, acceptance improved 0.978%, decode speed 0.552%, and end-to-end throughput 0.344%; all family-bootstrap intervals still include zero.
- Bootstrap: 10,000 resamples, seed 42, source-session family as the cluster; both runs retained together. There are only **five test families**, so these intervals are limited evidence, not strong population guarantees or an independent estimate of temporal run noise.
- No functional task execution/quality score was performed. Generated tools were never executed. No production promotion.

## Actual training attempt

Compared with the earlier ~10,370-position MBPP smoke run, this used ~16× more unique useful training positions and real engineering contexts:

- 644 normalized, locally secret-scanned contexts; split by source-session family before generation: 412 train / 75 dev / 157 test. Gitleaks v8.30.1 ran without networking and flagged zero candidates; scanner clearance is not proof that all sensitive content is absent. Raw data remains private.
- Exact deployed quantized verifier generated on-policy native-MTP trajectories: **340 train contexts / 162,964 useful positions**, **54 dev contexts / 23,292 useful positions**. Complete prompts exceeding the training budget were skipped (72 train, 21 dev); no prefix truncation or fabricated hidden states.
- This is **below the proposed 1–2M useful-position corpus**. Do not confuse 948,686 input-token exposures across two epochs with unique supervised data. Most continuations naturally ended in tool calls.
- Captured actual post-final-norm target states after acceptance decisions, before proposer buffer reuse, including async and speculative execution. Rejected rows and terminal output beyond the authoritative API trajectory are excluded. Final unobserved hidden rows are not padded.
- Native architecture and stock warm start; frozen embedding and runtime-RTN-effective draft LM head. First depth uses `(x[t+1], h[t], p[t]) → x[t+2]`; deeper depths feed the draft's own previous hidden state, advance positions, and attend only the base prefix plus their own branch cache. Teacher-forced continuation tokens train accepted-prefix behavior.
- Two epochs, 170 optimizer updates, accumulation 4, LR 5e-6, eight sampled roots/sequence, depth weights `[1, 0.5, 0.25, 0.125]`, seed 42, maximum complete sequence length 2,048. FP32 masters, FP16 XPU compute, BF16 native overlay, existing RTN packing on load. **No core fake-quant QAT.** Quant awareness here means quantized-verifier features plus post-quantization checkpoint evaluation.
- Step 0 and intermediate exported/RTN-effective candidates evaluated on dev every 20 updates; stock eligible. Step 60 selected: weighted dev objective **1.247637 → 1.238551 (−0.728%)**. Offline improvement did not become a demonstrated serving win.
- Corpus job: 48m24s, including its eight-context canary. Training job: 8m08s. Peak XPU allocation **17,704,458,240 bytes**; reserved **20,082,327,552 bytes**. Verifier unloaded during training. No cloud rental used.

## Backend and declared configuration

- Intel Arc Pro B70 32GB on `inference-host`; final observed power cap 275W. No continuous thermal/power trace retained.
- `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, revision `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`.
- Local model: `/home/mike/inference/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`.
- Image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`; vLLM `0.27.2rc1.dev77+gac7509e2b`, torch `2.13.0+xpu`, Transformers `5.15.0`.
- Native MTP4, C1, FP16 runtime compute, target GPTQ INT4 g128, existing draft RTN INT4 g128, FP8 KV, configured context 212,992. Same tokenizer/chat template; explicit restricted bash/read/write/replace tool profile. Each trace's thinking setting retained.
- Temporary reference launcher differences from persistent deployment: balanced performance mode, graph capture sizes `[1,2,4,8]`, existing uniform-prefill guard, loopback binding. Production prefix caching/default thinking restored; unique request salts impose cold prefixes. All ABBA cells match except the draft overlay.
- Training corpus sampling: temperature 0.6, top-p 0.95, top-k 20, seed 42; natural termination; maximum response 1,792 within the 2,048 complete-sequence cap.
- Test sampling: temperature 0, seed 42, natural termination, response cap 1,024; complete test prompts admitted up to 32,768 total tokens, so testing is not restricted to the short training context. Four synthetic warmups excluded per cell. Capture **off** for timed requests.
- Persistent launcher SHA256 remained `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`; temporary servers stopped.

This is a **development run**, not a standard-publishable/community-comparable benchmark: no full BetterBench, standard concurrency/long-context suite, complete environment capture, or functional quality suite. No BF16-trained control was run; stock vs tuned does not isolate quantized-feature training from workload adaptation or the recursive objective.

## Verification and artifacts

Real public-boundary journey:
1. Native capture canary: four `/tokenize` + `/v1/chat/completions` requests, 146 useful positions, zero/partial/full acceptance, thinking/tool/EOS/length cases.
2. One-update recursive XPU training on those features; stock and trained head reloaded into the same decoder. All four output-token sequences matched across capture-on/stock/tuned; each reload accepted 101/184 drafts.
3. Eight private contexts validated (1,696 useful positions), then the train/dev corpus generated on the same native path.
4. Full training above, checkpoint frozen, then held-out ABBA through the real API with per-request metrics and token-identity checks.

The first capture canary failed because an extra async round beyond the output cap tripped a bound. It was fixed with a regression test; the successful retry is retained separately. Earlier CPU-test staging omitted two existing packer fixtures; after staging them, those tests passed.

Final targeted checks in the pinned CPU-only container: **107 passed, 1 skipped** (26.50s). The skipped check requires the optional pinned-source-root environment variable; real serving was exercised separately above. No runtime packages were installed into Pi.

```bash
docker run --rm --network none \
  -v /home/mike/b70-evals/20260914-mtp-real-attempt/source:/work:ro -w /work \
  --entrypoint python \
  vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f \
  -m pytest -q -p no:cacheprovider tests/test_qwen38_mtp_native_capture.py \
  tests/test_qwen38_train_mtp.py tests/test_qwen38_mtp_trace_corpus.py
```

Private artifacts on `inference-host`:
`/home/mike/b70-evals/20260914-mtp-real-attempt/`

- `capture-canary-v2-output/`, `training-canary/`, `stock-reload-output/`, `tuned-reload-output/`
- `private-train-v1/`, `private-dev-v1/`, `training-private-v1/`
- `heldout-{A1,B1,B2,A2}/launch-config.json` and `launcher.sh`; corresponding `*-output/request-*/{request,response,measurement}.json`
- `commands/` retains the exact executed shell scripts and `analyze_abba.py`; run with `bash commands/b70-real-mtp-heldout-abba.sh` only using fresh output paths and an idle host (existing outputs fail closed).
- Analysis: `python3 commands/analyze_abba.py /home/mike/b70-evals/20260914-mtp-real-attempt` (exclusive output; existing summary must not be overwritten).

Only this aggregate report, aggregate summary, and the analysis source are published. Prompts, per-request raw results, features, scanner reports, and heads are not committed.

**Next decision:** retain stock. This attempt validates feasibility, not the proposed performance benefit. A future experiment needs broader independent workload families and a larger useful-token corpus, with a fresh final test set; the current test set is now consumed. Do not automatically expand or promote based on this result.
