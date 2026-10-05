# Qwen stock native-MTP baseline — development result

Completed 2026-10-05 on the existing B70. This was **inference only**, with no optimizer updates or model changes. It is not the forthcoming CUDA/BF16 training baseline.

## Protocol and observed result

Real native vLLM `/v1/chat/completions`, streaming API token IDs and `/metrics`; stock MTP4 → no-spec → no-spec → stock MTP4 (ABBA). Each cell ran two functional canaries and one excluded warmup, then the same 12 synthetic prompts across coding, prose, reasoning, structured, repetitive and high-entropy categories. Total: **48 measured requests**, all transport checks passed; both canaries passed in every cell. Natural EOS, maximum 512 generated tokens; temperature 0, seed 42, thinking disabled, concurrency 1, prefix caching disabled. Rendered prompt IDs were checked against `/tokenize`; per-request metrics waited for exact generated-token accounting. No generated code or tools were executed.

| Cell | Accepted drafts / draft pass | Draft acceptance | Median decode tok/s | Median API E2E tok/s |
|---|---:|---:|---:|---:|
| A1 stock MTP4 | 2.4025 | 60.0624% | 97.6415 | 95.4872 |
| B1 no-spec | — | — | 33.9037 | 33.8110 |
| B2 no-spec | — | — | 33.9043 | 33.8100 |
| A2 stock MTP4 | 2.3702 | 59.2550% | 97.4325 | 94.9873 |

Acceptance excludes bonus tokens and warmups. Joint survival at draft positions 1–4 was 84.11/65.27/50.82/40.05% in A1 and 83.15/64.80/50.36/38.71% in A2. These are unconditional accepted-position counts divided by native draft passes, **not conditional per-depth accuracy**. Draft-pass counters are not asserted to count every target/verifier invocation.

All cells generated 5,179 measured tokens, but token sequences were not identical. Stock repeats matched 8/12 prompts; no-spec repeats matched 7/12. Cross-arm matches were 9/12 (A1/B1) and 8/12 (A2/B2). Therefore this is **not a bitwise greedy-equivalence or lossless-speed qualification**. The structured-output checker accepted one JSON prompt and rejected the other in every cell. Canaries are not comprehensive functional-quality evidence. Most long responses hit the token cap; no long-context/generalization claim is made. State drift and deployment VRAM overhead were not measured.

## Exact runtime and intentional differences

- Target: existing `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, revision `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`.
- Image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.
- Target GPTQ INT4-g128, FP16 compute, FP8 KV; existing S+M1 RTN INT4 draft machinery. **Not full-BF16 target inference.**
- Context 212,992; batch-token budget 8,192; graph sizes 1/2/4/8; balanced mode; mamba cache align; existing five reference patches and uniform-prefill guard; 275 W.
- Control difference: remove only `--speculative-config` from the otherwise identical temporary launcher. Disabled prefix cache and thinking, balanced/graph/guard settings and loopback port 8001 are declared differences from the persistent production launcher.
- Production launcher remained SHA256 `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`. Owned containers stopped; host left idle; no production promotion.

## Retained evidence and reproduction

Commands, source snapshots/hashes, exact launchers, server and workflow logs, container/runtime details, raw SSE responses, requests, token IDs, before/after counters and failures remain in:

`inference-host:/home/mike/b70-evals/20261005-qwen38-native-mtp-baseline/`

Archived in the existing R2 store:

`r2:ml-archive/2026-10-05/cache/b70-evals/20261005-qwen38-native-mtp-baseline/raw.tgz`

Archive SHA256: `d7b2334893752a5c4f22b963492e60dca942ce8906c3dd13aa5460db33c2e3bf`.

Executed on the idle B70: `nohup timeout --signal=TERM --kill-after=120s 3600s python3 -u run.py > workflow.log 2>&1`; exit **0**. `run.py` SHA256 `e48c6be9ff2ebb0d1a82513b9a51735cce71b978cf3e1ca53771412caba726e3`. Dependencies are existing benchmark helpers snapshotted from repository HEAD `a098acb9`, not a substitute decoder. Restore the archive into a new run directory before reproducing; do not overwrite this completed evidence. `protocol.json` freezes the prompts/configuration; `comparison.json` contains the raw aggregate and token-match results.

## Interpretation and next experiment

The stock native head is a useful control: approximately 2.37–2.40 accepted draft tokens per draft pass on this development set. It has not reached the proposed ≥3 accepted-draft target. Do not pool these B70 numbers with a CUDA/BF16 run or treat them as a trained-head improvement.

Fresh primary-source inspection confirms the checkpoint has one MTP layer and native serving recursively reuses it, with token/state and attention-cache machinery. The CUDA pilot should train this existing head, not replace it with the Glimmer final-state MLP prototype:

- https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json
- https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/model_executor/models/qwen3_5_mtp.py
- https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/v1/spec_decode/step3p5.py
