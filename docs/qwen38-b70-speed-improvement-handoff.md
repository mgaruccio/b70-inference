# Qwen3.8 B70 speed-improvement handoff

## Goal

Investigate why the Intel Arc Pro B70 Qwen agent-eval rate is materially below the Qwen3.8 model-card decode figures, without changing the evaluation contract or conflating synthetic serving throughput with agent performance.

## Completed baseline

The completed server-side row used:

- row: `qwen38-b70-gptq-int4-mtp4`
- model: `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` at revision `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`
- image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
- server: vLLM `0.27.2rc1.dev77+gac7509e2b`, Intel XPU, one visible B70
- launch: GPTQ INT4, BF16 MTP draft, MTP-4, FP8 KV, 131,072 max context, 0.88 memory target, 8,192 scheduler tokens, no prefix cache, `qwen3_xml` tool parser
- eval: `configs/qwen38-b70-gptq-int4-mtp4-pi-local.toml`; Pi harness `0.83.0`, Docker runtime, high thinking, 98,304 context, 32,768 max output, one concurrent rollout, 3,600-second rollout budget
- taskset: pinned five-task Harbor Terminal-Bench 2.1 cohort

Result: **3/5 solved**; one ordinary unsolved outcome and one 3,600-second budget timeout. Output:

```text
~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T051024Z-qwen38-c1-full/
```

## Measured rate

From 68 OpenAI chat-completion calls persisted in `traces.jsonl`:

| Metric | Observed |
| --- | ---: |
| Completion tokens | 139,776 |
| Sum of API-call duration | 4,897.19 s |
| Aggregate completion rate | **28.54 tok/s** |
| Median per-call rate | **17.10 tok/s** |
| p10 / p90 per-call rate | 4.81 / 36.56 tok/s |

This is an agent-workload measure, not a pure decode benchmark: it includes long prompts, thinking/tool responses, prefill, and each completion's end-to-end API duration. It must not be compared directly to the model card's short fixed-prompt decode rows.

The same full eval was MTP-4 enabled:

```text
--speculative-config {"method":"mtp","num_speculative_tokens":4}
```

## Why the number still warrants investigation

The model card reports, on its B70 reference system, approximately 83.7 tok/s for a BF16-draft MTP-4 fixed decode and 102.6 tok/s at its recommended non-thinking C1 sampling. The workload difference explains part of the gap, but the current agent median of 17.10 tok/s is low enough to require controlled measurement before accepting it as the host baseline.

The current full run did not retain the vLLM container logs after shutdown (`docker run --rm`), so it does **not** contain MTP acceptance, prefill/decode split, or service-level TTFT telemetry. Do not infer those values from the trace alone.

## 2026-08-23 controlled C1 outcome

The unchanged BF16 launcher was restarted with persisted host and vLLM logs. The public endpoint passed LAN and Tailnet health checks. Controls used the evaluation sampling, one request at a time, `max_tokens=128`, and the same payloads in both arms.

| Control | BF16 draft | Draft-INT4 overlay | Outcome |
| --- | ---: | ---: | --- |
| Short (575 prompt tokens, g128; n=5 median) | 38.99 tok/s | 53.20 tok/s | +36.4% |
| Medium coding (3,138 prompt tokens, g128) | 24.52 tok/s | 30.56 tok/s | +24.6% |
| Long (80,076 prompt tokens, g128) | 1.69 tok/s, completed | HTTP 500 | overlay rejected |

The optional overlay mounted `patch_draft_lmhead_int4.py` and `patch_draft_mtp_int4.py`, applied both patches after the existing MTP pair, and set both `B70_DRAFT_*_INT4=1` flags. It was not a concurrency experiment and did not add the mixed-batch v5 patch.

The overlay's long request killed the engine with `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` in `async_tensor_h2d`; vLLM reported 5.31 GiB of graph-capture memory before serving. It therefore cannot satisfy the unchanged 98,304/32,768 evaluation contract, despite its short-control improvement. No rollout smoke or full cohort was run. The BF16 server was restored; its public p8k smoke completed (8,263 prompt tokens, 16 completion tokens, 5.33 s).

Artifacts:
- BF16 controls: `~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T163646Z-c1-service-controls/`
- Draft controls and OOM response: `~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T165919Z-c1-draft-int4-service-controls/`
- Restored-BF16 p8k smoke: `~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T170432Z-bf16-restored-p8k-smoke.json`
- Server logs: `~/inference/logs/qwen38-speed/20260823T163300Z-control/` and `~/inference/logs/qwen38-speed/20260823T165609Z-draft-int4-debug/`

The short prompt is deliberately repetitive and has lower MTP acceptance than natural Pi traffic, so it is only a matched overlay comparison—not a model-card or agent-rate claim. Do not re-enable this overlay until a separate memory-safe configuration preserves the 131,072-token serving ceiling and passes the same long control.

## Recommended next-session plan

### 1. Improve service-level control measurements

The first C1 controls are recorded above, but their repetitive short prompt is not agent-representative. Before another serving variant, use a tokenizer-verified short/medium/long suite with natural Pi-like prompts; retain per-request TTFT, generated tokens, wall time, output tok/s, MTP acceptance, exact launch/image/model/host evidence, and logs outside the `--rm` container. Keep the evaluator on `inference-host` using `127.0.0.1:8000/v1`; do not try C2/C5 until C1 is stable.

### 2. Keep the eval control immutable

Use the same taskset digest, Pi harness version, task order, sampling, 98,304/32,768 token limits, and Docker runtime. Record a separate row/output directory for every serving variant. Do not attribute a score change to throughput when context, sampling, or concurrency also changed.

### 3. First candidate: optional INT4 draft overlay

The model card documents an optional runtime overlay that quantizes only the draft LM head and five MTP linears, retaining the GPTQ target and BF16 verification head. It claims a large fixed-decode improvement but explicitly says draft logits change.

The cookbook describes this as requiring its two draft patches plus:

```text
B70_DRAFT_LMHEAD_INT4=1
B70_DRAFT_MTP_INT4=1
```

Treat it as a new experimental row. First repeat the service controls, then one rollout smoke, then the full pinned cohort. Do not replace the BF16-draft baseline.

### 4. Second candidate: mixed-batch correctness before Qwen concurrency

The current Qwen launcher has not mounted the cookbook's `patch_gdn_mixed_split_v5.py`. The cookbook says it is required for mixed speculative/non-speculative batches at C>=2. Add and validate that patch before trying a concurrent Qwen evaluation.

Suggested progression:

```text
C1 control -> C2 fixed tool-call smoke -> C2 short agent smoke -> C5 fixed tool-call smoke -> C5 evaluation
```

The model card's C5 claims use roughly 8k coding sessions, not five simultaneous 98k-token agent contexts. Measure actual KV headroom and queueing; do not assume C5 is valid for this evaluation contract.

### 5. Prefix caching is an experiment, not a default

The baseline disables prefix caching. The model card reports a small C1 decode regression when cache is on and warns that concurrent-session reuse can be poor. Test it only after the service-level control suite, especially on repeated Pi system/tool prefixes.

### 6. Do not chase no-op changes

- The eval sends explicit sampling values; vLLM's generation-config warning concerns defaults and is not evidence that sampling was ignored.
- Moving ports or changing firewall rules does not improve local decode speed.
- The `qwen3_xml` parser is required for correct tool execution and should remain enabled; changing parsers is not a speed experiment.

## Primary sources

- B70 GPTQ/MTP model card: <https://huggingface.co/SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16>
- B70 cookbook Qwen recipe: <https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/main/docs/qwen38-27/QWEN38-VLLM-XPU.md>
- B70 cookbook Pi-agent tool configuration: <https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/main/docs/qwen38-27/PI-AGENT-BACKEND.md>
- vLLM XPU support: <https://docs.vllm.ai/en/latest/models/hardware_supported_models/xpu/>
