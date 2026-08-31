# Qwen3.8-27B B70 default configuration

Last verified: 2026-08-31 on `inference-host` (Intel Arc Pro B70, `xe`).

## Active default — 212,992-token C1, no KV offload

The persistent host launcher is `~/inference/launchers/start-qwen38.sh`, sourced from [`scripts/start-qwen38.sh`](../scripts/start-qwen38.sh). It is the default Qwen service—not a temporary experiment.

- Model: `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`
- Image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
- API: `http://inference-host:8000/v1`, model ID `qwen38`; no API-key authentication is configured.
- Context and scheduling: `max-model-len=212992`, `gpu-memory-utilization=0.95`, `max-num-seqs=1`, `max-num-batched-tokens=8192`, FP8 KV cache, prefix caching, and `mamba-cache-mode=align`.
- Performance implementation: MTP-4; XPU graph; Draft-INT4 S+M1; v5 mixed split; the five patches run in this order: `patch_mtp_nightly.py`, `patch_mtp_boundary.py`, `patch_gdn_mixed_split_v5.py`, `patch_draft_lmhead_int4.py`, `patch_draft_mtp_int4.py`.
- Environment: `B70_MTP_BF16_DRAFT=1`, `B70_DRAFT_LMHEAD_INT4=1`, `B70_DRAFT_MTP_INT4=1`, `VLLM_XPU_ENABLE_XPU_GRAPH=1`, `VLLM_TARGET_DEVICE=xpu`, `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`, `ZE_AFFINITY_MASK=0`, and `PYTORCH_ALLOC_CONF=expandable_segments:True`.
- Power: `power/control=on`, Xe `power1_cap=230000000`.
- **No KV offload:** no offload mount, `KV_XFER_CONFIG`, or KV-offload CLI flag is present.

### Output semantics and request defaults

- `--chat-template-content-format openai` pins the public OpenAI chat-message format.
- Thinking is enabled by default; clients can disable it per request with `chat_template_kwargs: {"enable_thinking": false}`.
- `--reasoning-parser qwen3` separates thinking from the final answer. vLLM 0.27 returns it as `choices[0].message.reasoning` (not the older `reasoning_content` name); final text is `message.content`.
- The server overrides the model's bundled sampling defaults with Qwen's thinking defaults: `temperature=0.6`, `top_p=0.95`, `top_k=20`, and `min_p=0`. Clients may override these request fields.
- For an explicit non-thinking request, use `enable_thinking=false` and Qwen's non-thinking settings: `temperature=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0`, and `presence_penalty=1.5`.
- Request-specific limits and penalties (`max_completion_tokens`/`max_tokens`, `stop`, `seed`, `frequency_penalty`, `presence_penalty`, `repetition_penalty`, and structured-output controls) remain caller-owned OpenAI API parameters; do not set a server-wide value unless the application contract requires it.
- Tool calling remains enabled with `--enable-auto-tool-choice --tool-call-parser qwen3_xml`.

### Effective-settings audit

| Layer | Knob(s) | Effective setting / ownership |
| --- | --- | --- |
| Model execution | image, model, quantization, dtype | Pinned image above; local GPTQ-Int4 model; `gptq`, `float16`. |
| Context and cache | `max-model-len`, memory utilization, KV dtype, Mamba cache, offload | `212992`, `0.95`, FP8, `align`, and no offload. |
| Scheduling | sequence and batch limits, prefix cache | C1 (`max-num-seqs=1`), `max-num-batched-tokens=8192`, prefix cache enabled. |
| Speculation | speculative method and token count | MTP with four speculative tokens; five documented B70 patch layers. |
| XPU | graph and B70 environment | XPU graph enabled; documented Xe/Level Zero/Draft-INT4 environment values above. |
| Chat format | content format, default thinking | OpenAI content format; thinking enabled by default; caller may set `enable_thinking=false`. |
| Reasoning | parser and response field | `qwen3`; reasoning is `message.reasoning`, final answer is `message.content`. |
| Thinking sampling | `temperature`, `top_p`, `top_k`, `min_p` | Server defaults `0.6`, `0.95`, `20`, `0`; callers can override. |
| Non-thinking sampling | sampling and presence penalty | Caller owns it; recommended explicit values are `0.7`, `0.8`, `20`, `0`, and `1.5`. |
| Request controls | token limit, stop, seed, penalties, repetition, `n`, logprobs | Caller-owned OpenAI request parameters; no hidden server defaults beyond the thinking sampling row. |
| Reasoning controls | `include_reasoning`, `thinking_token_budget` | Caller-owned. `include_reasoning` defaults to true in the API. No server thinking-budget cap is imposed. |
| Tools and structured output | tool choice/parser, response format, structured outputs | Auto tool choice plus `qwen3_xml`; callers own tools, tool choice, response format, and structured-output schema. |
| Transport | streaming, IDs, priority, cache salt | Caller-owned API controls; no authentication is configured on the endpoint. |

### End-to-end verification

Public API evidence: `~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260831T020608Z-default-semantics-e2e/public-api-semantics.json`.

1. `GET /v1/models` returned `qwen38`.
2. A default-thinking `POST /v1/chat/completions` returned the private chain in `message.reasoning` and only `OK` in `message.content`.
3. A streamed default-thinking request emitted 11 reasoning deltas.
4. An explicit `enable_thinking=false` request returned `reasoning: null` and only `OK` in `content`.
5. A non-thinking function-call request returned a parsed OpenAI `tool_calls` object for `get_weather({"city":"Paris"})` with finish reason `tool_calls`.

## Capacity and performance evidence

- Highest tested successful `max-model-len`: **212,992**. The boundary between it and 229,376 was not narrowed.
- No-offload receipt: `~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260830T201650Z-champion-max-context-nooffload-high/`.
- At 212,222 prompt + 106 completion tokens: TTFT 444.738 s, prefill **477.184 tok/s**, and client post-first decode **35.686 tok/s**. Decode is `(completion_tokens - 1) / (stream end - first SSE choices chunk)`.
- Native 8 GiB KV offload did not extend the tested boundary: 262,144, 245,760, and 229,376 failed initialization both with and without it; 212,992 succeeded without it.
- The retained short-context performance reference used `max-model-len=131072`, `gpu-memory-utilization=0.88`, and `max-num-seqs=64`. Its five-run streaming `g256` median was **95.432 tok/s** (95.419–95.458), using the same client post-first decode method. It is not the active default.
- C1 active-context sweep receipt: `~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260830T214516Z-champion-context-sweep/`: 32,016 / 64,776 / 97,546 / 129,547 prompt tokens measured prefill 1,396.507 / 1,107.730 / 833.241 / 719.508 tok/s and post-first decode 81.007 / 71.717 / 63.699 / 57.346 tok/s.

## References

- https://docs.vllm.ai/en/stable/features/reasoning_outputs/ — Qwen3 parser, thinking controls, and `message.reasoning` migration.
- https://docs.vllm.ai/en/latest/cli/serve/ — generation-config overrides and serve options.
- https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html — Qwen thinking/non-thinking sampling settings.
- https://docs.vllm.ai/en/latest/api/vllm/config/cache/ — context length and KV allocation tradeoff.
