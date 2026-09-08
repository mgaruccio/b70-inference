# Qwen3.8 on one B70: sustained-speed progress

Started 2026-09-08 UTC. This is a working research log for a later public post, not a claim of a new record.

## Request and first-pass scope

> ok let's take a shot at qwen 3.8 and see if we can get performance up. start by figuring out current best performance on the B70 and then let's make that our target. this will be a lot more saturated than glimmer but shouls still be some room to improve on what the community has done.
>
> keep notes on our progress for later posting

The user selected **single-user sustained speed** over concurrency or long-context latency, and authorized briefly stopping idle Glimmer to measure the existing Qwen configuration, then restoring Glimmer. This pass establishes the community target and a fresh local baseline. No model, kernel, launcher, power, or context settings are changed.

## Community evidence, fetched 2026-09-08

These are the strongest relevant public reports found in this search, not an exhaustive or independently audited leaderboard. `pN/gN` means prompt/output tokens. Every row below is single-request on **one Intel Arc Pro B70**, except the explicitly excluded two-card comparison.

| Source and workload | Reported result | Conditions and interpretation |
| --- | ---: | --- |
| AnnoyingTechnology, sustained p476/g512 | **85.61 tok/s median, six measured runs after warmup** | 210 W cap, vLLM XPU, Frozenlock-derived GPTQ W4A16 target, draft-only INT4 MTP4, FP8 KV, exact verifier graph via `interactivity`. Best clearly documented sustained cell found. |
| Same project, diversified ~p512/g128 | **91.98 tok/s median, five prompt families** | 210 W; more representative short-prompt diversity, but shorter output than the sustained cell. |
| Same project, favorable p512/g128 | **118.60 tok/s median, range 118.48–118.67** | Earlier **275 W** profile, forced 128-token output, cold unique prefixes/cache disabled; not a sustained/general-purpose rate. An isolated 118.83 row is also reported; do not promote that over the deployed median. |
| SergiioB, p512/g128 | **112.65 tok/s median, n=5 (111.58–117.73)** | 230 W configured cap, cache off, GPTQ target, draft-INT4 S+M1, MTP4, image `f01e24f6`, FP8 KV, 131072 configured context; BF16-draft control 81.20. Different cell from sustained g512. |
| SergiioB, current calibrated short C1 | **106.7 tok/s, n=5 (103.2–111.3)** | Prefix cache on/current-stack report; the same source reports only 56.8 on the different LMX short-prompt set. Prompt/acceptance dependence is substantial. |
| Jonathan Mann reproduction | **84.56 tok/s median client decode** | Researcher found an independent BF16-draft MTP4/GPTQ/FP8-KV reproduction. Supporting evidence for the stack, not a matched p476/g512 replication. |
| 0xSero, llama.cpp SYCL Q4_K_M/f16 KV | **33.3–33.4 tok/s** around 2.5k–40k | One-card non-speculative reference. Its ~51 tok/s and 84.3 easy-counting MTP figures use **two** B70s and are excluded from our one-card target. |

Primary sources:

- https://github.com/AnnoyingTechnology/intel-arc-b70-llm-inference — result overview, model path, power and caveats. The 85.61 / 91.98 / 118.60 numbers were read directly here; do not misattribute mirrored results as independent reproductions.
- https://github.com/AnnoyingTechnology/intel-arc-b70-llm-inference/blob/main/docs/benchmarks-and-quality.md — full cells, quality gates, rejected paths and exact-context limitations.
- https://github.com/AnnoyingTechnology/intel-arc-b70-llm-inference/blob/main/docs/xpu-single-user-tuning-2026-08-27.md — controlled balanced 84.60 → interactivity 85.61 tok/s (+1.19%), same p476/g512, six runs/side; rejected fusion regressed serving despite winning its microbenchmark.
- https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/qwen38-27/QWEN38-VLLM-XPU.md — 2026-08-19 recipe, short/long/concurrent distinctions, sampling and acceptance sensitivity.
- https://jonathanmann.tech/blog/qwen38-intel-arc-b70/ — published August 18, updated August 21, 2026; independent stack reproduction found by the research pass.
- https://github.com/0xSero/qwen38-b70/blob/main/BENCHMARKS.html — August 17 one/two-card SYCL comparison.
- https://huggingface.co/Qwen/Qwen3.8-27B — official model identification and native context/MTP architecture.
- https://docs.vllm.ai/en/latest/models/hardware_supported_models/xpu/ — official XPU hardware guidance; hardware support alone is not proof that a particular patched Qwen recipe is correct.

### Target decision

1. **First community target: 85.61 tok/s sustained p476/g512**, at the source's 210 W operating point. This is a target to reproduce under matched conditions, not a universal speed floor for arbitrary prompts.
2. Keep **118.60 tok/s p512/g128 at 275 W** as a separate short-cell reference, not the primary optimization objective or a reason to increase our power cap.
3. Improve our own fixed sustained cells against a same-session control, recording median, range, actual token counts, cache state and output validity. Aim for a repeatable gain that exceeds run-to-run noise; a higher unmatched number alone does not beat the community.
4. The quantized targets differ: our launcher uses SergiioB, while the sustained community recipe uses a Frozenlock-derived target. The latter repository reports rejecting its tested SergiioB setup on one of seven canaries. That is a reason to measure correctness, not proof that our current setup is corrupt or permission to swap weights silently.

## Existing local evidence (historical, not today's result)

The canonical [golden configuration](qwen38-b70-golden-config.md), last verified August 31, records:

- One B70; `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`.
- Pinned image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.
- Default: **212992-token C1**, memory utilization 0.95, FP8 KV, no KV offload, MTP4, draft-INT4 S+M1, mixed-split v5, XPU graphs and prefix caching; batched-token cap 8192.
- Historical **95.432 tok/s** short g256 median (95.419–95.458), but under the older **131072 / 0.88 / 64-sequence** server configuration. This is neither a fresh result nor a matched sustained-community comparison.
- Historical 212222-token prompt + 106-token output: 35.686 tok/s post-first decode and 477.184 tok/s input/TTFT. Do not compare this long-context row with a short prompt.
- Older `qwen38-b70-next-session.md` still describes 131k; the newer golden configuration and actual launcher take precedence.

Read-only preflight on September 8 found Glimmer (`glimmer-tb21-prefix-c8`) resident at localhost:18080, with zero running and waiting requests. The host cookbook checkout was `4ea5e596b3a01428b1416211cf3ecbde15b5bcda`. Glimmer is preserved as a stopped container during the measurement, not removed.

## Defined public-API baseline process

### Environment and preconditions

- Host: `inference-host`; local B70 only, no hosted GPU charges.
- Existing host launcher: `/home/mike/inference/launchers/start-qwen38.sh`; unchanged.
- Before switching: verify zero running/waiting Glimmer requests, no unexpected running containers, and launcher availability. Save launcher bytes/hash and readable Xe power-cap values in `preflight.json`.
- Execute via `ssh -o BatchMode=yes -o ConnectTimeout=8 inference-host 'python3 -u -'`, with the bounded orchestrator retained in the lead's background-task command history. The orchestrator uses `docker stop --time 30 glimmer-tb21-prefix-c8`, then `bash /home/mike/inference/launchers/start-qwen38.sh`.
- Wait up to 600 seconds for `GET http://127.0.0.1:8000/health`; confirm `qwen38` in `GET /v1/models`. No install or source patch edits.

### Workloads and executable API boundary

Use `POST http://127.0.0.1:8000/v1/chat/completions` through Python's standard-library HTTP client. Each saved `*-request.json` is an exact replayable payload, for example:

```sh
curl -N -H 'Content-Type: application/json' \
  --data-binary @code-1-request.json \
  http://127.0.0.1:8000/v1/chat/completions
```

Common parameters: `model=qwen38`, `temperature=0`, `seed=42`, `stream=true`, `stream_options.include_usage=true`, `chat_template_kwargs.enable_thinking=false`. Each request has a distinct `cache_salt`; server prefix caching stays enabled, but the measured requests deliberately do not reuse prefixes. Decode-speed trials use `max_tokens=512` and `ignore_eos=true`; these are **forced-length stress measurements**, not completed-answer quality claims.

Two fixed prompt families, each with one warmup then five measured trials:

**Code:**

> Write a detailed tutorial on implementing a bounded LRU cache in Python using collections.OrderedDict. Include a complete class, explain get and put behavior, discuss edge cases, and include tests. Continue with a worked example and complexity analysis. Be precise and use meaningful prose and code.

**Prose:**

> Explain how a relational database executes a SQL query, from parsing and planning through indexing, joins, transactions, and returning rows. Write a detailed technical tutorial with concrete examples and tradeoffs, including common performance pitfalls. Use complete sentences and avoid repeating yourself.

These are new local control prompts, **not** the community's unpublished-in-the-summary p476 payload. Record actual `usage.prompt_tokens`; do not call them p476 or p512 without measuring.

After speed trials, natural-stop smoke requests (no `ignore_eos`) check:

1. `What is 19 + 23? Reply with only the integer.` → exactly `42`.
2. `Reply with only this exact JSON object and no markdown: {"answer":42}` → JSON parsing to that object.
3. `Output only Python source, without markdown fences: define a function add(a, b) that returns a + b.` → Python AST matching that function, without executing generated code.

### Metrics, evidence, expected results and cleanup

- Save each payload, timed SSE events, usage, finish reason, content, TTFT to first choices and first content, total latency, and both post-first decode conventions.
- Primary local decode: `(completion_tokens - 1) / (stream_end - first_content_chunk)`. Also retain first-choices timing for comparison with the historical golden-config metric. SSE chunks are not assumed to be individual tokens.
- Require `[DONE]`, nonempty content, server usage and a finish reason; forced trials must return exactly 512 tokens with `finish_reason=length`. Report separate code/prose medians and ranges, excluding warmup.
- Save server log and `/metrics` before/after for cache and MTP inspection. Passing three small canaries is only a smoke check, not broad quality or token-level parity evidence.
- On success **or failure**, stop the temporary `qwen38` container (existing launcher uses `--rm`), restart the preserved Glimmer container, wait up to 600 seconds for health, and verify `muse-glimmer-gptq` in `/v1/models`.
- Raw host evidence: timestamped `~/b70-evals/qwen38-b70-gptq-int4-mtp4/*-sustained-baseline/`, including `summary.json`. Do not commit raw output or host runtime state.

## Run log

- 2026-09-08: Fresh external research completed; sustained target selected; existing local state and source discrepancies recorded.
- 2026-09-08: User authorized temporary GPU switchover. First background task `b22809b2c` exited 127 before SSH: the runner's Fish shell rejected Bash heredoc syntax, so the GPU service was untouched. Retried the exact stored command through explicit Bash as task `bb71e72ee`; it completed with exit 0.
- Raw host evidence: `/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260908T052530Z-sustained-baseline/`. Lead-side command and console output: `/home/mike/code/local-dev-model/.pi/tasks/session-121568-121568/b22809b2c.json` (original command) and `bb71e72ee.output` (successful run). These are local runtime artifacts, not repository files.

### Fresh unchanged baseline

| Fixed workload | Prompt / output tokens | Median post-first-content decode | Range, n=5 | Median TTFT |
| --- | ---: | ---: | ---: | ---: |
| LRU-cache tutorial | 68 / 512 | **89.647 tok/s** | 89.029–91.494 | 0.1564 s |
| SQL-execution tutorial | 63 / 512 | **74.553 tok/s** | 74.484–74.575 | 0.1141 s |

Observed facts:

- **Actual configured power cap was 275 W** (`power1_cap=275000000`), not the 230 W recorded in the historical golden-config note and not the community sustained target's 210 W. No power setting was changed. Instantaneous power/energy was not measured; a cap is not consumption.
- Launcher SHA-256: `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`, matching the checked-in launcher. The server identified vLLM `0.27.2rc1.dev77+gac7509e2b`, max length 212992, C1, FP8 KV and MTP4.
- All ten measured speed requests, both warmups and three natural-stop canaries completed. Speed requests each returned exactly 512 tokens, `finish_reason=length`, nonempty content and `[DONE]`.
- **3/3 natural-stop smoke checks passed:** `42`, JSON `{"answer":42}`, and the exact AST for `def add(a, b): return a + b`. These do not establish broad quality or completed-answer throughput. The forced tutorial outputs end mid-explanation/code as expected at 512 tokens.
- Zero prefix-cache hits were reported across 878 queried prompt tokens, consistent with the distinct request salts. Whole-run speculative counters (including warmups and smoke, not per-family): 4287 accepted of 7580 drafted tokens, 1895 draft cycles. Do not attribute the whole-run acceptance fraction to either measured family separately.
- **Repeat stability is not established:** the five code trials produced **three distinct continuations** despite temperature 0 / seed 42; prose produced one. Differences included wording and code layout, not just whitespace. The requests also differed in cache salt. This is not proof of corruption, but it prevents claiming exact greedy reproducibility or assigning small speed differences solely to a kernel change. Investigate same-salt/no-cache and no-spec controls before a lossless-quality claim.
- The engine log reports `cudagraph_capture_sizes: [1, 2, 4, 8]`, maximum 8, and `FULL_AND_PIECEWISE`. Size 5 is absent; the exact-graph hypothesis is grounded in this configuration, though actual per-step dispatch was not profiled.
- Startup emitted Transformers video-processor documentation errors for `min_frames`/`max_frames`. The service still became healthy and all requests completed; do not describe the log as error-free.
- Cleanup succeeded: Qwen stopped, and the original Glimmer container restarted healthy with model ID `muse-glimmer-gptq`.

**Decision:** keep **85.61 tok/s p476/g512 at 210 W** as the external sustained reference; keep **89.647 code / 74.553 prose** as our measured local controls at the observed 275 W cap. Do not average the two into a headline, call this a matched replication, or claim a community lead. **No optimization was applied and no gain is claimed.**

## First candidate after baseline

The startup configuration confirms graph sizes `[1, 2, 4, 8]`, with no exact size-5 graph. Confirm actual five-token MTP verifier dispatch and support for `--performance-mode interactivity` in the pinned image, then compare an isolated exact-graph candidate with the unchanged control. The community's +1.19% result is a small, already-known opportunity, not a new discovery. First bound the observed greedy continuation variation so that output-content changes cannot masquerade as a kernel win. Keep prompts, power, weights, context, cache semantics and quality checks matched; do not silently increase power, shorten context or swap quantizations. No such A/B has been launched in this pass.
