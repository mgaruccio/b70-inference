# Qwen3.8 on one B70: sustained-speed progress

Started 2026-09-08 UTC. This is a working research log for a later public post, not a claim of a new record.

**Latest outcome:** short-context decode is GEMM-bound. On the production graph path, `gemm_kernel` is **93.39%** of Self XPU; that time splits about evenly between INT4 W4A16 (**47.29%**, 1.095 ms avg) and the unquantized target `lm_head` `aten::mm` of shape `[5,5120]×[5120,248320]` (**46.10%**, 4.271 ms avg). Compiled no-graph traces show GDN and flash-attn as a few percent at this prompt length. Glimmer was restored after a first-restart segfault. **No kernel or launcher was changed.**

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

## Follow-up: repeatability and exact-five graphs

User authorized continuation after the baseline milestone. No permanent service change is planned for this experiment.

### Fresh research and pinned-source checks

- https://docs.vllm.ai/en/stable/usage/reproducibility/ — fixed seed/greedy sampling alone does not guarantee online reproducibility. The earlier `/serving/reproducibility/` URL returned 404; the correct section is `usage`.
- https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/docs/usage/reproducibility.md — pinned documentation is NVIDIA-only for batch invariance. Current XPU support uses a newer/different Triton-attention path; do not import it into this native-XPU experiment.
- https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/worker/xpu_model_runner.py — pinned XPU graph adapter.
- https://docs.vllm.ai/en/stable/design/prefix_caching/ — salt participates in cache identity. Distinct cold salts prevent prefix reuse; repeated fixed salt is a separate warm-cache diagnostic.
- https://github.com/vllm-project/vllm/pull/34936 — performance-mode behavior; `interactivity` captures more than only size 5.
- https://github.com/vllm-project/vllm/issues/54698 — pinned-stack graph-replay report; inspect actual local dispatch, not just configuration labels.

Read-only inspection inside the existing pinned-image container confirmed:

- `config/vllm.py:1914–1922,1953–1993`: explicit capture sizes override the generated list; dispatch uses the nearest padded captured size. `interactivity` includes every small size, so it is a broader change.
- `config/compilation.py:648–651,692–700`: explicit list supported, with maximum inferred from the largest entry.
- `engine/arg_utils.py:1642–1660`: `--compilation-config` and `--performance-mode` are accepted.
- `entrypoints/openai/chat_completion/protocol.py:415–423`: `return_token_ids=true` exposes prompt IDs and streamed generated-token deltas.
- `compilation/cuda_graph.py:33–123`: built-in graph statistics report unpadded tokens, padded tokens, runtime mode and count. Enable `cudagraph_metrics=true` equally in all cells rather than adding a custom tracing patch.

A transient SSH banner timeout cleared on the bounded retry. A remote Fish heredoc inspection failed before executing; the corrected `python -c` inspection succeeded. Neither changed services.

### Predeclared experiment

- **A1 / A2:** balanced, explicit `[1,2,4,8]`.
- **B:** balanced, explicit `[1,2,4,5,8]`. Only the presence of size 5 changes relative to A.
- **N:** diagnostic no-MTP control; remove only `--speculative-config` from A's launcher. Not a speed competitor and not assumed to have identical execution numerics.
- All use the same original launcher/weights/patches, observed 275 W cap, C1/212992/FP8 KV settings, plus the same existing graph-stat logging option. Assert the launcher hash and power cap before proceeding. No new batch-invariance mode, eager execution, kernel changes, quantization or context reduction.
- Sequence: **A1 → N → B → A2**. Before speed work in each cell: natural-stop canaries plus exact raw token-ID prompt lengths **1–128, 133, 197, 261** (131 probes), using `/v1/completions`, token ID 42 repeated N times, `max_tokens=1`, `ignore_eos=true`, `logprobs=1`. Require reported prompt count N and a finite selected-token logprob. Failure aborts the experiment and restores Glimmer; passing is a bounded shape check, not broad semantic parity.
- A1/B/A2: prior exact code/prose prompts, one warmup and five measured 512-token trials per prompt. N: the same code prompt, one warmup/five trials. A1 also runs one warmup/five code trials with one fixed salt, separate from cold measurements.
- Requests retain `temperature=0`, `seed=42`, thinking disabled and forced 512 output tokens. Enable `return_token_ids=true`; require token-ID counts to match usage. Use matching per-trial cold salt labels across freshly started cells, unique within each cell. Save per-request accepted/drafted/cache-counter deltas, output sequences and hashes.
- Compare **within the same prompt**, not code versus prose. Exclude warmups. Check B against both A1 and A2 for drift; group by token sequence if outputs differ. Do not attribute small rate differences to the graph when content differs. Inspect actual `5 → 5` versus padded `5 → 8` graph statistics before claiming the mechanism.
- Run entrypoint: `ssh -o BatchMode=yes -o ConnectTimeout=10 inference-host 'python3 -u /tmp/qwen-b70-exact-five-20260908.py'`. The complete disposable runner copies itself into the timestamped host `*-exact-five/runner.py`; each cell retains its exact launcher, request/result JSON, timed SSE, boundary responses and server log. No new persistent testing service or source patch.
- Cleanup: stop temporary Qwen and restart/health-check the preserved Glimmer container in `finally`, including after failed gates. No candidate promotion during this run.

Local preflight checks passed Python syntax and `bash -n` on all three launcher variants; parsed shell arguments confirmed valid JSON and isolated graph/spec flag changes. These are supplemental checks, not substitutes for the real API experiment.

**Attempt 1:** task `bdda7ac20`, host directory `20260908T054736Z-exact-five/`, failed before inference: the pinned CLI rejected `--observability-config {"cudagraph_metrics":true}`. The first local syntax/JSON checks did not catch unsupported CLI flags. Glimmer restoration passed. Corrected to the actual `--cudagraph-metrics` flag (`engine/arg_utils.py:1480`), validated both graph configurations with the pinned `EngineArgs` parser inside the running container, and made readiness fail fast if the launcher exits. The rejected attempt is a harness error, not a Qwen graph result.

**Attempt 2:** corrected task `b74df0aea`, host directory `20260908T055830Z-exact-five/`, started A1 successfully. Three natural-stop canaries passed with returned token counts matching usage. Raw prompt lengths 1–4 returned HTTP 200 with finite logprobs; exact length **5 returned HTTP 400**, aborting the candidate campaign. The first runner did not retain that error body, so its cause is not yet asserted. Glimmer restoration passed.

**Hypothesis rejected before A/B:** A1's native graph-stat table at 06:01:20 reports **unpadded 5 / padded 5 / zero padding / FULL / count 9**, despite the displayed capture-size configuration `[1,2,4,8]`. This disproves our assumption that ordinary C1 MTP4 verifier replay was padded to eight in this cell. Do not carry the community's +1.19% `interactivity` gain over to our setup or launch a redundant exact-five performance sweep. Configuration lists alone were insufficient evidence.

### Narrow diagnostic continuation

No candidate was deployed. The next run intentionally diagnoses the baseline rather than treating a failing gate as a performance pass:

- Two cells, **D-MTP4** and **N-no-MTP**, same weights/power/context/graph-stat instrumentation. Remove only the speculative-config flag for N; compare it as a diagnostic, not a speed improvement.
- Each cell: the three natural-stop canaries, then one warmup/five cold code repeats and one warmup/five fixed-salt code repeats, retaining complete prompt/output token IDs and per-request cache/acceptance deltas.
- Raw exact lengths **4/5/6, 68/69/70, 132/133/134**, using the earlier token-ID payload. Record HTTP status and full error body as well as finite selected-token logprobs. Failed probes stay failed; only continue diagnostic probes while `/health` responds, abort on server-side failure. No graph candidate or promotion follows these diagnostics.
- Entry point: `ssh -o BatchMode=yes -o ConnectTimeout=10 inference-host 'python3 -u /tmp/qwen-b70-mtp-diagnostic-20260908.py'`; copied runner and raw artifacts live in a timestamped `*-mtp-diagnostic/` directory. Restore Glimmer in `finally` as before.
- Parallel read-only source follow-up examines the community's prefill/decode-classification fix; no patch application is authorized by a source suggestion alone.

**Diagnostic completed:** task `b52fa4149`, host directory `/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260908T060708Z-mtp-diagnostic/`, exit 0; Glimmer restoration passed. Both cells kept the observed 275 W cap before/after. All six natural-stop canaries passed. This completes the diagnostic protocol, not a claim that the earlier failure was fixed.

| Code p68/g512 control, five measured trials | Median tok/s | Range | Distinct output-token sequences |
| --- | ---: | ---: | ---: |
| MTP4, distinct salts | **91.599** | 90.964–91.649 | 1 |
| MTP4, fixed salt | **90.295** | 89.593–90.944 | 2 |
| No MTP, distinct salts | **33.842** | 33.822–33.847 | 3 |
| No MTP, fixed salt | **33.842** | 33.817–33.844 | 1 |

Interpretation:

- **Variation is not MTP-only.** No-MTP distinct-salt trials also varied, with identical 68 prompt token IDs. Compared with their first output, divergences began at zero-based output positions 245 or 60. Two common output hashes occurred in both the MTP and no-MTP cells. This does not prove universal speculative/non-speculative parity or identify the underlying numerical cause.
- **The intended warm-cache control did not establish a warm cache.** Every measured fixed-salt request still reported **zero prefix-cache hits / 68 queried tokens**, just like distinct-salt requests. The artifact labels `warm-code-*` describe intent, not observed cache reuse. Call these *fixed-salt repeats*, not cached-turn measurements; no cache-on/off causal conclusion is supported.
- MTP distinct-salt output tokens were identical across five trials, but draft counts still varied (564–568 proposed, 373–374 accepted per request). Small timing changes can reflect speculative acceptance without changing final content.
- All nine neighboring-length probes passed in **both** cells, including 5/69/133. These probes ran **after** the longer generation workload; unlike the original failure they were not the first short-prompt sequence after startup. The previous HTTP 400 remains real but unexplained; success under a different history does not clear it.
- The MTP graph log again showed **FULL 5→5**, including intervals with 147 and 245 such steps. The exact-five padding opportunity remains rejected.
- This is an unchanged-stack characterization, not a 91.599-versus-89.647 optimization gain. Request history, instrumentation and returned token tracing differ from the first baseline. No new configuration was promoted.

**Cold-order reproduction completed:** task `be338efa0`, host directory `/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260908T062117Z-cold-order-repro/`, replayed the original fresh-server order—three canaries, then exact raw lengths 1–128/133/197/261. No long-generation warmup or classifier patch. Canaries and lengths 1–4 passed; **length 5 failed again**, now retaining the HTTP 400 body: `{"error":{"message":"Out of range float values are not JSON compliant: nan","type":"BadRequestError","param":null,"code":400}}`. This is a repeated NaN response failure, not an unsupported request field. The run stopped at the first failed probe and restored Glimmer. Entry point: `ssh -o BatchMode=yes -o ConnectTimeout=10 inference-host 'python3 -u /tmp/qwen-b70-cold-order-repro-20260908.py'`.

### Source follow-up: possible prefill/decode alias

The public investigation identifies a legacy-runner classifier that uses only batch shape. An MTP4 five-token prefill can therefore look like a five-token decode verification group; aligned chunking can expose the same alias at `64*N+5`. This began as a candidate explanation; the subsequent local guarded trial below corroborates the classification defect on our reproduced input.

Sources read in the focused follow-up:

- https://github.com/vllm-project/vllm-xpu-kernels/issues/548
- https://github.com/vllm-project/vllm/pull/53059
- https://raw.githubusercontent.com/AnnoyingTechnology/intel-arc-b70-llm-inference/main/docs/gdn-64n5-investigation-pause-2026-08-25.md
- https://raw.githubusercontent.com/AnnoyingTechnology/intel-arc-b70-llm-inference/main/docker/patches/patch_uniform_decode_prefill.py

The community runtime patch was fetched and read directly. It strictly matches the old `_is_uniform_decode` implementation and one live call site, adds `has_prefill` from `num_computed_tokens_cpu < num_prompt_tokens`, and requires `not has_prefill` before shape-based classification. It preserves intentional `force_uniform_decode` overrides. It compiles the modified source before writing and fails closed on changed anchors. It targets legacy `gpu_model_runner.py`, not Model Runner V2. Upstream call-site/regression coverage still needs checking before local use.

This is materially different from adding a space to troublesome prompts: dispatch correction preserves input tokens, whereas prompt-padding containment changes the model trajectory. At this source-review stage, no patch had been applied. The reported upstream regression coverage includes neighboring/exhaustive prompt lengths, chunked and mixed prefills, no-spec one-token aliases, genuine decode, forced capture, quality canaries and repeat checks.

The runtime graph table, rather than the static size list, remains our evidence that the measured baseline's ordinary MTP4 steps were already FULL 5→5. The source review does not claim to have fully reconstructed how the header's list became that effective descriptor inventory.

### Temporary classifier-guard validation

The reproduced cold-order NaN and the public prefill-alias investigation justify testing the minimal guard, not assuming that it will fix all nondeterminism. **No permanent launcher change or prompt-padding workaround.**

- The directly fetched community script is retained as `/tmp/qwen-b70-patch-uniform-decode-prefill.py`, with source attribution. At runtime it is copied into the experiment directory and mounted read-only into a disposable Qwen container, applied after the existing five patch layers and before `vllm serve`.
- A read-only snapshot of the actual installed pinned `gpu_model_runner.py` contains one `_is_uniform_decode` call (line 4066), with keyword arguments. Its classifier and call-site strings each match the community patch exactly; this is not a blind apply to an arbitrary vLLM version.
- Local tests against a temporary fake-package copy of that installed source passed: exactly the two intended replacements, whole-module compile, idempotent reapply, and fail-without-writing for separately changed classifier/call anchors. Nine pure-classifier cases cover genuine MTP/no-spec decode, five-token and one-token prefills, chunked/mixed aliases, nonuniform batches, and true/false forced-capture overrides.
- Supplemental runner checks passed Python syntax, `bash -n`, read-only patch-mount validation, and preservation of MTP4/graph arguments. The independent read-only review completed: the single keyword call, source wiring, CPU-counter timing and forced-capture override were checked, with no untracked contract break found. This is source review, not a substitute for the live API gates.
- Public-API validation keeps **275 W, same weights/patches, MTP4, C1/212992, FP8 KV and `[1,2,4,8]`**. Sequence: same cold-order three canaries → exhaustive raw **1–128,133,197,261** finite-logprob probes → prior code/prose p68/p63 g512 warmup plus five trials → a **49,925-token** raw prefill with one output token/finite logprob. No input padding. Stop on any failed probe and restore Glimmer.
- Entry point: `ssh -o BatchMode=yes -o ConnectTimeout=10 inference-host 'python3 -u /tmp/qwen-b70-prefill-guard-trial-20260908.py'`. Each timestamped `*-prefill-guard-trial/` directory retains the exact runner, patch, launcher, payloads, response/token/metric data and logs.

**Guard validation completed:** task `b207376dc`, host directory `/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260908T063104Z-prefill-guard-trial/`, exit 0. The log confirms the guard was applied to the installed `vllm/v1/worker/gpu_model_runner.py`. Retained patch SHA-256: `baa4647398874c19175ea74fe6f5d8dd6c2d83fc4bd0e5f2a68558afd983f5ad`.

Observed real-API results:

- **132/132 finite-logprob probes passed:** raw exact lengths 1–128,133,197,261, then 49,925. The cold-start order that repeatedly failed at length 5 now passes without padding or changing its five prompt tokens.
- Representative selected-token logprobs: p5 **−2.526634693**, p69 **−1.365266681**, p133 **−0.487957239**, p49925 **−0.010509976**. Each returned the exact requested prompt-token count plus one output token and HTTP 200. The 49,925-token request completed in **36.466 s**; one output token is not a decode-throughput benchmark.
- **3/3 natural-stop canaries passed** (arithmetic, JSON and exact Python AST).
- Runtime mechanism evidence: the five-token prefill now records **5→5, zero padding, PIECEWISE**; genuine MTP decode still records **5→5, FULL**. This changes classification, not verifier graph width or input length.
- Code p68/g512: **90.398 tok/s median**, 89.154–91.720, n=5, two output-token sequences. Prose p63/g512: **74.648 tok/s median**, 74.640–74.670, n=5, one sequence. All ten measured requests had zero prefix-cache hits and exact 512-token output. Warmups excluded.
- Power cap remained **275 W** before/after. The container was removed and original Glimmer restored healthy with `muse-glimmer-gptq`.

**Keep/kill decision:** keep this **already-published community guard as a validated prerequisite for subsequent experiments**; kill the speculative five-versus-eight graph speed hypothesis. The guarded code/prose rates remain in the previous range, with different request history and code trajectories, so no improvement or formal no-regression percentage is claimed. The guard does **not** fix all greedy variation, prove broad model quality, validate warm prefix reuse, or establish mixed-concurrency safety.

**Deployment status:** guard and exact test runner are retained in the host experiment directory; no permanent model, launcher, kernel, power or context change was made. The existing default Qwen launcher still lacks this guard and must not be described as having passed the new cold-order regression. Do not silently resume optimization from that unguarded default. A future default promotion needs the guard explicitly included in the maintained launcher/patch source; it is not implied by this temporary test.

**Next performance cut:** use the guarded cell as the controlled research baseline and investigate MTP acceptance/depth on fixed sustained prompts, rather than adding a graph size that runtime already uses. Keep 210 W community results separate from our 275 W data, preserve the same prompts/output lengths, and make any gain content- and acceptance-aware. No further optimization campaign was launched in this pass.

## Follow-up: guarded MTP-depth sweep

The user authorized this next experiment with “ok cool do it”. This tests depths, not a permanent default promotion.

Fresh external check: https://docs.vllm.ai/en/latest/features/speculative_decoding/ — `num_speculative_tokens` is the number proposed per step, not the accepted length; more speculation is not guaranteed faster. Gains depend on hardware, model and sampling. Lossless-speculation theory does not imply bit-identical online results, and latest docs do not prove compatibility with this older XPU build. Local source/runtime checks remain required.

### Frozen protocol

- Order: **K4-A → K2 → K3 → K6 → K4-B**. The final K4 repeat bounds baseline drift; rejected candidates are not ranked as speed results.
- Same pinned image, original launcher, GPTQ target/draft settings, validated prefill-classification guard, FP8 KV, **212992-token C1**, batched-token cap 8192, observed **275 W** cap, `performance-mode=balanced`, explicit graph list `[1,2,4,8]` and built-in graph-stat logging. Only `num_speculative_tokens` changes; resulting internal graph/state geometry is a consequence of that depth. Do not confuse the static graph list with observed replay descriptors.
- Assert original launcher hash and the exact previously validated guard hash. Keep original launcher/weights/power untouched; mount the guard into disposable containers as in the successful guard trial.
- Before timing each cell: three natural-stop canaries and the **same 140 finite-logprob raw probes**. Lengths are 1–128 plus the union of `128/192/256 + (K+1)` for K=2,3,4,6. This covers all candidate aliases without changing pre-benchmark request order between depths.
- Then use the identical earlier code/prose chat payloads (p68/p63, g512, temperature0, seed42, thinkingfalse, forced output, returned token IDs): one warmup and five measured trials per prompt. Salt labels match between freshly started cells and are unique within each cell. Verify actual prefix-cache hits rather than assuming cache state.
- Save per-request draft/acceptance/cache counters, actual token IDs and timings. Compare code against code and prose against prose, with output-trajectory differences exposed. Do not average unlike prompts into a headline or count a faster but incorrect candidate as a win.
- Reject a failed candidate, preserving its error and log. Stop it before another cell; abort the matrix on failed K4 control, device-loss/hang evidence, unsafe cleanup, or exhausted runtime budget. Restore the original Glimmer container and verify its model endpoint in `finally`.
- Entry point: `ssh -o BatchMode=yes -o ConnectTimeout=10 inference-host 'python3 -u /tmp/qwen-b70-mtp-depth-sweep-20260908.py'`. Exact runner/guard, per-cell launchers, requests, SSE/token outputs, metrics, finite-probe responses and logs are retained under a timestamped host `*-mtp-depth-sweep/` directory.

Supplemental preflight passed: Python and generated Bash syntax; parsed CLI arguments are identical after replacing only the MTP-depth JSON value with a common sentinel; all cells preserve context/C1/graph flags. The 140-probe union is unique and identical across cells. These checks do not substitute for live startup and API gates.

**Sweep completed:** task `b171acb70`, host directory `/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260908T132303Z-mtp-depth-sweep/`, exit 0. All five cells completed; none were rejected. Glimmer restoration passed.

### Depth results

All numbers are client post-first-content decode tok/s, median of five measured trials after one warmup. Code is p68/g512; prose is p63/g512; forced 512-token outputs, not complete-answer benchmark scores.

| Guarded cell | Code median (range) | Prose median (range) |
| --- | ---: | ---: |
| K4-A | **90.949** (90.273–91.602) | **74.585** (74.498–74.597) |
| K2 | 74.826 (72.598–74.887) | 66.893 (66.840–66.901) |
| K3 | 85.998 (84.892–86.020) | 70.732 (70.686–70.742) |
| K6 | **95.678** (94.851–97.278) | 69.532 (69.521–69.558) |
| K4-B | **91.558** (88.511–91.666) | **74.631** (74.570–74.651) |

### Content-matched interpretation

- K6's five code outputs all shared token hash `54d4bf7e847a…`, matching all five K4-A outputs and three of five K4-B outputs. Restricting K4-B to those three gives **91.661 tok/s**, rather than its mixed-content 91.558 median. K6's gain is therefore **+5.20% versus K4-A** and **+4.38% versus content-matched K4-B**. This range describes two controls, **not** a statistical confidence interval or a general coding-workload claim.
- Matched-code baseline drift was **+0.783%** from A to B, smaller than the observed K6 gain. K6's observed range also remained above the matched K4 rates in this run; independent-session reproduction and a broader coding cohort remain untested.
- K6 and both K4 prose cells generated the **same full token sequence** (`31616e525564…`). K6 was **6.78–6.83% slower**. Prose baseline drift was only **+0.061%**, so this is a real counterexample to claiming a general K6 win from these samples.
- K2/K3 were slower on both fixed prompts. Their prose outputs shared a different hash (`63229b9a3b87…`), so do not label those prose differences as content-matched throughput deltas. Their common-code-hash subsets were also slower than K4: K2 **74.861** (n=3), K3 **86.005** (n=4).
- Code variation persists in some cells (K4-A/K2/K3/K6/K4-B: **1/3/2/1/3** distinct measured token sequences). Passing canaries is not universal greedy parity.
- Accepted draft tokens per draft cycle, measured over the five cold trials, increased more for code (**2.627 at K4-A → 3.305 at K6**) than prose (**1.971 → 2.097**). That supports the interpretation that extra drafting pays off on this code prompt but adds insufficient accepted prose tokens. These ratios are not component-level profiling or guaranteed emitted tokens per cycle.

### Verification and disposition

- **700/700 finite-logprob probes** and **15/15 natural-stop canaries** passed. In particular K3 started and passed on this guarded recipe; older startup failures are not reproduced here.
- All **50 measured requests** returned exactly 512 token IDs matching server usage, nonempty content, `finish_reason=length` and completed SSE. Prompt token-ID arrays matched across every cell within each family. Every measured request reported **zero prefix-cache hits**.
- Power cap stayed **275 W** before/after every cell. The retained guard hash stayed `baa4647398874c19175ea74fe6f5d8dd6c2d83fc4bd0e5f2a68558afd983f5ad`; original model/launcher bytes were unchanged.
- Runtime logs confirmed the requested MTP depths and actual FULL replay groups: K2 **3→3**, K3 **4→4**, K4 **5→5**, K6 **7→7**. The static capture-size header alone was not used as execution evidence.
- **Keep K4 as the general guarded research baseline.** Retain K6 as a promising **code-prompt-specific candidate**, not an unrestricted replacement. K2/K3 do not win these cells. Do not select a universal winner by averaging code and prose.
- No persistent launcher promotion, power increase, shorter context, new model conversion or dependency upgrade was made. Original Glimmer was restored healthy; the default Qwen launcher still does not contain the validated guard. A wider coding-quality/long-session test would be required before treating K6 as a deployment-ready coding profile.

The result is a modest local optimization opportunity with a measured downside, **not a community leaderboard claim**: the external sustained reference uses a different prompt, quantized target and 210 W cap.


## Follow-up: kernel profile

The user asked to profile Qwen and look for kernel-level headroom. This is a diagnostic, not a speed run. Profiled tok/s must not be used as a promotion metric.

Fresh external check: [vLLM profiling](https://docs.vllm.ai/en/latest/contributing/profiling/) warns that profiling slows inference; traces are for developers. Older env-var docs: [v0.11.0 profiling](https://docs.vllm.ai/en/v0.11.0/contributing/profiling.html). This image already accepts `--profiler-config.profiler=torch` plus `/start_profile` and `/stop_profile`, as used on Glimmer 2026-09-05. PyTorch records `ProfilerActivity.XPU` as Self XPU. Inner kernels can be hidden inside graph replay, so a compiled no-graph cell is diagnostic only.

### Frozen protocol

- Same pinned image, original Qwen launcher, GPTQ target, validated prefill guard, MTP-4, FP8 KV, **212992-token C1**, observed **275 W** cap. Only profiler flags and, in the second cell, graph disablement change.

- Order: **graph** (production XPU graphs, `delay_iterations=4`, `max_iterations=20`) then **compiled** (`VLLM_XPU_ENABLE_XPU_GRAPH=0`, `cudagraph_mode=NONE`, 4+4 iterations). Skip eager: it also disables compilation and is a worse proxy of serving.

- Workload: LRU-cache code prompt, thinking off, temperature 0, seed 42, unique salts. Three natural-stop canaries, one 32-token warmup, then on the graph cell an unprofiled 32-token decode before `/start_profile`. Profiled forced lengths: 64 graph / 32 compiled.

- Rank **Self XPU** from `profiler_out_0.txt`. Do not sum `gemm_kernel` with its wrappers: `gemm_kernel` is the device body of both `_xpu_C::int4_gemm_w4a16` and `aten::mm`. Chrome-trace CPU ops (`aten::to`, `empty_strided`) are not device time.

- Restore original Glimmer and verify `muse-glimmer-gptq`.

### Results

Host artifact: `/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260908T211205Z-kernel-profile/`. Lead task `b688c9981` (profiling completed; first Glimmer `docker start` segfaulted during KV setup). Retry `bc3f7c4d6` restored the container. Live check: `GET /v1/models` returned `muse-glimmer-gptq` / 131072; a non-stream `POST /v1/chat/completions` with `max_tokens=8` returned `finish_reason=length` and 8 completion tokens (`chatcmpl-80b78acf125a5bf6`).

Graph log: FULL replay **5→5**. Compiled log: XPU graph disabled and CUDA graph capture skipped. All six graph canaries/requests and five compiled canaries/requests passed transport checks. Unprofiled graph 32-token decode: **87.059 tok/s**. Profiled graph 64-token decode: 72.384 tok/s (profiler tax). Compiled warmup 80.259 tok/s; compiled profiled 7.088 tok/s. Those profiled rates are not a new baseline.

**Production graph, Self XPU (do not add parent+child):**

| Name | Self XPU | Calls | Avg | Role |
| --- | ---: | ---: | ---: | --- |
| `gemm_kernel` | **93.39%** (155.752 ms) | 90 | 1.731 ms | device body of both GEMMs |
| `_xpu_C::int4_gemm_w4a16` | **47.29%** (78.873 ms) | 72 | 1.095 ms | GPTQ W4A16 |
| `aten::mm` | **46.10%** (76.879 ms) | 18 | **4.271 ms** | dense target `lm_head` `[5,5120]×[5120,248320]` |
| `_vllm_fa2_C::varlen_fwd` | 0.91% | 72 | 21 µs | full attention |

Graph chrome also recorded draft-looking INT4 vocab shapes `[1,5120]→248320` (`B70_DRAFT_LMHEAD_INT4=1`). Target `lm_head` is unquantized (`quantization_config.lm_head: false`).

**Compiled no-graph, Self XPU (kernel visibility, not serving):**

| Name | Self XPU | Calls | Avg | Role |
| --- | ---: | ---: | ---: | --- |
| `gemm_kernel` | **91.01%** (138.260 ms) | 1316 | 105 µs | device body |
| `_xpu_C::int4_gemm_w4a16` | **79.15%** (120.246 ms) | 1120 | 107 µs | GPTQ W4A16 |
| `aten::mm` | **11.86%** (18.013 ms) | 196 | 92 µs | mostly `[5,5120]×[5120,96]`, not the vocab matmul |
| `_xpu_C::gdn_attention` | 3.29% | 192 | 26 µs | GDN wrapper |
| `gdn::gated_delta_rule_spec_kernel` | 2.23% | 192 | 18 µs | spec GDN |
| `gdn::causal_conv1d_spec_kernel` | 1.05% | 192 | 8 µs | spec conv |
| `_vllm_fa2_C::varlen_fwd` | 1.03% | 80 | 20 µs | full attention |

Dominant compiled INT4 shapes at verify width M=5 (G128 packed `K/8`):

- `[5,6144]×[768,5120]` — GDN output / value path, 18.2 ms chrome, 260 calls

- `[5,5120]×[640,34816]` — fused MLP gate+up (`17408×2`), 17.7 ms, 260 calls

- `[5,17408]×[2176,5120]` — MLP down, 17.4 ms, 260 calls

- `[5,5120]×[640,16384]` — GDN in-proj, 12.8 ms, 260 calls

260 calls ≈ 4 profiled decode steps × 65 layer-like launches. Full-attention INT4 `[5,5120]→14336` is far smaller (4.5 ms, 68 calls). At this ~68-token prompt, GDN and flash are not the bottleneck.

### Keep/kill and next cut

- **Keep** the guarded MTP4 graph cell as the research baseline. This profile does not change depth or default launchers.

- **Largest serving-path opportunity:** the dense target `lm_head` (`[5,5120]×[5120,248320]`, 4.271 ms/call, ~46% of graph Self XPU). Next bounded test, if authorized: isolate that matmul (quantize vs keep dense, numerical check, then public-API decode). Do not assume GPTQ `lm_head` is free.

- **Second opportunity:** oneDNN INT4 W4A16 at M=5 for MLP/GDN projections (~47% of graph Self XPU). Glimmer already found strategy-override gains of ~1% on similar kernels; do not expect a large serving win without new evidence.

- **Kill as short-context kernel targets:** GDN spec kernels and flash-attn. They matter at long context; they are a few percent here.

- First Glimmer restore after the 212k Qwen cell segfaulted (`Segfault encountered` after model load, before KV ready). A later `docker start` on an idle GPU succeeded. Future Qwen jobs should treat Glimmer bring-up as a health gate, not a single `docker start`.

No persistent Qwen/Glimmer launcher, weight, power, or kernel change was made.
