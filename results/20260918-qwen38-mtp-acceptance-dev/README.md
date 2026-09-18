# Acceptance-aligned saved-head diagnostic — development only

**Completed development result: a different saved head wins the new diagnostic and shows a promising native trend, but primary acceptance/decode intervals still include zero. Retain stock; no promotion or generalization claim.** This follows the [next-session handoff](../../docs/qwen38-mtp-acceptance-next-session.md) and [frozen protocol](protocol.md), reusing saved heads rather than retraining.

## Native dev serving result

Frozen candidate: **LR 1e-6 decay, final step 652**. Stock → candidate → candidate → stock, **38 matched dev contexts × 4 cells = 152 completed requests**, eight families, zero skipped timed requests. Capture was off; four synthetic warmups per cell were excluded. Exact inputs, prompt token IDs, budgets, runtime configuration, and candidate overlay hash were checked across cells ([validation](native-validation.json)).

| Pooled metric | Stock | Selected head | Relative change | Family-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| Accepted draft tokens / speculative pass | 2.321616 | 2.402994 | **+3.505%** | **−0.876% to +9.143%** |
| Decode tokens/sec | 81.2582 | 83.1645 | **+2.346%** | **−0.629% to +6.328%** |
| End-to-end tokens/sec | 73.6373 | 75.4096 | +2.407% | +0.399% to +4.831% |
| Draft acceptance fraction | 58.0404% | 60.0748% | +2.0344 percentage points | — |
| Median request latency | 2.81375 s | 2.57711 s | −8.41% | — |
| Median TTFT | 0.405033 s | 0.404954 s | essentially unchanged | — |

The primary acceptance and supporting decode intervals include zero. The end-to-end interval excludes zero in this small dev sample, but does not override uncertainty in the primary metric or establish fresh-test generalization. Both candidate cells exceeded both stock cells on acceptance and decode. Stock A1/A2: **2.31353/2.32994** accepted drafts/pass; candidate B1/B2: **2.42227/2.38456**. Raw aggregate run values and the paired 10,000-draw family bootstrap (seed 42, both repeats retained per family) are in [`native-summary.json`](native-summary.json).

Native depth survival stock → candidate: **82.424→83.490%, 63.821→65.586%, 48.539→51.321%, 37.377→39.902%**. Decode rate is `sum(generated_tokens−1)/sum(request_decode_time_seconds)`; end-to-end rate is `sum(generated_tokens)/sum(client_wall_seconds)`. Native accepted drafts/pass excludes the bonus token, unlike the runtime's logged mean acceptance length. Native counters may include terminal candidates discarded by the API. Completion counters exactly matched API token counts; prefix-cache hits were zero.

Output variability remains substantial: stock repeats matched exactly on **20/38** contexts, candidate repeats **20/38**; cross-head matches **17/38** and **21/38**. Only **17/38** were identical in all four cells. That post-hoc subset is reported in the JSON, not used as the primary result. Natural output totals differ (stock 29,036, candidate 29,944 tokens); no generated tool calls were executed or functional quality score inferred.

### Decision

The saved heads were **not all equivalent under the better-aligned development diagnostic**. The lower-LR final checkpoint, not the previous sampled-dev CE winner, ranks best offline and yields a similarly directed native trend. However, **CE and joint-prefix ranking agree on this new greedy/root-matched dataset**, so the experiment does not isolate the selection metric from the changed sampling policy and coverage. Nor does similarity of relative improvements prove dense/native decision parity: the offline mean (3.00) and scheduled runtime mean (2.40) use different trajectory/root/pass weighting and backend numerics.

**Retain stock and stop at a development result.** No extra checkpoint sweep, objective change, training, or automatic fresh-test generation is justified here. A future confirmatory experiment would freeze this candidate and require independent, unconsumed test families before claiming generalization. The optional second native comparison of the former CE winner was not run: the bounded primary comparison remains uncertain, and the new CE/prefix rankings do not disagree. Both prior test sets remain consumed.

This is not a standard-publishable/community-comparable benchmark: no full BetterBench, concurrency/long-context suite, continuous thermal trace, or functional quality regression suite.

## Offline result

A new **greedy** native dev capture changed the shortlist winner from the previous LR 5e-6/step-400 head to **LR 1e-6/final step 652**. Both CE and joint-prefix agreement on this new diagnostic rank the latter first. Thus this is not evidence that changing the metric alone fixes selection: sampling policy and root coverage also changed from the previous temperature-0.6 dev evaluation.

Primary post-RTN-effective dense scores, **2,432 identical roots per head/stage**:

| Head | Common weighted CE | Mean accepted draft prefix | Change vs stock | Families improved / 8 |
|---|---:|---:|---:|---:|
| Stock | 0.857187 | 2.903372 | — | — |
| LR 5e-6 decay, step 400 (previous winner) | 0.786081 | 2.937911 | +1.190% | 5 |
| **LR 1e-6 decay, final step 652** | **0.768754** | **3.004523** | **+3.484%** | **7** |
| LR 5e-6 flat, step 400 | 0.804203 | 2.928043 | +0.850% | 4 |

CE uses common depth weights `[1,.5,.25,.125]`. The mean prefix is the sum of **joint** survival probabilities, not the sum of marginal accuracies. The selected head's survival is **90.090%, 79.770%, 70.107%, 60.485%**, versus stock **88.898%, 77.714%, 66.776%, 56.949%**. Its prefix histogram `[L=0,1,2,3,4]` is `[241,251,235,234,1471]`; stock is `[270,272,266,239,1385]`.

The selected head improved seven families; the eighth declined by 0.015625 mean accepted draft tokens. Family deltas range from −0.015625 to +0.179688. Families have unequal context counts, and eight families do not establish generalization.

BF16-core means: stock **2.955181**, prior winner **2.975329**, LR 1e-6 final **3.039474**, flat **2.971217**. Both stages retain the existing frozen RTN-effective draft LM head. Dense reconstructed RTN weights are screening evidence, **not bitwise native INT4-kernel proof**. Aggregate depth CE, marginal agreements, survival counts, histograms, and family dispersion for both stages are in [`offline-summary.json`](offline-summary.json).

### Data, alignment, coverage

- Existing 54 ordered dev contexts → **38 admitted, 16 excluded for prompt budget**, across **8 dev families**. No truncation, test reuse, new training corpus, or optimizer.
- Native greedy capture generated **14,870 tokens**, 33,848 prompt tokens and 48,708 observed hidden rows. Finishes: 35 tool calls, three length caps. Generated tools were never executed.
- Of 48,642 possible aligned base rows: **33,810 boundary exclusions**, **114 terminal incomplete-horizon exclusions**, **14,718 eligible roots**, zero hidden/mask exclusions. Every admitted sequence had at least 64 eligible roots; sampling selected 64 per sequence (2,432 total, 16.524% of eligible roots).
- Root sampling is uniform without replacement, sorted, with seed 42 combined deterministically with the prompt ID. Identical roots/labels are used for every head and precision stage. Private reports retain root identities and token digests, not public trace content.
- Root `t` consumes response token `x[t+1]` with actual hidden `h[t]` and position `p[t]`, predicting `x[t+2]`. Four-label horizon required; the first verifier-generated token is not counted as a draft label. Each recursive branch sees only its own nodes and the base prefix through `t`. No synthetic terminal hidden padding.
- Labels come from the exact quantized native verifier at **temperature 0**, not the existing temperature-0.6 sampled feature cache. They remain observed trajectories subject to backend repeat variation, not a deterministic universal oracle.

## Environment and real E2E path

Intel Arc Pro B70 32GB on `inference-host`, 275 W power cap; model `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, revision `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`.

Pinned image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`; installed vLLM reports `0.27.2rc1.dev77+gac7509e2b.xpu`. Fresh official-source research and installed source confirmed the token shift, unchanged first-pass positions, recursive +1 position, normalized hidden path, and contiguous greedy rejection semantics. Source URLs and their effects are in the protocol.

Capture/native cells use the existing temporary launcher: GPTQ target INT4 g128, existing draft RTN INT4 g128, FP16 compute, FP8 KV, native MTP4, C1, configured context 212,992, balanced mode, graph sizes `[1,2,4,8]`, uniform-prefill guard, loopback API. Prefix cache is enabled but unique salts impose cold requests. Per-context thinking and four-tool template preserved. Greedy seed 42, natural termination, response cap 1,792, complete-sequence limit 2,048, minimum available response budget 128. Four synthetic warmups excluded per cell; capture-on warmups also register capture controls.

Completed public-boundary journey: idle host → temporary stock server → actual `/tokenize` and `/v1/chat/completions` greedy dev capture → stop server → evaluation-only CLI on real B70 features → saved-head selection frozen → native overlay load through the existing loader/RTN path → capture-off matched dev ABBA → cleanup and persistent-launcher verification. Capture-on timings were not used as throughput evidence.

## Verification

Pinned CPU-only container, integrated scorer plus existing trainer, capture, and merger suites:

```bash
docker run --rm --network none --user 1000:1000 \
  -e PYTHONDONTWRITEBYTECODE=1 -v "$ROOT/source:/work:ro" -w /work \
  --entrypoint python "$IMAGE" -m pytest -q -p no:cacheprovider \
  tests/test_qwen38_mtp_acceptance.py tests/test_qwen38_train_mtp.py \
  tests/test_qwen38_mtp_native_capture.py tests/test_qwen38_mtp_merge_captures.py
```

**115 passed, 2 skipped** in 48.14s. The two checks were then rerun as container root with `B70_MTP_TEST_VLLM_ROOT=/opt/venv/lib/python3.12/site-packages/vllm`: **2 passed** in 2.04s (`test_root_worker_preserves_host_bind_mount_owner`, `test_actual_pinned_source_seam_and_api_contract`). Thus all 117 collected tests passed across the two invocations. CPU-only Sysman initialization warning was expected; real XPU scoring succeeded separately. The analysis script also passed a synthetic paired-zero-delta, empty-stable-subset, and exclusive-output check.

Read-only correctness review found no blockers in root eligibility, native alignment, joint-prefix math, branch-cache isolation, fixed roots/labels, precision-stage routing, or no-optimizer behavior. Common CE is derived from the scorer's per-depth values in the aggregate analysis; its tie-break rule was declared in the protocol before scoring (no tie occurred), not chosen post hoc.

One initial SSH preflight and one background test launch were rejected by fish's Bash-heredoc syntax; both failed before the intended commands ran. Explicit Bash wrappers corrected them. No benchmark observations were dropped. The Prime Lab preview/patch UI was unavailable (missing socket); shell/file tools were used directly.

## Exact private commands and artifact locations

Private root: `/home/mike/b70-evals/20260918-mtp-acceptance-dev/` on `inference-host`.

- `commands/capture.sh`: exact launcher preparation, readiness, warmups, greedy API capture, cleanup. Run from the lead host via `bash`; the script SSHs to inference-host. Existing output paths refuse reuse.
- `commands/score.sh`: exact XPU CLI invocation of `scripts/experiments/qwen38_mtp_acceptance.py`, all three saved-head paths, `--roots 64 --max-length 2048 --logits-chunk 64`; stock automatic, both stages.
- `commands/tests.sh`, `integrated-tests.log`: integrated CPU verification.
- `commands/abba.sh`, `commands/analyze_abba.py`: frozen selected-head capture-off native comparison and paired family-bootstrap aggregation.
- `protocol.md`, `selection.json`, `installed-alignment.txt`: run protocol, selected saved head, installed-source checks.
- `greedy-capture/`, `greedy-warmup/`, `greedy-dev/`: launch arguments, logs, raw capture and exact API outputs.
- `private-prompts/{dev,admitted-dev}.jsonl`: original ordered dev file and admission-frozen subset; never committed.
- `offline-prefix.json`, `offline-score.log`: full private evaluation report and runtime log. Only sanitized aggregates are published here.

No ML dependencies were installed into Pi. No cloud resources, persistent launcher edits, test-set access, training, cache deletion, or production restart. Final cleanup verified **no running Docker containers or vLLM/scorer processes**, **275 W cap**, and **81 GiB free**. Persistent launcher SHA256 remained `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`. Private `dev-{A1,B1,B2,A2}/` launch/log artifacts and corresponding `*-output/request-*/` raw requests, token IDs, responses, and measurements are retained on inference-host, not in Git.
