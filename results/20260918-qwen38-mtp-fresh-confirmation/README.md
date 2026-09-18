# Fresh-family MTP holdout — positive exploratory result

**The frozen LR 1e-6 / final-step-652 head improved native acceptance and speed on a small, previously unused holdout.** Acceptance **+5.17%**, decode speed **+4.08%**, end-to-end speed **+3.41%**; all three paired family-bootstrap intervals exclude zero. **Exploratory only: eight families, 19 contexts, no functional quality score or production promotion. Stock remains configured.**

This follows the [acceptance-aligned development diagnostic](../20260918-qwen38-mtp-acceptance-dev/README.md). No checkpoint was selected using these test outputs, and no new training occurred. The [protocol](protocol.md) records the user-approved reduction from the initial 20-family feasibility minimum to the available eight fresh families, **before generation**.

## Native held-out result

Stock → candidate → candidate → stock (**ABBA**), **19 frozen contexts × four cells = 76 completed requests**, zero skipped timed requests. Four synthetic warmups per cell excluded; capture off. Only the saved native MTP head overlay differs.

| Pooled metric | Stock | Frozen candidate | Relative change | Family-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| Accepted draft tokens / speculative pass | 2.854116 | 3.001585 | **+5.167%** | **+1.500% to +8.455%** |
| Decode tokens/sec | 93.9713 | 97.8024 | **+4.077%** | **+1.330% to +6.550%** |
| End-to-end tokens/sec | 82.1671 | 84.9671 | **+3.408%** | **+1.255% to +5.695%** |
| Draft acceptance fraction | 71.3529% | 75.0396% | +3.6867 percentage points | — |
| Median request latency | 3.05739 s | 2.89244 s | −5.395% | — |
| Median TTFT | 0.467694 s | 0.467807 s | essentially unchanged | — |

Both candidate cells exceeded both stock cells on acceptance and decode:

| Cell | Accepted drafts/pass | Decode tok/s | Output tokens |
|---|---:|---:|---:|
| A1 stock | 2.906600 | 95.3597 | 6,257 |
| B1 candidate | 2.992415 | 97.5397 | 6,305 |
| B2 candidate | 3.010807 | 98.0667 | 6,300 |
| A2 stock | 2.804476 | 92.6608 | 6,441 |

Unconditional native depth survival, stock → candidate: **88.469→91.315%, 77.028→80.000%, 65.163→68.748%, 54.752→60.095%**.

Primary analysis includes every selected context and both repeats. Bootstrap resamples source-session families, retaining both stock/candidate repetitions together, 10,000 draws, seed 42. Decode is `sum(generated−1)/sum(server_decode_seconds)`; end-to-end is `sum(generated)/sum(client_wall_seconds)`. Accepted drafts/pass excludes the bonus token. Native speculative counters may include terminal drafts discarded by the API. Per-request completion counters exactly matched API token counts and prefix-cache hits were zero. All run summaries, pooled counts and intervals are in [`native-summary.json`](native-summary.json).

### Output variability and limited quality checks

- Exact token matches: **12/19** for stock repeats, **12/19** for candidate repeats; cross-head matches **12/19** and **13/19**.
- **Nine contexts** matched in all four cells. This post-hoc subset showed acceptance +1.26% and decode +1.31%, with intervals including zero. It is a diagnostic only, not a replacement for the all-context primary result.
- Generated tool calls: **54 stock, 58 candidate**. All **112** passed tool-name, strict JSON argument parsing and supplied JSON-schema checks; zero structural failures or assistant-role errors. No generated tools were executed.
- Length-capped responses: **4/38 stock, 2/38 candidate**; the other responses ended with tool calls. These counts and all structural checks are in [`output-checks.json`](output-checks.json).
- Structural validity and token-output similarity are **not functional task success or semantic quality equivalence**. No ground-truth task execution or external model judge was used. The pinned vLLM docs explicitly warn that floating-point/batch numerics can vary across repeats.

## Holdout freshness and admission

No unused families remained in the previously filtered pools. Outcome-blind acquisition used the existing strict local engineering-trace extractor and actual 24-hour inactivity cutoff:

1. Loaded 965 eligible session files from selected project/related worktree roots; skipped 45 recent and 11 invalid files. Existing lineage resolution keeps forks with their originating family; missing/cyclic lineage is rejected.
2. Produced 2,015 normalized candidate contexts. Excluded **1,951** by the union of **157 prior families / 1,428 assigned records** across both prior train/dev/test collections, also checking IDs and canonical normalized-message hashes. Previously skipped-long assigned data was excluded too. The recent dev data is already within that union.
3. Remaining: **64 contexts / eight fresh families**. The user explicitly approved an exploratory small-sample run rather than waiting for more families. No performance outputs existed at this point.
4. Offline Gitleaks 8.30.1 found and excluded **one context**. Existing `filter-scan` validation/deduplication retained **63 contexts / eight families**, with zero prior family/context overlap. Scanner clearance is not proof that every sensitive datum is absent. Its intermediate train/dev/test filenames were pooled solely for this new test-only acquisition; none of these families had been used for training or selection, and none were trained here.
5. Exact native `/tokenize` admission found **49 eligible contexts**, with 14 excluded for the frozen prompt budget. Deterministic family/context ordering and the predeclared four-context-per-family cap selected **19 contexts across all eight families**; 30 otherwise-eligible contexts were left unselected. Membership, order, token counts and per-request budgets were saved **before any test completions**.

Counts are retained in [`preparation-summary.json`](preparation-summary.json), [`scan-summary.json`](scan-summary.json), and [`admission-summary.json`](admission-summary.json). Raw records, family identifiers, source membership, scanner findings and token IDs remain private. **All eight evaluated families are now consumed**, including their unselected contexts; they must not be relabeled as a future independent holdout. Family separation and exact-context deduplication do not guarantee semantic independence between related engineering tasks.

## Runtime, verification, and cleanup

- Intel Arc Pro B70 32GB on `inference-host`, 275W cap. Model `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, revision `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`.
- Pinned image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`; native MTP4/C1, GPTQ target INT4 g128, existing draft RTN INT4 g128, FP16 compute, FP8 KV, 212,992 configured context.
- Same temporary reference launcher: balanced mode, graph sizes `[1,2,4,8]`, uniform-prefill guard, loopback API. Existing prefix cache with unique salts for cold requests; per-context thinking and four-tool profile preserved.
- Greedy temperature 0, seed 42, natural termination, response cap 1,792, sequence limit 2,048, minimum remaining response budget 128. No prompt truncation. Same sampling, budgets, normalized requests and rendered token IDs verified across all **76** requests.
- Candidate SHA256 stayed `1af9142095c1d387847c83f27bdc330df8d34f81d0d73683c593ef5768babde5`; both candidate launch configs recorded it. All matched runtime/patch settings and capture-off mode were checked ([validation](native-validation.json)).
- The existing inference/capture/trainer path had passed 117 targeted checks in the preceding diagnostic; it was not modified here. The new structural checker passed **eight synthetic cases** in the same pinned CPU-only image, then processed the actual API outputs. Script syntax and result accounting checks passed. The real E2E journey was tokenizer admission → frozen membership → stock/candidate/candidate/stock API generation → counter/output/config validation → cleanup.
- One post-scan summary parser initially assumed the existing CLI emitted JSON instead of `key=value` text. The scan/filter had already succeeded; the parser was corrected and the summary rebuilt from those same artifacts. No model response or benchmark observation was discarded or rerun. The initial official-docs endpoint was rate-limited; fresh pinned GitHub documentation was accessible.
- Final state: **no running containers or vLLM/scorer jobs**, **275W**, **81GiB free**. Persistent launcher SHA256 unchanged: `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`. No cloud use, ML installs into Pi, training, old-cache deletion, production restart or promotion.

## Exact commands and private evidence

Lead private staging: `/home/mike/b70-mtp-private-confirm-20260918/`.

```bash
python3 /home/mike/b70-mtp-private-confirm-20260918/prepare.py
bash /home/mike/b70-mtp-private-confirm-20260918/scan.sh
bash /home/mike/b70-mtp-private-confirm-20260918/run.sh
```

The last command SSHs to inference-host; all inference dependencies stay in the pinned Docker image. Existing output paths are exclusive: these are the executed commands, not safe overwrite/rerun instructions.

Host evidence: `/home/mike/b70-evals/20260918-mtp-fresh-confirmation/`:

- `commands/run.sh`, `commands/admit.py`, `protocol.md`, `selection.json`: exact native preparation/launch/warmup/generation/analysis and frozen-head selection.
- `private-prompts/fresh-candidates.jsonl`, `private-prompts/admitted-test.jsonl`, `private-prompts/prior-membership.json`, `admission-private.json`: private inputs, exclusions, frozen order and token budgets.
- `test-{A1,B1,B2,A2}/` and matching `*-output/request-*/`: launch configurations, server logs, exact raw requests/responses, token IDs and native per-request metric snapshots. `*-warmup/` outputs are retained but excluded.
- `commands/analyze_abba.py`, `commands/check_outputs.py`, `commands/test_output_checks.py`: aggregate analysis and structural-validation source, also published alongside this report.
- `native-summary.json`, `output-checks.json`, `native-validation.json`: sanitized numerical outcomes and validation.

Synthetic checker command on inference-host:

```bash
docker run --rm --network none --user 1000:1000 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$ROOT/commands:/work:ro" -w /work --entrypoint python "$IMAGE" \
  test_output_checks.py
```

## Decision

**Positive small-holdout evidence for the frozen head, not a broad deployment guarantee.** Acceptance and decode meet the predeclared positive-interval criterion on these eight fresh families. The user-approved sample reduction, uneven family sizes, only two repetitions per head, output variability, and missing functional quality checks limit the conclusion. The family bootstrap does not estimate all temporal/hardware variation; stock A1/A2 variability remains visible.

Keep this head as the sole candidate for any future deployment decision. Do not retrain or reselect using this consumed test set. **Stock remains configured:** promotion was not authorized and would still require an explicit quality/capacity decision. This is not standard-publishable/community-comparable: no full BetterBench, concurrency/long-context sweep, continuous thermal trace or functional quality suite.
