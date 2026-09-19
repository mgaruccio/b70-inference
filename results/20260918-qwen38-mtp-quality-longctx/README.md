# Frozen MTP head: coding quality and cold long-context measurements

**Completed 2026-09-18/19. Development measurements; retain stock, no promotion.** Full HumanEval+ coding checks and both cold-context sweeps completed. The candidate's coding score was slightly lower and long-context performance was mixed, not a consistent improvement. See [protocol](protocol.md) for the predeclared workload and disclosed setup corrections.

## Executable coding quality — completed

All four cells generated and evaluated all164 HumanEval+ v0.1.10 tasks, in stock/candidate/candidate/stock order. Identical prompts, prompt token IDs and sampling settings were checked across cells. One greedy, non-thinking completion/task, seed42, 2,048-output-token cap, no skipped or truncated input prompts. Official full base+extra tests ran in isolated CPU-only, offline containers.

| Cell | Head | Base pass@1 | Base+extra pass@1 |
|---|---|---:|---:|
| A1 | Stock | 159/164 (96.95%) | 151/164 (92.07%) |
| B1 | Candidate | 160/164 (97.56%) | 151/164 (92.07%) |
| B2 | Candidate | 159/164 (96.95%) | 150/164 (91.46%) |
| A2 | Stock | 161/164 (98.17%) | 152/164 (92.68%) |

Mean per-cell base+extra scores: **stock92.38%, candidate91.77% (−0.61 percentage points)**. These are repeated pass@1 measurements, not pass@2 or328 independent tasks/head. They do not establish quality equivalence or a statistically resolved quality regression.

- **Regression flag:** HumanEval/93 passed both stock cells but failed one candidate cell. No task passed both stock cells and failed both candidate cells.
- Within-head base+extra pass/fail disagreements: stock HumanEval/134; candidate HumanEval/93. No task passed both candidate cells while failing a stock cell.
- Raw answers were identical on151/164 tasks across stock repeats and152/164 across candidate repeats. Greedy decoding was not perfectly repeatable.
- Output caps were reached on A1:116,132; B1:116; B2:116; A2:none. These samples remain in the reported scores; none was retried or excluded.
- Full aggregate/per-task results: [quality-summary.json](quality-summary.json). No private engineering traces or generated tool calls were executed.

This public benchmark may overlap model pretraining. It is a coding regression check, not proof of unseen-task generalization, broad reasoning, tool-use correctness, or long-context semantic correctness.

## Evaluator validation and corrections

The official Docker image's installed package was0.4.0.dev2 despite its release tag. Its canonical set scored163/164 because the HumanEval/32 special oracle skipped success/progress bookkeeping. The current official upstream implementation already fixes that bug. The evaluator was rebuilt from the original immutable image with **complete official source pinned to `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`**, package0.4.0.dev44; see [Dockerfile](commands/Dockerfile.evalplus), [image ID](evaluator-image.txt) and [runtime identity](evaluator-runtime.txt).

The repaired evaluator scored **164/164 canonical solutions on both base and base+extra tests** before model generation. Earlier failures are retained remotely: an incomplete one-task canary rejected by EvalPlus, the upstream oracle bug, an archive build without SCM metadata, and a wrapper output-filename mismatch after upgrading. These are evaluator setup failures, not model failures. Tests, time limits and pass expectations were not weakened. Dataset and prompt-record hashes remained unchanged.

## Reproduction and evidence

Remote root: `/home/mike/b70-evals/20260918-mtp-quality-longctx/` on `inference-host`. Runtime/data stay outside Pi. Frozen candidate and vLLM configuration are in [protocol.md](protocol.md).

Commands run on inference-host:

```sh
R=/home/mike/b70-evals/20260918-mtp-quality-longctx
bash "$R/commands/evaluate.sh" "$R/data/canonical-full.jsonl" "$R/eval-canary-fixed.json"
bash "$R/commands/run.sh" quality
python3 "$R/commands/summarize_quality.py" "$R"
bash "$R/commands/run.sh" long
```

These commands require new output paths; do not overwrite completed evidence. The committed runner includes the disclosed `/server_info` fallback; the original runner is retained remotely as `commands/run-before-server-info-fallback.sh`.

Raw evidence: `quality-{A1,B1,B2,A2}-output/request-*/{request,response,measurement,tokenize}.json`, per-cell launch configurations/server logs, sanitized samples and `quality-*-evaluation.json`. Failed initial long-context evidence: `long-A-output/{run,summary}.json`; the resumed comparison writes `long-verified-{A,B}-output/`.

Supplementary checks:12 existing long-context harness tests passed in the pinned runtime; shell/Python helper syntax passed; a synthetic164-task four-cell test verified the new quality aggregator's paired flags and repeat counts; the official sanitizer entry-point smoke passed. These checks supplement, not replace, the656 real generated samples and their executable evaluations.

## Cold-context results — completed

Both heads completed all eight lengths, **48/48 valid measured requests per head**, plus eight excluded warmups. All 56 tokenized requests/head matched exactly across heads, including sampling settings. Every request used its exact declared prompt length and generated128 output tokens. No measured request failed, timed out or OOMed. The maximum tested input was212,000 tokens (212,128 including output), not a proven capacity ceiling.

Values are **median [inclusive IQR]** over six requests per point. Decode is client-side `(128−1)/(stream end−first nonempty chunk)`, not per-token ITL; speculative bursts can contain multiple tokens. E2E is output tokens divided by total request time. Relative changes compare medians, not paired confidence intervals.

| Prompt tokens | Stock TTFT (s) | Candidate TTFT (s) | Stock decode (tok/s) | Candidate decode (tok/s) | Decode change | E2E rate change |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.289 [0.00036] | 0.289 [0.002] | 69.32 [2.32] | 73.20 [5.56] | +5.60% | +4.758% |
| 8,192 | 4.190 [0.008] | 4.185 [0.015] | 61.97 [1.01] | 61.31 [5.14] | −1.05% | −0.161% |
| 16,384 | 9.121 [0.007] | 9.109 [0.009] | 62.28 [3.42] | 61.63 [4.94] | −1.04% | −0.056% |
| 32,768 | 21.123 [0.002] | 21.115 [0.003] | 62.75 [3.28] | 61.29 [0.18] | −2.33% | −0.174% |
| 65,536 | 53.865 [0.006] | 53.859 [0.006] | 54.39 [3.38] | 53.17 [3.23] | −2.25% | −0.072% |
| 120,000 | 134.347 [0.015] | 134.356 [0.016] | 56.07 [2.91] | 56.08 [1.66] | +0.00% | −0.017% |
| 160,000 | 213.934 [0.013] | 213.989 [0.011] | 47.71 [2.94] | 46.42 [1.80] | −2.69% | −0.067% |
| 212,000 | 343.800 [0.046] | 343.923 [0.011] | 36.52 [8.13] | 37.85 [9.82] | +3.63% | +0.012% |

At212k, total median request time was347.314s stock versus347.274s candidate. Cold prefill dominates: a modest decode-rate change barely changes end-to-end time. Input-tokens/TTFT is a **prefill proxy**, not directly measured engine prefill throughput; at212k it was616.637 versus616.418 input tok/s. Full medians/IQRs for TTFT, proxy prefill, decode, E2E rate and total time, plus all96 individual measured rows, are in [long-summary.json](long-summary.json).

### Native speculative acceptance

All96 measured requests had valid before/after counters: exactly one completed request and128 generated tokens, four proposed tokens/draft, nonnegative deltas, monotone per-position acceptance, and accepted-position totals matching total acceptance. These are medians [IQR] of **accepted tokens per draft block (maximum4)**, not teacher-forced top-1 accuracy.

| Prompt tokens | Stock accepted/draft | Candidate accepted/draft |
|---:|---:|---:|
| 512 | 1.826 [0.135] | 1.989 [0.154] |
| 8,192 | 1.698 [0.048] | 1.670 [0.233] |
| 16,384 | 1.813 [0.141] | 1.796 [0.277] |
| 32,768 | 2.047 [0.136] | 1.954 [0.051] |
| 65,536 | 1.965 [0.187] | 1.944 [0.230] |
| 120,000 | 2.606 [0.167] | 2.578 [0.101] |
| 160,000 | 2.432 [0.165] | 2.342 [0.174] |
| 212,000 | 2.103 [0.660] | 2.205 [0.749] |

### Configuration, capacity and limitations

- Cache disablement was checked against each temporary launcher's hash/flag and vLLM initialization log. `/server_info` was404, so the harness records its documented operator-assertion fallback. Additionally, **every retained metrics snapshot reported `enable_prefix_caching="False"`**. This is runtime evidence, not successful `/server_info` verification.
- Both server logs reported16.8GiB model-loading allocation,8.23GiB available KV cache memory and226,397 KV-cache tokens. These are initialization/capacity figures, **not a measured transient GPU-memory peak**. Configured context remained212,992.
- Stock ran2026-09-18 16:57:30–18:30:48 UTC; candidate ran2026-09-19 02:03:08–03:36:29 UTC. The **7h32m inter-cell gap**, fixed order and uncontrolled thermal/time effects limit causal interpretation. Six within-server repetitions do not replace interleaved server-level repeats. The large212k decode IQR also counsels against overinterpreting its median improvement.
- Candidate preparation initially failed before any measurements because the guard path under `/tmp` had become a directory. The identical, hash-verified guard retained with stock was staged under the experiment directory; only the missing candidate cell was resumed (`run.sh long-candidate`). No stock measurements or failed performance samples were discarded/repeated.
- These deterministic synthetic engineering prompts test performance/capacity, not semantic recall or answer correctness at long context. Combined with the slightly lower coding score, this round **does not support promoting the candidate** or extrapolating the earlier small engineering-holdout gain to all workloads.

Aggregation command: `python3 "$R/commands/summarize_long.py" "$R"`. It validated all actual lengths/output counts, matched all requests, verified cache metrics and classified speculative counter validity before producing the aggregate.

## Final verification and disposition

All owned inference/evaluation containers stopped; `docker ps -q` was empty. Power cap275W,79GiB disk free. Persistent launcher SHA256 remained `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`; candidate SHA256 remained `1af9142095c1d387847c83f27bdc330df8d34f81d0d73683c593ef5768babde5`. Raw evidence and completed checkpoints were retained on inference-host. No training, checkpoint reselection, production restart or promotion occurred.

**Decision: retain stock and stop this round.** Full repository standard-publishable coverage remains incomplete (no MBPP+/GSM8K/IFEval, broad tool-use or long-context semantic evaluation, interleaved long-context server repeats, or transient-memory/energy measurement). These omissions are limits, not implicit authorization for another experiment.
