# Next session: acceptance-aligned Qwen MTP validation

Prepared after the million-position experiment, results commit `129498f7`; cache-merger implementation `c3212a54`. **Executed 2026-09-18:** see the [acceptance-aligned dev result](../results/20260918-qwen38-mtp-acceptance-dev/README.md). LR 1e-6/final step 652 won the new greedy diagnostic; native acceptance trended +3.51% and decode +2.35%, but both confidence intervals include zero. Retain stock; no retraining or promotion. The original bounded plan is preserved below.

**Fresh holdout follow-up:** the frozen step-652 head subsequently achieved acceptance **+5.17%** and decode **+4.08%**, with positive family-bootstrap intervals on **19 contexts / eight unused families** ([report](../results/20260918-qwen38-mtp-fresh-confirmation/README.md)). The user explicitly approved this smaller exploratory sample. All eight families are now consumed, including unselected contexts. Structural tool-call checks passed; functional quality was not measured in that round. No promotion or new training; stock remains configured.

**Coding quality + cold-context follow-up, completed 2026-09-18/19:** [full report](../results/20260918-qwen38-mtp-quality-longctx/README.md). Full HumanEval+ ABBA: stock151/164 and152/164, candidate151/164 and150/164 (mean−0.61 percentage points). One candidate-repeat regression flag; no task passed both stock cells and failed both candidate cells. Both heads completed all eight cold lengths through212,000 prompt tokens plus128 output tokens,48 valid measured requests/head. Long-context decode changes were mixed (−2.69% to+3.63% across8k–212k); E2E changes there were negligible (−0.174% to+0.012%). The sequential sweep had a7h32m inter-cell gap, not interleaved server repeats. All temporary containers stopped; launcher/head hashes unchanged. **Retain stock; this round does not support promotion.** Public coding quality is now measured, but tool-use and long-context semantic correctness remain unmeasured. No further training or experiments are implied.

## Original goal and next question

> Improve speculative acceptance rate and resulting tokens/sec for the exact Qwen 27B quant + inference backend used locally.

The selected deployment target is the existing B70 **vLLM/XPU GPTQ model**, not a new llama.cpp/GGUF experiment. Keep the native architecture, exact quant/backend, tokenizer/template, and runtime draft RTN packing.

**Next question:** Does checkpoint ranking by cross-entropy miss a better accepted-prefix/runtime-acceptance checkpoint, or is the learned improvement failing to generalize/transfer to serving?

Do not generate another large corpus or start another training sweep first. Reuse the cached data and saved checkpoints to answer this bounded question.

## What actually happened

- Built and exercised native quantized-verifier capture, recursive MTP4 training, native BF16 overlay loading, and existing RTN packing.
- Expanded to **1,043,893 useful supervised positions**, 2,607 distinct full trajectories, 913 prompt contexts, 126 training families. Exact duplicate trajectories were removed; repeated epochs were not counted as new data. Distinct trajectories still share prefixes.
- Trained three settings, two epochs / 652 updates each. Selected **LR 5e-6, depth weights `[1,.5,.25,.125]`, step 400** using a common post-RTN dev CE criterion. Offline objective improved **7.92%**.
- Fresh-test ABBA: 63 contexts / 10 families, 252 completed requests. Acceptance/pass **+0.010%**, decode throughput **−0.019%**, end-to-end throughput **+0.330%**. All family-bootstrap intervals include zero. **No demonstrated serving gain; no promotion.**
- Dev top-token agreement did not track the loss gain: depth 1 **83.895→84.330%**; depth 2 **81.250→80.592%**. Depths 2–4 had only **304 sampled positions each**. These are clues, not a proven root cause.
- Significant native repeat variability remains: stock output tokens matched on only 29/63 test prompts across repeats. Do not attribute every changed output or small timing delta to tuning.

Read first:
- [Million-position report](../results/20260915-qwen38-million-position-mtp/README.md), `summary.json`, and `analyze_abba.py` beside it.
- [Earlier 163k-position attempt](../results/20260915-qwen38-native-quant-aware-trace-tune/README.md).
- `AGENTS.md` and `BENCHMARKING_STANDARDS.md`.

## Critical distinction: labels, policy, and acceptance

**Existing training/dev trajectories were generated at temperature 0.6; serving benchmarks used greedy temperature 0.** Comparing draft argmax against a sampled continuation measures agreement with that observed trajectory, not necessarily agreement with the greedy verifier.

Keep these separate:
1. CE and token agreement against existing sampled trajectories.
2. Joint accepted-prefix agreement against observed **greedy-verifier** trajectories.
3. Actual native serving acceptance and throughput, under an explicitly matched sampling policy.

For the greedy diagnostic, generate a **small development-only greedy capture** from the existing dev prompts using the same native runtime. This is not a new large training corpus. Alternatively, use an exact-runtime verifier-label export if already available and verified; do not substitute approximate BF16/dense logits and call them deployment truth. Greedy backend numerics can still vary across runs, so even an observed greedy trace is not a universal deterministic oracle.

For stochastic deployment, sampled-label argmax agreement is not a substitute for the sampler's acceptance rule. Treat that as a separate native serving measurement, not a renamed greedy metric.

## Bounded work sequence

### 1. Freeze a small diagnostic protocol

Before looking at new scores, declare:
- Same ordered dev prompt set, token limits, tools/template/thinking settings, cold-prefix policy, and sampling policy for every candidate.
- Root eligibility and deterministic root selection; identical roots/labels for all offline candidates.
- Primary offline diagnostic: mean jointly accepted draft-prefix length. Also report depth-wise survival, ordinary CE/top-token agreement, eligible-root coverage, and per-family dispersion.
- Primary runtime metric: native accepted draft tokens/speculative pass; supporting decode rate, end-to-end latency, TTFT, and output variability.
- Development tier only. No production changes, cloud spend, architecture change, or new training in this phase.

Initial candidate shortlist (already saved; no optimizer needed):
1. Stock.
2. `training-lr5e6_decay/tuned-mtp.step0400.safetensors` — previous CE winner.
3. `training-lr1e6_decay/tuned-mtp.safetensors` — final step 652, that run's common-CE winner.
4. `training-lr5e6_flat/tuned-mtp.step0400.safetensors` — flat-depth run's common-CE winner.

Do not expand to all checkpoints unless the shortlist reveals a useful ranking difference.

### 2. Add an evaluation-only joint-prefix diagnostic

Reuse `qwen38_train_mtp.py` helpers rather than building another trainer/harness. Inspect `sequence_depths`, `sample_roots`, `depth_losses`, `evaluate`, `evaluate_export`, and native loading/quantization before editing. If an evaluation-only CLI is needed, add the smallest one; **no such new flag is assumed to exist today**.

For each eligible root `t`, use the existing native alignment:
- Depth 1: embedding `x[t+1]`, observed target hidden `h[t]`, position `p[t]`, predict `x[t+2]`.
- Depth `d`: embedding `x[t+d]`, previous draft hidden, position `p[t]+d−1`, predict `x[t+d+1]`.
- Each branch attends the base prefix through `t` and only its own appended nodes. No future-base or sibling leakage.
- Respect actual observed hidden coverage; no terminal padding or synthetic zero hidden rows.

For aligned labels define `c_d = 1[argmax(draft_logits_d) == label_d]`, `J_d = product(c_1...c_d)`, and accepted prefix length `L = sum(J_1...J_4)`.

**Do not sum independent per-depth accuracies:** after the first disagreement, later predictions contribute zero accepted tokens. Teacher-forced later steps are acceptable for this diagnostic only because their contributions are masked by earlier failures.

Use response-relevant roots with a full valid four-label horizon, respecting the actual first-draft boundary and native position convention. Report roots excluded near termination. Increase coverage beyond the previous eight roots/sequence; prefer all eligible roots in bounded batches if affordable, otherwise predeclare a larger fixed sample. Avoid giant root×sequence attention allocations.

Evaluate saved BF16 export and runtime-RTN-effective weights; make post-RTN results primary. Dense RTN-effective evaluation is **not bitwise proof of native INT4-kernel parity**. An all-roots offline mean also is not identical to runtime verification scheduling; serving remains decisive.

### 3. Verify the metric before using its ranking

Targeted synthetic tests:
- First mismatch at each depth and full acceptance produce prefix lengths 0,1,2,3,4.
- Later correct predictions after an earlier mismatch do not count.
- Token/position shifts, prompt boundary, short/terminal sequences, and observed `T−1` hidden coverage are correct.
- Root labels/order match across checkpoints; future/sibling KV isolation is preserved.
- Export and post-RTN scoring are clearly distinguished; no optimizer or weight mutation occurs.

Then exercise the real path on a small greedy dev capture. If offline/native decisions materially disagree, diagnose alignment/cache/precision differences first. Do not dismiss a mismatch as quantization noise without evidence, or introduce another architecture to hide it.

### 4. Rank on dev, then measure native dev acceptance

Compare stock and the shortlisted saved heads on the same dev-only greedy diagnostic. Show whether CE, top-token agreement, and joint-prefix ranking disagree; report actual evaluated counts, not only percentages.

Run matched, repeated native dev serving cells for stock and the most informative one or two candidates. Reuse the existing temporary launcher and `--measure` client. Capture must be off for timed cells; preserve raw private per-request metrics and token IDs. Keep within-head repeats to estimate runtime variability.

Decision rules:
- Joint-prefix/native acceptance improves while CE ranking misses it: test acceptance-aligned selection before changing training.
- Offline prefix scores improve but native acceptance does not: investigate the scorer/runtime interface, numerical parity, or scheduling/distribution mismatch before retraining.
- Neither improves: the saved heads did not learn a useful draft-decision improvement under this diagnostic. Only then propose a bounded objective or quantization-aware-forward experiment, with a specific hypothesis.
- Changes remain within repeat variability: report uncertainty; do not promote or choose a favorable post-hoc subset as the primary result.

### 5. Fresh final evaluation only after selection is frozen

**Both previous test sets are consumed.** Do not use either test set, test outputs, or identical-output subsets to select another checkpoint. Do not turn an existing training family into a nominal fresh holdout.

If dev results justify another candidate, obtain new independent test families and freeze the final protocol before evaluating them. If no fresh holdout is available, stop at a clearly labeled development result rather than claim generalization.

## Execution, tests, and cleanup

All ML runtimes, feature tensors, generated prompts, and heads stay on `inference-host`, outside Pi/Git. Existing user permission covers selected local engineering traces after filtering; it does not authorize publishing them or sending them to external model providers. Generated tool calls are never executed.

Before implementation next session, satisfy the repository's fresh external-research and defined-E2E gates. Relevant pinned sources:
- https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/v1/spec_decode/step3p5.py
- https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/model_executor/models/qwen3_5_mtp.py
- https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/v1/worker/gpu_model_runner.py

Check installed pinned code too; do not import newer-version position/normalization conventions.

Real E2E acceptance process: idle B70 → temporary exact-runtime stock server → small greedy **dev** capture through `/tokenize` and `/v1/chat/completions` → stop verifier → frozen offline prefix scoring → load candidate through the existing native overlay → capture-off matched dev serving comparison. Expected: valid aligned features, finite/export-consistent scores, explicit terminal-token accounting, exact request accounting, and measured—not inferred—acceptance. Preserve commands, configuration, failures, raw private outputs, and aggregate results. Stop temporary servers and verify the persistent launcher hash afterward.

Run targeted tests in the pinned CPU-only image, then the real XPU/API journey. Existing tests include `tests/test_qwen38_train_mtp.py`, `tests/test_qwen38_mtp_native_capture.py`, and `tests/test_qwen38_mtp_merge_captures.py`; add only focused tests for the new scorer. Unit tests alone are not E2E evidence.

## Resume locations / invariants

Repository: `/home/mike/code/b70-inference`.

Private host root:
`/home/mike/b70-evals/20260915-mtp-scale-attempt/`

- `private-prompts/dev.jsonl` — 54 dev contexts; 38 were admitted by the previous 2,048-token capture budget. Fix admission/order consistently for the new diagnostic.
- `fresh-dev/heldout/` — existing **temperature-0.6** dev features, not greedy ground truth.
- `dataset/train/` — 1.04M-position cache; do not regenerate it for this diagnostic.
- `training-{lr5e6_decay,lr1e6_decay,lr5e6_flat}/` — saved checkpoints, per-depth metrics, logs.
- `selection.json` — previous common-CE ranking.
- `commands/b70-mtp-scale-attempt.sh`, `commands/b70-mtp-scale-heldout-abba.sh` — exact prior commands; these are not safe rerun commands without fresh output paths.
- `heldout-{A1,B1,B2,A2}*` — consumed test artifacts; inspection only for reporting, not reselection.

Pinned image:
`vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`

Model:
`/home/mike/inference/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`

Persistent launcher:
`/home/mike/inference/launchers/start-qwen38.sh`

Expected SHA256:
`63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`

Last verified: temporary containers stopped; power cap 275W; production launcher unchanged. Around 82GiB free before the final small benchmark. **Recheck host idleness, power, and free space next session; do not treat historical status as live.** Never delete private caches/heads or restart production implicitly.

Use background jobs for long runs. Completion notifications failed to reach the lead reliably in this session: several workers/jobs had finished while the lead incorrectly reported them pending. On an explicit status request, inspect actual task/host evidence; never infer “still running” from a missing notification. Do not add a new job manager or poll merely to wait.

Deliver next: a small dev-only metric/ranking report, a native acceptance comparison if justified, explicit uncertainties, and the resulting decision. Commit/push code and aggregate documentation; keep private data out of Git.
