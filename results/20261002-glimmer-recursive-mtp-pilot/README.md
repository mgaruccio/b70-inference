# Frozen Glimmer recursive MTP pilot — 2026-10-02

**Completed development pilot, not a production benchmark.** Corrected source: `adbb09a264407d8986771e931a6a64f90fccbd8a`. The real model CLI completed all 144 comparisons; 102 targeted tests passed in the same cloud runtime. Both cloud instances were terminated and Shadeform returned `instances: []` at `2026-10-02T19:08:54.543245+00:00`.

## Result

**No useful speculative acceleration in this small run.** Best aggregate acceptance was **0.0161 draft tokens per verification pass**, versus the proposed **≥3** threshold. Best individual prompt/depth result was 0.0678. All three heads were slower than target-only decoding. This establishes a functioning experiment path, **not** that recursive MTP cannot work: training was only 100 updates × 4 roots, on 24 short sequences (1,665 tokens; 1,449 eligible roots), roughly 3.4–3.6 seconds of optimization per head. Neither the conventional nor recursive head learned useful first-step prediction here.

All variants trained depth 8 and evaluated prefixes 1/2/4/8. The fixed baseline uses eight independent transition blocks, not an optimized EAGLE implementation or separately trained depth-specific ceilings. All use rank 64, seed 20261002, the same ordered training roots and hyperparameters, frozen target/embedding/output weights, and last-update checkpoints. Parameter budgets intentionally differ: **10,276,864 fixed vs 1,284,608 shared**. `shared-state` adds state supervision with coefficient 0.2; teacher KL weight is zero.

| Variant | Draft depth | Accepted drafts/pass¹ | Median paired decode speed² |
|---|---:|---:|---:|
| fixed-ce | 1 | 0.016129 | 0.9135× |
| fixed-ce | 2 | 0.016129 | 0.8773× |
| fixed-ce | 4 | 0.016129 | 0.8146× |
| fixed-ce | 8 | 0.016129 | 0.7129× |
| shared-ce | 1 | 0.006658 | 0.9091× |
| shared-ce | 2 | 0.006658 | 0.8718× |
| shared-ce | 4 | 0.006658 | 0.8083× |
| shared-ce | 8 | 0.005319 | 0.7123× |
| shared-state | 1 | 0.006658 | 0.9061× |
| shared-state | 2 | 0.006658 | 0.8706× |
| shared-state | 4 | 0.006658 | 0.8084× |
| shared-state | 8 | 0.005319 | 0.7122× |

¹ Total accepted **drafts** / total verification passes, across 12 prompts. Excludes anchor, correction and bonus tokens. ² Median per-prompt candidate decode tokens/sec divided by paired baseline decode tokens/sec; includes diverged pairs and is therefore diagnostic, not a lossless speed claim. Ratios below one mean slower. Each prompt has one repeat; timing uncertainty is not estimated.

Target-only median decode throughput was **21.123 tokens/sec**. Depth-1 candidates were approximately 19.1–19.3 tokens/sec and depth-8 approximately 15.0–15.1. Draft generation cost was about 1.9 ms/pass at depth 1 and 14.0 ms/pass at depth 8. Candidate target calls/generated token ranged 0.9375–1.0 versus baseline 1.0. Maximum allocated GPU memory was 59.67 GB fixed and 59.63 GB shared (decimal GB); these totals include the frozen 30B target, not just head overhead.

## State stability

Mean cosine similarity to actual target states, sample-weighted across separate, **untimed candidate-chosen-path** diagnostics from the depth-8 runs:

| Variant | Step 1 | Step 2 | Step 4 | Step 8 |
|---|---:|---:|---:|---:|
| fixed-ce | 0.4970 | 0.4195 | 0.3541 | 0.3162 |
| shared-ce | 0.5140 | 0.4260 | 0.3256 | 0.2471 |
| shared-state | 0.5168 | 0.4333 | 0.3411 | 0.2709 |

State supervision modestly improved later-step similarity versus shared CE, but not acceptance or speed. First-step quality is already inadequate; drift alone does not explain failure. Deeper diagnostic states follow the candidate's chosen sequence, including rejected proposals; they are not an acceptance-conditioned accuracy curve. Counts and normalized MSE are in `r1/summary.json`; full per-root records are in the compressed raw evaluation.

## Fidelity qualification

The explicitly approved `--record-divergence` diagnostic mode retained the actual target verifier, greedy acceptance/rejection, and cache rollback; it did not relax acceptance. **102/144 pairs matched target-only output exactly (70.83%); 42 differed.** Median first differing token was zero-based position 50; all baseline and candidate outputs had 64 tokens. Every prompt's 12 target-only runs produced one identical sequence, so baseline repeatability did not establish block-verifier equivalence.

Strict SDPA and eager runs previously stopped on divergence. Same-prefix diagnostics showed BF16 near-tie sensitivity: one cached single-token forward tied tokens 21 and 402 at logit 16.875, while a block forward gave token 402 logit 17.0 and token 21 logit 16.75. A cloned preceding cache reproduced the block result. This supports a batching/numerical explanation for that observed mismatch, **not a proof that every mismatch is harmless**. [PyTorch documents](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html) that batched and sliced calculations need not be bitwise equal. There is no lossless-production claim.

Slowdown also holds on exact-output pairs: depth-1 median ratios were 0.9135× fixed, 0.9161× shared CE, and 0.9060× shared state (8 pairs each); depth-8 approximately 0.712× (10 pairs each).

## Why the first run is not a clean comparison

The initial prototype reapplied the trunk final RMSNorm to states that were already final-normalized. That violates zero-update residual identity and produced odd/even state flips. `v0-confounded-*` and `commands/v0-glimmer_recursive_mtp.py` preserve the actual historical diagnostic implementation and results; **do not use them as a clean architecture comparison**.

A read-only B70 probe of the BF16 norm stored in the separate GPTQ checkpoint found 3,319 negative channel scales out of 6,656. This is a proxy norm, **not bytewise verified against the original cloud BF16 checkpoint**. On saved original teacher roots, the proxy repeated-norm control changed mean adjacent-state cosine from 0.5951 to −0.0816. `norm-control.json` records the control without publishing raw model weights. The probe command was `ssh inference-host 'python3 -' < commands/glimmer_norm_inspect.py`; it read only the safetensors header and norm tensor.

R1 retrained every head using the post-normalized gated residual transition, with no second trunk norm. Teacher capture, data schedule, training budget, verifier and cache logic were unchanged. Checkpoints now require `postnorm-gated-residual-v1`; obsolete heads are refused. Zero-update identity tests cover FP32/BF16, both head layouts and recursive depths 1–8. Correcting the representation improved state similarity but did **not** produce useful drafting in this training budget.

## Real end-to-end execution and reproduction

Target: public `meta-models/Muse-Glimmer-30B`, pinned revision `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`, BF16 SDPA. Environment: one Shadeform Massed Compute **A100 SXM4 80GB**, Ubuntu 22.04, NVIDIA driver 580.126.09, Python 3.10.13, PyTorch 2.14.1+cu130, Transformers 5.15.1, Accelerate 1.15.0. Model/runtime were isolated from interactive Pi. The occupied local RTX 5080 was not used. B70 was only used for the read-only norm probe.

The actual input boundary was the standalone CLI against the official 30B target, not mocks or a projection-only proxy. Test data: `scripts/experiments/glimmer_recursive_mtp.jsonl`, 24 training sequences and 12 heldout raw continuation prompts, two each coding/prose/reasoning/structured/repetitive/high-entropy. Outputs were greedy, capped at 64 new tokens. Each candidate really proposed recursively, verified `[anchor, drafts...]` in one cached target call, rolled back rejected KV, and continued from actual target state. Full-prefix diagnostics ran separately and were excluded from timing. No service or production launcher was changed.

Exact setup and run commands are retained in `commands/glimmer-mtp-r1-20261002.yaml` and `commands/glimmer-pilot-r1.sh`. From repository root in the pinned GPU runtime, the executed sequence was:

```bash
python -m pytest -q -p no:cacheprovider tests/test_glimmer_recursive_mtp.py
python -S scripts/experiments/glimmer_recursive_mtp.py validate
# Initial teacher capture; reused unchanged for R1:
python scripts/experiments/glimmer_recursive_mtp.py capture --output capture.pt
python scripts/experiments/glimmer_recursive_mtp.py train \
  --capture capture.pt --output-dir heads --rank 64 \
  --updates 100 --batch-size 4 --seed 20261002
python scripts/experiments/glimmer_recursive_mtp.py evaluate \
  --record-divergence --capture capture.pt \
  --heads heads/fixed-ce.pt heads/shared-ce.pt heads/shared-state.pt \
  --max-new-tokens 64 --repeats 1 --output evaluate-diagnostic.json
```

Observed: **102 tests passed in 2.64 s**, validation accepted 36 sequences, training saved three newly tagged heads, and evaluation returned `status: complete` with 144 pairs. Small-model/oracle unit tests supplement rather than replace this real GPU journey. To inspect raw evaluation without PyTorch: `gzip -dc r1/evaluate-diagnostic.json.gz`.

Run metadata's `code_commit` is empty because the cloud upload was not a Git checkout; `r1/code-commit.txt` records the uploaded source commit. Timing and environment are original observations, not synthesized receipts. Historical canary and strict-failure evidence remain in this folder. R1 is the authoritative corrected comparison.

## Artifacts, budget and cleanup

- Published: source/tests/protocol, commands, compressed per-pair evaluations, summaries, training JSON and runtime/test/evaluation logs. Research sources and the predeclared E2E contract are in [`scripts/experiments/glimmer_recursive_mtp.md`](../../scripts/experiments/glimmer_recursive_mtp.md).
- Local full backup: `/home/mike/b70-evals/20261002-glimmer-recursive-mtp-pilot/`, corrected run in `r1/`. Contains `capture.pt` and all three head checkpoints; model weights and binary captures/checkpoints are **not committed**.
- Initial instance `028a1baa-11b1-42b3-9833-74d9b276a668`: created 15:33:28 UTC, confirmed absent 17:49:56 UTC, approximately **$3.14** at $1.38/hour.
- R1 instance `11d74cf3-1d1d-41a5-aa80-a26ae6d214c3`: created 18:03:31 UTC, terminated after successful backup; confirmed absent by 19:08:54 UTC. Conservative creation-to-confirmation estimate **$1.51**.
- Combined observed-time estimate **≤$4.65**, well below the approved $25 cap. Not a reconciled provider invoice. Both launches had native three-hour wall-clock automatic teardown; manual `sky down -y` followed successful backups. Final provider query returned no instances.

## Interpretation and next decision

This is commissioning evidence and a very small training pilot, not a decisive test of the recursive hypothesis. The shared head is eight times smaller; state supervision helped drift modestly; none met the acceptance criterion. **Do not port these heads to Qwen or production.** A subsequent bounded experiment should first establish that a head can learn useful step-1 prediction with representative data and a substantially larger training budget, then revisit depth and serving numerics. Alternating blocks, IQuest training/deployment asymmetry, dynamic recursion, Qwen, optimized speculative baselines, top-k/KL evaluation and production integration were not performed. IQuest serving code shows recursive reuse; its unavailable training implementation does not establish the proposed independent-training asymmetry.
