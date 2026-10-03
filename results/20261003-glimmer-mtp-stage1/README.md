# Glimmer MTP Stage-1: actual 20,000-update training

**Status:** completed and backed up. This is a trained one-step shared head,
not yet a depth-4/8 recursive MTP result. The selected-best one-step acceptance
is below the predeclared 50% gate; recursive comparisons remain gated.

## Observed results

- Frozen official `meta-models/Muse-Glimmer-30B`, revision
  `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`.
- NVIDIA A100 SXM4 80GB; BF16 / SDPA; PyTorch 2.14.1+cu130,
  Transformers 5.15.1, Python 3.10.13.
- Rank-128 shared residual head: **2,562,560 learned parameters**.
- **20,000 actual updates**, batch64: **1,280,000 root exposures**
  (not unique roots). Training timer **536.48 seconds**, including in-loop
  validation and decoding probes; excludes model loading and data generation.
- Dataset capture: **3,300,355 sequence tokens**, including prompts.
  Train:3,000,166 tokens /2,137,968 eligible response roots;
  validation:150,156 /107,856; test:150,033 /108,360.
  All479 teacher shards were backed up locally; weights/states are not in Git.
- Common objective:0.25 sequence CE +1.0 forward teacher KL, frozen trunk,
  one-step depth1. LR horizon50,000; warmup1,000; no batching changes.

Selected best is update20,000, chosen by heldout teacher KL:

| Metric | Observed |
|---|---:|
| Heldout teacher argmax agreement (1,024 roots) |52.93%|
| Heldout teacher top-5 recall |81.25%|
| Heldout teacher KL |1.7673|
| Heldout cosine distance |0.5591|
| Actual cached depth-1 draft acceptance (12 validation prompts) |43.48% (230/529)|
| Exact greedy output identity |11/12 pairs|

Actual in-loop acceptance progressed32.64% →37.00% →39.85% →43.48%
(at updates5k/10k/15k/20k). Validation KL decreased3.6052 at1k to1.7673
at20k. The latest checkpoint is still improving; the preapproved next experiment
is continuation toward50k with the existing schedule, not ungated recursive arms.

## Timing and fidelity qualification

On the **11 exact-output validation pairs**, median decoding speedup was
**1.2986×** (median total-generation speedup1.2903×). Across all12 diagnostic
pairs it was1.3114×. One pair diverged at generated token57; lengths matched.
These are single-pass validation probes, not the sealed final60-prompt,
interleaved/repeated production benchmark. They do not establish lossless BF16
serving or ≥3 accepted speculative tokens/pass. At depth1 acceptance is about
0.435 accepted drafts/pass; anchor/correction/bonus tokens are excluded.

## Commands, evidence and resource closure

Commands executed on the training host from `~/mtp-training-code`:

```bash
RUN=/home/shadeform/mtp-training-run
P=scripts/experiments/glimmer_recursive_mtp.py
python "$P" train --capture "$RUN/capture/index.json" --output-dir "$RUN/stage1" \
  --variants shared-ce --rank 128 --train-depth 1 --updates 20000 \
  --schedule-updates 50000 --warmup-updates 1000 --batch-size 64 \
  --ce-weight 0.25 --kl-weight 1.0 --seed 20261002 --checkpoint-every 1000 \
  --validation-every 1000 --validation-roots 1024 --validation-batch-size 8 \
  --probe-every 5000 --probe-new-tokens 64 \
  --validation-prompts scripts/experiments/glimmer_mtp_validation.jsonl \
  --record-divergence
python "$P" validate-head --capture "$RUN/capture/index.json" \
  --head "$RUN/stage1/checkpoint-best.pt" --split validation \
  --validation-roots 1024 --batch-size 8 --max-new-tokens 64 \
  --record-divergence --output "$RUN/stage1/validation.json"
```

Raw logs, resumable best/last checkpoints and selected-head `validation.json`:
`/home/mike/b70-evals/20261002-glimmer-mtp-training/stage1/`.
The initial launch failed before any update because the required validation
prompt argument was omitted. The corrected run completed; its background task
returned exit0 after backup and GPU termination. Shadeform API verification at
2026-10-03T15:49:35Z confirmed owned instance
`0613084a-7ee9-42f5-900f-ae57e0e2b292` absent. No GPU remains provisioned for
this run. The final test prompts were not exercised.
