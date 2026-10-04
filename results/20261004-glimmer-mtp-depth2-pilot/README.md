# Glimmer genuinely depth-2-trained recursive MTP pilot

**Development experiment. Both heads trained successfully for 2,000 actual optimizer updates. Recursive training improves deeper offline predictions but does not improve realized speculative acceptance. No production promotion.**

## Matched training

Two independent checkpoint arms, each with one **shared** rank-128 gated residual transition (2,562,560 parameters):

- `control-one-step`: loss unrolled at depth 1.
- `recurrent-two-step`: loss unrolled at depth 2, feeding the first **predicted state** back through the same block; teacher hidden states are targets, not replacement recurrent inputs. Training token embeddings are teacher-forced; inference feeds back predicted tokens.

Both start from the same amplitude-supervised update-2,000 selected checkpoint, with fresh optimizers and identical seed/data order, batch 64, LR 3e-4, cosine horizon 2,000, warmup 100, CE 0.25, teacher KL 1.0, angular state loss 0.2 and relative RMS loss 0.2. Glimmer trunk, embeddings, norm and LM head remain frozen. Existing 479 teacher-capture shards are reused; no regeneration or data-generation batching changes. Each arm consumes 128,000 root exposures (not a claim of distinct roots). Depth 2 intentionally receives twice the supervised loss-position exposures: 256,000 versus 128,000. Both final-update checkpoints are evaluated, rather than comparing different best-selection objectives. Best update happens to be 2,000 for both.

Saved checkpoint optimizer states independently confirm step 2,000, trained depths 1/2, matching initialization and final sampler tensors. Each training log contains 201 finite logged rows, recording update 1 and every tenth update; not 2,000 log rows. Training-loop elapsed time is 33.28s / 53.40s; the full setup, restore, training and decoding-evaluation workflow takes 25m18s. This is not a serving benchmark.

## Real decoding results

Fixed 12 validation prompts across six categories, 64 generated tokens each. Draft acceptance excludes anchor/correction/bonus tokens.

| Probe depth | Accepted drafts/pass: control / recurrent | Draft acceptance: control / recurrent | Exact-output pairs: control / recurrent |
|---|---|---|---|
| 1 | 0.4951 / 0.4663 | 49.5088% / 46.6281% | 12/12 / 12/12 |
| 2 | 0.5542 / 0.5385 | 27.9959% / 27.1984% | 12/12 / 12/12 |
| 4 | 0.5385 / 0.5416 | 13.8110% / 13.9135% | 10/12 / 10/12 |

The original goal of **at least 3 accepted speculative drafts per verification pass is not met**. Depth-2 training trades some first-step acceptance for stronger deeper offline predictions; it does not beat the one-step control in realized acceptance at depth 2.

Offline validation uses 1,024 fixed unique validation roots, batch 8, seed 314159:

| Offline depth | Teacher KL: control / recurrent | Raw state MSE: control / recurrent | Predicted RMS: control / recurrent |
|---|---|---|---|
| 1 | 1.6402 / 1.6256 | 10.6769 / 9.4903 | 3.8202 / 3.5206 |
| 2 | 6.3028 / 3.1154 | 19.1362 / 15.0025 | 4.8926 / 4.2390 |
| 4 | 14.5575 / 5.2177 | 68.5399 / 35.9917 | 8.7555 / 6.3803 |

Teacher RMS is approximately 2.82 throughout. Directional cosine error does not improve (depth 2: 0.5213 / 0.5341; depth 4: 0.6138 / 0.6233), so stronger token KL and amplitude do not establish universally better representations.

Median candidate/baseline **decode-only timing ratios for exact pairs only** are 1.3206x / 1.3026x at depth 1, 1.3337x / 1.3296x at depth 2 and 1.2417x / 1.2545x at depth 4. These small diagnostic probes are not production throughput or statistically supported superiority claims. Depth 4 fails exact greedy equivalence on `val-prose-01` and `val-reasoning-01` for both arms (median first divergence position 54). Its cause remains unresolved; numerical sensitivity is not proven. Failed configurations cannot be promoted or called lossless. One seed, 12 prompts and short contexts do not satisfy the full benchmarking standard; final test data remains sealed.

## Executed end-to-end process and environment

One dedicated Shadeform/massedcompute A100-SXM4-80GB, MIG disabled, 81,920MiB, 400W, driver 580.126.09, advertised $1.38/hour; provider-side 60-minute/$2 cap. Frozen `meta-models/Muse-Glimmer-30B` revision `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`, BF16/SDPA, Torch 2.14.1+cu130, Transformers 5.15.1, CUDA 13.0, Python 3.10.12. Both arms use the same physical GPU. RTX 5080 and unrelated B70/cloud workloads are untouched.

1. Before rental, run focused CPU checks (**87 passed**) and exercise actual tiny CPU trainer writes through the exact shell reload gate. Fixtures are not real Glimmer training evidence.
2. Restore the prior checkpoint, its unchanged referenced manifest and capture index from private R2. Verify SHA256, actual manifest loader, equality with the training corpus and held-out separation. Restore all 479 existing shards on the GPU, verifying SHA256.
3. Verify actual SSH-job interpreter CUDA availability, BF16 allocation/matmul/synchronization and full non-MIG GPU. Bare-image bootstrap installs `python3-venv` only when required. Run eight-file regression selection: **206 passed, 17 subtests** (`tests.log`).
4. At the actual application CLI boundary run:

   ```bash
   bash ~/mtp-training-code/scripts/experiments/glimmer_depth2_pilot.sh
   ```

   [Exact recipe](run.sh) performs one real update at each depth, reloads `checkpoint-last.pt`, checks finite training, changed weights, same initialization/sampler and parameter counts (`depth2-smoke.json`), then automatically trains both arms to update 2,000. Offline validation runs every 500 updates. Public `validate-head` runs depths 1/2/4 for both final checkpoints. Chosen-path drift diagnostics are outside timed decoding. Exact commands/environment and full per-prompt outputs are retained in metadata/validation JSON.
5. Independently reload final optimizer checkpoints on CPU and confirm update 2,000, depths/exposures, finite losses/gradients, initialization identity and final sampler state including RNG tensors.
6. Back up **45 output files** to existing R2 and locally; independently verify sizes and SHA256. Workflow, backup and teardown exit codes are all zero. Provider API inspection confirms the owned instance `2d225e34-e624-4faf-91cb-6177b8e6150b` is gone.

Full checkpoint/log backup: `/home/mike/b70-evals/20261002-glimmer-mtp-training/recursive-depth2-run3/outputs/`. R2: `r2:ml-archive/2026-10-04/cache/b70-evals/20261002-glimmer-mtp-training/recursive-depth2-run3/outputs/`. Weight files, private signed URLs and account credentials are not committed.

Two earlier bounded attempts failed: the first did one actual smoke update at each depth, then the recipe incorrectly expected variant-named resumable checkpoints instead of single-variant `checkpoint-last.pt`; the second failed SSH verification when the provider recycled an IP with a different host key. The filename contract was corrected and exercised against actual trainer writes. SSH now uses the provider instance UUID as `HostKeyAlias` with `accept-new`, retaining changed-key rejection without deleting unrelated known-host entries or disabling verification. Both failed instances were deleted; their local logs remain under sibling `recursive-depth2-run` / `recursive-depth2-run2` directories. Neither is counted as the completed 2,000-update experiment.

The existing [Grafana dashboard](https://hermes.tailc35014.ts.net:3000/d/glimmer-mtp/?refresh=20s) has no live GPU source after teardown. An unreachable live metric source is then expected, not evidence of failed training; use historical data or these durable artifacts. No replacement telemetry service was added.

## Fresh external sources informing the approach

- https://arxiv.org/abs/2401.15077 — EAGLE studies feature autoregression with token input; this final-normalized-state head is a prototype, **not** an EAGLE reproduction.
- https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html — batched and sequential floating-point results need not be bitwise identical; this motivates measuring exact-output fidelity, not assuming a cause for observed failures.
- https://docs.nvidia.com/datacenter/tesla/mig-user-guide/getting-started-with-mig.html — MIG mode needs configured GPU/compute instances; this run instead requires a full non-MIG GPU and proves actual CUDA execution.
- https://docs.shadeform.ai/api-reference/instances/instances-create — explicit `shade_cloud`, owned instance request and provider auto-delete controls.
- https://docs.python.org/3.10/library/venv.html — venv bootstrap depends on ensurepip; the provider image needed the OS venv prerequisite.

This bounded training run is complete. Results justify further architecture investigation, not a longer identical run or Qwen/serving promotion. No further paid run is launched by this report.
