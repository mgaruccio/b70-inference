# Glimmer amplitude-supervision pilot

**Tier: development. Both arms trained successfully. Amplitude supervision reduces magnitude drift, but does not provide a meaningful speculative-decoding improvement. No deployment promotion or recursive-depth training success is claimed.**

## Experiment and results

Two independent heads receive 2,000 updates / 128,000 training-root exposures each, starting from the same prior angular-supervised selected 10k checkpoint. Both selected checkpoints are at update 2,000. Fresh optimizers, seed/data order, batch 64, rank 128, LR 3e-4, warmup 100 and cosine horizon 2,000 match. Both use CE 0.25 + teacher KL 1.0 + angular state loss 0.2. Only `shared-state-norm` adds relative RMS error with weight 0.2:

```python
ratio = sqrt((mean(predicted_state**2) + 1e-8) / (mean(teacher_state**2) + 1e-8))
loss_norm = mean((ratio - 1)**2)
```

This term is FP32 with detached teacher targets. Angular loss remains normalized MSE plus cosine distance. Each checkpoint has one shared gated-residual transition block, trained **only at depth 1**; depth 2/4 below means recursive deployment probes, not training at those depths. No downstream normalization is reapplied between head transitions. Teacher/trunk, embeddings, norm and LM head remain frozen.

| Probe depth | Accepted drafts/pass, angular | Accepted drafts/pass, +RMS | Draft acceptance, angular / +RMS | Exact-output pairs, angular / +RMS | Median decode speedup, angular / +RMS |
|---|---:|---:|---|---|---|
| 1 | 0.4834 | 0.4834 | 48.3431% / 48.3431% | 12/12 / 12/12 | 1.3539x / 1.3457x |
| 2 | 0.5354 | 0.5416 | 27.0408% / 27.3566% | 12/12 / 12/12 | 1.3238x / 1.3253x |
| 4 | 0.5170 | 0.5354 | 13.2684% / 13.7448% | 10/12 / 10/12 | 1.2187x / 1.2154x |

Timing ratios include only exact-output pairs and are diagnostic small-probe decode timings, not full serving throughput or valid lossless speedup claims for a configuration that fails equivalence. Draft counts exclude anchor/correction/bonus tokens. Conditional acceptance and full per-prompt records are in the raw validation JSON. The original goal of at least three accepted speculative drafts per verification pass is **not met**.

| Offline depth | Predicted RMS, angular / +RMS | Teacher RMS | Raw state MSE, angular / +RMS | Teacher KL, angular / +RMS |
|---|---|---:|---|---|
| 1 | 4.4544 / 3.9211 | 2.8230 | 13.6927 / 11.0307 | 1.6040 / 1.5956 |
| 2 | 6.4573 / 5.1516 | 2.8237 | 32.0827 / 20.7740 | 6.9399 / 6.3125 |
| 4 | 14.4920 / 9.6457 | 2.8207 | 192.9401 / 82.6302 | 19.3934 / 15.4750 |

At depth 4, RMS-error falls from 19.1628 to 6.6030 and raw MSE falls about 57.2%, but cosine distance worsens from 0.5577 to 0.5944. Thus amplitude improves, not full representational quality. Deeper recursion remains inaccurate and adds overhead. The pre-pilot directional checkpoint also has only 0.5230 accepted drafts/pass and 10/12 exact pairs at depth 4 (`prior-recursion/validation.json`).

Depth-4 decoding fails the exact greedy token-equivalence contract: both final arms diverge on two prompts, median first divergence at output position 54. These are **correctness failures**, not merely poor acceptance, and block deployment promotion and lossless speedup claims. Their cause remains unresolved; prior near-tie diagnostics do not establish the cause of these new divergences. One seed, 12 validation prompts, 64 output tokens and 1,024 fixed offline validation roots do not support statistical superiority or production claims. The final test split remains sealed for tuning.

## Real end-to-end process

Environment: one rented NVIDIA A100-SXM4-80GB on Shadeform/massedcompute ($1.38/hour); frozen `meta-models/Muse-Glimmer-30B`, revision `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`; BF16 / SDPA; Torch 2.14.1+cu130; Transformers 5.15.1; CUDA 13.0; Python 3.10.13. Both arms run on the same physical GPU. Exact arguments/environment are retained in metadata and raw reports. RTX 5080 is untouched.

1. Before renting, run focused CPU tests: **68 passed and 3 subtests**. Rehearse the exact shell-embedded gates/report with isolated fixtures (not training evidence), including mismatched tensor-sampler rejection and exclusion of divergent pairs from speedups.
2. Cleanly restore the unchanged directional checkpoint, its referenced `state-supervision/train-manifest.json` and capture index from existing private R2 objects. Verify SHA256, actual manifest loader, corpus equality and held-out separation. Restore all 479 original teacher shards onto the GPU; no regeneration or batching changes.
3. Run the complete seven-file GPU test selection: **194 passed and 17 subtests** (`tests.log`). Check an actual batch of 64 for state/token alignment, frozen teacher, detached targets, finite gradients and RMS-loss scale (`state-preflight.json`).
4. Execute [the exact run recipe](run.sh) at the public training/validation CLI boundary:

   ```bash
   bash ~/mtp-training-code/scripts/experiments/glimmer_amplitude_pilot.sh
   ```

   It first performs one actual update per arm and reloads saved checkpoints (`state-smoke.json`), probes the prior head at depth 4, trains both arms for 2,000 updates, then runs `validate-head --probe-depth 1`, `2`, and `4` for each selected-best head. Validation seed 314159; 1,024 unique roots; fixed 12 prompts spanning six categories; 64 new tokens. Per-prompt chosen-path drift diagnostics run separately and outside decode timing for one prompt/category at depths 2/4.
5. Independently reload final checkpoints on CPU: actual optimizer step 2,000 for both, matching initialization identity and final sampler state including tensor RNG values, training depth 1, 128,000 root exposures, finite logged losses. Logs record every tenth update plus the initial update; they are not 2,000-row logs.
6. Back up **41 output files** to R2 and locally, independently verify their sizes/SHA256, then delete only the owned rented instance. Workflow, backup and teardown exit codes are all zero; provider inspection confirms no owned MTP instance remains.

Local full checkpoint backup: `/home/mike/b70-evals/20261002-glimmer-mtp-training/amplitude-pilot/outputs/`. R2 backup: `r2:ml-archive/2026-10-04/cache/b70-evals/20261002-glimmer-mtp-training/amplitude-pilot/outputs/`. Private signed-URL manifests are not published.

The existing [Grafana dashboard](https://hermes.tailc35014.ts.net:3000/d/glimmer-mtp/?refresh=20s) loses its live source after the rented GPU is deleted. “Metric source unreachable” then means **no live host**, not training failure. Prometheus retains the historical `mtp_train_update{stage="amplitude-pilot"}` series at 2,000 for both variants; select a time range covering the run. This result directory is the durable completion record, without introducing a replacement metrics service.

## Research informing the experiment

Fresh official PyTorch documentation distinguishes angular cosine loss (direction, scale-invariant) from squared error (magnitude-sensitive):
- https://docs.pytorch.org/docs/2.14/generated/torch.nn.CosineEmbeddingLoss.html
- https://docs.pytorch.org/docs/2.14/generated/torch.nn.MSELoss.html

The relative RMS term is this experiment's choice, not a formula recommended by those pages. These results justify neither a longer identical run nor Qwen/serving promotion. Actual recursive-depth training and the depth-4 fidelity issue are separate decisions; no further paid run is automatically launched.
