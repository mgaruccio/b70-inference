# Glimmer one-step hidden-state supervision: matched training ablation

**Tier: development. Both arms trained successfully; no observed decoding advantage from state supervision. This is not recursive-depth success or a standard publishable serving benchmark.**

## Results

Both arms start from the same previously selected 48,000-update one-step checkpoint. Each receives **10,000 new optimizer updates / 640,000 root exposures**, with a fresh optimizer, identical seed/data order, and the same fixed-validation selection rule. Both selected checkpoints are at update 10,000. The final sampler states match exactly, including tensor-valued generator states.

| Metric | Token-only control | State-supervised |
|---|---:|---:|
| Accepted / proposed speculative drafts | 250 / 512 | 248 / 513 |
| One-step draft acceptance | 48.8281% | 48.3431% |
| Teacher argmax agreement, 1,024 fixed validation roots | 58.3008% | 58.5938% |
| Mean teacher KL | 1.53927 | 1.53675 |
| Normalized state MSE | 1.10955 | 0.85536 |
| State cosine distance | 0.55477 | 0.42768 |
| **Raw state MSE** | **12.39678** | **13.18571** |
| Predicted-state RMS / teacher RMS | 3.65831 / 2.82299 | 4.34423 / 2.82299 |
| Median paired decode speedup, exact-output pairs only | 1.35329x | 1.35412x |
| Median baseline / candidate decode tokens/sec | 21.0346 / 28.4505 | 21.0616 / 28.5220 |
| Exact token-output matches | 12 / 12 | 12 / 12 |

State supervision reduces **normalized** state error by 22.91%, but does not improve acceptance (-0.485 percentage points) or realized decode speed in this small probe. Both remain below the earlier 50% one-step acceptance gate. Only one seed and 12 validation prompts were measured; the small acceptance difference is not evidence of a statistically reliable quality ranking.

Importantly, this loss is `0.2 * (normalized MSE + cosine distance)`: it rewards directional alignment, **not state amplitude**. Predicted RMS and raw state MSE worsen. Do not describe this as improved full-state fidelity or recursive stability. A next experiment should address amplitude as well as direction and then measure actual depth-2/4 drift and acceptance; do not simply extend identical one-step training.

## Frozen model, data and intentional difference

- Teacher: `meta-models/Muse-Glimmer-30B`, revision `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`, frozen trunk, embedding, normalization and LM head.
- Same existing capture: 479 SHA256-verified shards; no regeneration. Train: 3,000,166 sequence tokens / 2,137,968 available roots. Fixed validation: 1,024 unique roots, seed 314159. The final test split stays sealed.
- Same 2,562,560-parameter, rank-128, postnorm gated-residual head; same batch 64, LR 3e-4, 500 warmup updates and 10,000-update cosine horizon.
- Both use sequence CE weight 0.25 plus forward teacher KL weight 1.0. **Only the shared-state arm enables the state loss, weight 0.2.** `shared-ce` intentionally ignores that configured auxiliary weight.
- This is a fresh, matched warm-start ablation, not a continuation of the original optimizer/RNG cursor. Reset optimizer/schedule apply equally to both arms.
- State contract: `h[t] + embedding(token[t+1]) -> predicted h[t+1]`, whose LM-head projection predicts `token[t+2]`. Teacher states are final-normalized and detached. No teacher hidden state is fed back after the root.

## Real end-to-end process and observed results

Environment: one NVIDIA A100-SXM4-80GB, 500 W; BF16 teacher; SDPA; Torch 2.14.1+cu130; Transformers 5.15.1; CUDA 13.0; NVIDIA driver 580.126.09; Python 3.10.13. Both arms ran on the same physical GPU. Full environment and exact arguments remain in each raw validation and metadata JSON.

1. Restore the original capture, selected-best head **and its referenced training manifest** from existing R2 objects into `/home/shadeform/mtp-training-run`. Verify every object against its original SHA256. A clean local R2 restore checks the actual manifest loader, corpus equality and held-out separation before rental.
2. Run the existing test suite plus `tests/test_glimmer_mtp_state_recipe.py` in the isolated GPU runtime: **178 tests and 17 subtests passed**.
3. Check a real batch of 64: correct shifted token/state alignment, detached teacher targets, frozen teacher parameters, finite head gradients and auxiliary-loss scale. Initial supervised loss 1.7282635; auxiliary component 0.3276241.
4. Run both arms through one actual public-CLI optimizer update, save and reload the checkpoints, and require matching initialization and tensor-aware sampler state. Losses: control 1.4006394, supervised 1.7282635. Gate passed.
5. Execute the exact training and validation recipe:

   ```bash
   bash ~/mtp-training-code/scripts/experiments/glimmer_mtp_state_supervision.sh
   ```

   Source: [`scripts/experiments/glimmer_mtp_state_supervision.sh`](../../scripts/experiments/glimmer_mtp_state_supervision.sh). It runs `glimmer_recursive_mtp.py train --variants shared-ce shared-state --init-head ... --train-depth 1 --updates 10000 ...`, then the actual `validate-head` CLI independently for each selected-best checkpoint. Checkpoint/validation cadence is 1,000 updates; real decoding probes run every 5,000 updates. Approximate per-arm training time including in-run validation/probes: 262.9 s control, 260.9 s supervised.
6. Validation uses 12 real prompts, two each for code, prose, reasoning, structured output, repetitive output and high-entropy generation; max-new-tokens 64, greedy decoding, fresh caches. Baseline and candidate use the same target/configuration and prompt. Acceptance excludes anchor/correction/bonus tokens. Decode throughput excludes prefill; raw timings retain prefill and total generation separately. These are direct Transformers decoding measurements, **not an actual serving-stack benchmark**.
7. Back up all outputs, including smoke checkpoints, through object-specific R2 PUT URLs, download locally, and independently verify **31 files** by SHA256. Confirm both real last checkpoints contain update 10,000, optimizer step 10,000, and 640,000 exposures. Delete only this experiment GPU. Workflow, backup and teardown all exited 0; provider live-instance query confirmed no MTP instance remains.

The successful workflow took 23m04s including provisioning, downloads, restoration, preflight, training, validation and backup. Full GPU task wall durations including the two failed rentals (11m44s and 11m36s) give a conservative **$1.0672 GPU-usage upper-bound at $1.38/hour**, below the $3 experiment allowance. This is not a provider invoice and excludes R2 fees. The earlier configuration-parse failure allocated no GPU.

## Failures and limitations retained

- First launch: invalid `infra`/`region` combination, before resource allocation.
- First rented attempt: checkpoint restored without the external `train-manifest.json`; no optimizer update. Fixed by packaging the unchanged sidecar and checking it before rental.
- Second rented attempt: both arms reached update 1, then the diagnostic gate used Python dictionary equality on tensor-valued sampler states. Fixed with `torch.equal` and CPU tests of the exact embedded gate, mismatch rejection and report generation.
- The earlier 48k initialization checkpoint still reproduces a divergence at token 57: identical emitted prefix, reversed verifier top-two tokens, 0.0625 logit margin on each path. This is consistent with a near-tie numerical difference; it is not a general correctness proof. The final two candidates matched all 12 measured prompts, but **no global lossless-decoding claim is made**.
- No depth-2/4/8 recursive training or evaluation, fixed-depth/EAGLE comparison, dynamic stopping, Qwen transfer, serving integration, or >=3 accepted speculative tokens/pass result is claimed.
- The remote runtime was an rsynced source copy outside Git, so its raw `code_commit` fields are empty and preserved unchanged. The trainer source was unchanged from repository commit `89e30939`; the exact new recipe is published alongside its regression tests. No checkpoint metadata was rewritten.

## Artifacts

- [`comparison.json`](comparison.json): metrics and paired conclusions.
- `state-preflight.json`, `state-smoke.json` and `tests.log`: original actual-batch, save/reload-gate and test outputs.
- Each arm directory: original `metadata.json`, `validation.json` and compressed per-update `training.jsonl.gz`.
- Full local backup, including weights, tests, preflight, smoke and failures: `/home/mike/b70-evals/20261002-glimmer-mtp-training/state-supervision-r2/outputs/`. Failed attempt backups are in neighboring `state-supervision/` and `state-supervision-r1/` directories.
- Private R2 backup: `r2:ml-archive/2026-10-04/cache/b70-evals/20261002-glimmer-mtp-training/state-supervision-r2/outputs/`. Initialization and its sidecar are in the neighboring `initialization/` prefix. No presigned URLs, account credentials, weights or generated training corpus are committed to Git.
- This reduced development run deliberately does not satisfy the repository's full standard benchmark checklist (BetterBench, serving benchmark, long-context sweep and larger quality evaluation).

## Research informing the approach

- [EAGLE paper, latest accessed revision v3](https://arxiv.org/abs/2401.15077): feature-level autoregression and the advanced token address feature uncertainty. This motivates testing state supervision, but our final-normalized representation and small MLP are **not an EAGLE reproduction**.
- [Cloudflare R2 boto3 documentation](https://developers.cloudflare.com/r2/examples/aws/boto3/): use the R2 endpoint, `auto` region and object-specific presigned URLs. Existing capture and checkpoints were staged privately through R2 instead of repeated desktop-to-cloud bulk transfers.
