# Glimmer recursive-head 20k continuation

**Development training completed: 20,000 additional actual depth-2 updates, 22,000 depth-2 updates in this checkpoint lineage. Modest acceptance improvement; original >=3 accepted drafts/pass goal remains unmet. No production promotion.**

Warm-start from the prior depth2 update2000 `recurrent-two-step/checkpoint-last.pt`, one shared rank128 gated residual block (2,562,560 parameters). Frozen Glimmer trunk/embedding/norm/LM head; existing 479 teacher shards, no regeneration. New optimizer, restarted deterministic sampler, cosine horizon20000, warmup200, batch64, LR3e-4, seed20261002, CE .25, KL1, angular .2, RMS .2. This is not exact optimizer-state resume. Predicted hidden states feed back through both steps, without teacher-state substitution; training token embeddings are teacher-forced. Final update20000, not a different best-selection checkpoint, is evaluated; best update happens to be20000.

| Probe depth | Accepted drafts/pass, initial -> final | Draft acceptance, initial -> final | Exact output, initial / final |
|---|---|---|---|
| 1 | .4663 -> .4681 | 46.6281% -> 46.8085% | 12/12 / 12/12 |
| 2 | .5385 -> .5585 | 27.1984% -> 28.1865% | 12/12 / 12/12 |
| 4 | .5416 -> .5458 | 13.9135% -> 14.0167% | 10/12 / 10/12 |

Same-GPU initial/final evaluation: fixed12 validation prompts spanning six categories,64 generated tokens;1024 fixed offline validation roots,batch8,seed314159. Teacher KL at depths1/2/4 changes1.6256/3.1154/5.2177 ->1.4817/2.6114/4.3093. Raw state MSE changes9.4903/15.0025/35.9917 ->8.8772/12.6236/23.0693. State supervision improves offline predictions more than realized acceptance. Depth4 still violates exact greedy output equivalence on2/12 prompts; cause unresolved and no lossless deployment claim. Decode timing ratios in comparison.json exclude divergent pairs, are diagnostic only, and do not establish production throughput/statistical superiority. One seed, short prompts and small validation selection are not the full benchmarking standard. Final test data remains sealed.

## Executed process and evidence

Dedicated full non-MIG A100-SXM4-80GB (400W), Shadeform/massedcompute Beltsville,$1.38/hour advertised,60minute/$2 provider cap. Frozen meta-models/Muse-Glimmer-30B revision a4e59da52a7bc87ae7251dd5545c0dd437c44b68,BF16/SDPA,Torch2.14.1+cu130,Transformers5.15.1,CUDA13.0. Exact environment/arguments are in shared-state-norm.json and raw validation artifacts. Other workloads/RTX5080 untouched.

1. Before rental,91 focused CPU tests passed. The exact recipe smoke/reload/report was exercised with real tiny CPU trainer writes and fixtures (not GPU training evidence), and the actual prior Glimmer checkpoint initialized identical shared-block weights.
2. Clean R2 checkpoint/manifest/index restore verified SHA256, actual manifest loader, training-corpus equality and held-out independence. All479 existing shards were restored and SHA verified on the GPU. Proved actual job-interpreter CUDA BF16 allocation/matmul/synchronization (`readiness.json`).
3. Actual rented-runtime tests: **210 passed,17 subtests** (`tests.log`). Public CLI via [run.sh](run.sh): `bash ~/mtp-training-code/scripts/experiments/glimmer_mtp_continue20k.sh`. First perform one actual update and reload `checkpoint-last.pt`, proving fresh optimizer step1 and changed weights; then train all20000 updates without an approval pause, checkpoint every1000, offline validation every2000. Evaluate initial and final heads at depths1/2/4 on the same physical GPU. Chosen-path diagnostics are outside timed decoding.
4. Independently reload final checkpoint on CPU: all saved optimizer steps20000, trained depth2,1,280,000 root exposures,2,560,000 loss-position exposures. All2001 logged rows have finite losses/gradient norms (every10th update plus update1, not20000 rows). Training-loop elapsed486.51seconds; full setup/train/evaluate/backup workflow31minutes.
5. **32 output files** backed up to existing private R2 and locally; independently verify sizes/SHA256 (`r2-backup.json`). Workflow/backup/teardown exit0. Owned instance0861bcc5-c99e-4751-99dc-15091860041d deletion requested after backup; no production serving changes.

Full checkpoint/logs: `/home/mike/b70-evals/20261002-glimmer-mtp-training/recursive-depth2-continue20k/outputs/`. R2: `r2:ml-archive/2026-10-04/cache/b70-evals/20261002-glimmer-mtp-training/recursive-depth2-continue20k/outputs/`. Weights/account credentials/private signed URLs are not committed. Existing Grafana live source disappears after GPU release; historical data/raw artifacts distinguish finished training from failure. No parallel telemetry service.

Fresh primary research informing the warm-start: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html distinguishes loading model parameters from restoring optimizer/training state; https://docs.pytorch.org/docs/main/generated/torch.optim.Optimizer.load_state_dict.html documents optimizer/scheduler ordering. A fresh optimizer/schedule is deliberate and disclosed, not claimed as exact resume.

The user requests continued training without approval pauses. The next research stage trains the same shared transition at depth4 from this final checkpoint, rather than extending identical depth2 training indefinitely. Output-equivalence failures continue to block deployment, not the explicitly authorized frozen-model research run.
