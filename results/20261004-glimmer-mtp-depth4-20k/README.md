# Glimmer shared-block depth4 training — completed

**Development result: 20,000 actual depth4 updates completed. Offline prediction improved, but realized speculative acceptance regressed. No production promotion.**

Warm-start from the completed depth2 continuation checkpoint (22,000 depth2 updates in lineage), new optimizer/schedule and deterministic sampler restart, not exact optimizer resume. One shared rank128 gated-residual block,2,562,560 parameters; frozen Glimmer trunk/embed/norm/LM head. Unroll4 predicted-state transitions, teacher-forced token embeddings, losses at all4 steps weighted1,1,.8,.8. Batch64,LR3e-4,cosine horizon20000,warmup200,seed20261002,CE.25,KL1,angular.2,RMS.2. Reuse479 unchanged teacher shards; no regeneration. Evaluate final update20000; best update is also20000.

| Probe depth | Accepted drafts/pass, incoming depth2 -> final depth4 | Exact-output pairs, incoming / final |
|---|---|---|
| 1 | .4587 -> .4048 | 10/12 / 10/12 |
| 2 | .5574 -> .5020 | 10/12 / 10/12 |
| 4 | .5542 -> .5190 | 11/12 / 11/12 |

Offline teacher KL at depths1/2/4:1.4817/2.6114/4.3093 ->1.5489/2.5297/3.3371. Raw state MSE:8.8772/12.6236/23.0693 ->8.5937/11.1974/15.4258. Stronger deep offline prediction does not imply better accepted drafts; first-step acceptance worsens. Original >=3 accepted drafts/pass goal remains unmet.

**Fidelity warning:** both the incoming and final heads fail exact greedy equivalence on this GPU, including depths1/2. The same incoming head passed depths1/2 on the prior GPU run. This is observed run-to-run instability, not proof of a numerical/cache cause or proof the new training introduced the failures. Median divergence position57.5 at depths1/2,58 at depth4. All configurations in this run fail equivalence, so even exact-pair-only decode timing ratios in raw reports are diagnostics, not valid lossless acceleration/production throughput claims. No promotion. One seed,12 short validation prompts and1024 offline roots are not the full benchmarking standard; final test remains sealed.

## Executed end-to-end process

Dedicated Shadeform/massedcompute full non-MIG A100-SXM4-80GB,$1.38/hour advertised,60minute/$2 cap. Frozen meta-models/Muse-Glimmer-30B revision a4e59da52a7bc87ae7251dd5545c0dd437c44b68,BF16/SDPA,Torch2.14.1+cu130,Transformers5.15.1,CUDA13.0. Full environment/arguments in shared-state-norm.json; other workloads/RTX5080 untouched.

1. Pre-rental exact recipe gate/report exercised via real tiny CPU trainer writes and fixtures (not GPU evidence). Actual incoming checkpoint initialized identical shared-block parameters with a depth4 call bound. Clean R2 checkpoint/manifest/index restore verified SHA256, actual manifest loader,corpus equality and held-out separation.
2. Restore all479 shards and verify SHA; prove actual job-interpreter CUDA BF16 allocation/matmul/synchronization. Rented runtime: **210 tests passed,17 subtests**.
3. Execute [run.sh](run.sh) at public CLI boundary: `bash ~/mtp-training-code/scripts/experiments/glimmer_mtp_depth4.sh`. One actual depth4 update, reload checkpoint-last.pt, prove fresh optimizer step1,changed weights,root64/loss256,same parameter count; then all20000 updates without approval pause. Checkpoint1000,offline validation2000 on1024 fixed roots,batch8,seed314159. Public validate-head compares incoming and final heads on same physical GPU at probe1/2/4,fixed12 validation prompts across6 categories,64 new tokens; chosen-path diagnostics outside decode timings.
4. Independently reload final checkpoint on CPU: **all optimizer steps20000**,depth4,1,280,000 root exposures,5,120,000 loss-position exposures,2001 finite logged loss/gradient rows (every10th update+update1). Training-loop elapsed823.34seconds; full workflow39m13s.
5. **32 output files** backed up to existing R2 and locally; independently verify sizes/SHA (`r2-backup.json`). Workflow/backup/teardown exit0; owned instance35f26878-219d-48d6-ad50-f3b758734cd2 deletion requested after backup. Existing Grafana live source goes away after release; raw/historical records show successful completion, not failed training.

Full checkpoint/logs: `/home/mike/b70-evals/20261002-glimmer-mtp-training/recursive-depth4-20k/outputs/`. R2: `r2:ml-archive/2026-10-04/cache/b70-evals/20261002-glimmer-mtp-training/recursive-depth4-20k/outputs/`. No weights,private signed URLs or account credentials committed.

Fresh primary research: https://raw.githubusercontent.com/pytorch/pytorch/main/torch/_tensor.py documents detach disconnecting a tensor from autograd. Existing training detaches teacher/root targets but preserves recurrent predicted-state gradients through all4 calls. No new architecture was needed for this depth4 stage.

The user requests continued training without approval pauses. Next is the explicitly proposed alternating A-B-A-B architecture, not another identical single-block extension. Compare against this trained pure depth4 control on the same new GPU; no claim it will improve results before measurement.
