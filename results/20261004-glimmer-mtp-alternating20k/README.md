# Glimmer alternating A/B depth4 — 20,000 actual updates

**Development training experiment. Not an upstream-DFlash baseline or a production/lossless speed result.**

One alternating head contains two independent learned blocks reused A→B→A→B. Both start from the same completed depth2 checkpoint used to initialize the previously trained pure/shared depth4 control. Frozen Glimmer, original479-shard corpus, rank128, batch64,20kupdates, fresh cosineoptimizer/warmup200, CE.25/KL1/angular.2/RMS.2, seed20261002. Alternating has5,125,120parameters vs2,562,560shared: twice parameter budget, not an equal-size comparison.

Actual checkpoint and optimizer verified at update20,000;1,280,000rootexposures;5,120,000losspositionexposures;both blocks present;2001loggedtrainingrows. Same final sampler as shared control. Both final checkpoints evaluated on the SAME dedicated full A100-SXM4-80GB with Torch2.14.1+cu130/Transformers5.15.1,12fixedvalidationprompts,64newtokens,greedy,BF16SDPA. Target pinned meta-models/Muse-Glimmer-30B@a4e59da52a7bc87ae7251dd5545c0dd437c44b68. No corpus regeneration or base-model adaptation.

| Draft depth | Shared accepted drafts/verify | Alternating accepted drafts/verify | Exact outputs, shared / alternating |
|---|---:|---:|---:|
|1|0.4048|0.4348|10/12 /10/12|
|2|0.5020|0.5120|10/12 /10/12|
|4|0.5190|0.5354|11/12 /11/12|

Accepted drafts exclude the guaranteed target/bonus token. These small increases do not meet the proposal's≥3accepted-drafts/pass criterion. Both have greedy-output divergence; no lossless speedup or deployment promotion is justified. Offline depth4teacher KL3.3371→3.3013. **Matched official-DFlash baseline and cache/numerical diagnosis are the next priority, not another training variant.**

Observed verification:217GPUtests+17subtests passed; real one-update smoke proved both blocks changed, fresh optimizerstep1,2xparameters and checkpointreload; real20ktraining and six public validate-head calls completed. Training elapsed826.86seconds (includes final offline validation). Workflow ended with workflow_exit=0/backup_exit=0/teardown_exit=0.32R2/localfiles individually checked for size/SHA256. Provider instance5e7db814-b8c3-498e-8822-773c30146d0a deleted and its absence independently confirmed. A later CPU-only baseline-test SSH attempt against the already deleted training lease timed out; it was not a training failure.

Exact executable scripts and run-assets: `/home/mike/b70-evals/20261002-glimmer-mtp-training/alternating-depth4-20k/`. Raw comparison is committed beside this report; full checkpoints/logs at `r2:ml-archive/2026-10-04/cache/b70-evals/20261002-glimmer-mtp-training/alternating-depth4-20k/outputs/`. No secrets or model weights are committed. Earlier shared-control training used a different physical GPU; only the final inference comparison here is same-GPU.
