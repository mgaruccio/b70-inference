# Substantive frozen-Glimmer MTP training — review draft

Status: user approved continuing with the staged approach. No GPU provisioned for this run. Fresh primary-source checks below establish the dataset/loss approach; an independent researcher review is still in flight. Numerical budgets are our proposed bounded research experiment, not published reproduction settings. Main-run timing remains conditional on the real Stage-0 profile.

## Authority and goal

The user asks: **“ok let's plan out the actual training run now then”**, following a commissioning run of only 100 updates × 4 roots on 1,665 training tokens. The user has approved **$75 total cloud spend**, including approximately $4.65 already spent. B70 and Shadeform may be used; Glimmer or Qwen may be tested. This plan keeps **Glimmer first**, target frozen, and does not port to Qwen or production.

Original scientific question: can a small shared state transition produce useful speculative proposals through depths 4–8, competitive with independent blocks, without draft overhead erasing savings? Primary success remains **≥3 accepted speculative draft tokens per target verification pass**, excluding anchor/correction/bonus tokens, together with an actual decode speedup.

This run must distinguish:
1. The head/training path cannot learn useful first-step prediction.
2. First-step prediction works, but recursive quality drifts at depth.
3. Recursion remains useful and actually improves speed.

A failure at (1) must not be presented as a finding about (2) or (3).

## Hardware and spend

- One dedicated Shadeform A100 SXM4 80GB using the previously exercised BF16 CUDA runtime. Previously observed rate: $1.38/hour; re-query price/availability before launch. No automatic acceptance of a more expensive offer.
- Keep the occupied RTX 5080 untouched. Do not interrupt B70 serving. The user subsequently authorized deleting B70 model weights and other unnecessary files **if needed**. Read-only inventory found approximately 97 GB of model directories, including 23 GB Glimmer GGUF and 20 GB OpenVINO. These are possible cleanup candidates, **not confirmed unused**; process model-flag inspection alone does not establish that. Inspect serving configuration and preview the exact deletion set before cleanup, preserving active models and requested artifacts. The initial 3.3M-token capture fits the desktop without deletion; B70 cleanup is a contingency for expanded/reusable captures.
- Remaining approved spend: approximately **$70.35**. Initial allocation: **36 instance-hours total**, including setup, data generation/capture, training, evaluation and backup. At the observed rate, this is $49.68; cumulative spend approximately $54.33. Remaining approximately $20.67 is contingency, not authority to launch parallel instances blindly.
- Use a native wall-clock auto-termination cap and shorter stage timeouts. Terminate manually after successful artifact backup. Account for provisioning, idle time and any second instance in cumulative spend.
- Do not assume that a given number of updates fits the allocation. Measure teacher tokens/sec, training updates/sec, RAM/VRAM and transfer speed at the start; revise the schedule before the main run if the estimate exceeds the allocation.

## Dataset proposal

Use prompt/document-level train/validation/test splits, assigned **before** teacher generation and state capture. No random root-level split. Normalize and deduplicate documents/prompts before splitting; keep evaluation prompts separate from training source families. Pin public dataset revisions and record licenses/selection rules. Do not silently substitute a smaller dataset.

Target state capture budget:
- **3.0 million total training sequence tokens**, including prompts and responses.
- **150,000 validation sequence tokens** and **150,000 test sequence tokens**.
- Count actual eligible training roots separately; aim for **at least 1.5 million response/continuation roots** with complete eight-step futures. Token count must not be inflated by prompt tokens that receive no training loss.
- BF16 width-6656 states require 13,312 bytes/token: 3.3 million captured tokens are approximately **43.93 GB**, before small token IDs/metadata. The desktop currently has approximately 72 GB free; reserve at least 20 GB and retain only one full state-copy there. Recheck free space before execution.
- Ordinary text/code plus frozen-Glimmer-generated continuations, so the head sees actual target rollout states rather than only human-written next-token labels. Source mix should cover coding, prose, reasoning and structured output; repetitive/high-entropy cases are explicit validation/test strata.
- Implement a 40% coding / 60% general-instruction prompt mix using `nvidia/OpenCodeInstruct` (CC BY 4.0, its `input` field) and `HuggingFaceH4/ultrachat_200k` (MIT, first user prompt from `train_sft`). Regenerate responses with frozen Glimmer rather than trusting the datasets' original assistant outputs. Pin resolved immutable dataset revisions, retain attribution and source IDs, deduplicate across sources and split before generation. Do not claim that the general-instruction source has exact reasoning/prose/structured quotas; validation/test explicitly cover all six domains.
- Use 512–1024-token sequences under the existing safe context limit. Apply the official Glimmer tokenizer/chat template where applicable; raw continuations are separately labelled. Never double-add BOS or mix chat and raw-format conventions silently.
- Save actual post-final-norm states and tokens in bounded tensor shards; no full-vocabulary teacher-logit archive. Derive teacher logits from saved teacher states and the frozen output head. The saved states must represent the same teacher trajectory as the token sequence.
- Generated tokens require a measured throughput/cost projection. If generation is slower than the budget permits, reduce/alter the proposed data mix explicitly before training; do not quietly turn this into corpus-only training or another tiny run.

## Stage 0 — prove the learning path and profile it

Real pinned Glimmer target, not mocks:
1. Re-run existing cache/projection tests and the real-model alignment canary. Confirm all target/embedding/output parameters remain frozen.
2. Fit a fixed set of 64 actual teacher-root windows with a single rank-128 transition. Use teacher next-token argmax as the small-set diagnostic target. Proposed check: ≥95% training-set top-1 agreement within 1,000 updates; no heldout claim. If it cannot fit this small set, stop and diagnose rather than running a large matrix.
3. Benchmark at least 1,000 meaningful train updates with batch sizes 32/64/128 as memory allows. Choose one feasible common batch size before comparisons; starting proposal **64 roots**.
4. Benchmark teacher generation and state capture on representative sequences, not a one-token input. Extrapolate runtime and disk usage for the full proposed dataset.
5. Verify checkpoint save/resume on the real training CLI and exercise one real validation run. CPU/unit checks supplement this public-boundary test; they do not replace it.

These are prerequisites, not the substantive training outcome.

## Stage 1 — useful first-step prediction

- Train one rank-128 post-normalized gated residual transition, with the target fully frozen, on the substantive dataset.
- Start with **20,000 updates × 64 roots = 1.28 million root exposures**. Count this separately from unique roots and dataset tokens. Do not label repeated/overlapping windows unique data.
- Proposed optimizer: AdamW, initial peak LR 3e-4, weight decay .01, global gradient clipping 1.0, approximately 5% warmup followed by cosine decay. Final settings should reflect the fresh recipe review and real stability profile; do not retroactively tune on test results.
- Common token/distillation objective: `0.25 * CE(actual sequence token) + 1.0 * KL(teacher_distribution || head_distribution)` at temperature 1, with teacher logits derived from saved true post-normalized states using the frozen exact Glimmer projection. The coefficients are our predeclared research choice, not a claimed published recipe. Forward KL provides target-distribution supervision; generated-sequence CE adds observed-token supervision. Log the two terms separately. No full-vocabulary saved-logits product.
- Validate every 1,000 updates on a fixed, source-disjoint validation-root sample: sequence-token CE, teacher argmax top-1 agreement, top-5 recall, teacher KL, normalized state error/cosine, and throughput. Report train and validation separately.
- Run small **actual cached depth-1 speculative decoding** probes every 5,000 updates on validation-only prompts. Offline token accuracy alone is not sufficient.
- Proposed continuation gate: ≥50% first-draft acceptance on the fixed validation decoding probe, with no category entirely failing. This is an engineering gate, **not** the ≥3 acceptance success criterion; 50% alone is not enough for useful deep speculation.
- If improving but below the gate, extend to at most **50,000 updates**, subject to the measured cost/time allocation. If plateaued, try one bounded rank-256 capacity check on the same validation data before declaring the current MLP head inadequate. No architecture sweep or trunk adaptation yet.
- If neither capacity setting learns useful step 1 within its predeclared budget, stop and report that exact result. Do not spend the remainder on an uninformative 4/8-step speed matrix.

Checkpoint selection: use validation metrics only. Keep last and validation-selected checkpoints plus optimizer/scheduler/RNG state. Keep the final test set sealed until the prescribed comparison. Resume must not reset optimizer or data order inadvertently.

## Stage 2 — controlled recurrence comparison

Only after Stage 1 passes:
- Compare **independent blocks**, **one shared block**, and **one shared block plus state supervision**. Start from the same validated one-step checkpoint; initialize each independent block from that checkpoint. This isolates recurrence rather than comparing unrelated bad initializations.
- Use the selected rank, identical root ordering/exposure, optimizer schedule and target teacher policy. Parameter budgets are intentionally different and must be reported.
- Proposed common continuation budget **60,000 updates × 64 roots** per variant:
  - first 10,000 updates: unroll depth 2;
  - next 10,000: depth 4;
  - remaining 40,000: depth 8.
- At depth 8, retain the original per-step token weights `[1,1,.8,.8,.5,.5,.5,.5]`, renormalized to active steps during the curriculum. Supervise every active step; never reset predicted hidden state to teacher hidden after the root.
- All variants use that same CE+KL objective; report them as distilled fixed, shared, and shared+state arms rather than CE-only. State supervision is the isolated additional objective in the third arm; coefficient .2. Preserve the common warm start, token data/order, schedule and loss weights across arms.
- During teacher-forced token training, recursive state feedback is still the head's own state. Do not claim this is self-fed-token training. If actual on-policy head-token training is needed, its teacher states must be recomputed for those changed tokens; pairing predicted-token paths with original states is invalid. Treat that as a subsequent bounded decision, not an invisible objective change.
- Validate at checkpoints on depths 1/2/4/8; retain actual first-step acceptance as a regression check. Track accepted-prefix conditional quality and state drift separately from rejected/unconditional proposal statistics.
- Repeat the fixed baseline and best shared arm with a second seed if the first comparison is promising and contingency allows. A one-seed difference is not a robust architectural finding.

## Stage 3 — real heldout speculative-decoding test

Preconditions: prescribed training finished or documented early stop; validation-selected checkpoints fixed; final test prompts untouched; same target/model revision/backend/hardware and decoding policy for every arm.

- Proposed final test: **60 prompts**, ten each coding/prose/reasoning/structured/repetitive/high-entropy; up to 128 new greedy tokens; two paired timing repeats, randomized/interleaved candidate-versus-baseline order. Maintain context plus output plus verification within the existing safe cache limit.
- Evaluate all three variants at depths 1/2/4/8 against same-target, no-speculation decoding. Use real cached block verification and rollback; no full-prefix fallback in the timed path.
- Metrics: accepted drafts/pass; per-depth conditional acceptance and top-k/KL diagnostics; target calls/generated token; draft and verification latency; decode and end-to-end tokens/sec; peak VRAM; state norms, normalized distance and cosine versus depth; exact output-match/divergence positions; per-domain and pooled results.
- State diagnostics are untimed and sampled on a declared subset, rather than repeating an expensive full-prefix diagnostic on every timing pair. Report whether each curve is chosen-path, accepted-prefix or rejected-prefix conditional.
- Preserve the approved numerical-fidelity diagnostic mode, with strict mode available. **Do not claim lossless equivalence** if baseline/block outputs differ. Report speed separately on exact-output pairs and all diagnostic pairs; equal lengths do not mean equal outputs. Re-run the same-prefix cache/numerical diagnostic on any newly suspicious failure before attributing it to arithmetic.
- Primary success: ≥3 accepted **drafts** per verification pass and >1× measured decode speed. A suggested practically interesting speed gate is ≥1.1× median paired speedup, with uncertainty/domain variation reported; not a production SLA.
- No Qwen port merely because loss decreased. No production/optimized-EAGLE/DFlash comparison claim from this standalone HF path.

Executable planned commands and real E2E expectations are specified below. They are implementation acceptance requirements, **not commands already exercised** for the substantive run.

## Necessary implementation scope before training

Extend the existing standalone experiment, not a new platform:
- Bounded teacher-data capture/loading with proper token/root accounting and source-disjoint splits.
- Checkpoint/resume for head, optimizer, LR schedule and RNG state.
- Configurable active training depth/curriculum and validation intervals.
- Offline teacher-agreement/top-k/KL/state metrics and small real cached decoding validation probes.
- Preserve the exact post-normalized transition and existing verifier/cache semantics; tests cover them plus loading, masking, resume and curriculum boundaries.
- Progress through ordinary training stdout/logs and checkpoint validation results. No new service, dashboard, database, evidence sealing or generalized audit system.

No target finetuning/LoRA, alternating blocks, dynamic stopping, Qwen training, production stack integration or unavailable IQuest training-asymmetry reproduction in this run.

## Follow progress and closure

Before launch, publish the exact cluster name, log command and stage budget. Training logs must show updates completed/planned, root/loss-position exposures, LR, train/validation metrics, validation acceptance and ETA—not just “running.” Report milestone completion only after inspecting actual outputs. Do not rely on a missing background notification to keep claiming a finished job is active.

Retain the selected/last head checkpoints, resumable optimizer state, split/source selections, teacher-state shards within the agreed disk budget, validation histories and final per-pair results. Keep large tensors outside Git; commit/push the approved protocol and small outcome artifacts. Back up before manual termination, verify the instance is gone, and report observed-time cost separately from invoiced cost.

## Pre-launch decision gates

1. Integrate the independent research review if it identifies concrete errors in the selected recipe. The fresh primary sources below already establish a scoped external-research basis.
2. Verify measured generation throughput and actual eligible-root/token/storage counts support the proposed capture before committing to full data generation.
3. Stage-1 gate and extension/stop policy are part of the user-approved staged approach; do not bypass them to fill the budget.
4. Pass the real public-CLI Stage-0 train/resume/validate/evaluate journey below before substantive optimization.

Budget increase is resolved: **$75 total**, not $75 additional. No GPU has been launched for this run.

## Fresh primary-source findings affecting this design

- [DFlash-UltraChat model card](https://huggingface.co/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat), inspected 2026-10-02: assistant responses were regenerated by the target. We adopt this alignment technique, not DFlash architecture/performance.
- [UltraChat 200k card](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/raw/main/README.md): MIT; `prompt_id`, `prompt`, `messages`; 207,865 `train_sft` records. Select prompts, pin the revision, regenerate Glimmer responses.
- [OpenCodeInstruct card](https://huggingface.co/datasets/nvidia/OpenCodeInstruct/raw/main/README.md): CC BY 4.0; five million records; `input` is the coding question. Preserve attribution and select a streaming subset, not its original assistant outputs.
- [Speculators losses](https://raw.githubusercontent.com/vllm-project/speculators/main/docs/user_guide/loss_functions.md): forward teacher-to-draft KL is its default distribution loss; hard teacher-argmax CE differs from sequence-token CE. Native MTP uses its own CE; our hybrid is a predeclared experiment, not that published recipe. Its overlap/acceptance equality concerns distribution-preserving stochastic speculation, not the greedy acceptance metric here.
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html): batching/slicing need not agree bitwise. Preserve fidelity qualification and target acceptance rules.

## Defined public-boundary E2E process — planned, not executed

Environment/preconditions: isolated dedicated A100 GPU runtime outside Pi; pinned official BF16 Glimmer and existing CUDA/Transformers versions; sufficient disk; no unrelated GPU jobs; source-disjoint validation/test prompts. Stage 0 uses the same real generated-data path with smaller token budgets (e.g. 20k/10k/10k), complete future windows, and masked prompt/padding positions. Expected retained outputs: tensor shards/index, head/resume checkpoints, training/validation logs and per-pair real-decoding JSON. Large tensors stay outside Git.

The following additive public CLI names/flags define the implementation interface. `$P` is `scripts/experiments/glimmer_recursive_mtp.py`; `$RUN` is the dedicated cloud experiment directory. They are future acceptance steps, not fabricated observed commands.

```bash
python "$P" prepare-prompts --output "$RUN/prompts.jsonl" --seed 20261002 \
  --coding-fraction 0.4 --max-prompts 12000
python "$P" capture-generated --prompts "$RUN/prompts.jsonl" \
  --output-dir "$RUN/capture" --train-token-budget 3000000 \
  --validation-token-budget 150000 --test-token-budget 150000 \
  --max-prompt-tokens 512 --max-new-tokens 512 --generation-batch-size 4
# First exercise these paths on the small Stage-0 sample, fit diagnostic and profile.
python "$P" train --capture "$RUN/capture/index.json" --output-dir "$RUN/step1" \
  --variants shared-ce --rank 128 --train-depth 1 --batch-size 64 \
  --updates 20000 --schedule-updates 50000 --warmup-updates 1000 \
  --ce-weight 0.25 --kl-weight 1.0 --seed 20261002 \
  --checkpoint-every 1000 --validation-every 1000 --probe-every 5000 \
  --validation-prompts scripts/experiments/glimmer_mtp_validation.jsonl
# Only if still improving but below the gate; preserve the original 50k LR horizon:
python "$P" train --capture "$RUN/capture/index.json" --output-dir "$RUN/step1" \
  --resume "$RUN/step1/checkpoint-last.pt" --updates 50000
# Only after Stage 1 real acceptance passes:
python "$P" train --capture "$RUN/capture/index.json" --output-dir "$RUN/comparison" \
  --variants fixed-ce shared-ce shared-state --init-head "$RUN/step1/checkpoint-best.pt" \
  --rank 128 --batch-size 64 --updates 60000 --schedule-updates 60000 \
  --curriculum 2:10000,4:10000,8:40000 --warmup-updates 3000 \
  --ce-weight 0.25 --kl-weight 1.0 --state-weight 0.2 --seed 20261002 \
  --checkpoint-every 1000 --validation-every 1000 --probe-every 5000 \
  --validation-prompts scripts/experiments/glimmer_mtp_validation.jsonl
python "$P" validate-head --capture "$RUN/capture/index.json" \
  --head "$RUN/comparison/shared-state-best.pt" --split validation \
  --output "$RUN/validation.json"
python "$P" evaluate --record-divergence --capture "$RUN/capture/index.json" \
  --heads "$RUN/comparison/fixed-ce-best.pt" "$RUN/comparison/shared-ce-best.pt" \
          "$RUN/comparison/shared-state-best.pt" \
  --eval-data scripts/experiments/glimmer_mtp_test.jsonl \
  --max-new-tokens 128 --repeats 2 --diagnostic-prompts 6 \
  --output "$RUN/evaluate.json"
```

Stage-0 public-boundary acceptance: refuse incompatible/misaligned data; change only head weights in the tiny-fit diagnostic; reach its teacher-agreement gate; compare interrupted/resumed vs uninterrupted trajectories at equal seed/budget in the same runtime; validate only validation roots/prompts, never test/optimization rows; produce nonempty real cached-decoding results; evaluate the actual target verifier/cache path and explicitly record fidelity. Mock/projection-only/unit paths do not replace this journey. Also verify 102 existing tests plus scoped new masking/sharding/resume/curriculum tests in the isolated runtime.

Cleanup: back up while sufficient time remains; `sky down -y <exact-cluster>`; verify the created provider instance is absent. Retain the small Stage-0 reproduction sample and checkpoints. Any B70 cleanup gets a separate exact-path preview after checking active serving dependencies; never broadly delete `/home/mike/inference/models`.
