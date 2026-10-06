# Qwen3.8 serving-like recurrence: proposed experiment

Status (2026-10-06): **design and local baseline checks only**. User chose serving-like recurrence, starting with design/correctness validation before new paid compute. No new training implementation, rental, optimizer updates, serving cells or promotion is authorized by this document. Development tier only.

## Why this experiment

The [saved-checkpoint sweep](../results/20261006-qwen38-kl-dev-checkpoint-sweep/README.md) did not establish a better head than KL step2618: its 0.0645% acceptance lead over step1870 was smaller than identical-head variation. More identical epochs are not the next intervention.

The existing trainer has a concrete distribution gap: `sequence_depths` consumes ground-truth `tokens[root+d]` at later depths (`scripts/experiments/qwen38_train_mtp.py:521-551`), while native serving consumes the previous proposed token and MTP state. The live checker explicitly validates this feedback (`scripts/experiments/qwen38_mtp_live_parity.py:245-247`). This is one recurrent native MTP layer, not independent parallel heads; the trainer rejects models with `mtp_num_hidden_layers != 1` (`:103`).

This proposal fixes **within-block draft histories**, not the entire distribution of accepted serving contexts. Roots still come from the full archived corpus. Training on newly accepted/rejection-corrected contexts would be a separate experiment.

## Exact conditional targets

Use the existing zero-based indexing. At root `r`, let `c = x[:r+2]`, including the known seed token `x[r+1]`. Captured target state `h[r]` and that seed drive the first MTP prediction. Let `s_d` denote MTP output state and `y_d` the greedy draft:

- `s_1 = MTP(x[r+1], h[r], p[r], root_prefix_cache)`; predicts `y_1` at original token index `r+2`.
- `y_d = argmax Process(W s_d)`, using the serving proposal policy, detached as a discrete token choice.
- `s_d = MTP(y_{d-1}, s_{d-1}, p[r]+d-1, branch_cache)` for `d >= 2`.
- Teacher distribution `P_d = Teacher(next_token | c + y_1 + ... + y_{d-1})`.

Thus `P_1` corresponds to captured `h[r+1]`, and `P_d` corresponds to captured `h[r+d]` **only when the entire preceding draft history matches the captured token history**. Equality of the current token alone is insufficient after an earlier divergence. Preserve the native MTP position convention; do not shift it to the target token position.

Generate a whole greedy block with the current student, then run the frozen, unquantized BF16 target causally on `c + y[:K-1]`. The final `K` target rows predict exactly `y_1..y_K`; gather post-final-norm rows and use the frozen shared output head, with matching logits processing. This needs the **full target model**, not only its captured states/head.

After a mismatch, remaining teacher rows are valid counterfactual proposal-prefix targets. They are not the verifier's accepted continuation. Record the first greedy rejection; discard the rejected suffix when constructing any next actual serving context. Do not substitute the corrected target token into the history of that same rejected proposal block.

Initial target replay should recompute each root prefix from a fresh target state. Qwen's hybrid target has recurrent state as well as attention KV; copying/cropping only attention KV is not a safe target-cache fork. Optimizing teacher-prefix reuse is deferred until equivalence is demonstrated.

### Completed source/cache audit

- Student root cache is the MTP's own rows `[0, r]`, length `r+1`, with a fresh branch object per root; append the earlier draft nodes at successive depths. The target hybrid cache is separate. Single-row MTP decode attends every supplied KV, so future/sibling cache leakage is invalid (`prefix_cache:511-518`, `NativeMTP.forward:222-239`). Verify the pinned Transformers cache append semantics and unchanged base tensors explicitly.
- Native corpus capture retains only accepted verifier rows, `keep = len(outputs)`; its metadata's `source=native-quantized-verifier` is a generic label, not evidence this BF16 run was quantized. Its hidden rows cannot supervise divergent histories. A partial-rejection first pass selects `keep-1`, not the last scheduled row; retain this contract when reconstructing roots.
- Require contiguous native positions and `teacher_position = student_query_position + 1`. Legacy gapped positions are accepted by the old record validator, but must be rejected on the new branch path. Confirm the restored 374/64 corpus satisfies this guard; do not silently reinterpret legacy RoPE positions.
- An existing exact-token replay capture contract already exists in `b70_mtp_training.py:181-270`: speculation off, prefix caching off, one active request, `/v1/completions` with token-ID `prompt` and `max_tokens=1`, post-final-norm prompt rows. Native capture and target-only replay capture are mutually exclusive; use separate disposable cells.
- Reuse that native target-only mechanism for the correctness gate, not a new service/API. The `qwen38_mtp_replay.py` CLI currently accepts completed generation cells, not arbitrary branch tensors: its disposable caller needs a minimal adapter for exact branch IDs. Do not fabricate generation summaries or claim the unchanged CLI already supports online branches.
- Replay capture is capped at **128 sequences per process** and a configured total token budget (`b70_mtp_training.py:18-19,184-185`). It is a bounded validation mechanism, not an unchanged backend for approximately30,000 training-root replays per arm. Before full training, choose and validate a direct frozen-target backend within the trainer or account for bounded process restarts. Do not silently widen collector limits or build an unrequested scalable capture service.

## Minimal controlled treatment

Proposed, not implemented:

1. Preserve the existing teacher-forced CE anchor at every depth, with its original labels, masks and normalization. Never apply ground-truth CE labels to a divergent free-running branch.
2. Preserve depth-one captured teacher KL.
3. Replace the *deeper KL branches* with current-student greedy recurrence and teacher targets recomputed on their exact draft histories. Keep a separate teacher-forced graph for the unchanged CE anchor.
4. Keep forward KL `KL(P_teacher || P_student)`, temperature1, coefficient1, depth4/eight roots, and weights `[1,1,.8,.8]`. No reverse KL, state matching, rejected-suffix decay, depth8 training or LR sweep in this first treatment.
5. Generate proposals with a BF16 serving-export view of the current MTP parameters. Detach token choices and all teacher computations, but preserve gradients through the student recurrent states, parameter casts and student prefix KV. Validate this export view against the existing BF16 export/reload boundary; training-FP32 argmax is not automatically serving argmax.
6. Keep the feature default-off. Off must retain the old outputs, gradients, reports and dependency-loading behavior. No new serving API, persistent launcher, service or generalized artifact infrastructure.

The change is a **deeper-KL branch/history and teacher-target intervention**, not a claim that all training is on-policy. When drafts match ground truth, the treatment should reduce to the existing CE+KL objective within declared numerical tolerance; test this with identical weights and roots.

Greedy proposal/teacher processing must match the existing serving configuration. Distillation temperature1 controls the loss, not the serving sampler's temperature0. Stochastic proposals or relaxed verification are out of scope.

## Full experiment scope and controls

- Restore all **374 train / 64 dev** captures and the pinned original BF16 target. Preserve canonical IDs, seed42, LR1e-6, max length2048, logits chunk64 and accumulation1.
- If authorized, train **two matched arms from the same stock initialization**: existing CE+KL control and recurrence treatment. Ten complete epochs / 3,740 optimizer updates per arm; checkpoint every374 updates. Do not substitute reduced data, a shorter schedule, a quantized teacher or a warm-start treatment against a stock-start control.
- Historical step2618 remains the serving reference; historical training/throughput is contextual evidence, not a substitute for matched same-lease controls.
- Report sequences, updates, root counts, teacher replay counts, divergent branches, first-rejection positions, KL pairs and loss normalization separately by depth. Failures/OOMs remain in the record. Preserve existing auxiliary missing-row behavior on the unchanged captured path; do not fabricate target rows.
- Predeclare checkpoint selection by accepted draft tokens/pass, bonus excluded, on **all64 dev requests**. Evaluate stock plus all ten epoch heads per arm at D8; exact ties use E2E median then earlier step. This selection has substantial serving cost and must be included in the budget.
- No coefficient/depth/checkpoint selection using test requests. The prior64 test requests are a reused held-out regression set, not a new blind confirmation set.
- Compare selected treatment against the matched selected control and incumbent2618 using fresh interleaved repeats on the same lease. Keep no-spec and stock controls. Acceptance, E2E/decode speed, functional quality and exact-output fidelity are separate metrics.
- Define paired request-level analysis before launch; bootstrap entire requests, not correlated tokens/passes. Retain raw counts and per-cell results. Small positive differences without uncertainty/repeat evidence are not a demonstrated win. Do not pool CUDA BF16 with B70 quantized results or across leases.

## Required correctness and real-system test process

This process must be finalized with executable commands **before implementation**. Existing public boundaries are the trainer CLI/export, parity client and native `/v1/chat/completions` serving endpoint. No new endpoint is needed.

### Local fixtures (supplementary, not real-model proof)

- Force draft equality, then divergence at depths1/2/3. Check exact teacher prefix tokens, target-row/label offsets, absolute MTP positions and masks. Equality must recover existing KL; an early divergence must prevent reuse of every later captured teacher row.
- Compare batched causal teacher scoring with fresh-prefix sequential scoring. Test independent root caches, branch order and repeated roots; detect sibling cache leakage.
- Check off-path identity; frozen teacher/embedding/head; detached token choices; deeper student-state/base-KV gradients; chunked vs dense loss/gradient equivalence; BF16 export/reload proposal equality.
- Cover empty roots, short observed captures, boundaries, nonfinite replay targets and zero valid pairs. Document EOS handling from the pinned proposer before implementing it; do not invent an EOS cutoff that silently changes fixed-depth proposal behavior.

### Real-model gate (requires separate compute authorization)

Environment: isolated external benchmark runtime, owned GPU with no unrelated work; unquantized `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; pinned vLLM0.27.1 image `vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`; existing CUDA compatibility/V1 runner settings. Recheck availability, price and memory before authorization/launch.

1. Use the existing parity request/check CLI against disposable native D4/D8 servers, stock and incumbent2618. Use deterministic canonical train diagnostics plus all64 dev requests for serving regression; cover prefill, all-accepted, partial and zero acceptance.
2. Retain the actual proposal token histories, post-final-norm target rows (including rejected verification rows), per-depth MTP inputs/outputs, positions and slot mappings using the existing disposable hook.
3. Replay those *same draft histories* through the frozen target; compare every teacher row/logit with native verifier rows. Compare full-prefix versus incremental execution across rejection/corrected continuation, including hybrid recurrent state. Force branch divergence in diagnostic scoring if live requests do not cover it; do not count that diagnostic as serving throughput.
4. Compare the proposed training branch and BF16 export/reload against native per-depth proposal tokens/state/positions/cache behavior. Retain frozen-head argmax and top-two margins as well as numeric errors. Student MTP cache and target hybrid cache are different objects.
5. Retain no-spec exact-output/functional controls and stock-overlay identity. The previous BF16 near-tie exceptions and no-spec divergence remain disclosed; they do not authorize fresh exceptions. New disagreements stop the gate for diagnosis/user decision. Do not report strict parity or exact-distribution preservation when those gates remain open.
6. Profile real full-prefix teacher replay with the actual corpus length/root distribution. Estimate complete two-arm training, dev checkpoint selection and repeat serving costs. Do not promise the historical four-hour cap can fit this substantially more expensive objective.

Correctness-only validation is **zero full-corpus training updates**, not a shortened substitute for the full experiment. Actual full training needs a separately approved complete-schedule budget after profiling. Retain exact commands, versions, inputs, exports, raw trace/metrics/output artifacts and every failed/completed cell in the existing benchmark archive workflow. Verify archive readback, delete only the owned lease, and confirm provider absence. No promotion.

This is development preparation, not a standards-complete performance result. BetterBench, long-context/concurrency qualification and the full publication checklist remain required before community-comparable claims.

## Evidence obtained in this design session

Current source/report baseline verified at commit `f4685282`. Existing targeted CPU contracts ran in the external ML environment:

```sh
cd /home/mike/code/b70-inference
/home/mike/b70-evals/20261002-glimmer-mtp-training/state-check-env/bin/python -m pytest -q tests/test_qwen38_train_mtp.py -k 'real_hf_sequential_cache or depth_four_loss_reaches or kl_teacher_rows'
```

Observed: **4 passed, 78 deselected in 2.76s**, exit0. These cover existing HF sequential/sliced-cache behavior, recursive gradient connectivity and captured teacher-row indices; they do **not** test a new on-policy implementation, full target replay or live serving. New captures0, new full-corpus updates0, new serving cells0. No new lease was created.

Design verification also checked the local report link/code fence and **208 symbolic teacher-prefix/row mappings** across roots0..15 and depths1/4/8. All passed. This is index algebra, not model execution or evidence of live parity.

## Fresh primary research affecting the proposal

- [Pinned Qwen MTP source, vLLM0.27.1](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/model_executor/models/qwen3_5_mtp.py), fetched in this session: separate embedding/hidden norms, embedding-first concat, one step-indexed layer interface and shared logits head. Model-specific source and live calls override generic independent-head MTP descriptions.
- [Draft-OPD](https://arxiv.org/html/2605.29343v2): score draft-induced prefixes, including rejected proposals. Its reverse-KL/rejected-suffix weighting choices are additional hypotheses, deliberately not included here.
- [Speculative decoding algorithm](https://arxiv.org/html/2211.17192): distinguish proposed histories from rejection-corrected accepted continuations; stochastic exact sampling requires the correct acceptance/residual rules. This experiment retains existing greedy native verification.

## Readiness / next authorization

Design prepared and source/cache audit completed; baseline CPU contracts passed. **Not implementation-ready or training-ready:** new teacher replay/export equality tests, actual restored-corpus position checks, a budgeted full-training teacher backend, finalized executable real-system steps and the real-model gate remain required. No performance gain is claimed.

Next proposed authorization: implement the default-off path and tests, with an explicitly capped **correctness/profiling-only** GPU lease to execute the real-system gate. Obtain its budget separately; then propose the full-corpus experiment budget using observed costs. Do not invoke old lease-creation scripts merely to prepare this design.
