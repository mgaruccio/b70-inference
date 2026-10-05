# Native MTP alignment audit and proposed cached-corpus ablation

Status: **source audit and experiment design only**. No objective implementation, new rental, live parity trace or ablation training has been launched. This document follows the full-corpus CE-only report; it is not a claim that distillation fixes serving acceptance.

## Audit findings

The full corpus was captured through native **MTP4 speculative verification**, not target-only replay: `run-pilot.py` enables the native-capture patch, launches depth 4 for capture, and runs the native corpus generator. The verifier still uses the full unquantized BF16 target. Accepted target hidden rows are retained; rejected draft suffixes are not teacher targets.

Source references below refer to repository commit `17b72e58` (trainer unchanged from `1daf7b31`), the runtime patch directory `patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/`, and the published workflow snapshot in this directory. **The directory name is not the full run's runtime version:** the full run used the pinned vLLM 0.27.1 image in `pilot-protocol.json`. Source comments/tests derived from another pinned build cannot alone establish parity with that image.

| Boundary | Observed evidence | What remains unproven |
|---|---|---|
| Target hidden normalization | `b70_mtp_training.py:21-29` places capture after target final norm; native capture rejects MTP-specific hidden overrides. Trainer `qwen38_train_mtp.py:216-225` separately normalizes embedding/hidden and concatenates embedding first. The v0.27.1 MTP source confirms that order and a final MTP norm. | This establishes the intended tensor boundary, not numerical equality of HF and vLLM kernels or every post-rejection row. No evidence justifies switching cached inputs to pre-final-norm. |
| Base token/position alignment | Trainer `aligned_inputs` uses `(x[j+1], h[j], p[j]) -> x[j+2]`; positions are not independently shifted/rebased. Native capture `:211-216` validates actual GPU positions against the committed contiguous stream. | `test_alignment_matches_installed_vllm_first_pass` (`tests/test_qwen38_train_mtp.py:277-304`) executes installed source only when vLLM is available. It was skipped in the reported trainer tests. It is not a passed live-image parity gate. |
| Recursive token inputs | Trainer `sequence_depths:483-485` uses ground-truth `tokens[root+d]`; serving uses previous sampled drafts. | **Concrete distribution gap:** deeper training is teacher-forced, not free-running. Auxiliary state/KL terms on the same recurrence retain this gap. |
| Recursive position/cache behavior | Trainer uses `positions[root] + d - 1`, copies the root prefix KV, and propagates previous MTP state. Existing reference checks exercise HF sequential-cache behavior. | Live proposer branch positions, rejection/slot reuse, and partial-acceptance row selection need validation against the exact serving image. |
| Captured row selection | Native capture `:244-258` takes `keep = len(outputs)`, checks accepted draft IDs, then copies `hidden_states[:keep]`. | Need to compare these rows with the actual proposer-selected inputs after partial rejection, not merely prefill. |
| State/KL diagnostic | `run-pilot.py:178-201` calls teacher-forced `sequence_depths`, compares depth-d/root-r output with target hidden row `r+d`, and computes teacher-to-head KL. | Only four dev captures / 32 roots. This is not free-running drift, general HF/vLLM parity, or speculative acceptance. |

No definite normalization or base-label off-by-one bug was found. The demonstrated mismatch is the branch input distribution. The live parity questions are blockers for interpreting state loss as an exact serving-state target, not evidence that a specific parity bug exists.

**Audit corrections:**

- The proposed KL below is `KL(teacher || student)`. Reversing it is a different loss and does not reproduce the current diagnostic.
- `[441270,29920,29920,29920]` are supervised counts across the **entire ten-epoch CE run**, not per epoch. State/KL pairs require observed future rows, so their count may be smaller and must be reported separately.
- “Shared embedding/head” means MTP reuses the target's embedding and output head, each frozen; it does not mean embedding and LM-head tensors are tied to one another.

## Proposed experiment (not launched)

### Tier, baseline and scope

Development tier, unchanged CUDA BF16 target and native shared-head architecture. Reuse all archived **374 train / 64 dev** captures; preserve canonical IDs/messages and the original 64 held-out test requests. No reduced pilot, quantized teacher, regenerated corpus, persistent launcher change, production promotion, or extra capture infrastructure.

Baseline: the published CE-only full-data run, trained from stock for ten epochs with minimum dev CE selecting step 1,496. For treatment arms, initialize from the **same stock head**, not the selected candidate; otherwise objective and initialization both change. Existing CE-only evidence may be reused if stack/data/seed remain matched. If a parity correction changes the training path, rerun the full-data CE control as well and label it a new comparison.

Keep LR 1e-6, seed 42, accumulation 1, four recursive depths, eight roots, depth weights `[1,1,0.8,0.8]`, max length 2,048, logits chunk 64, and **ten complete epochs / 3,740 optimizer updates per trained arm**, with validation every 374 updates. Frozen target, embedding and output head; no new trainable parameters. Never replace complete epochs with an arbitrary short update budget.

### Loss arms

1. **CE-only control:** existing loss, auxiliary weights zero. Require unchanged outputs/gradients/report behavior on the zero-weight path.
2. **Primary CE + KL:** proposed `lambda_kl = 1`, `T = 1`, no state term. For student state `s` and detached captured teacher state `h`, form logits using the same frozen BF16 output head `W`:

   `L_KL = T^2 * sum_v p_teacher(v) * (log p_teacher(v) - log p_student(v))`

   where `p_teacher = softmax(W h / T)` and `p_student = softmax(W s / T)`. Compute softmax/log-softmax and accumulation in FP32, with chunked logits to bound memory. Do not substitute hard pseudo-labels or a quantized teacher. Detach teacher logits/states, not the student recurrence.
3. **Separate secondary CE + state arm:** proposed `lambda_state = 0.1`, no KL, using `mean((s-h)^2) / clamp(mean(h^2), eps)` per valid pair. This relative MSE penalizes amplitude as well as direction; it is not unit-normalized MSE/cosine. Raw future-state equality is a regularization hypothesis, not guaranteed by equal shape or normalization boundary. Keep this arm separate to identify the term responsible for any effect.

Coefficients are **proposed fixed values**, not implemented or tuned results. Do not search them using held-out test performance. A combined state+KL arm, temperature sweep, sampled-token recurrence, or additional updates would be separate experiments, not silently folded into these arms.

### Teacher rows and masks

At base row `j`, output state `s[j]` predicts token `x[j+2]`. Its teacher-state comparison is `h[j+1]`, whose frozen-head logits predict the same token. For branch depth `d` at root `r`, compare to `h[r+d]`; the label remains `x[r+d+1]`. This is the existing diagnostic's mapping, pending live validation.

Apply CE to the original supervised mask. Add auxiliary loss only where the **future hidden row is actually observed**, finite, in bounds, and corresponds to the supervised token. Never fabricate the last missing hidden row, pad it, or discard whole training sequences just because one auxiliary pair is unavailable. For four-depth paired diagnostics require `r+4 < observed_hidden_rows`. Count/report valid pairs independently at each depth.

Preserve existing per-depth CE normalization and weights. Normalize each auxiliary term over its own valid pair count; zero valid pairs give zero auxiliary loss, not NaN or a dropped record. Preserve one connected backward through recursive state and base KV. Masked positions and teacher tensors must receive no gradients. Future ground-truth teacher states are **not** valid targets for arbitrary sampled draft histories, so adding free-running branches needs a different design rather than blindly reusing these targets.

### Validation and checkpoint selection

Retain minimum weighted dev **BF16-export CE** among stock and all ten epochs as the selection criterion for direct continuity; the new auxiliary training objective must not silently redefine `dev_selection`. Record CE, KL, relative-state loss, norm/cosine diagnostics and pair counts separately. Evaluate all 64 dev captures; no subset substitutes for validation. The four-capture drift diagnostic can remain a secondary continuity diagnostic, explicitly labeled as such.

The 64 test requests have already been inspected in the CE-only report. They remain disjoint from gradient/checkpoint selection, but are now a **reused held-out serving regression set**, not a newly untouched blind discovery set. Do not choose coefficients/depth/checkpoints from them or claim independent confirmatory evidence from reuse.

## Defined real-system test process before implementation/launch

### Preconditions and data

- Restore the seven ordered R2 archive parts by streaming extraction, verifying the existing part/whole hashes; avoid a second local 14GB copy. Retrieve the same model revision and captures into an isolated benchmark environment outside interactive Pi.
- Use the exact serving image and compatibility environment from `pilot-protocol.json`; verify physical unquantized BF16 operation. The earlier lease is deleted. Fresh compute/budget authorization is needed for live validation and full training; do not invoke the historical provider creation script while merely publishing this design.
- Keep the owned instance idle of unrelated GPU work and use disposable containers only.

### Gate A: live alignment/fidelity, before treating auxiliary targets as valid

Use the existing disposable capture/overlay path and actual `/v1/completions` or `/v1/chat/completions` boundary. Run fixed public train/dev examples through stock native MTP at depths 4 and 8, plus no-spec and a complete stock-weight overlay control. The small trace set is a **diagnostic**, not a replacement training/evaluation corpus.

Retain the actual installed proposer/model source and exact request/server argv. Inspect prefill, full acceptance, partial rejection, and zero draft acceptance where observed. Compare input token IDs, GPU positions, selected target hidden rows, MTP feedback hidden, and KV prefix/slot boundaries at each branch with the offline reconstruction. Existing capture records supply target rows; if existing hooks do not expose the live proposer inputs needed for this comparison, record that as a blocker and obtain approval for the smallest disposable trace hook—not a new persistent tracing service.

Expected: exact token/index/position agreement for the same committed history and branch prefix; correct post-final-norm boundary and embedding-first order. Establish and declare a numerical tolerance for HF/vLLM comparisons **before examining candidate results**. Require identical discrete argmax predictions on the trace or report disagreements explicitly. If rejection cases are not observed, say they are untested rather than declaring parity. Stock-versus-no-spec repeat nonidentity from the CE run must be diagnosed/reported; do not require or claim a bitwise identity gate passed when it did not.

### Gate B: objective implementation and full-data trainer CLI

Only after Gate A, implement default-off objective terms in the existing trainer. Targeted tests must cover teacher-to-student KL direction, temperature scaling, future-row indexing at all depths, finite extreme logits, masks/missing terminal rows, detached teacher/frozen shared weights, zero-weight equivalence, and recursive gradients.

Exercise the **real trainer CLI** on the existing captures, then each authorized arm for the complete 374/64 ten-epoch schedule. A one-update smoke checks memory/gradient/export only; it does not substitute for full training. Expected: 3,740 actual updates, all 374 sequences per epoch, all 64 dev validations at each epoch, correctly selected BF16 checkpoint, complete pair counts and no teacher/head changes. Exact auxiliary CLI argv is pending implementation; no nonexistent runnable flags are asserted here.

### Gate C: serving comparison at the public API

For each selected candidate, use the same nine-cell sequence as the CE run: no-spec once, then stock/candidate/candidate/stock at depths 4 and 8, all 64 requests per cell. Keep sampling, token budget, cache policy, warm-up, target/overlay loading and sandbox checks unchanged. Retain exact rendered tokens, response tokens/SSE, finish reasons, per-request timings and speculative counters; acceptance excludes bonus tokens.

Report pooled acceptance denominators, per-position acceptance, median decode/E2E speed, output lengths, functional scores, stock/candidate repeat identity and no-spec identity. Do not claim a material gain based on a ~1% timing delta without uncertainty/repeat evidence, or interpret teacher-forced auxiliary metrics as acceptance. If standard qualification is desired later, BetterBench, `vllm bench serve`, long-context, larger fidelity and fixed quality-suite work remain separate unmet requirements.

### Cleanup and failures

Retain failures, exact environment/source/commands, captures, all checkpoints and raw results in the existing artifact location. Verify complete remote archive readback before requesting deletion of only the owned instance; confirm subsequent provider absence. Never stop unrelated workloads, reuse private signed manifests in publication, change persistent launchers, or pool this with quantized B70 results.

## Primary research affecting this design

- [Pinned vLLM v0.27.1 Qwen3.5 MTP source](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/model_executor/models/qwen3_5_mtp.py): separate hidden/embedding norms, embedding-first concat, shared target-weight loading and final MTP norm; directly inspected for this audit.
- [vLLM proposer source](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/spec_decode/llm_base_proposer.py): validate against the installed pinned source, not a moving-main reference; live branches/rejection still require measurement.
- [Qwen3.8 public config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json): native MTP metadata, not a complete original training-target recipe; the run's exact model revision remains pinned in its protocol.
- [ONNX Runtime Qwen MTP explanation](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/qwen-3.6-mtp.md): supports the post-final-norm and shifted-token interpretation, but concerns an adjacent Qwen version/runtime and does not prove this run's parity.
- [PyTorch distillation tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html): teacher-to-student KL, detached teacher, temperature-squared scaling, and separate state-matching hypotheses.

## Decision

Prepare KL as the first objective ablation, with state matching separate. **Do not launch on the strength of this source audit alone.** Close the pinned-runtime live trace gaps first, then execute the authorized full-data schedule; if capacity/authorization is unavailable, report the blocker rather than substituting a pilot or quantized teacher.
