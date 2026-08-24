# P0 greedy ~8k divergence tracer and MTP committed-history state machine

**Status:** research specification only. This artifact does not implement the tracer, change a runtime, replace the live BF16 service, or claim a speed win.

**Scope:** P0 (the greedy ~8k divergence tracer) and GPT-pro backlog item 2 (MTP committed-history / XPU GDN state semantics). Items 1, 3–14 are out of scope except for the dependency map in Part B.

## Authority, pins, and evidence labels

The authoritative request and non-goals are the spawn prompt. The frozen serving/evaluation contract is unchanged: 98,304 input / 32,768 output, 131,072 serving ceiling, and no Harbor/Prime cohort in this slice.

Pinned stack:

- image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
- vLLM: `0.27.2rc1.dev77+gac7509e2b`, commit `ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9`
- `vllm-xpu-kernels`: `0.1.12.3`; the closest published source tag is `v0.1.12`, commit `1796aa8bc8db4ac68d9cd19636cef88f3af81d2b`
- model: `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` at `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`

Every statement below is marked or written as one of:

- **Measured:** observed in the last-session artifacts or handoff.
- **Source-derived:** directly visible in the pinned source or the cited kernel source.
- **Hypothesis/unknown:** an interpretation that must be tested by the tracer or by a single-request state trace.

## Last-session facts that motivate P0

**Measured** in `docs/qwen38-b70-next-session-handoff-20260823.md` and the two progress documents:

- BF16 and draft-INT4 matched on the fixed short and coding controls.
- The ~8k control did not match, although both responses finished by `length`.
- The candidate under the quality gate was draft-INT4, graph capture cap 64, MTP-4; it must not replace BF16 until the token-level cause is known.
- The existing output artifacts contain `token_ids: null`, so they establish output-level mismatch but cannot identify a proposal, target, acceptance, or state boundary.

The exact comparison artifacts are:

- `~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T205000Z-greedy-bf16/`
- `~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T205000Z-greedy-gdn/`
- `~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T205000Z-greedy-comparison.json`

The `8k.json` files report `prompt_tokens: 8459`, `completion_tokens: 128`, and `finish_reason: "length"`; `comparison.json` reports `content_equal: false` for this case and equality for short/coding. These are facts about that run, not a claim about all prompts.

---

# Part A — P0 tracer specification

## A.1 Diagnostic question and non-goals

The tracer must answer one question for one deterministic request:

> At the first generated position where BF16 and draft-INT4/cap64/MTP-4 diverge, did the draft proposal differ, did target verification differ, did rejection/bonus bookkeeping differ, or did a prior partial-accept GDN/KV state become different?

It is an observational diagnostic. It must not:

- alter target logits, sampler outputs, accepted-token counts, slot mappings, GDN arguments, or graph capture;
- replace `GPUModelRunner` semantics with a proxy path;
- dump full 248k-vocabulary logits on every step by default;
- run Harbor/Prime or a cohort;
- touch the live B70 server.

A diagnostic trace may pay synchronization and logging overhead. Its timings are not performance measurements.

## A.2 Exact hooks and records

The following hooks are the smallest complete path. A future Composer implementation should wrap or instrument these functions in a disposable runtime overlay and leave the surrounding control flow byte-for-byte equivalent.

### A.2.1 Batch/context and draft-token identity

| Phase | Exact source hook | Record |
| --- | --- | --- |
| Round entry | `vllm/v1/worker/gpu_model_runner.py:GPUModelRunner.execute_model` and its call to `_prepare_inputs` ([pinned source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/worker/gpu_model_runner.py#L4277-L4361)) | `run_id`, request id and stable batch row, graph mode/cap, MTP depth, `context_len` **before** the speculative forward, scheduled token count, and the CPU `scheduler_output.scheduled_spec_decode_tokens` list. The scheduler list is the authoritative per-request draft-token input for alignment. |
| Spec metadata | `GPUModelRunner._calc_spec_decode_metadata` ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/worker/gpu_model_runner.py#L2909-L2982)) | `num_draft_tokens`, `cu_num_draft_tokens`, `draft_token_ids`, `logits_indices`, `target_logits_indices`, and `bonus_logits_indices`. Clone only the small ID/index tensors to CPU for a trace. This function maps the scheduled input rows; it does not itself decide acceptance. |
| MTP proposal | `vllm/v1/spec_decode/step3p5.py:Step3p5MTPProposer.propose` and `_sample_draft_tokens_for_step` ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/spec_decode/step3p5.py#L259-L348), [continuation](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/spec_decode/step3p5.py#L394-L463)) | For each `spec_step_idx`, the proposal token IDs, request row, and input/output hidden-state shapes. The proposer runs MTP steps 0…3 for MTP-4 and stacks the resulting IDs. |
| Optional draft-logit probe | `vllm/model_executor/models/qwen3_5_mtp.py:Qwen3_5MTP.compute_logits` ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/model_executor/models/qwen3_5_mtp.py#L279-L299)) | Only for the first mismatch or an explicitly requested sample: draft top-1 ID, a small top-k ID/value slice, and `spec_step_idx`. Do not materialize or serialize the full vocabulary every round. `Qwen3_5MultiTokenPredictor.forward` is the optional hidden-state boundary ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/model_executor/models/qwen3_5_mtp.py#L139-L183)). |

`context_len` means the committed/request-visible length at the start of the round, not the optimistic length with all drafts appended. Prefer `self.input_batch.num_computed_tokens_cpu` / the corresponding request state after scheduler correction. Record prompt length separately. This avoids falsely aligning a candidate that accepted a different prefix.

### A.2.2 Target verification and rejection

| Phase | Exact source hook | Record |
| --- | --- | --- |
| Target forward boundary | `GPUModelRunner._model_forward` ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/worker/gpu_model_runner.py#L3946-L3976)) plus the target `self.model.compute_logits(sample_hidden_states)` call in `GPUModelRunner.execute_model` ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/worker/gpu_model_runner.py#L4548-L4590)) | Map target rows through `SpecDecodeMetadata.target_logits_indices` and `bonus_logits_indices`. For greedy tracing, record `target_ids = argmax(target logits)` for every drafted position and the target bonus ID. Record a small logit/top-k slice only around the first mismatch. |
| Rejection entry | `GPUModelRunner._sample` ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/worker/gpu_model_runner.py#L3751-L3780)) and `vllm/v1/sample/rejection_sampler.py:RejectionSampler.forward` ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/sample/rejection_sampler.py#L38-L201)) | Capture the draft IDs, target-row IDs, bonus ID, and the returned `sampled_token_ids` tensor before any output cleanup. |
| Greedy decision | `vllm/v1/sample/rejection_sampler.py:rejection_sample` and `rejection_greedy_sample_kernel` ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/sample/rejection_sampler.py#L394-L469), [kernel](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/sample/rejection_sampler.py#L715-L769)) | For `temperature=0`, the kernel compares each draft ID to target argmax in order, emits the target ID at the first reject, fills the rest with `-1`, and emits the bonus only when every draft is accepted. This is the canonical acceptance event. |
| Commit count | `GPUModelRunner._update_states_after_model_execute` ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/worker/gpu_model_runner.py#L1613-L1669)) | Record `(output_token_ids != -1).sum(dim=1)` before mamba postprocess as `num_accepted_tokens`. It is the number of committed output positions for the next state step: a first reject after `k` draft tokens normally yields `k+1` (the target replacement), while a full accept of `n` drafts yields `n+1` including the bonus. It is not identical to `accepted_prefix_len`. |

The tracer must derive `accepted_prefix_len` from the contiguous accepted draft prefix, not from the total output width. For a greedy event:

```text
accepted_prefix_len = first i such that drafted_ids[i] != target_ids[i]
                       or len(drafted_ids) when all positions match
```

The first rejected position's target ID is a committed replacement token; `bonus_token` is non-null only for a full draft accept. For random sampling, retain the same fields but treat the greedy equality rule as invalid and use the sampler's accepted/recovered output markers.

### A.2.3 GDN and state-boundary correlation

These hooks are needed to distinguish an acceptance error from a state error. They should log metadata and indices, not full state tensors on every round.

| Exact source hook | Record |
| --- | --- |
| `vllm/v1/attention/backends/gdn_attn.py:GDNAttentionMetadataBuilder.build` ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/attention/backends/gdn_attn.py#L210-L367)) | `num_spec_decodes`, `num_spec_decode_tokens`, `spec_query_start_loc`, `spec_token_indx`, `spec_state_indices_tensor`, and `num_accepted_tokens` passed to the current target forward. |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:QwenGatedDeltaNetAttention.forward_xpu` ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L953-L1000)) and `vllm/_xpu_ops.py:_gdn_attention_core_xpu_impl` ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/_xpu_ops.py#L116-L201)) | Boundary-only call record: layer name, active token count, state-index shapes, and accepted-count values. Do not change the `torch.ops.vllm.gdn_attention_core_xpu` or `_xpu_C.gdn_attention` ABI. |
| `vllm/v1/worker/mamba_utils.py:preprocess_mamba` ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/worker/mamba_utils.py#L1158-L1263)) | `prev_state_idx`, `curr_state_idx`, accepted-token bias (`num_accepted_tokens - 1`), source/destination block, and whether a copy is scheduled. |
| `vllm/v1/worker/mamba_utils.py:postprocess_mamba_align_gpu` and `MambaSpecDecodeGPUContext.run_fused_postprocess` ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/worker/mamba_utils.py#L859-L1016), [caller](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/worker/mamba_utils.py#L1310-L1363)) | The postprocess decision, source/destination state columns, `accept_token_bias`, and the postprocess accepted-count value. |

A state checksum is optional and only needed for the first divergence or a forced reject/full-accept probe. If enabled, checksum per layer/state type and selected state slot; never serialize the whole recurrent state in the normal trace.

## A.3 Minimal JSONL event schema

Emit one event per request per speculative verification round. Keep the schema stable enough that BF16 and candidate traces can be joined without knowing their batch order:

```json
{
  "schema": "qwen38-b70-p0/v1",
  "run_id": "20260824T...-gdn-cap64-mtp4",
  "variant": {"draft": "bf16", "graphs": "cap64", "mtp": 4},
  "request_id": "single-8k",
  "step": 0,
  "context_len": 8459,
  "drafted_ids": [101, 102, 103, 104],
  "target_ids": [101, 102, 999, 104],
  "draft_target_pairs": [
    {"index": 0, "draft_token_id": 101, "target_token_id": 101},
    {"index": 1, "draft_token_id": 102, "target_token_id": 102},
    {"index": 2, "draft_token_id": 103, "target_token_id": 999},
    {"index": 3, "draft_token_id": 104, "target_token_id": 104}
  ],
  "accepted_prefix_len": 2,
  "bonus_token": null,
  "first_divergence_index": 2,
  "num_accepted_tokens": 3
}
```

Required fields are `step`, `context_len`, `drafted_ids`, `target_ids`, `accepted_prefix_len`, `bonus_token`, `first_divergence_index`, and the per-position draft/target token IDs. `num_accepted_tokens` is included because it is the exact input to the next GDN state transition. Use `null` for `first_divergence_index` on a full draft match and set `bonus_token` to the target bonus ID in that case.

Optional fields may be added without changing the required contract:

```json
{
  "graph_mode": "FULL_AND_PIECEWISE",
  "graph_cap": 64,
  "mtp_depth": 4,
  "prompt_len": 8459,
  "target_topk": [[101, 0.0]],
  "draft_topk": [[101, 0.0]],
  "state": {
    "spec_state_indices": [[700, 701, 702, 703, 704]],
    "spec_query_start_loc": [0, 5],
    "accepted_count_before_forward": 1,
    "mamba_prev_state_idx": 10,
    "mamba_curr_state_idx": 10,
    "mamba_copy": false
  }
}
```

The optional state object is an observation of the metadata boundary. It must not be used to substitute or repair the state machine.

## A.4 Comparison procedure on the existing ~8k pair

This is a two-run, single-request comparison, not a cohort.

1. Preserve the existing artifacts as the output-level control. Reconstruct the exact request from the same existing harness/payload used to make the artifacts; do not infer the repeated prompt from the generated answer. Assert that the tokenizer and prompt token IDs are identical between runs and that the prompt length is 8,459 tokens as reported by both `8k.json` files.
2. Launch disposable BF16 and candidate servers separately with the pinned image, model revision, tokenizer, target quantization, max model length, KV settings, and sampling parameters unchanged. The live BF16 service is not a test target to replace.
3. Set `temperature=0`, `seed=12345`, the same `max_tokens`/stop settings, and one request only. Apply the existing MTP/boundary/metadata overlays exactly as used by the candidate artifact; the tracer is additive and observational.
4. Save JSONL traces outside the disposable container. Also save the request hash, image digest, vLLM commit, model revision, graph mode/cap, MTP depth, and the unmodified public response artifact.
5. Join rows by `(request_id, step, context_len)`. Do not join by raw output position after a mismatch: a reject changes the committed prefix and therefore changes the next round's context. If a row cannot be joined, report the earliest missing context rather than silently padding it.
6. Compare in this order:
   - `drafted_ids`: first proposal difference;
   - `target_ids`: first target-argmax difference at the same pre-step context;
   - `accepted_prefix_len`, bonus, and `num_accepted_tokens`;
   - GDN state indices/copy decision immediately before the first target difference;
   - final committed token stream and finish reason.
7. Classify the first divergence:
   - **draft-only:** draft IDs differ but target IDs and committed output remain equal;
   - **target numerical/state:** target IDs differ with equal input IDs, or target IDs diverge immediately after a different accepted-count/state transition;
   - **accept/bonus bookkeeping:** target IDs agree but output/bonus/accepted count differs;
   - **metadata/graph:** the first difference occurs only when graph capture/padding or state-index metadata changes;
   - **unresolved:** traces do not cover the boundary or the binary provenance is unknown.
8. Stop after the first failing pair and its reproduction matrix. Do not start the five-task Harbor cohort until the target-token parity gate is closed.

The existing `comparison.json` is an output-level summary only. The new `comparison.json` produced by the tracer should include the first divergent `(step, context_len, index)`, the corresponding draft/target IDs, acceptance count, state-index excerpt, and paths to the two JSONL traces.

## A.5 Reproduction matrix

Run the same single 8k request and trace schema for each row. Keep target verification and all non-draft settings fixed.

| Row | Draft | Graph setting | MTP | Purpose |
| --- | --- | --- | ---: | --- |
| Control | BF16 | cap 64 / normal pinned graph setting | 4 | Reference target/draft/state trace. |
| Candidate | draft-INT4 overlay | graph capture cap 64 | 4 | Reproduce the quality failure in the strongest measured candidate configuration. |
| Graph isolation | draft-INT4 overlay | graphs disabled | 4 | Decide whether the first mismatch requires graph replay/padding or is present in eager execution. |
| Depth isolation | draft-INT4 overlay | graph capture cap 64 | 2 | Decide whether the mismatch requires MTP-4 state/row scheduling or survives at MTP-2. |

Record the actual `CUDAGraphMode`/capture cap reported by the runtime rather than relying on a launcher label. Do not add cap 128 to this matrix: the last-session cap-128 startup failure is a known serving constraint, not a useful parity control. If the no-graph row cannot be compared because the launcher changes another setting, mark it invalid rather than attributing its result to graph mode.

## A.6 Composer implementation notes (future work, not implemented here)

A narrow future implementation should be a reversible tracer injector plus an offline comparator, for example:

```text
patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/patch_p0_tracer.py
scripts/compare_qwen38_b70_p0_traces.py
```

The injector's runtime targets are the exact modules/functions in A.2:

- `vllm/v1/worker/gpu_model_runner.py` for round/context, metadata, target-logit, rejection, and accepted-count observation;
- `vllm/v1/spec_decode/step3p5.py` for per-MTP-step proposal IDs;
- `vllm/v1/sample/rejection_sampler.py` for target-vs-draft decisions and bonus;
- `vllm/v1/attention/backends/gdn_attn.py` and `vllm/v1/worker/mamba_utils.py` for state metadata/copy decisions;
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` or `vllm/_xpu_ops.py` only at the Python-to-XPU call boundary;
- `vllm/model_executor/models/qwen3_5_mtp.py` only for an optional draft-logit/hidden probe.

The existing metadata overlay is separate and must remain separate: `patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/patch_gdn_metadata.py` and its README state that it changes static metadata buffers only and does not change `GPUModelRunner`, `_xpu_C.gdn_attention`, or kernels ([local README](../patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/README.md)). Apply the tracer after the existing MTP/boundary/metadata sequence, or use import-time wrappers so the metadata patcher's anchors are not invalidated.

**Must not touch:**

- `GPUModelRunner` scheduling, input-index construction, rejection semantics, accepted-token correction, or graph dispatch. Hooks may copy values for observation but may not rebind or mutate them.
- The GDN ABI: `torch.ops.vllm.gdn_attention_core_xpu`, `_xpu_C.gdn_attention`, `spec_state_indices_tensor`, `num_accepted_tokens`, and their argument order.
- `vllm-xpu-kernels` compiled `.so` files, SYCL source, kernel launch geometry, or a kernel rewrite.
- The target model's verification numerics or target token stream.
- The 98,304 / 32,768 contract, live launcher, model weights, or an agent-evaluation configuration.

Trace artifacts must be written to an explicitly mounted host directory and the disposable container must be removed after each run. Do not make logging permanent in the serving image.

## A.7 Draft-INT4 keep/kill rule

**Keep as an active research platform only if all of the following hold:**

1. The tracer identifies the first divergent round and token index.
2. The cause is either fixed, or independently shown to be a diagnostic-only draft proposal difference that cannot change target verification/committed tokens.
3. BF16 and candidate target-token streams are identical on the fixed short, coding, and ~8k greedy controls at `temperature=0`, `seed=12345`; the graph-off and MTP-2 rows do not reveal an unexplained state-dependent failure.
4. The candidate still preserves the 131,072 serving ceiling and does not introduce a graph/prefill OOM.

**Kill/freeze the platform** if the first divergence remains unexplained, target tokens change after a partial accept, the failure survives graph-off/MTP-2 without a bounded cause, or the only evidence is output-level mismatch. A persistent draft-logit mismatch with correct target tokens may be retained as a short-decode curiosity, but it is not permission to use draft-INT4 for agent-quality claims or deployment. No row in this spec is a speed claim.

---

# Part B — GPT-pro item 2 state machine

## B.1 Source-derived state inventory

### MTP hidden state

**Source-derived:** `Qwen3_5MultiTokenPredictor.forward` selects one MTP layer with `spec_step_idx % num_mtp_layers`, transforms the supplied hidden state, normalizes it, and returns the result; it does not expose a committed-history hidden-state store ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/model_executor/models/qwen3_5_mtp.py#L139-L183)). `Step3p5MTPProposer.propose` runs the first pass, samples a draft token, then runs subsequent `spec_step_idx` values and stacks the draft IDs ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/spec_decode/step3p5.py#L276-L348), [continuation](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/spec_decode/step3p5.py#L374-L463)).

Therefore, after accepting `k` of `n` draft tokens, the **MTP hidden tensors from that proposal are not committed as a reusable hidden-history object**. The next proposal starts from the accepted target hidden/input boundary and runs its MTP steps again. The proposer does have draft-model attention/KV metadata and slots while producing a round: `Step3p5MTPProposer.build_per_group_and_layer_attn_metadata` builds per-layer metadata and `_get_slot_mapping` maps draft layers to slots ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/spec_decode/step3p5.py#L58-L154)). That is a KV/cache concern, not persistent MTP hidden replay.
**Source-derived detail:** the MTP constructor creates its predictor layers as `Qwen3_5DecoderLayer(..., layer_type="full_attention")`, so these draft-layer KV entries are regular attention-cache state, distinct from the target model's GDN conv/SSM state ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/model_executor/models/qwen3_5_mtp.py#L116-L123)).

**Hypothesis to verify:** the exact physical reuse/overwrite behavior of every draft MTP full-attention KV slot depends on the scheduler's block table and the actual launch's cache mode. The source shows the slot-mapping path but no named `rollback_mtp_hidden_state()` function.

### Target GDN conv state

**Source-derived:** On XPU, `QwenGatedDeltaNetAttention.forward_xpu` calls `torch.ops.vllm.gdn_attention_core_xpu` ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L953-L1000)). `_gdn_attention_core_xpu_impl` passes `spec_state_indices_tensor` and `num_accepted_tokens` to the compiled `_xpu_C.gdn_attention` entry point ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/_xpu_ops.py#L116-L201)).

At source tag `v0.1.12`, the spec path is explicit in `csrc/xpu/gdn_attn/causal_conv1d.hpp`:

- `causal_conv1d_spec_kernel` starts from `cache_indices[batch, num_accepted_tokens - 1]` (clamped at zero), so the previous committed/accepted state is selected rather than a fresh head state.
- It walks every speculative token through the causal window.
- It checkpoints the trailing conv state at each `cache_indices[batch, t_local]`, so a later accepted prefix can select the corresponding column ([blob](https://github.com/vllm-project/vllm/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/gdn_attn/causal_conv1d.hpp), blob SHA `5dcb4f07754eafb4057a09a3e3fb126690ee3241`; spec code around [L477-L526](https://github.com/vllm-project/vllm/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/gdn_attn/causal_conv1d.hpp#L477-L526) and [L687-L825](https://github.com/vllm-project/vllm/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/gdn_attn/causal_conv1d.hpp#L687-L825)).

The conv state is consequently **speculative write + accepted-column rollback/selection**, not fresh-per-round state.

### Target GDN delta/SSM state

**Source-derived:** `gated_delta_rule_spec_kernel` uses the same `num_accepted_tokens - 1` initial column, iterates the speculative tokens sequentially, and writes each resulting recurrent state to that token's dedicated cache slot ([blob](https://github.com/vllm-project/vllm/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/gdn_attn/gated_delta_rule.hpp), blob SHA `445ff26ad62ff1205aedb4e2e9faab3798481da1`; [spec kernel](https://github.com/vllm-project/vllm/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/gdn_attn/gated_delta_rule.hpp#L302-L550)). The C++ interface routes the speculative buffers and accepted counts into that path ([interface](https://github.com/vllm-project/vllm/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/gdn_attn/gdn_attn_interface.cpp#L477-L589), blob SHA `3576a3a412963d0517cea4d6446e258efe08b4db`).

Thus the current source model is:

```text
committed state at accepted column A_prev - 1
  -> recurrently consume all target/draft rows
  -> write H_0 ... H_n to speculative state columns
  -> rejection selects A_cur
  -> next round starts from column A_cur - 1
```

There is no source evidence of replaying accepted MTP trunk hiddens after the rejection. The accepted recurrent state is retained by slot selection and, when a mamba block boundary requires it, by explicit state migration.

### KV and block migration

**Source-derived:** `_update_states_after_model_execute` counts the non-`-1` output positions and makes that count available to the next input preparation ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/worker/gpu_model_runner.py#L1613-L1669)). `GPUModelRunner._prepare_inputs` synchronizes/copies the accepted count for the next metadata build ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/worker/gpu_model_runner.py#L2153-L2225)). The regular attention/KV path then uses the next logical sequence length and slot mappings; no general-purpose “erase every rejected KV row” function appears in this path. Rejected physical slots are therefore expected to be ignored by the corrected logical prefix and overwritten when reused.

For GDN's conv/SSM cache in `mamba-cache-mode=align`, `preprocess_mamba` computes a running state block, and when the accepted prefix moves to another block it copies the previous state with an accepted-token bias before the next forward ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/worker/mamba_utils.py#L1158-L1263)). The shared copy body documents the two cases: conv state copies the sliding-window suffix, while temporal state selects the accepted speculative column ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/worker/mamba_utils.py#L117-L247)). The fused postprocess makes the same decision on GPU and resets the accepted count to 1 when a state block becomes the new running block ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/worker/mamba_utils.py#L250-L383)).

## B.2 State transition after accepting `k` of `n`

Use `A_cur` for the number of committed output positions, because that is what the runner passes into the next state transition:

- first reject at draft index `k` (`0 <= k < n`): `accepted_prefix_len = k`, one target replacement is emitted, so `A_cur = k + 1`;
- all `n` drafts accepted: the target bonus is emitted, so `A_cur = n + 1`;
- `A_prev` is the corresponding count from the preceding round and is the input used while processing this round's speculative state.

The state transition is:

1. **Before the forward:** metadata carries the prior accepted count and per-request speculative state columns. GDN uses column `A_prev - 1` as its initial conv/SSM state.
2. **MTP proposal:** MTP hidden tensors for steps 0…`n-1` are produced in the proposer and discarded after the proposal; draft-layer KV slots are addressed by draft slot mappings.
3. **Target verification:** target/GDN consumes the full scheduled speculative group and writes per-token GDN checkpoints. The target rejection sampler compares `drafted_ids` with target IDs for greedy mode.
4. **Acceptance:** the output tensor contains the accepted prefix, a target replacement or bonus, then `-1` placeholders. `GPUModelRunner._update_states_after_model_execute` records `A_cur`.
5. **Logical KV prefix:** scheduler/request state advances only by the committed positions. Rejected speculative suffix rows are not part of the next logical prefix; regular KV slots are reused/overwritten rather than explicitly cleared.
6. **GDN state:** if the running mamba block does not change, the next round selects the accepted checkpoint column directly. If a block boundary is crossed, `postprocess_mamba_align_gpu`/`preprocess_mamba` migrates the accepted conv/SSM state into the new running block with the appropriate bias and resets the neutral in-block accepted count.
7. **Next round:** MTP begins again from the accepted target boundary; GDN starts from `A_cur - 1`. No accepted MTP hidden trunk is replayed from a persistent hidden-state cache.

### ASCII state diagram

```text
Round r: committed prefix L, previous accepted count A_prev
    │
    ├─ prepare metadata: state columns [0..n], logical KV prefix L
    │
    ├─ MTP proposer: hidden h_0..h_(n-1) + draft KV slots (round-local)
    │
    ├─ target/GDN forward:
    │      init conv/SSM = state_column[A_prev - 1]
    │      write conv checkpoints C_0..C_n and delta states H_0..H_n
    │
    ├─ rejection sampler: compare draft IDs with target IDs
    │      accept k drafts, then target replacement; or accept all + bonus
    │      A_cur = k + 1, or n + 1 on full accept
    │
    ├─ commit: logical KV prefix becomes L + A_cur
    │      rejected suffix rows are ignored/reused, not logically committed
    │
    ├─ GDN postprocess:
    │      same block  -> next round selects state_column[A_cur - 1]
    │      new block   -> copy accepted conv/SSM state, reset in-block count
    │
    └─ Round r+1: discard MTP hidden tensors; restart MTP from accepted target boundary
```

### Component classification

| Candidate interpretation | MTP hidden state | GDN conv/delta state | Regular target/draft KV |
| --- | --- | --- | --- |
| Fresh head state every round | **Yes for proposal hidden tensors only**: no committed MTP hidden object is carried across proposer calls. | **No**: initial state is selected from the accepted column. | **No**: logical prefix and slot mappings persist. |
| Committed prefix | **Logical input/target boundary only.** | **Yes**, through accepted state-column selection and block migration. | **Yes**, through scheduler/request logical length. |
| Speculative write + rollback | **Draft rows/slots are speculative and later ignored/reused.** | **Yes**, checkpoint every token, then select/copy the accepted column; this is not an in-kernel undo pass. | **Logical rollback/overwrite**, with no explicit full-cache erase observed. |
| Replay of accepted trunk hiddens | **No source hook or persistent store found.** | **No; recurrent state checkpoints avoid replaying the accepted trunk.** | **No replay is needed for already-cached regular attention rows.** |

The table is a source-derived map, not proof that every compiled wheel byte matches the source tag. The tracer must capture the state-index values at the first mismatch.

## B.3 Xe2 versus CUDA rollback/recompute

### What the pinned source establishes

The XPU interface contains a split Xe2 path, but its `#ifdef VLLM_XPU_ENABLE_XE2` branch selects `chunk_causal_conv1d_xe2` and `chunk_gated_delta_rule_xe2` for **non-spec prefill** (`num_prefills > 0`). The speculative branch calls the generic `gdn::causal_conv1d` and `gdn::gated_delta_rule` with `spec_state_indices_tensor` and `num_accepted_tokens` ([interface](https://github.com/vllm-project/vllm/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/gdn_attn/gdn_attn_interface.cpp#L269-L316), [delta dispatch](https://github.com/vllm-project/vllm/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/gdn_attn/gdn_attn_interface.cpp#L569-L657)). The relevant Xe2 source blobs are nevertheless recorded for the prefill path:

- `csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_xe2.hpp`, blob SHA `ccb6006d8e5ebf02c3ee33fbafd093a3c086e0fc` ([source](https://github.com/vllm-project/vllm/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_xe2.hpp));
- `csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_tiled_xe2.hpp`, blob SHA `21a73ccc3aa46da79a84b54c1cef58225848e5a6` ([source](https://github.com/vllm-project/vllm/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_tiled_xe2.hpp));
- `csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_xe2.cpp`, blob SHA `beec71093f9a36db1f621f5e1f6c4a758da918e3`, and `chunk_gated_delta_rule_kernels_xe2.hpp`, blob SHA `88d232e3893e1693db007b4e06a524df7358f978` ([directory](https://github.com/vllm-project/vllm/tree/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/gdn_attn/xe_2)).

The CUDA/Triton source uses the same state-index algorithm: `causal_conv1d_fwd_kernel` offsets its initial window by `num_accepted_tokens - 1` and documents the accepted-prefix rolling window ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/model_executor/layers/mamba/ops/causal_conv1d.py#L860-L957)); `fused_recurrent_gated_delta_rule_fwd_kernel` loads the accepted state index and writes each token's final state to its per-token index ([source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py#L102-L166)).

**Source-derived conclusion:** at the state-machine level, the pinned XPU speculative source matches CUDA's accepted-column/checkpoint approach. It does not recompute the accepted trunk after rejection, and it does not rely on a fresh recurrent head state each round. It computes the speculative rows once, stores checkpoints, and selects the accepted checkpoint next time.

### What remains unknown

**Unknown:** the installed `vllm-xpu-kernels 0.1.12.3` package was reported by the last-session audit as compiled-only, with `libgdn_attn_kernels_xe_2.so` and no editable source/build headers (`docs/qwen38-b70-mlxfast-research-progress-20260823.md`, “Actual installed B70 source audit”; `docs/qwen38-b70-next-session-handoff-20260823.md`, “Known blockers”). The published source tag is `v0.1.12`, not a manifest proving the exact `.3` wheel build.

Therefore this spec does **not** claim bitwise equivalence or identical performance between Xe2 and CUDA. The missing artifact is the wheel/build provenance mapping `libgdn_attn_kernels_xe_2.so` and its exported GDN entry points to a source commit. There is no missing source symbol for the speculative path: the relevant source symbols are `gdn::causal_conv1d_spec_kernel` and `gdn::gated_delta_rule_spec_kernel` in the two cited headers. If binary behavior ever disagrees with this map, obtain that build manifest or an upstream source checkout before changing the ABI or kernels.

## B.4 What item 2 blocks

### Item 3 — separate MTP hidden update from vocabulary projection

Blocked until the trace proves which MTP rows are required to advance draft-layer attention/GDN/KV state and which row is only a vocabulary projection. A last-row-only projection is safe only if the history rows still execute the required state transitions and the next proposal receives the same accepted boundary. The exact projection hook is `Qwen3_5MTP.compute_logits`; the state boundary is the proposer and per-layer metadata above. See the item 3 dependency in `docs/qwen38-b70-research-plan-20260824.md`.

### Item 4 — adaptive speculative depth

Blocked until the state machine is validated for partial and full accepts. Changing `n` changes `spec_state_indices_tensor` width, `spec_query_start_loc`, the number of speculative slots, and the MTP `spec_step_idx` sequence. An adaptive controller must not leave GDN state, draft KV, or accepted-count correction at the old depth. The last session also measured MTP-3 startup failure, so adaptive depth cannot be inferred from a static MTP-4 result. See item 4 in `docs/qwen38-b70-research-plan-20260824.md`.

### Item 11 — GDN speculative kernels

Blocked until the Python metadata/state map and the exact wheel/source provenance are closed. A kernel change must preserve the per-token conv/SSM checkpoint layout, accepted-column selection, mamba block migration, mixed-batch partitioning, and the existing XPU ABI. The source files to profile are the XPU GDN interface and the `causal_conv1d_spec_kernel` / `gated_delta_rule_spec_kernel`; the Xe2 chunk files are not evidence that the spec path uses a separate Xe2 rollback kernel. See item 11 in `docs/qwen38-b70-research-plan-20260824.md` and the installed-source blocker in `docs/qwen38-b70-mlxfast-research-progress-20260823.md`.

## B.5 Follow-up state trace after P0 (single request only)

Once the P0 tracer exists, run one disposable request with controlled acceptance outcomes at MTP-4:

1. reject at the first draft position;
2. accept a middle prefix and reject the next position;
3. accept all drafts and observe the bonus path.

For each round, capture `num_accepted_tokens`, `spec_state_indices_tensor`, conv/SSM initial and final selected columns, mamba block-copy decision, draft KV slot mapping, and the next round's context length. Compare the next target logits to a non-spec reference at the same committed prefix. This is a correctness trace, not an end-to-end speed test and not a Harbor evaluation.

---

## Research/source ledger

Local measured sources:

- `docs/qwen38-b70-next-session-handoff-20260823.md`
- `docs/qwen38-b70-mlxfast-research-progress-20260823.md`
- `docs/qwen38-b70-research-plan-20260824.md`
- `~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T205000Z-greedy-bf16/`
- `~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T205000Z-greedy-gdn/`
- `~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T205000Z-greedy-comparison.json`
- `patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/README.md`
- `patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/patch_gdn_metadata.py`

Pinned vLLM source:

- [`qwen3_5_mtp.py`](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/model_executor/models/qwen3_5_mtp.py)
- [`gpu_model_runner.py`](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/worker/gpu_model_runner.py)
- [`step3p5.py`](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/spec_decode/step3p5.py)
- [`rejection_sampler.py`](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/sample/rejection_sampler.py)
- [`gdn_attn.py`](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/attention/backends/gdn_attn.py)
- [`mamba_utils.py`](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/worker/mamba_utils.py)
- [`qwen_gdn_linear_attn.py`](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py)
- [`_xpu_ops.py`](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/_xpu_ops.py)
- CUDA comparison sources [`causal_conv1d.py`](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/model_executor/layers/mamba/ops/causal_conv1d.py) and [`fused_recurrent.py`](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py)

Pinned XPU-kernel source at `v0.1.12` / `1796aa8bc8db4ac68d9cd19636cef88f3af81d2b`:

- [`gdn_attn_interface.cpp`](https://github.com/vllm-project/vllm-xpu-kernels/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/gdn_attn/gdn_attn_interface.cpp), blob SHA `3576a3a412963d0517cea4d6446e258efe08b4db`
- [`causal_conv1d.hpp`](https://github.com/vllm-project/vllm-xpu-kernels/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/gdn_attn/causal_conv1d.hpp), blob SHA `5dcb4f07754eafb4057a09a3e3fb126690ee3241`
- [`gated_delta_rule.hpp`](https://github.com/vllm-project/vllm-xpu-kernels/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/gdn_attn/gated_delta_rule.hpp), blob SHA `445ff26ad62ff1205aedb4e2e9faab3798481da1`
- [`xe_2/chunk_causal_conv1d_xe2.hpp`](https://github.com/vllm-project/vllm-xpu-kernels/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_xe2.hpp), blob SHA `ccb6006d8e5ebf02c3ee33fbafd093a3c086e0fc`
- [`xe_2/chunk_causal_conv1d_tiled_xe2.hpp`](https://github.com/vllm-project/vllm-xpu-kernels/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_tiled_xe2.hpp), blob SHA `21a73ccc3aa46da79a84b54c1cef58225848e5a6`
- [`xe_2/chunk_gated_delta_rule_xe2.cpp`](https://github.com/vllm-project/vllm-xpu-kernels/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_xe2.cpp), blob SHA `beec71093f9a36db1f621f5e1f6c4a758da918e3`
- [`xe_2/chunk_gated_delta_rule_kernels_xe2.hpp`](https://github.com/vllm-project/vllm-xpu-kernels/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp), blob SHA `88d232e3893e1693db007b4e06a524df7358f978`

The published source and the installed `0.1.12.3` binary are deliberately distinguished. The latter's exact build provenance remains an unresolved concern until the runtime wheel manifest or matching upstream checkout is available.
