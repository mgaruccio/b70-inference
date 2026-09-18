# Frozen development protocol: acceptance-aligned saved-head diagnostic

Declared before new candidate scores. Development only: no training, new architecture, production promotion, cloud allocation, or use of consumed test sets.

## Question and candidates

Does dev CE ranking miss a saved head with better joint accepted-prefix agreement and native acceptance? Use the existing exact Qwen GPTQ/B70/XPU/native-MTP4 stack. Candidates, in order:

1. Stock model MTP tensors.
2. `training-lr5e6_decay/tuned-mtp.step0400.safetensors` (previous CE winner).
3. `training-lr1e6_decay/tuned-mtp.safetensors` (step 652).
4. `training-lr5e6_flat/tuned-mtp.step0400.safetensors`.

All paths are under the existing private `/home/mike/b70-evals/20260915-mtp-scale-attempt/` root on `inference-host`. Do not expand this shortlist based on favorable subsets.

## Data and offline metrics

Use all 54 ordered records of `private-prompts/dev.jsonl`, with the existing tokenizer/template, restricted four-tool profile, and per-record thinking flag. Generate one new native stock greedy trajectory per admitted record: temperature 0, seed 42, top-p .95/top-k 20 (inactive under greedy), natural termination, response cap 1,792, complete-sequence limit 2,048, minimum available response budget 128. Do not truncate prompts. Record every admission/exclusion and retain the same ordered admitted prompts and budgets for native comparison. Unique cache salts enforce cold prefixes.

Use **64 uniformly sampled roots without replacement per sequence**, capped by eligible count, seed 42 and sorted root order, fixed across all heads/stages. Eligibility requires that x[t+1] is a response token (first draft follows the first verifier-generated response token), all four labels x[t+2:t+6] are supervised and present, and the root hidden/position is observed. Exclude terminal incomplete horizons; never pad missing hidden rows. Report boundary/terminal exclusions and eligible/evaluated counts. This fixed sample increases deeper coverage over the previous eight roots/sequence without unbounded attention allocations.

Primary offline metric: post-RTN-effective dense mean jointly accepted prefix length, L = sum_d product_{i<=d}(draft_argmax_i == greedy_observed_label_i), depth 4. Report prefix histogram 0..4, unconditional survival by depth, CE and marginal top-token agreement on the same roots, sequence/family counts and dispersion. Common CE weights [1,.5,.25,.125]. Score both BF16-export and RTN-effective core stages with the existing frozen runtime-effective LM head; post-RTN is primary. No optimizer, checkpoint updates, or weight exports. Chunk logits and use isolated root branch caches. Preserve exact root/label identity in private outputs.

Observed greedy labels are not a universal deterministic oracle. Dense RTN-effective scoring does not establish bitwise INT4-kernel parity. All-roots/random-root agreement is not the runtime scheduling distribution.

## Native comparison and decision rule

After offline results are frozen, select the highest post-RTN joint-prefix candidate (tie: lower common CE, then shortlist order). Run stock versus that candidate even if stock remains the offline winner, to diagnose transfer; the candidate is the best nonstock head. If it differs from the previous CE winner, a second comparison against the previous CE winner is permitted to answer the ranking question, not to select on consumed tests. At most two nonstock heads.

Use capture-off temporary native servers, unchanged target quantization, RTN packing, FP16 compute, FP8 KV, MTP4, C1, 212,992 configured context, balanced mode, graph sizes [1,2,4,8], existing uniform-prefill guard, loopback API, cold prefixes. Match the admitted dev records and generation budgets above. Four synthetic warmups excluded per server. Initial cell order stock/candidate/candidate/stock (ABBA); any second candidate gets its own ABBA. Retain every failure, request, exact API token IDs, per-request counter deltas, TTFT, decode time, wall time, and output lengths privately.

Primary native metric: total accepted draft tokens / total native speculative passes. Supporting pooled decode and end-to-end rates, latency/TTFT, depth survival, and within-head token-output variability. Family-bootstrap 95% intervals with 10,000 draws, seed 42, retaining paired repeats. No identical-output subset as primary. Native counters may include terminal drafts discarded by the API; report this limitation rather than equating counters with emitted tokens.

If offline/native materially disagree, investigate alignment/cache/precision/scheduling before retraining. If neither improves or deltas are within variability, retain stock and report uncertainty. These are dev results, not fresh-test generalization. Both prior test sets remain consumed.

## E2E process, environment, evidence, cleanup

Preconditions verified at 2026-09-17 20:39 EDT: inference-host has no running Docker/vLLM jobs, 82 GiB free, 275W cap, all shortlisted heads present. Persistent launcher SHA256 `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`.

Pinned image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`. Model: `/home/mike/inference/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` (revision `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`). All ML dependencies/tensors remain on inference-host.

Executable journey: existing `qwen38_mtp_native_corpus.py prepare` temporary stock capture launcher -> start/warm server -> `generate --trace-split heldout --trace-temperature 0 --sequence-limit 2048 --min-response-tokens 128 --max-tokens 1792` against `/tokenize` and `/v1/chat/completions` -> stop server -> targeted pytest in the pinned CPU-only image -> evaluation-only CLI on real XPU features -> existing `prepare --weights ... --no-capture` and `generate --capture-off --measure` for native ABBA -> stop temporary containers, check processes/power/free space and persistent launcher hash. Exact invocations will be retained in private `commands/` and summarized with results.

Expected checks: no corrupt/misaligned features, finite scores, identical roots/labels across heads, prefix survival monotonicity and histogram accounting, exact API token/counter accounting, capture disabled for timing, no production changes. Private artifacts use a fresh `/home/mike/b70-evals/20260918-mtp-acceptance-dev/` root; outputs fail closed rather than overwrite old data. Public Git contains only code, protocol, and aggregate findings. Never delete private caches/heads or run generated tool calls.

## Fresh external sources affecting the approach

- https://raw.githubusercontent.com/vllm-project/vllm/ac7509e2b/vllm/v1/sample/rejection_sampler.py — freshly fetched: `rejection_greedy_sample_kernel` sets a persistent rejected flag on first draft/target-argmax mismatch. This requires joint-prefix scoring rather than summed marginal accuracies; bonus token is not a draft acceptance.
- https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/v1/spec_decode/metrics.py#L20-L49 and #L104-L118 — native position counters count contiguous accepted drafts. This protocol deliberately uses accepted drafts/pass (no bonus), unlike vLLM's logged mean acceptance length, which adds one bonus token.
- https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/v1/spec_decode/llm_base_proposer.py#L829-L868 and https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/v1/spec_decode/utils.py#L43-L83 — first-pass token shift preserves target positions; recursive steps increment positions by one.
- https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/model_executor/models/qwen3_5_mtp.py#L136-L183 and https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/v1/worker/gpu_model_runner.py#L2909-L2982 — native normalized embedding/hidden concatenation and verifier token alignment support the existing helpers.
- https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py#L160-L350 — native packing/kernel dispatch is not the dense RTN reconstruction; native serving remains decisive. The existing deployment draft RTN patches, not an inferred default quantization, govern this run.
- https://docs.vllm.ai/en/latest/features/speculative_decoding/ — output variability can arise from backend precision and batch numerics. Paired repeats remain necessary; newer per-request metric APIs are not assumed present in this pinned image.
