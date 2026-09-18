# Frozen fresh-family confirmation protocol

Declared before target generation or candidate results. Tier: **fresh-family exploratory holdout**, not an adequately powered confirmation or the repository's full standard-publishable benchmark. No training, architecture/quantization change, cloud allocation, or production promotion.

**User-approved amendment before scanning/admission/generation:** initial outcome-blind extraction found only eight unused families / 64 contexts. The user explicitly chose “Run eight exploratorily” rather than gathering more families or changing to public tasks. The original 20-family feasibility guard below is therefore waived for this run; keep every other rule, admit the available fresh families, cap at four contexts per family, report actual counts and limited power. No performance outputs were observed before this amendment.

## Frozen hypothesis and head

Does the development-selected LR 1e-6 decay, final step 652 head improve native greedy accepted drafts/pass and decode throughput on unused engineering session families?

Candidate: `/home/mike/b70-evals/20260915-mtp-scale-attempt/training-lr1e6_decay/tuned-mtp.safetensors`, SHA256 `1af9142095c1d387847c83f27bdc330df8d34f81d0d73683c593ef5768babde5`. Baseline: stock native MTP from the same GPTQ model. This candidate is frozen; no test-driven checkpoint selection, objective changes, subset selection, or extra checkpoints.

## Outcome-blind data acquisition and admission

Use the existing strict `qwen38_mtp_trace_corpus.py` extractor on selected local engineering roots: b70-inference, local-dev-model, localmodels, local-serving, agentic-orch, model-router, weathermodeling, winr, including their worktree session directories for existing lineage resolution. Use the actual clock; preserve the 24-hour inactivity exclusion. Forks resolve to their original session family; missing/cyclic lineage fails closed. Do not expose session/prompt bodies to external models or publish them.

Exclude the **157-family union of all six prior train/dev/test splits**, including all assigned records even if previously too long to generate. The recent acceptance-dev set is a subset of the second dev split. Also exclude old context IDs and canonical normalized-message hashes. If any context in a newly encountered family duplicates a prior context, exclude that entire family. Keep source membership and exact checks privately.

Run existing Gitleaks 8.30.1 in a network-disabled namespace, redacting logs, then use the existing `filter-scan` validation/deduplication path. Its hash-based train/dev/test filenames are intermediate partitions only: **none of these newly acquired families are used for training or selection**, and fresh families from their union are eligible for this test-only experiment. Never reclassify a previously assigned training family. Scanner clearance is not proof of absence of sensitive content.

Before any generation, render every candidate with the exact native `/tokenize` boundary and existing four-tool profile/per-context thinking. Admit prompts leaving at least 128 tokens in a 2,048-token complete sequence; no prompt truncation. Order families deterministically by SHA256 of `fresh-confirmation-42:` plus family ID, and contexts by the same prefix plus context ID. Select at most **32 families**, at most **four admitted contexts per family**. Use all if 20–31 families remain. **Stop before generation if fewer than 20 independent eligible families remain.** This minimum is an explicit feasibility guard, not a formal power guarantee. Freeze exact selected IDs/order, token counts, per-request response budgets, and source membership before generating any test response. Unselected fresh families remain unused.

## Matched native experiment

Stock/candidate/candidate/stock (**ABBA**), one request at a time, four synthetic warmups excluded per cell. Reuse the existing temporary capture-off native launcher and `qwen38_mtp_native_corpus.py generate --capture-off --measure` path. Only the BF16 head overlay differs; the existing native loader performs unchanged RTN packing. No capture instrumentation active during measurements.

All cells: greedy temperature 0, seed 42, top-p .95/top-k 20 (inactive under greedy), natural EOS/tool termination, response cap 1,792, complete-sequence limit 2,048, minimum available response budget 128; same ordered frozen test contexts and budgets. Unique cache salts impose cold prefixes. Preserve per-record thinking and restricted bash/read/write/replace tool schema; never execute generated tools.

Runtime: existing B70 32GB, 275W cap, same model/tokenizer/template/target GPTQ INT4 g128, draft RTN INT4 g128, FP16 compute, FP8 KV, native MTP4, C1, context 212,992; balanced mode, graph sizes `[1,2,4,8]`, existing uniform-prefill guard, loopback serving. Prefix cache enabled with unique salts. Image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`; model revision `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`.

## Outcomes and interpretation

Primary: pooled native **accepted draft tokens/speculative pass**, excluding bonus tokens. Supporting decode rate `sum(generated−1)/sum(server_decode_seconds)`, end-to-end rate `sum(generated)/sum(client_wall_seconds)`, TTFT/latency, depth survival, emitted token counts, and finish reasons. Use paired source-family bootstrap, 10,000 draws, seed 42; retain both runs per head per sampled family. Report all four cells and relative effects with 95% percentile intervals. No response or family dropped after outcomes; failures remain recorded and invalidate a complete-run claim rather than being silently replaced.

Output checks: exact token matches within and across heads; API usage/counter accounting; generated tool-call names and arguments against the supplied JSON schemas; finish reason and truncation counts. These are output-divergence and structural-validity checks, **not semantic task-success scores**. Native numerics can vary across repeats. No external model judge receives private outputs. Without executable task ground truth, functional quality/general equivalence remains unproven and no promotion is authorized.

A supported performance confirmation requires positive lower confidence bounds for both primary acceptance and decode speed, with no clear structural-validity failure introduced by the candidate. Otherwise report an inconclusive or negative test and retain stock. An end-to-end-only win does not override an inconclusive primary metric. No sequential stopping, favorable-subset headline, retraining, or implicit rerun with a different head. This test set becomes consumed after evaluation regardless of outcome.

## E2E execution, evidence, cleanup

Preconditions checked 2026-09-18 10:17 EDT: idle host/no Docker or vLLM jobs, 81GiB free, 275W, candidate hash above, persistent launcher SHA256 `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`.

Real journey: local filtered/excluded/scanned sources → private staging on inference-host → exact-runtime stock `/tokenize` admission → freeze test set → stock synthetic warmup and timed `/v1/chat/completions` → stop stock → candidate overlay load/warmup/timed calls twice → stock repeat → paired analysis and output checks → stop all owned containers and verify launcher hash, power and disk. Check exact normalized requests and rendered token IDs match across cells; capture disabled; zero prefix hits; exact native completion-counter/API accounting. No optimizer or tensor capture is needed.

Private staging: `/home/mike/b70-mtp-private-confirm-20260918/` on the lead host, and `/home/mike/b70-evals/20260918-mtp-fresh-confirmation/` on inference-host. Retain exact commands, membership/admission decisions, scanner logs, launch configs, raw requests/responses/token IDs/counters, failures and aggregate reports privately. Git receives code/commands without trace bodies and aggregate documentation only. Stop temporary servers on success or failure; never delete old caches/heads or restart production.

## Fresh external research

https://raw.githubusercontent.com/vllm-project/vllm/ac7509e2b/docs/features/speculative_decoding/README.md — freshly retrieved pinned official docs: rejection is theoretically/algorithmically lossless within hardware precision, but vLLM does not guarantee stable logprobs across runs; floating-point and batching effects can change outputs. Therefore measure within-head repeats, do not call token differences alone a quality regression, and retain semantic-quality uncertainty. The docs.vllm.ai fetch was rate-limited; the official pinned GitHub source was accessible. Existing dev E2E and 117 passing targeted tests establish the unchanged execution path; this experiment must still exercise fresh test data through the actual API.
