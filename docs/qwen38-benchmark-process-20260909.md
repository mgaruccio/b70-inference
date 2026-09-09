# Qwen3.8 INT4-DFlash2 standard benchmark campaign

Status: pre-fix benchmark execution and artifact analysis complete, with preserved DFlash boundary failures. The [result directory and completed checklist](../results/20260909-qwen38-dflash2-rtn-standard/) classify this as development-tier, not publication-ready; quality sensitivity and peak-memory measurement remain unresolved.

## Protocol and external sources

Follow [BENCHMARKING_STANDARDS.md](../BENCHMARKING_STANDARDS.md). The imported document differs from the supplied Downloads copy only in its Section 2.3 cross-reference (quality tests are Section 9, not 8).

Fresh primary-source inspection on 2026-09-09:

- [BetterBench, commit 1de941d256ddd633a8c117963ba72aebe4b4d5e4](https://github.com/GGZ14/BetterBench/tree/1de941d256ddd633a8c117963ba72aebe4b4d5e4): the intended community harness. Its full run includes eight corpus categories, prefill and client concurrency. Use 20 measured passes/category, three warmups, greedy sampling and seed 42. Preserve its defaults otherwise (including concurrency 1/2/4/8/16, 48 requests/level). Its prefill sweep explicitly records depths that exceed the advertised context. Its interleaved A/B command requires two resident endpoints and has no server-restart hook; two Qwen stacks cannot coexist on this B70. These runs are sequential, not interleaved A/B.
- [Pinned vLLM serving benchmark source](https://github.com/vllm-project/vllm/blob/73029d42441321b631779db3475031f5ec26dd6c/vllm/benchmarks/serve.py): use the actual `vllm bench serve` CLI, fixed seed, random 512-input/128-output workload, greedy decoding, ignored EOS, 48 requests, three warmups and client concurrency 1/2/4/8. Save detailed JSON, latency percentiles and exact argv. Synthetic random-token serving results are separate from BetterBench's real-text corpus.

## Comparators and non-goals

- Candidate: the existing partial RTN INT4/G128 DFlash2 graph configuration, K7, strict target rejection, FP16-compute GPTQ target, FP8 target KV, BF16 draft KV, 32768 configured tokens, server max-num-seqs=1, prefix caching off. Retained draft modules remain BF16; this is not a fully INT4 drafter or calibrated GPTQ.
- Matched drafter baseline: identical configuration and pinned image, original BF16 DFlash2 weights instead of the RTN artifact. This isolates drafter quantization.
- Current best deployed reference: optimized MTP4, using the persistent launcher's pinned image and five patches, with benchmark-only cold-cache/non-thinking settings and the previously validated uniform-prefill guard/graph settings. It must be measured rather than substituting the target-only control. Comparison against DFlash2 is a **stack/configuration bundle**, including different images, algorithms and context ceilings; it cannot isolate quantization gains. The serving benchmark client stays pinned to the newer image on all arms.
- Do not alter persistent launchers, original weights, power limits, or promote a candidate. Glimmer remains stopped. The current 32K candidate is not a 200K replacement. Report configured limits, runtime capacity estimates and successfully tested context separately.
- The target weights and strict rejection rule are unchanged; no relaxed/cascade acceptance or lossy target-head change is under test. Initially Section 9 was considered inapplicable to drafter-only quantization. However, the observed greedy-output variation means this campaign cannot establish output invariance. Treat quality sensitivity as unresolved and the Section 9 divergence/regression suite as an outstanding publication gate, not an unconditional N/A. No quality-preserving optimization claim is made.

## Defined public-boundary end-to-end process

Environment/preconditions: `inference-host`, single Arc Pro B70, idle GPU/no running containers, 275 W cap, unchanged persistent launcher SHA `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`. Benchmark code/dependencies run on the inference host or inside its pinned disposable vLLM container, never in Pi's Python environment. HTTP listens on localhost only.

For each configuration:

1. Save exact launcher/argv and source snapshots. Start the disposable container; verify `/health` and `/v1/models` match the requested model and context. Save hardware/runtime metadata, relevant environment, model/draft configs and `python -m vllm.collect_env` output. Never dump credentials or unrelated environment variables.
2. Exercise natural-stop canaries and the existing finite-output boundary checks. A correctness/server failure stops the cell and is retained, not silently retried.
3. Run unmodified BetterBench `run --greedy --seed 42 --passes 20 --warmup 3` with no phase selection or quick mode. Preserve JSON, HTML and console output. Preserve all failures and unsupported prefill depths. Compare identical corpus IDs/seed/config; sequential order is a limitation, particularly for small effects.
4. Run pinned `vllm bench serve` at client concurrency 1/2/4/8. Save each command, detailed JSON and stdout. Client concurrency measures queuing against server C1; it is **not** active multi-sequence decoding.
5. Run the repository cold context sweep via `/tokenize` and streamed `/v1/completions`: 512, 8192, 16384, 32768, 65536, 120000, 160000 and the maximum practical prompt near each configured limit, leaving room for 128 output tokens. Use deterministic unique prompt content, temp 0, ignore EOS, one warmup and six measured trials per supported point. Save exact rendered token IDs/counts, request/response/SSE, latency, usage and metric deltas. Retain HTTP rejection evidence for unsupported points; do not fabricate six successful measurements or call an incomplete sweep compliant. A 32768-token prompt plus 128 output tokens cannot fit a 32768 context; the initial near-limit probe was 32640 (see the failure and explicit follow-up below).
6. Capture raw `/metrics` around each benchmark phase and each long-context request, including all exposed speculative counters/position metrics. Retain memory/KV sizing from server logs and distinguish static allocation from sampled device usage.
7. Expect all supported requests to complete successfully; forced long-context and serving requests must produce exactly 128 tokens. Report medians/IQR, failures, capacity regression and methodology differences without hiding outliers. Assemble raw artifacts and an explicit publication checklist before assigning a final tier.
8. On success or failure, stop only the owned disposable `qwen38` container; verify launcher, power cap and stopped Glimmer state remain unchanged. Preserve artifacts on the host and copy publishable small results to `results/20260909-qwen38-dflash2-rtn-standard/`.

Exact executed commands, paths, outcomes and the final checklist will be recorded with the results, not inferred from this plan.

## Boundary failure and explicit follow-up

The first INT4 cold sweep (`int4-long`, source `code-v2`) passed all six measured trials at 512, 8192 and 16384 prompt tokens. Its 32640-token warmup crashed the engine with `RuntimeError: Expected spec_token == num_spec_decodes * (num_speculative_tokens + 1) to be true, but got false`. The subsequent health check returned 503; the cell aborted and its disposable container was removed. Keep the failure, partial results and full server traceback. This is not an OOM or successful 32K validation.

Follow-up process, without changing inference code/configuration: test BF16 at the same exact boundary to distinguish a shared DFlash2 failure from an INT4-specific failure. Then run both arms with an explicitly selected conservative `--near-limit 32000`, keeping every required standard length in the sweep. The failed 32640 attempt remains a separate artifact; the conservative point is not a silent replacement. This tests a practical near-limit point, not a proof of the largest possible working context. The MTP4 reference retains the standard 190000-token near-limit point.

Observed follow-up: BF16 reproduced the same 32640-token engine assertion after passing 512/8192/16384, so the failure is not unique to INT4. Both conservative sweeps completed six valid measured runs at 512/8192/16384/32000 and preserved actual HTTP 400 context-limit rejections at 32768/65536/120000/160000. Their 24 measured request payloads match exactly. MTP4 completed full BetterBench and serving clients plus six valid measured trials at every cold point through 190000 input tokens. All three arms' raw results and final analysis are retained. These cells all precede the separately requested [boundary fix](qwen38-dflash2-boundary-fix-20260909.md); do not attribute their throughput to the fixed runtime.

The matched full BetterBench run showed 12/160 differences in output length across arms despite identical prompt IDs/counts. Fixed-output vLLM requests also were not bit-identical across all runs, including within each unmodified arm at different client concurrency. Preserve these observations; strict rejection and unchanged target weights do not establish bitwise output or quality invariance. No relaxed/cascade acceptance was used, and no lossy-acceptance quality-suite result is claimed.
