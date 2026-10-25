# Intel 1.5 / Qwen3.8-27B: native MTP versus EAGLE-4

Status: preparation; no comparison result yet. Initial tier: development / engineering evaluation, not a standard-publishable benchmark. The repository publication checklist remains applicable before broader performance claims.

## Authoritative request

> can you do a comparison with their published 27b recipe with the standard drafter and then swap in our new eagle-4 drafter with a minimum of changes? that feels cleanest

> ok excellent, let's get that comparison made then please. if that means we need to update the drafter to work with fp8 kv cache we should do that

The user explicitly approved updating the drafter for FP8 KV cache if required. Only Intel's published recipe is the baseline. Do not substitute another baseline or change the target checkpoint between arms.

## Primary sources and pins

- Intel release: https://github.com/intel/llm-scaler/releases/tag/vllm-1.5
- Intel source commit: `77c547307e0f0289044c8a610bd9603c19c2c483`.
- Published release instructions: https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/README.md
- Image: `intel/llm-scaler-vllm:1.5@sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b` (release declares functional equivalence to `0.26.0-b2`).
- Model: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Model index: https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/model.safetensors.index.json
- Official index reports 55,562,855,904 bytes of tensors across 18 shards.

Intel explicitly validates Qwen3.8-27B and native MTP in the 1.5 release; some command examples still name Qwen3.6. The model-name substitution and tensor parallelism of one for the available single B70 are adaptations, not a claim of verbatim reproduction of a published Qwen3.8 command.

## Baseline contract

Use Intel's documented symmetric INT4 target quantization, FP16 compute, `fp8_e4m3` attention KV, FP16 Mamba SSM cache, and native `qwen3_5_mtp` with four speculative tokens. Start from the documented FP8 KV recipe: memory utilization 0.8, max model length 40000, batch-token budget 8192, max sequences 64, block size 64, eager execution, and prefix caching disabled. Verify every flag and the INT4 library path against the pinned image before launch.

Keep the target checkpoint, target quantization, compute/state/cache precision, resource settings, GPU power, requests, and measurement method identical between arms. Keep the published Intel image unchanged for the first baseline. Record the EAGLE-4 integration diff against that exact image. If integration requires shared runtime corrections, measure and report a separate native-MTP control with those corrections; do not attribute a bundled difference solely to the drafter. Do not loosen correctness thresholds, silently alter resource settings, or promote a different arithmetic path just to complete the comparison.

## Preconditions and public-boundary test process

1. Exclusive `inference-host` B70, no other GPU workload, temporary containers only, loopback API binding, unchanged persistent services. Capture hardware, power, driver, image digest, package/source revisions, complete launch arguments, cache layout/scales, and memory allocation.
2. Obtain the official pinned checkpoint and Intel image with enough disk for both and a 16-GiB free-space floor. Storage cleanup was explicitly authorized; initially remove only the two identified older Needle experiment directories.
3. Start Intel native MTP through its ordinary serving CLI. Use `/health`, `/v1/models`, `/tokenize`, `/v1/chat/completions`, and `/metrics` to verify startup, model identity, prompt accounting, finite meaningful outputs, and speculative counters. Run short arithmetic, code, and structured-output canaries plus an excluded warmup before measured requests.
4. Retain a deterministic request set, exact rendered prompt tokens, greedy sampling with seed 0, output cap 512, ignored EOS, no prefix reuse, and concurrency 1. Exercise 8K and 32K input classes supported by the declared 40000-token envelope. Retain the same request bytes for every arm. Validate each response's output length, token IDs where exposed, text, finish reason, and metric accounting.
5. Establish ordinary-target output references on the same Intel image/checkpoint/precision with speculation disabled. Repeat reference requests to expose runtime nondeterminism rather than treating it as candidate error or concealing it. Compare MTP and EAGLE outputs against those references, including first divergence and full-output match rate. Investigate failed gates before performance promotion.
6. Run paired native-MTP / EAGLE-4 / EAGLE-4 / native-MTP cells, with identical warmup and ten measured requests per workload per cell. Each cell uses a fresh temporary server. Save failures, retries, and all measurements; no cherry-picking.
7. Also run Intel's documented `vllm bench serve` public-boundary probe with fixed seed, 32768 input tokens, 512 output tokens, random-range-ratio 0, concurrency 1, ignored EOS, and temperature 0. Preserve the exact executable command and raw JSON. Verify options against the shipped CLI rather than assuming newer flags.
8. Report medians and dispersion for post-first-token decode, TTFT, full request latency, accepted/proposed tokens, draft rounds, accepted tokens per round, and exposed depth metrics. Report target/drafter memory, KV capacity, and the actually tested context limit. Tree acceptance percentages are not directly comparable to a linear draft's percentage without explaining the different proposal budgets.
9. Retain launch files, source modifications, environment output, client commands, requests/responses/token IDs, metrics, server/kernel logs, benchmark JSON, and cleanup outcomes in a dedicated run directory. Do not distribute raw secrets or package draft weights into the report.
10. Stop and remove owned temporary containers/listeners after each cell; preserve artifacts and verify the disk-space floor. No persistent deployment changes.

## Integration gates

Inspect the actual Intel-patched vLLM 0.26 source and native FP8 ABI before implementation. Establish scale direction/granularity/storage, target and draft cache allocation/grouping, quantized writes and reads, accepted-row compaction, ordinary fallback, GDN state precision, and graph/eager lifecycle. FP8 bytes must never be interpreted as unscaled FP16 values. Keep unsupported shapes fail-closed.

Use focused CPU/interpreter tests, native XPU cache/scale/compaction tests, and the real HTTP journey above. Unit tests and a mock runtime do not substitute for the public-boundary run. Do not claim an existing qualified kernel launch remains active if the new runtime chooses a different layout.

## Sharing

The report must state the evaluation tier and all deviations from the Intel template. It must not claim standard-publication compliance until the full repository checklist is satisfied. EAGLE-4 is source-available, not OSI open source; any distributed implementation must preserve its LICENSE/NOTICE, attribution, modification notices, and corresponding-source obligations. Commercial permissions are separate from successful technical validation.
