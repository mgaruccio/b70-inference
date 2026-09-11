# DSpark per-layer context normalization correction

Development-tier experiment; no persistent service promotion. Corrects the proven grouped-XPU RMSNorm weight-row bug documented in `../20260911-qwen38-dspark-reference-parity/observed-results.md`.

## Exact change and source

Canonical overlay commit `8a19b58`, SHA256 `0640edc7a72c4b6650bb6846cdc988c87883dad7c0cb86d36684513a1c070643`. Only the opted-in DSpark context-K normalization uses one native RMSNorm call per layer with that layer's weight vector. Allocation, epsilon and non-opted-in behavior remain unchanged. Old overlays fail closed rather than receiving an in-place upgrade.

Primary caller source: https://raw.githubusercontent.com/vllm-project/vllm/73029d42441321b631779db3475031f5ec26dd6c/vllm/model_executor/models/qwen3_dflash.py

## Real validation performed

Environment: inference-host, Intel Arc Pro B70, pinned XPU image `sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4`, vLLM `73029d424`, torch `2.13.0+xpu`, transformers `5.15.0`. Same GPTQ-Int4 target, FP16 shared embedding/head, FP8 target KV, BF16 DSpark checkpoint and draft KV, K7 greedy, eager C1 8192 as the pre-fix capture.

Executed `bash run-verified-candidate.sh` on inference-host:

1. Full canonical regression suite in the pinned image: **25 tests passed, zero skipped** (`pinned-image-tests.txt`).
2. `bash run-candidate.sh`: identical real API prompt unarmed then armed, with 16 output tokens; first-proposal tensors only, no replacement sampler. Capture-on/off prompt and output IDs matched. Exact 62 checkpoint weight identities verified.
3. `bash run-reference-command.sh`: unmodified SHA-pinned official HF model replayed on the captured input, then independently replayed again. All stages finite; official self-repeatability bitwise exact.
4. Host invariants checked before/after; launcher hash, 275 W cap and stopped Glimmer unchanged. All temporary containers removed. Large capture/reference tensors remain host-local, excluded from Git.

Detailed metadata/commands/API streams/source hashes are under `capture-01/`; numerical metrics are `reference-01/comparison.json`. Each native/reference pair uses identical captured inputs. Pre-fix and post-fix captures are separate server runs, so cross-run target auxiliary tensors are not claimed bitwise identical.

## Numerical result

Compared with the pre-fix native/reference comparison:

- Layer-1 context K normalization RMSE: **0.55872 → 0.005524**; layers 2–4 similarly fall to approximately 0.0055.
- Layer-1 attention cosine: **0.321716 → 0.999385**.
- Final hidden cosine: **0.506508 → 0.999679**; RMSE **1.84517 → 0.047265**.
- Base-logit cosine: **0.847810 → 0.999886**.
- Proposal IDs: **0/7 → 7/7 exact matches** with official reference. Corrected native and official both produce `[30057,286,264,436,34810,18887,303]`.

Residual BF16/attention numerical differences remain. This is one real first-proposal fixture, not universal bitwise parity, distribution-fidelity proof, or a measured speedup.

## Performance journey

`bash run-performance.sh` reuses the prior diagnostic public-API client and shared finite/functional gates unchanged:

- Eager: code/math/prose × thinking off/on × target temperature 0/1 × seeds 42/43/44, 36 requests with 512 forced output tokens.
- FULL_DECODE_ONLY graphs: identical matrix, capture sizes `[7,8]`.
- Graph 64K: seven requests of 65,536 input and 128 output tokens, max context 65,664.

Only intended algorithm change versus matching pre-fix cells is per-layer context normalization. Fresh containers, C1, prefix cache disabled, fixed model/image/power. Compare native acceptance counters and decode throughput with prior target-only and DSpark cells. Historical MTP4 is not a contemporaneous control. The separate official FP8 target control remains a distinct experiment, not an explanation inferred from these numbers.

## Observed performance results

`bash run-performance.sh` completed successfully on inference-host. All shared gates and both 36-request matrices passed; 64K completed one warmup plus six valid measured requests. All three cells report `host_unchanged: true`; their temporary containers were removed. Raw requests/SSE/metrics/server logs are retained under `norm-eager/`, `norm-graph/`, and `norm-graph-64k/`.

`performance-comparison.json` verifies all 36 request payloads and rendered prompt IDs match the corresponding pre-fix cell in each short-context comparison. This is a sequential development comparison, not interleaved or a universal quality/output-identity claim.

- Eager all-request median decode: **25.161 → 40.107 tok/s** (1.594×). Weighted emitted tokens per step: **2.169 → 3.497**; accepted/proposed fraction: **16.70% → 35.67%**.
- Graph all-request median decode: **40.208 → 62.774 tok/s** (1.561×). Weighted emitted tokens per step: **2.158 → 3.482**; accepted/proposed fraction: **16.54% → 35.45%**.
- Corrected graph group medians (nine requests each): thinking-off/temp0 **71.712**, thinking-off/temp1 **63.582**, thinking-on/temp0 **62.418**, thinking-on/temp1 **47.005 tok/s**.
- Graph 64K median post-first-token decode: **16.976 → 26.445 tok/s**. Corrected inclusive IQR **1.496 tok/s**; median TTFT **57.370 s**. Warmup excluded.

The historical MTP4 64K result (~55.75 tok/s) still exceeds corrected DSpark. It used a different pinned image and MTP-specific patches, so it is not a contemporaneous isolated algorithm comparison. This correction is a substantial measured improvement, not a claim that DSpark now wins or all acceptance causes are resolved. The genuine FP8 target control is a separate pending diagnostic. No benchmark result is standard-compliant or promoted by this development record.
