# Qwen3.8-27B B70 golden configurations

Last verified: 2026-08-30 on `inference-host` (Intel Arc Pro B70, `xe`).

Use these profiles deliberately. The performance golden and the maximum-context profile have different operating shapes.

## Performance golden — live `qwen38`

This is the active service on port 8000.

- Model: `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`
- Image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
- Served model ID: `qwen38`
- Power: `power/control=on`, Xe `power1_cap=230000000`
- Context: `131072`; GPU-memory utilization: `0.88`; `max-num-seqs=64`; `max-num-batched-tokens=8192`
- FP8 KV cache, MTP-4, XPU graph enabled, prefix caching enabled.
- Patches in order: `patch_mtp_nightly.py`, `patch_mtp_boundary.py`, `patch_gdn_mixed_split_v5.py`, `patch_draft_lmhead_int4.py`, `patch_draft_mtp_int4.py`.
- Environment: `B70_MTP_BF16_DRAFT=1`, `B70_DRAFT_LMHEAD_INT4=1`, `B70_DRAFT_MTP_INT4=1`, `VLLM_XPU_ENABLE_XPU_GRAPH=1`, `VLLM_TARGET_DEVICE=xpu`, `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`, `ZE_AFFINITY_MASK=0`.
- **No KV offload flags** in this profile.

The host launcher is `~/inference/launchers/start-qwen38.sh`. Do not replace it with the older BF16-only cache-off launcher.

### Verification

A public streaming `/v1/chat/completions` test with thinking disabled, 256 generated tokens, measured client post-first-token decode at **95.806 tok/s median** across five runs (95.793–95.832). This is a short-prompt `g256` measurement, not a long-context claim.

## Maximum-context proven profile — temporary C1 experiment

Receipt: `~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260830T193253Z-champion-max-context/`.

- Highest **tested** successful `max-model-len`: **212,992** (the boundary between this and 229,376 was not narrowed).
- Required changes from the performance golden: `gpu-memory-utilization=0.95`, `max-num-seqs=1`, native KV offload (`--kv-offloading-backend native --kv-offloading-size 8`) with the filesystem tier mounted at `/kv-offload`, and `--mamba-cache-mode align`.
- Keeps the performance golden's model, image, MTP-4, Draft-INT4 S+M1, v5 mixed-split, FP8 KV, prefix cache, XPU graph, and 230 W cap.
- A streamed near-limit request used 212,222 prompt tokens and generated 106 tokens (212,328 total): TTFT 447.948 s, prefill **473.765 tok/s**, post-first decode **37.598 tok/s**.
- `262144`, `245760`, and `229376` failed initialization; do not claim native 262k support on this B70 profile.

The experiment restores the performance golden afterward. Do not serve this C1/context profile as the normal 64-sequence endpoint.

## Why these profiles differ

`max-model-len` includes prompt and generated tokens. The long-context profile reserves more KV capacity and uses C1/offload; an active ~212k request therefore cannot retain short-context decode speed.

References used when defining the experiment:

- https://docs.vllm.ai/en/latest/api/vllm/config/cache/ — context length and KV allocation tradeoff.
- https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/ — hybrid Mamba/attention cache behavior and prefix caching.
