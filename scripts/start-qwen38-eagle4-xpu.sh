#!/usr/bin/env bash
# EAGLE-4's released Qwen3.8 drafter on one Arc Pro B70, using lossless chain verification.
# Independent XPU model adapter, not the upstream adaptive-tree harness.
# Research: https://github.com/eagle-hq/EAGLE-4/releases/tag/v0.1.0
# Checkpoint: EAGLE-Corp/Qwen3.8-27B-EAGLE4@f58a5985223040552ad404fba98f0dd27f5f676a
# Runs detached; BIND_HOST defaults to loopback. Set it to the host's Tailnet IP for remote access.
# Stop with: docker stop qwen38-eagle4
set -euo pipefail
ROOT=${B70_ROOT:-/home/mike/inference}
SCRIPTS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
COOKBOOK="$ROOT/src/intel-arc-pro-b70-inference-cookbook"
DRAFT="$ROOT/models/EAGLE4-Qwen3.8-27B"
IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
# Download once outside the serving container:
# hf download EAGLE-Corp/Qwen3.8-27B-EAGLE4 --revision f58a5985223040552ad404fba98f0dd27f5f676a --local-dir "$DRAFT"
for file in "$DRAFT/config.json" "$DRAFT/model.safetensors" "$SCRIPTS/eagle4_xpu/install.py" "$SCRIPTS/eagle4_xpu/model.py"; do
  test -f "$file" || { echo "Missing required file: $file" >&2; exit 1; }
done
# XPU FlashAttention cannot address the draft's 40-head cache with hybrid-page padding;
# use the runtime's XPU-supported Triton attention for the draft only.
# Avoid the image's uninstalled /workspace/vllm shadowing the installed package in registry subprocesses.
exec docker run --rm --detach --name qwen38-eagle4 --ipc=host --workdir /tmp \
 -p "${BIND_HOST:-127.0.0.1}:${PORT:-18099}:8000" \
 --device /dev/dri --group-add "$(stat -c '%g' /dev/dri/renderD128)" -v /dev/dri:/dev/dri:ro \
 -v "$ROOT/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16:/model:ro" \
 -v "$DRAFT:/draft:ro" -v "$SCRIPTS/eagle4_xpu:/overlay/eagle4_xpu:ro" \
 -v "$ROOT/launchers/libattn_kernels_xe_2-head256-k64.so:/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libattn_kernels_xe_2.so:ro" \
 -v "$COOKBOOK/patches/patch_gdn_mixed_split_v5.py:/patch_v5.py:ro" \
 -v "$COOKBOOK/patches/patch_uniform_decode_prefill.py:/patch_uniform.py:ro" \
 -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True \
 -e 'EAGLE4_SPEC={"method":"eagle3","model":"/draft","num_speculative_tokens":4,"quantization":null,"kv_cache_dtype":"auto","attention_backend":"TRITON_ATTN","enforce_eager":false}' \
 --entrypoint bash "$IMAGE" -lc \
 'set -e; python /overlay/eagle4_xpu/install.py; python /patch_v5.py; python /patch_uniform.py; exec vllm serve /model --quantization gptq --dtype float16 --kv-cache-dtype auto --max-model-len 16384 --gpu-memory-utilization 0.88 --max-num-seqs 1 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mamba-cache-mode align --language-model-only --served-model-name qwen38-eagle4 --host 0.0.0.0 --port 8000 --cudagraph-metrics --speculative-config "$EAGLE4_SPEC"'
