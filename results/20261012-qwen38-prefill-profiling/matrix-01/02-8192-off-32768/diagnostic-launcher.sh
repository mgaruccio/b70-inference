#!/usr/bin/env bash
# Default Qwen3.8 service for inference-host: 212,992-token C1, no KV offload.
set -euo pipefail

ROOT=/home/mike/inference
MODEL_DIR="$ROOT/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
COOKBOOK="$ROOT/src/intel-arc-pro-b70-inference-cookbook"
IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
RENDER_GID="$(stat -c '%g' /dev/dri/renderD128)"

exec docker run --rm --name qwen38-native-prefill-profile --ipc=host \
  -p "127.0.0.1:8000:8000" \
  --device /dev/dri --group-add "$RENDER_GID" -v /dev/dri:/dev/dri:ro \
  -v "$MODEL_DIR:/model:ro" \
  -v "$ROOT/launchers/libattn_kernels_xe_2-head256-k64.so:/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libattn_kernels_xe_2.so:ro" \
  -v "$COOKBOOK/patches/patch_mtp_nightly.py:/patch_mtp.py:ro" \
  -v "$COOKBOOK/patches/patch_mtp_boundary.py:/patch_boundary.py:ro" \
  -v "$COOKBOOK/patches/patch_gdn_mixed_split_v5.py:/patch_v5.py:ro" \
  -v "$COOKBOOK/patches/patch_draft_lmhead_int4.py:/patch_s.py:ro" \
  -v "$COOKBOOK/patches/patch_draft_mtp_int4.py:/patch_m1.py:ro" \
  -v "$COOKBOOK/patches/patch_uniform_decode_prefill.py:/patch_uniform_decode_prefill.py:ro" \
  -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e B70_MTP_BF16_DRAFT=1 -e B70_DRAFT_LMHEAD_INT4=1 -e B70_DRAFT_MTP_INT4=1 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v /home/mike/b70-evals/20261012-qwen38-prefill-profiling/matrix-01/02-8192-off-32768/torch:/profile \
  --entrypoint bash "$IMAGE" -lc \
  'set -e; python /patch_mtp.py; python /patch_boundary.py; python /patch_v5.py; python /patch_s.py; python /patch_m1.py; python /patch_uniform_decode_prefill.py; exec vllm serve /model --quantization gptq --dtype float16 --max-model-len 212992 --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 --port 8000 --max-num-seqs 1 --max-num-batched-tokens 8192 --no-enable-prefix-caching --mamba-cache-mode align --chat-template-content-format openai --default-chat-template-kwargs "{\"enable_thinking\":true}" --reasoning-parser qwen3 --override-generation-config "{\"temperature\":0.6,\"top_p\":0.95,\"top_k\":20,\"min_p\":0}" --enable-logging-iteration-details --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/profile --profiler-config.torch_profiler_with_stack=false --profiler-config.torch_profiler_record_shapes=false --profiler-config.torch_profiler_with_memory=false --profiler-config.ignore_frontend=true --profiler-config.max_iterations=16 --served-model-name qwen38 --language-model-only --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":4}" --enable-auto-tool-choice --tool-call-parser qwen3_xml'
