#!/usr/bin/env bash
# Flint production server (vLLM XPU + DFlash 2) — run by flint-inference.service.
# The API key comes from the systemd credential; it is never put on the command line.
set -euo pipefail
cd /home/mike/flint-vllm-spike
VLLM_API_KEY=$(< "$CREDENTIALS_DIRECTORY/api-key")
export VLLM_API_KEY
D=$PWD/driver-26.40
export LD_LIBRARY_PATH=$D/usr/lib/x86_64-linux-gnu:$D/usr/lib:$D/usr/local/lib
export LIBRARY_PATH=$D/usr/lib/x86_64-linux-gnu:$D/usr/lib
export CPATH=$D/usr/include
export OCL_ICD_VENDORS=$PWD/opencl-vendors
export NEOReadDebugKeys=1 ForcePreemptionMode=3
export VLLM_XPU_INC_WNA16_BACKEND=w4a16 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export VLLM_LOGGING_LEVEL=INFO
export PYTHONPATH=$PWD/serve${PYTHONPATH:+:$PYTHONPATH}
NUM_SPEC=${FLINT_NUM_SPEC:-4}
exec "$PWD/.venv/bin/vllm" serve "$PWD/gptq" \
    --host 0.0.0.0 --port 8080 --served-model-name flint \
    --middleware flint_auth.FlintAuth \
    --dtype bfloat16 --kv-cache-dtype auto --attention-backend TRITON_ATTN \
    --gpu-memory-utilization "${FLINT_GPU_UTIL:-0.97}" --max-model-len "${FLINT_MAX_MODEL_LEN:-auto}" \
    --max-num-seqs 1 --max-num-batched-tokens "${FLINT_MAX_BATCHED:-2048}" \
    ${FLINT_PREFIX_CACHING:---enable-prefix-caching} \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47 \
    -cc.mode=none \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"$PWD/dflash2\",\"num_speculative_tokens\":$NUM_SPEC}"
