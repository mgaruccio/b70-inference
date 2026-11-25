#!/usr/bin/env bash
set -euo pipefail
p=$HOME/inkling-vllm-xpu
runtime=$HOME/flint-vllm-spike
weights=$p/weights-ed40e7da625f594fc9c4990f5f66be24873c22d7
driver=$runtime/driver-26.40
VLLM_API_KEY=$(< "$CREDENTIALS_DIRECTORY/api-key")
[[ -n $VLLM_API_KEY ]]
export VLLM_API_KEY
export LD_LIBRARY_PATH=$driver/usr/lib/x86_64-linux-gnu:$driver/usr/lib:$driver/usr/local/lib
export LIBRARY_PATH=$driver/usr/lib/x86_64-linux-gnu:$driver/usr/lib CPATH=$driver/usr/include
export OCL_ICD_VENDORS=$runtime/opencl-vendors
# vLLM 4932e8596a (upstream 0085facc58 + #69444); XPU plugin e5f6619.
# Keep the verified Torch/Triton and CUTLASS Xe2 binary from the prior deployment.
stage=$p/sparse-69444
export PYTHONPATH=$stage/plugin:$stage/sparse:$runtime/serve
export INK2_CUTLASS_MOE=1 INK2_HYBRID_PREFIX_SDPA=1 INK2_SPARSE_CONV_HISTORY=1
export TRITON_CACHE_DIR=$stage/triton HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export NEOReadDebugKeys=1 ForcePreemptionMode=3 VLLM_LOGGING_LEVEL=INFO

exec "$stage/.venv/bin/python" -B -m vllm.entrypoints.cli.main serve "$weights" \
    --host 0.0.0.0 --port 8080 --served-model-name inkling-2-mini \
    --middleware flint_auth.FlintAuth \
    --language-model-only --dtype bfloat16 --max-model-len 409000 \
    --max-num-batched-tokens 2048 --gpu-memory-utilization 0.98 --max-num-seqs 32 \
    --block-size 64 --enable-prefix-caching --no-async-scheduling --kv-cache-dtype fp8 \
    --generation-config auto --chat-template "$weights/chat_template.jinja" \
    --enable-auto-tool-choice --tool-call-parser inkling --reasoning-parser inkling \
    -cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY \
    --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
