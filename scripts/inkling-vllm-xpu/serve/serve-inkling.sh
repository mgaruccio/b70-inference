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
# hc-ck1 = hc-s1 + SergiioB's CUTLASS Xe2 expert kernel (INK2_CUTLASS_MOE=1): lossless, +19-32% DSpark
# decode (results/20261123-inkling2-external-kernels-compare). Rollback: hc-s1 and drop INK2_CUTLASS_MOE.
export PYTHONPATH=$p/src/hc-ck1:$p/src/transformers-mini/src:$runtime/serve
export INK2_CUTLASS_MOE=1
export TRITON_CACHE_DIR=$p/.triton HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export NEOReadDebugKeys=1 ForcePreemptionMode=3 VLLM_LOGGING_LEVEL=INFO

exec "$runtime/.venv/bin/python" -B -m vllm.entrypoints.cli.main serve "$weights" \
    --host 0.0.0.0 --port 8080 --served-model-name inkling-2-mini \
    --middleware flint_auth.FlintAuth \
    --language-model-only --dtype bfloat16 --max-model-len 262144 \
    --max-num-batched-tokens 256 --gpu-memory-utilization 0.98 --max-num-seqs 1 \
    --block-size 64 --enable-prefix-caching --no-async-scheduling --kv-cache-dtype auto \
    --generation-config auto --chat-template "$weights/chat_template.jinja" \
    --enable-auto-tool-choice --tool-call-parser inkling --reasoning-parser inkling \
    -cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY \
    --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
