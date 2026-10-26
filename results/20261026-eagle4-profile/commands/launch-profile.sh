#!/usr/bin/env bash
# Exact delivered EAGLE target/tree flags, plus inactive-by-default profiler and labels.
set -Eeuo pipefail
RUN_DIR=${1:?staged source/output directory}
LABEL=${2:?unique label}
NAME="intel15-e4-$LABEL"
MODEL_DIR=/home/mike/inference/models/intel15-Qwen3.8-27B
DRAFT_DIR=/home/mike/inference/models/EAGLE4-Qwen3.8-27B
IMAGE=sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b
[[ -z $(docker ps -q) ]] || { echo 'GPU lane occupied' >&2; exit 1; }
if ss -ltn | awk '{print $4}' | grep -q ':18110$'; then echo 'Port occupied' >&2; exit 1; fi
(( $(df -Pk "$MODEL_DIR" | awk 'NR==2 {print $4}') >= 16*1024*1024 ))
test -f "$RUN_DIR/eagle-source/pyproject.toml"
test -f "$RUN_DIR/engine-6cae86c.patch"
mkdir "$RUN_DIR/trace"
exec docker run --rm --name "$NAME" \
 --device /dev/dri --group-add "$(stat -c '%g' /dev/dri/renderD128)" \
 --mount type=bind,src=/dev/dri/by-path,dst=/dev/dri/by-path,readonly \
 --shm-size 32g --ulimit core=0 -p 127.0.0.1:18110:8000 \
 -v "$MODEL_DIR:/llm/models/Qwen3.8-27B:ro" \
 -v "$DRAFT_DIR:/llm/models/EAGLE4-Qwen3.8-27B:ro" \
 -v "$RUN_DIR/eagle-source:/candidate:ro" \
 -v "$RUN_DIR/engine-6cae86c.patch:/tmp/engine.patch:ro" \
 -v "$RUN_DIR/trace:/tmp/eagle-profile" \
 -e ZE_AFFINITY_MASK=0 -e PIP_NO_INDEX=1 \
 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
 -e EAGLE4_PROFILE_STAGES=1 \
 --entrypoint /bin/bash "$IMAGE" -lc '
set -Eeo pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null
source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force >/dev/null
patch --fuzz=0 -p1 --directory=/opt/venv/lib/python3.12/site-packages < /tmp/engine.patch
cp -a /candidate /opt/eagle-candidate
/opt/venv/bin/python -m pip install --no-deps --no-build-isolation /opt/eagle-candidate
exec vllm serve "$@"
' -- /llm/models/Qwen3.8-27B \
 --served-model-name Qwen3.8-27B --host 0.0.0.0 --port 8000 \
 --tensor-parallel-size 1 --quantization sym_int4 --dtype float16 \
 --kv-cache-dtype fp8_e4m3 --mamba-ssm-cache-dtype float16 \
 --gpu-memory-utilization 0.9 --max-model-len 40000 \
 --max-num-batched-tokens 8192 --max-num-seqs 64 --block-size 64 \
 --enforce-eager --no-enable-prefix-caching --trust-remote-code \
 --attention-backend CUSTOM \
 --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/eagle-profile","torch_profiler_record_shapes":false,"torch_profiler_with_memory":false,"torch_profiler_with_stack":false,"torch_profiler_use_gzip":true,"max_iterations":12}' \
 --speculative-config '{"method":"eagle_harness","model":"/llm/models/EAGLE4-Qwen3.8-27B","num_speculative_tokens":15,"kv_cache_dtype":"float16","method_config":{"mode":"static","max_verify_width":16,"node_top_k":4,"tree_depth":14,"depth_stop_gain":0.0}}'
