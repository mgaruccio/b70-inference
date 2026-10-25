#!/usr/bin/env bash
# Candidate arm: preserve Intel target flags; add the EAGLE plugin and required hooks.
set -Eeuo pipefail
MODEL_DIR=${1:?absolute path to pinned Qwen3.8-27B checkpoint}
DRAFT_DIR=${2:?absolute path to EAGLE-4 draft checkpoint}
RUN_DIR=${3:?absolute path to staged source and engine patch}
LABEL=${4:?unique lowercase run label}
# 0.8 preserves the published budget; 0.9 is the explicitly approved comparison/control.
GPU_MEMORY_UTILIZATION=${5:-0.8}
case "$GPU_MEMORY_UTILIZATION" in 0.8|0.9) ;; *) echo 'Memory fraction must be 0.8 or 0.9' >&2; exit 1 ;; esac
[[ "$MODEL_DIR" = /* && "$DRAFT_DIR" = /* && "$RUN_DIR" = /* && "$LABEL" =~ ^[a-z0-9-]+$ ]]
# Verified local image ID after loading the pinned Intel digest archive.
IMAGE=sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b
NAME="intel15-e4-$LABEL"
for shard in $(seq -w 1 18); do
  test -f "$MODEL_DIR/model-000${shard}-of-00018.safetensors"
done
test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/model.safetensors.index.json"
test -f "$DRAFT_DIR/model.safetensors"
test -f "$RUN_DIR/eagle-source/pyproject.toml"
test -f "$RUN_DIR/engine-6cae86c.patch"
[[ -z $(docker ps -q) ]] || { echo 'GPU lane occupied' >&2; exit 1; }
if ss -ltn | awk '{print $4}' | grep -q ':18110$'; then
  echo 'Benchmark port occupied' >&2; exit 1
fi
(( $(df -Pk "$MODEL_DIR" | awk 'NR==2 {print $4}') >= 16*1024*1024 ))
docker image inspect "$IMAGE" >/dev/null
exec docker run --rm --name "$NAME" \
  --device /dev/dri --group-add "$(stat -c '%g' /dev/dri/renderD128)" \
  --mount type=bind,src=/dev/dri/by-path,dst=/dev/dri/by-path,readonly \
  --shm-size 32g --ulimit core=0 -p 127.0.0.1:18110:8000 \
  -v "$MODEL_DIR:/llm/models/Qwen3.8-27B:ro" \
  -v "$DRAFT_DIR:/llm/models/EAGLE4-Qwen3.8-27B:ro" \
  -v "$RUN_DIR/eagle-source:/candidate:ro" \
  -v "$RUN_DIR/engine-6cae86c.patch:/tmp/engine.patch:ro" \
  -e ZE_AFFINITY_MASK=0 -e PIP_NO_INDEX=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
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
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-model-len 40000 \
  --max-num-batched-tokens 8192 --max-num-seqs 64 --block-size 64 \
  --enforce-eager --no-enable-prefix-caching --trust-remote-code \
  --attention-backend CUSTOM \
  --speculative-config '{"method":"eagle_harness","model":"/llm/models/EAGLE4-Qwen3.8-27B","num_speculative_tokens":15,"kv_cache_dtype":"float16","method_config":{"mode":"static","max_verify_width":16,"node_top_k":4,"tree_depth":14,"depth_stop_gain":0.0}}'
