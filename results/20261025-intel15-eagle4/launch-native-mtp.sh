#!/usr/bin/env bash
# Intel's release recipe adapted only for Qwen3.8-27B, one B70, and a loopback test port.
# Run in the foreground; the caller owns the temporary container's lifecycle.
set -Eeuo pipefail
MODEL_DIR=${1:?absolute path to pinned Qwen3.8-27B checkpoint}
LABEL=${2:?unique lowercase run label}
[[ "$MODEL_DIR" = /* && "$LABEL" =~ ^[a-z0-9-]+$ ]]
# The optional none arm supplies the same-precision ordinary-target reference.
SPECULATION=${3:-mtp}
# Published baseline stays at 0.8; 0.9 is the explicitly approved matched control.
GPU_MEMORY_UTILIZATION=${4:-0.8}
case "$GPU_MEMORY_UTILIZATION" in 0.8|0.9) ;; *) echo 'Memory fraction must be 0.8 or 0.9' >&2; exit 1 ;; esac
case "$SPECULATION" in
  mtp) SPEC_ARGS=(--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":4}') ;;
  none) SPEC_ARGS=() ;;
  *) echo 'Speculation must be mtp or none' >&2; exit 1 ;;
esac
# Verified local image ID after loading the pinned Intel digest archive.
IMAGE=sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b
NAME="intel15-$SPECULATION-$LABEL"
test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/model.safetensors.index.json"
for shard in $(seq -w 1 18); do
  test -f "$MODEL_DIR/model-000${shard}-of-00018.safetensors"
done
[[ -z $(docker ps -q) ]] || { echo 'GPU lane occupied' >&2; exit 1; }
if ss -ltn | awk '{print $4}' | grep -q ':18110$'; then
  echo 'Benchmark port occupied' >&2; exit 1
fi
(( $(df -Pk "$MODEL_DIR" | awk 'NR==2 {print $4}') >= 16*1024*1024 ))
docker image inspect "$IMAGE" >/dev/null
# Use the released image's INT4 library path and entrypoint initialization.
# Do not force V2, change native kernels, or install the candidate into this baseline.
exec docker run --rm --name "$NAME" \
  --device /dev/dri --group-add "$(stat -c '%g' /dev/dri/renderD128)" \
  --mount type=bind,src=/dev/dri/by-path,dst=/dev/dri/by-path,readonly \
  --shm-size 32g -p 127.0.0.1:18110:8000 \
  -v "$MODEL_DIR:/llm/models/Qwen3.8-27B:ro" \
  -e ZE_AFFINITY_MASK=0 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  "$IMAGE" /llm/models/Qwen3.8-27B \
  --served-model-name Qwen3.8-27B \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --quantization sym_int4 --dtype float16 \
  --kv-cache-dtype fp8_e4m3 --mamba-ssm-cache-dtype float16 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-model-len 40000 \
  --max-num-batched-tokens 8192 --max-num-seqs 64 --block-size 64 \
  --enforce-eager --no-enable-prefix-caching --trust-remote-code \
  "${SPEC_ARGS[@]}"
