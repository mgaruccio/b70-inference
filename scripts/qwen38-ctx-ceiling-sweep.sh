#!/usr/bin/env bash
set -euo pipefail

# Disposable Qwen3.8 B70 KV-ceiling sweep. Run on inference-host only.
# Does not modify ~/inference/launchers/start-qwen38.sh.
# Method and results: docs/qwen38-b70-200k-context-20260830.md

ROOT=/home/mike/inference
MODEL_DIR="$ROOT/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
COOKBOOK="$ROOT/src/intel-arc-pro-b70-inference-cookbook"
IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
RENDER_GID="$(stat -c '%g' /dev/dri/renderD128)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT_DIR:-/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/${STAMP}-ctx-ceiling}"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
echo -e "name\tmax_model_len\tutil\thealth\tkv_tokens\tkv_gib\tmodel_gib\tdevice_free\tnotes" > "$SUMMARY"

export KV_XFER_CONFIG='{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","secondary_tiers":[{"type":"fs","root_dir":"/kv-offload"}]}}'

cleanup() {
  docker rm -f qwen38-ctx >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

run_one() {
  local name="$1" len="$2" util="$3"
  local log="$OUT/${name}.log"
  local health="fail" kv_tokens="" kv_gib="" model_gib="" device_free="" notes=""
  echo "=== $name max-model-len=$len util=$util ===" | tee -a "$OUT/run.log"
  cleanup
  docker run -d --name qwen38-ctx --ipc=host \
    -p 127.0.0.1:18000:8000 \
    --device /dev/dri --group-add "$RENDER_GID" -v /dev/dri:/dev/dri:ro \
    -v "$MODEL_DIR:/model:ro" \
    -v "$COOKBOOK/patches/patch_mtp_nightly.py:/patch_mtp.py:ro" \
    -v "$COOKBOOK/patches/patch_mtp_boundary.py:/patch_boundary.py:ro" \
    -v /home/mike/b70-evals/kv-offload-ssd:/kv-offload \
    -e KV_XFER_CONFIG \
    -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
    -e B70_MTP_BF16_DRAFT=1 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    --entrypoint bash "$IMAGE" -lc \
    "set -e; python /patch_mtp.py; python /patch_boundary.py; exec vllm serve /model --quantization gptq --dtype float16 --max-model-len ${len} --gpu-memory-utilization ${util} --kv-cache-dtype fp8 --port 8000 --max-num-seqs 1 --max-num-batched-tokens 8192 --enable-prefix-caching --mamba-cache-mode align --kv-offloading-backend native --kv-offloading-size 8 --kv-transfer-config \"\$KV_XFER_CONFIG\" --served-model-name qwen38 --language-model-only --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":4}' --enable-auto-tool-choice --tool-call-parser qwen3_xml" \
    >/dev/null

  local i=0 status=""
  while [ "$i" -lt 90 ]; do
    docker logs qwen38-ctx >"$log" 2>&1 || true
    if curl -fsS http://127.0.0.1:18000/health >/dev/null 2>&1; then
      health=ok
      break
    fi
    status="$(docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' qwen38-ctx 2>/dev/null || echo 'missing 0')"
    case "$status" in
      "running 0") ;;
      *)
        notes="container_${status}"
        break
        ;;
    esac
    i=$((i + 1))
    sleep 10
  done
  docker logs qwen38-ctx >"$log" 2>&1 || true
  if [ "$health" != ok ] && [ -z "$notes" ]; then
    notes="health_timeout"
  fi
  kv_tokens="$(rg -o 'GPU KV cache size: [0-9,]+ tokens' "$log" | tail -1 | rg -o '[0-9,]+' | tr -d ',' || true)"
  kv_gib="$(rg -o 'Available KV cache memory: [0-9.]+ GiB' "$log" | tail -1 | rg -o '[0-9.]+' || true)"
  model_gib="$(rg -o 'Model loading took [0-9.]+ GiB' "$log" | tail -1 | rg -o '[0-9.]+' || true)"
  device_free="$(rg -o 'Free memory on device \([^)]+\)' "$log" | head -1 || true)"
  if [ "$health" != ok ]; then
    local err
    err="$(rg -n 'OutOfMemory|OOM|not enough|too small|Error|ERROR' "$log" | tail -8 | tr '\t' ' ' | tr '\n' ';' || true)"
    notes="${notes}:${err}"
  fi
  echo -e "${name}\t${len}\t${util}\t${health}\t${kv_tokens}\t${kv_gib}\t${model_gib}\t${device_free}\t${notes}" | tee -a "$SUMMARY"
  cleanup
  [ "$health" = ok ]
}

echo "receipts: $OUT" | tee "$OUT/run.log"
run_one u088-l145k 145000 0.88 || true
run_one u088-l160k 160000 0.88 || true
run_one u088-l200k 200000 0.88 || true
run_one u094-l200k 200000 0.94 || true
run_one u095-l200k 200000 0.95 || true
echo DONE | tee -a "$OUT/run.log"
