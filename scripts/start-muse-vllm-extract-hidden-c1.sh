#!/bin/bash
# Temporary target-only serve: dump Muse hidden states at the DFlash aux layers.
# Replaces muse-vllm-xpu-c1. Restore with start-muse-vllm-dflash-c1-graph-draft-gptq.sh.
set -euo pipefail

IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
MODEL=/home/mike/inference/models/Muse-Glimmer-30B-GPTQ-Int4-sym-G128
OUT="${HIDDEN_CAPTURE_DIR:-/home/mike/b70-evals/muse-glimmer/20260826T-vllm-dflash-hidden-capture}"
SPEC="$OUT/extract-spec.json"
KV="$OUT/kv-transfer.json"
DRAFT_CFG="$OUT/extract-draft"
RENDER_GID="$(stat -c '%g' /dev/dri/renderD128)"
NAME=muse-vllm-xpu-c1
PORT=8000

mkdir -p "$OUT/hidden_states" "$DRAFT_CFG"
# Dummy draft dir so CLI SpeculativeConfig can load aux layer ids.
# Eagle3 IDs matching the live DFlash cell: target_layer_ids [1,13,25,37,49] -> (2,14,26,38,50)
cat > "$DRAFT_CFG/config.json" <<'EOF'
{
  "architectures": ["ExtractHiddenStatesModel"],
  "model_type": "extract_hidden_states",
  "eagle_aux_hidden_state_layer_ids": [2, 14, 26, 38, 50]
}
EOF
cat > "$SPEC" <<'EOF'
{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[2,14,26,38,50]}}}
EOF
cat > "$KV" <<'EOF'
{"kv_connector":"ExampleHiddenStatesConnector","kv_role":"kv_producer","kv_connector_extra_config":{"shared_storage_path":"/hidden_states","allow_custom_save_path":true,"use_synchronization_lock":true}}
EOF

test -f "$MODEL/config.json"

/usr/bin/docker rm -f glimmer-sycl glimmer-dflash2-sycl "$NAME" >/dev/null 2>&1 || true

/usr/bin/docker run -d --name "$NAME" -p "0.0.0.0:${PORT}:8000" \
  --device /dev/dri --group-add "$RENDER_GID" -v /dev/dri:/dev/dri:ro \
  -v "$MODEL:/model:ro" \
  -v "$DRAFT_CFG:/extract-draft:ro" \
  -v "$SPEC:/spec.json:ro" \
  -v "$KV:/kv.json:ro" \
  -v "$OUT/hidden_states:/hidden_states" \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "$IMAGE" -lc \
  'set -e; pip install -q vllm-xpu-kernels==0.1.13.2; exec vllm serve /model --quantization gptq --dtype float16 --max-model-len 2048 --gpu-memory-utilization 0.90 --port 8000 --max-num-seqs 1 --max-num-batched-tokens 2048 --no-enable-prefix-caching --no-enable-chunked-prefill --served-model-name muse-glimmer-gptq --language-model-only --reasoning-parser muse_glimmer --speculative-config "$(cat /spec.json)" --kv-transfer-config "$(cat /kv.json)"'

echo started "$NAME" extract-hidden-states on "$PORT" "out=$OUT"
/usr/bin/docker ps --filter name="$NAME"
