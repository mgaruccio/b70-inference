#!/bin/bash
# TB2.1 eval serving: verified C8 + prefix caching, GPTQ target/draft, DFlash K3.
# Ten Pi rollout agents share eight inference slots; extra requests queue.
# Requires the retained concurrency-sweep probes and frozen shortlist below.
# Preserve those assets: this is a host-specific recipe, not a standalone install.
# Prefix reuse avoids recomputing long agent histories (cold prefill still costs):
# https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
# Pair with local-dev-model/configs/glimmer-b70-tb21-pi-docker.toml.
# Stop the existing GPU server explicitly before launching; no automatic replacement.
set -euo pipefail
IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
MODEL="${MODEL:-/home/mike/inference/models/Muse-Glimmer-30B-GPTQ-Int4-sym-G128}"
DRAFT="${DRAFT:-/home/mike/inference/models/Muse-Glimmer-30B-assistant-GPTQ-Int4-sym-G128}"
PATCH="${PATCH:-/home/mike/b70-evals/muse-glimmer/concurrency-sweep-20260906T002815Z/c8/patch-vllm-dflash-gptq-context-kv.py}"
NAME="${NAME:-glimmer-tb21-prefix-c8}"
PORT="${PORT:-18080}"
RENDER_GID="$(stat -c '%g' /dev/dri/renderD128)"

test -f "$MODEL/config.json"
test -f "$DRAFT/config.json"
test -f "$DRAFT/quantize_config.json"
test -f "$PATCH"
test -f /home/mike/b70-evals/muse-glimmer/concurrency-sweep-20260906T002815Z/probes/patch_glimmer_draft_head.py
test -f /home/mike/b70-evals/muse-glimmer/concurrency-sweep-20260906T002815Z/shortlist.json
# No automatic deletion of existing containers; fail if NAME is already in use.
/usr/bin/docker run -d --name "$NAME" -p "127.0.0.1:${PORT}:8000" \
  --device /dev/dri --group-add "$RENDER_GID" -v /dev/dri:/dev/dri:ro \
  -v "$MODEL:/model:ro" -v "$DRAFT:/draft:ro" \
  -v "$PATCH:/patch-vllm-dflash-gptq-context-kv.py:ro" \
  -e VLLM_TARGET_DEVICE=xpu -e DFLASH_KV_MODE=none \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v "/home/mike/b70-evals/muse-glimmer/concurrency-sweep-20260906T002815Z/probes:/probes:ro" \
  -v "/home/mike/b70-evals/muse-glimmer/tb21-20260906/latency-diagnosis/prefix-c8:/artifacts" \
  -e PYTHONPATH=/probes -e GLIMMER_DRAFT_HEAD=shortlist \
  -e GLIMMER_DRAFT_HEAD_ARTIFACT_DIR=/artifacts \
  -v "/home/mike/b70-evals/muse-glimmer/concurrency-sweep-20260906T002815Z/shortlist.json:/shortlist.json:ro" \
  -e GLIMMER_DRAFT_HEAD_SHORTLIST=/shortlist.json \
  --entrypoint bash "$IMAGE" -lc '
    set -e
    pip install -q vllm-xpu-kernels==0.1.13.2
    python /patch-vllm-dflash-gptq-context-kv.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py
    python /probes/patch_glimmer_draft_head.py /opt/venv/lib/python3.12/site-packages/vllm
    exec vllm serve /model --quantization gptq --dtype float16 \
      --max-model-len 131072 --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 \
      --port 8000 --max-num-seqs 8 --max-num-batched-tokens 2048 \
      --enable-prefix-caching --served-model-name muse-glimmer-gptq \
      --language-model-only --reasoning-parser muse_glimmer --enable-auto-tool-choice --tool-call-parser muse_glimmer \
      --speculative-config '\''{"method":"dflash","model":"/draft","num_speculative_tokens":3,"quantization":"gptq"}'\''
  '
echo "Started $NAME: http://127.0.0.1:$PORT/v1 (C8, DFlash K3, 131072 context)"
