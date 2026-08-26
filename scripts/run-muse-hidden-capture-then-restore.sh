#!/bin/bash
# Stop the DFlash C1 cell, dump target hidden states, restore the known-good
# GPTQ DFlash cell. Durable scripts only; host /tmp is volatile.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${HIDDEN_CAPTURE_DIR:-/home/mike/b70-evals/muse-glimmer/20260826T-vllm-dflash-hidden-capture}"
mkdir -p "$OUT/hidden_states"
exec > >(tee -a "$OUT/capture.log") 2>&1

restore_dflash() {
  echo "=== restore DFlash GPTQ cell ==="
  bash "$ROOT/start-muse-vllm-dflash-c1-graph-draft-gptq.sh"
  bash "$ROOT/wait-vllm-health.sh" 420 8000
  curl -s http://127.0.0.1:8000/v1/models || true
  echo
}

trap restore_dflash EXIT

echo "=== capture start $(date -Is) out=$OUT ==="
bash "$ROOT/start-muse-vllm-extract-hidden-c1.sh"
i=0
while [ "$i" -lt 420 ]; do
  running=$(docker inspect -f '{{.State.Running}}' muse-vllm-xpu-c1 2>/dev/null || echo false)
  if [ "$running" != "true" ]; then
    echo "EXTRACT_CONTAINER_DEAD after ${i}s"
    docker logs muse-vllm-xpu-c1 >"$OUT/extract-docker.log" 2>&1 || true
    tail -80 "$OUT/extract-docker.log" || true
    exit 1
  fi
  if curl -sf -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "EXTRACT_HEALTH_OK after ${i}s"
    break
  fi
  sleep 5
  i=$((i + 5))
done
if [ "$i" -ge 420 ]; then
  echo "EXTRACT_HEALTH_TIMEOUT"
  docker logs muse-vllm-xpu-c1 >"$OUT/extract-docker.log" 2>&1 || true
  tail -80 "$OUT/extract-docker.log" || true
  exit 1
fi
python3 "$ROOT/vllm-extract-hidden-dump.py" "$OUT/hidden_states" 256
echo "=== dumps ==="
ls -l "$OUT/hidden_states"
echo "=== capture+restore done $(date -Is) ==="
