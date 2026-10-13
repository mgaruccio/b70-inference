#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
REMOTE=/home/mike/b70-evals/20261013-qwen38-int4-dispatch
SSH=(-o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no)
timeout -k 2s 25s ssh "${SSH[@]}" mike@inference-host "mkdir -p '$REMOTE'"
timeout -k 2s 30s scp "${SSH[@]}" "$HERE/compatibility.py" "mike@inference-host:$REMOTE/compatibility.py"
timeout -k 10s 180s ssh -T "${SSH[@]}" mike@inference-host 'bash -s' <<'HOST'
set -euo pipefail
OUT=/home/mike/b70-evals/20261013-qwen38-int4-dispatch
NAME=qwen38-int4-compat-20261013
if docker inspect "$NAME" >/dev/null 2>&1; then echo "Owned-name collision; refusing" >&2; exit 1; fi
cleanup() { if docker inspect "$NAME" >/dev/null 2>&1; then docker stop --time 5 "$NAME"; fi; }
trap cleanup EXIT
mkdir "$OUT/compatibility"
printf '%s\n' 'ONEDNN_VERBOSE=profile_create,dispatch ONEDNN_VERBOSE_TIMESTAMP=1; record_shapes=true; native synthetic M32 K128 N128; graph capture/replay; no latency benchmark' > "$OUT/compatibility/command.txt"
timeout -k 5s 120s docker run --rm --name "$NAME" --network none --device /dev/dri \
  --group-add "$(stat -c '%g' /dev/dri/renderD128)" --ipc=host \
  -v "$OUT/compatibility.py:/probe.py:ro" -v "$OUT/compatibility:/results" \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e ONEDNN_VERBOSE=profile_create,dispatch -e ONEDNN_VERBOSE_TIMESTAMP=1 \
  --entrypoint /opt/venv/bin/python \
  vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f \
  /probe.py 2>&1 | tee "$OUT/compatibility/console.log"
HOST
