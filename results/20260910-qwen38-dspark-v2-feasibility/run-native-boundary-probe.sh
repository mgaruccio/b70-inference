#!/usr/bin/env bash
set -euo pipefail
cd /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260910-dspark-v2-feasibility
name=dspark-boundary-probe
[[ -z "$(docker ps -q)" ]]
[[ "$(cat /sys/class/drm/card0/device/hwmon/hwmon2/power1_cap)" == 275000000 ]]
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
set -x
timeout --signal=INT --kill-after=20s 600s docker run --rm --name "$name" --network none \
  --device /dev/dri --group-add "$(stat -c %g /dev/dri/renderD128)" \
  -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -v "$PWD:/check:ro" -v /home/mike/inference/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16:/model:ro --entrypoint bash \
  vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4 \
  -lc 'set -e; /opt/venv/bin/python -P /check/patch-vllm-qwen38-xpu-boundary.py; /opt/venv/bin/python -P /check/check-qwen38-xpu-boundary-state.py --config /model/config.json --dtype float16 --alternating-depths'
sha256sum /home/mike/inference/launchers/start-qwen38.sh
cat /sys/class/drm/card0/device/hwmon/hwmon2/power1_cap
docker ps --format '{{.Names}}'
