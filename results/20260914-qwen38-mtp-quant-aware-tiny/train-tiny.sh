#!/usr/bin/env bash
set -euo pipefail
ssh inference-host bash -s <<'SH'
set -eu
r=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260914-mtp-quant-aware-tiny
test -z "$(docker ps -q)"
test "$(cat /sys/class/drm/card0/device/hwmon/hwmon2/power1_cap)" = 275000000
gid=$(stat -c '%g' /dev/dri/renderD128)
docker run --rm --name qwen38-mtp-train --network none --ipc=host --device /dev/dri --group-add "$gid" -v /dev/dri:/dev/dri:ro -v "$r:/experiment" -v /home/mike/inference/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16:/model:ro -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e OMP_NUM_THREADS=4 --entrypoint python3 vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f /experiment/source/qwen38_train_mtp.py --model /model --train-dir /experiment/dataset/train --eval-dir /experiment/dataset/heldout --output /experiment/tuned-mtp.safetensors --device xpu --steps 25 --grad-accum 4 --lr 5e-6 --seed 42 --max-length 1024 --logits-chunk 32
SH
