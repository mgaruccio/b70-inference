#!/usr/bin/env bash
# Real native XPU state gate. Run only after the baseline container is stopped.
set -euo pipefail
R=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMAGE=vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4
TARGET=/home/mike/inference/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
[ -z "$(docker ps -q)" ]
[ "$(cat /sys/class/drm/card0/device/hwmon/hwmon2/power1_cap)" = 275000000 ]
[ ! -e "$R/native" ] || { echo 'Refusing to overwrite native results.' >&2; exit 2; }
mkdir "$R/native"
for DTYPE in float16 bfloat16; do
  COMMAND=(timeout --signal=TERM --kill-after=120s 30m docker run --rm --name qwen38-native-verification --ipc=host --device /dev/dri --group-add "$(stat -c %g /dev/dri/renderD128)" -v /dev/dri:/dev/dri:ro -v "$R/code/scripts:/scripts:ro" -v "$TARGET/config.json:/target-config.json:ro" -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 --entrypoint bash "$IMAGE" -lc "set -e; python /scripts/patch-vllm-qwen38-xpu-boundary.py; python /scripts/check-qwen38-xpu-boundary-state.py --config /target-config.json --dtype $DTYPE --alternating-depths")
  printf '%q ' "${COMMAND[@]}" > "$R/native/$DTYPE.command.txt"; printf '\n' >> "$R/native/$DTYPE.command.txt"
  set +e
  "${COMMAND[@]}" 2>&1 | tee "$R/native/$DTYPE.console.txt"
  code=${PIPESTATUS[0]}
  set -e
  printf '%s\t%s\n' "$DTYPE" "$code" | tee -a "$R/native/exits.tsv"
  [ "$code" -eq 0 ] || exit "$code"
done
[ -z "$(docker ps -q)" ]
printf 'NATIVE_STATE_GATE_COMPLETE UTC=%s\n' "$(date -u +%FT%TZ)"
