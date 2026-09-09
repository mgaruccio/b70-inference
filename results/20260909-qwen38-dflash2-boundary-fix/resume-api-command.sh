#!/usr/bin/env bash
set -euo pipefail
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-dflash2-boundary-fix
scp -q /tmp/resume-qwen38-boundary-api.sh "inference-host:$R/resume-api-command.sh"
ssh -o BatchMode=yes -o ConnectTimeout=10 inference-host 'bash -s' <<'REMOTE'
set -euo pipefail
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-dflash2-boundary-fix
IMAGE=vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4
TARGET=/home/mike/inference/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
DRAFT=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-dflash2-quant/rtn-int4-g128
GUARD=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-dflash2/runner/patch_uniform_decode_prefill.py
[ -z "$(docker ps -q)" ]
[ "$(cat /sys/class/drm/card0/device/hwmon/hwmon2/power1_cap)" = 275000000 ]
[ "$(docker inspect -f '{{.State.Running}}' glimmer-tb21-prefix-c8)" = false ]
printf '63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4  /home/mike/inference/launchers/start-qwen38.sh\n' | sha256sum -c -
python3 - "$R/native-v2/float16.log" <<'PY'
import json,sys
result=json.loads(open(sys.argv[1]).read().splitlines()[-1])
assert result['result']=='PASS' and result['partial_cases']==224 and result['continuations']==896,result
print('NATIVE_FP16_GATE_PASSED')
PY
COMMAND=(docker run --rm --name qwen38-boundary-state-v2 --ipc=host --network none --device /dev/dri --group-add "$(stat -c %g /dev/dri/renderD128)" -v /dev/dri:/dev/dri:ro -v "$R/native-v2:/check:ro" -v "$TARGET:/model:ro" -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 -e PYTHONDONTWRITEBYTECODE=1 --entrypoint bash "$IMAGE" -lc 'set -e; python /check/patch-vllm-qwen38-xpu-boundary.py; python /check/check-qwen38-xpu-boundary-state.py --config /model/config.json --dtype bfloat16')
printf '%q ' "${COMMAND[@]}" > "$R/native-v2/bfloat16.command.txt"; printf '\n' >> "$R/native-v2/bfloat16.command.txt"
"${COMMAND[@]}" > "$R/native-v2/bfloat16.log" 2>&1
cat "$R/native-v2/bfloat16.log"
printf 'native-v2-float16\t0\nnative-v2-bfloat16\t0\n' >> "$R/cell-exits.tsv"
for ARM in bf16 int4; do
  EXTRA=()
  if [ "$ARM" = int4 ]; then EXTRA=(--draft-int4 "$DRAFT"); fi
  COMMAND=(timeout --signal=TERM --kill-after=30s 60m python3 -u "$R/code/scripts/experiments/qwen38_boundary_api.py" --guard "$GUARD" --out "$R/$ARM-api" "${EXTRA[@]}")
  printf '%q ' "${COMMAND[@]}" > "$R/$ARM-api.command.txt"; printf '\n' >> "$R/$ARM-api.command.txt"
  "${COMMAND[@]}" 2>&1 | tee "$R/$ARM-api.log"
  printf '%s-api\t0\n' "$ARM" | tee -a "$R/cell-exits.tsv"
done
{
 sha256sum /home/mike/inference/launchers/start-qwen38.sh
 cat /sys/class/drm/card0/device/hwmon/hwmon2/power1_cap
 docker inspect -f '{{.State.Running}}' glimmer-tb21-prefix-c8
 docker ps --format '{{.Names}}'
} | tee "$R/cleanup.txt"
REMOTE
