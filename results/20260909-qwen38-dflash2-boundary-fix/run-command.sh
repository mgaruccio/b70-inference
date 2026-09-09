#!/usr/bin/env bash
set -euo pipefail
REPO=/home/mike/code/b70-inference
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-dflash2-boundary-fix
ssh -n -o BatchMode=yes inference-host "test ! -e '$R' && mkdir -p '$R/code/scripts/experiments'"
tar -C "$REPO" -cf - scripts/start-qwen38.sh scripts/check-qwen38-xpu-boundary-state.py scripts/patch-vllm-qwen38-xpu-boundary.py scripts/experiments/qwen38_standard_bench.py scripts/experiments/qwen38_mtp_reference.py scripts/experiments/qwen38_long_context_bench.py scripts/experiments/qwen38_dflash2_probe.py scripts/experiments/qwen38_lossy_probe.py scripts/patch-vllm-qwen38-dflash2-bf16.py scripts/patch-vllm-qwen38-xpu-prefill.py tests/test_qwen38_xpu_boundary.py tests/test_qwen38_dflash2_probe.py tests/test_qwen38_standard_bench.py tests/test_qwen38_long_context_bench.py | ssh inference-host "tar -C '$R/code' -xf -"
scp -q /tmp/qwen38-boundary-api.py "inference-host:$R/code/scripts/experiments/qwen38_boundary_api.py"
scp -q /tmp/run-qwen38-boundary-fix.sh "inference-host:$R/run-command.sh"
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
cd "$R/code"
python3 -m unittest discover -s tests -p 'test_qwen38_*.py' -v > "$R/cpu-tests.txt" 2>&1
cat "$R/cpu-tests.txt"
for DTYPE in float16 bfloat16; do
  COMMAND=(docker run --rm --name qwen38-boundary-state --ipc=host --network none --device /dev/dri --group-add "$(stat -c %g /dev/dri/renderD128)" -v /dev/dri:/dev/dri:ro -v "$R/code:/code:ro" -v "$TARGET:/model:ro" -e VLLM_TARGET_DEVICE=xpu -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 -e PYTHONDONTWRITEBYTECODE=1 --entrypoint bash "$IMAGE" -lc "set -e; python /code/scripts/patch-vllm-qwen38-xpu-boundary.py; python /code/scripts/check-qwen38-xpu-boundary-state.py --config /model/config.json --dtype $DTYPE")
  printf '%q ' "${COMMAND[@]}" > "$R/native-$DTYPE.command.txt"; printf '\n' >> "$R/native-$DTYPE.command.txt"
  "${COMMAND[@]}" 2>&1 | tee "$R/native-$DTYPE.log"
  printf 'native-%s\t0\n' "$DTYPE" | tee -a "$R/cell-exits.tsv"
done
for ARM in bf16 int4; do
  EXTRA=()
  if [ "$ARM" = int4 ]; then EXTRA=(--draft-int4 "$DRAFT"); fi
  COMMAND=(python3 -u "$R/code/scripts/experiments/qwen38_boundary_api.py" --guard "$GUARD" --out "$R/$ARM-api" "${EXTRA[@]}")
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
