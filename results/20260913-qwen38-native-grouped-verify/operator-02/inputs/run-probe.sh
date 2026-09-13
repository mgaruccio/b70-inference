#!/usr/bin/env bash
# Lead-run ON inference-host, after explicit idle-host approval. No serving edits.
set -euo pipefail
R=$(cd -- "$(dirname -- "$0")" && pwd -P)
EXPECTED=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-native-grouped-verify
[[ $(hostname -s) == inference-host && $R == "$EXPECTED" ]]
[[ ${B70_IDLE_CONFIRMED:-0} == 1 ]]
BUILD=${1:?Pass successful build-NN}
CELL=${2:?Pass fresh operator-NN}
[[ $BUILD =~ ^build-[0-9]+$ && $CELL =~ ^operator-[0-9]+$ ]]
[[ $(cat "$R/$BUILD/exit-code.txt") == 0 ]]
NAME=b70-grouped-verify-$CELL
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "Refusing to reuse existing container $NAME" >&2
  exit 1
fi
O=$R/$CELL
mkdir "$O"
exec > >(tee "$O/driver.log") 2>&1
check_host() {
  date -Is
  test -z "$(docker ps -q)" || return 1
  sha256sum /home/mike/inference/launchers/start-qwen38.sh || return 1
  test "$(sha256sum /home/mike/inference/launchers/start-qwen38.sh | cut -d' ' -f1)" = 63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4 || return 1
  cat /sys/class/drm/card0/device/hwmon/hwmon2/power1_cap || return 1
  test "$(cat /sys/class/drm/card0/device/hwmon/hwmon2/power1_cap)" = 275000000 || return 1
  fuser /dev/dri/renderD128 || true
}
finish() {
  rc=$?
  trap - EXIT
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  # Preserve the original probe failure even if post-checking also fails.
  (set -e; check_host) || { if [[ $rc == 0 ]]; then rc=1; fi; }
  printf '%s\n' "$rc" > "$O/exit-code.txt"
  exit "$rc"
}
trap finish EXIT
set -x
check_host
uname -a
cp -a "$R/$BUILD/inputs" "$O/inputs"
cp "$R/run-probe.sh" "$O/run-probe.sh"
sha256sum "$R/$BUILD/build/libb70_grouped_verify.so"
docker image inspect vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f > "$O/image.json"
timeout --signal=TERM --kill-after=30s 45m docker run --pull=never --rm \
  --name "$NAME" --device /dev/dri \
  --group-add "$(stat -c %g /dev/dri/renderD128)" \
  --cpus=12 --memory=24g --memory-swap=24g --network=none \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -v "$R/$BUILD/build:/candidate:ro" -v "$O:/output" \
  --entrypoint /bin/bash \
  vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f -lc '
set -euxo pipefail
/opt/venv/bin/python -m torch.utils.collect_env > /output/collect-env.txt 2>&1
exec /opt/venv/bin/python -u -P /output/inputs/probe.py \
  --library /candidate/libb70_grouped_verify.so --output /output
'
