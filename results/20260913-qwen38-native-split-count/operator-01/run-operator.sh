#!/usr/bin/env bash
# Execute locally; all ML runs in the remote disposable pinned-image container.
set -euo pipefail
L=$(cd -- "$(dirname -- "$0")" && pwd)
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-native-split-count
CELL=${1:?Pass a fresh operator-NN directory name}
[[ $CELL =~ ^operator-[0-9]+$ ]]
ssh inference-host "bash -lc 'mkdir -p $R; mkdir $R/$CELL'"
scp "$L/probe.py" "$L/README.md" "$L/run-operator.sh" "inference-host:$R/"
set +e
ssh inference-host "bash -s -- '$CELL'" <<'REMOTE'
set -euo pipefail
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-native-split-count
O=$R/$1
exec > >(tee "$O/driver.log") 2>&1
check() {
 date -Is
 test -z "$(docker ps -q)"
 sha256sum /home/mike/inference/launchers/start-qwen38.sh
 test "$(sha256sum /home/mike/inference/launchers/start-qwen38.sh | cut -d' ' -f1)" = 63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4
 cat /sys/class/drm/card0/device/hwmon/hwmon2/power1_cap
 test "$(cat /sys/class/drm/card0/device/hwmon/hwmon2/power1_cap)" = 275000000
 fuser /dev/dri/renderD128 || true
}
check
cp "$R/probe.py" "$O/probe.py"
cp "$R/README.md" "$O/protocol.md"
cp "$R/run-operator.sh" "$O/run-operator.sh"
trap 'docker rm -f b70-native-split-probe >/dev/null 2>&1 || true' EXIT
set +e
timeout --signal=TERM --kill-after=30s 15m docker run --pull=never --rm --name b70-native-split-probe \
 --device /dev/dri --group-add "$(stat -c %g /dev/dri/renderD128)" \
 -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
 -v "$O:/output" --entrypoint /opt/venv/bin/python \
 vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f -u -P /output/probe.py
rc=$?
set -e
printf '%s\n' "$rc" > "$O/exit-code.txt"
docker rm -f b70-native-split-probe >/dev/null 2>&1 || true
check
exit "$rc"
REMOTE
rc=$?
set -e
scp -r "inference-host:$R/$CELL" "$L/"
exit "$rc"
