#!/usr/bin/env bash
set -euo pipefail
L=$(cd -- "$(dirname -- "$0")" && pwd)
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-gemm-catalog
CELL=${1:?Pass a fresh cell name}
INDEX=${2:?Pass -1 (auto), 0, 1, or 2}
MODE=${3:-timing}
[[ $CELL =~ ^[a-z0-9-]+$ && $INDEX =~ ^(-1|0|1|2)$ && $MODE =~ ^(timing|describe)$ ]]
ssh inference-host "bash -lc 'mkdir $R/$CELL'"
scp "$L/probe.py" "$L/run-probe.sh" "inference-host:$R/"
set +e
ssh inference-host "bash -s -- '$CELL' '$INDEX' '$MODE'" <<'REMOTE'
set -euo pipefail
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-gemm-catalog
O=$R/$1
exec > >(tee "$O/driver.log") 2>&1
check() {
 date -Is
 test -z "$(docker ps -q)"
 test "$(sha256sum /home/mike/inference/launchers/start-qwen38.sh | cut -d' ' -f1)" = 63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4
 sha256sum /home/mike/inference/launchers/start-qwen38.sh
 test "$(cat /sys/class/drm/card0/device/hwmon/hwmon2/power1_cap)" = 275000000
 cat /sys/class/drm/card0/device/hwmon/hwmon2/power1_cap
 fuser /dev/dri/renderD128 || true
}
check
cp "$R/probe.py" "$R/run-probe.sh" "$O/"
sha256sum "$R/binding-build/libb70_gemm_catalog.so" | tee "$O/library-sha256.txt"
# Reject accidentally exported oneDNN symbols before touching the GPU.
nm -D --defined-only "$R/binding-build/libb70_gemm_catalog.so" > "$O/dynamic-symbols.txt"
if grep -E ' (dnnl_|_ZN4dnnl)' "$O/dynamic-symbols.txt"; then
 echo 'Unexpected exported oneDNN symbol; reject library'; exit 1
fi
verbose=0
extra=()
if [ "$3" = describe ]; then
 verbose='debuginfo=10,dispatch,profile_exec'
 extra=(--describe)
fi
trap 'docker rm -f b70-gemm-catalog-probe >/dev/null 2>&1 || true' EXIT
set +e
timeout --signal=TERM --kill-after=30s 10m docker run --pull=never --rm \
 --name b70-gemm-catalog-probe --device /dev/dri \
 --group-add "$(stat -c %g /dev/dri/renderD128)" \
 -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
 -e "B70_M5_GATEUP_CATALOG_INDEX=$2" -e "ONEDNN_VERBOSE=$verbose" \
 -v "$R:/experiment:ro" -v "$O:/output" --entrypoint /opt/venv/bin/python \
 vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f \
 -u -P /output/probe.py --library /experiment/binding-build/libb70_gemm_catalog.so \
 --out /output/results "${extra[@]}"
rc=$?
set -e
printf '%s\n' "$rc" > "$O/exit-code.txt"
docker rm -f b70-gemm-catalog-probe >/dev/null 2>&1 || true
check
exit "$rc"
REMOTE
rc=$?
set -e
scp -r "inference-host:$R/$CELL" "$L/"
exit "$rc"
