#!/usr/bin/env bash
# Local orchestration; source/compiler files exist only on inference-host.
set -euo pipefail
L=$(cd -- "$(dirname -- "$0")" && pwd)
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-gemm-catalog
ssh inference-host "bash -lc 'mkdir $R; mkdir $R/preparation-01'"
scp "$L/README.md" "$L/prepare.sh" "inference-host:$R/"
set +e
ssh inference-host 'bash -s' <<'REMOTE'
set -euo pipefail
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-gemm-catalog
exec > >(tee "$R/preparation-01/driver.log") 2>&1
set -x
date -Is
test -z "$(docker ps -q)"
sha256sum /home/mike/inference/launchers/start-qwen38.sh
cat /sys/class/drm/card0/device/hwmon/hwmon2/power1_cap
cp "$R/prepare.sh" "$R/preparation-01/prepare.sh"
cp "$R/README.md" "$R/preparation-01/protocol.md"
mkdir "$R/compiler"
trap 'docker rm -f b70-gemm-compiler-copy >/dev/null 2>&1 || true' EXIT
docker create --name b70-gemm-compiler-copy intel/deep-learning-essentials@sha256:aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0 /bin/true
docker cp b70-gemm-compiler-copy:/opt/intel/oneapi/compiler/2026.0/. "$R/compiler/"
docker rm b70-gemm-compiler-copy
git init "$R/oneDNN"
git -C "$R/oneDNN" remote add origin https://github.com/uxlfoundation/oneDNN.git
git -C "$R/oneDNN" fetch --depth=1 origin 80afa71049cd69a3df32adcccb623b12cd7baa22
git -C "$R/oneDNN" checkout --detach FETCH_HEAD
git -C "$R/oneDNN" submodule update --init --recursive --depth=1
git clone --depth=1 --branch v0.1.12 --filter=blob:none --sparse https://github.com/vllm-project/vllm-xpu-kernels.git "$R/native"
git -C "$R/native" sparse-checkout set csrc/xpu/onednn
git -C "$R/oneDNN" rev-parse HEAD
git -C "$R/oneDNN" submodule status --recursive
git -C "$R/native" rev-parse HEAD
du -sh "$R/compiler" "$R/oneDNN" "$R/native"
date -Is
REMOTE
rc=$?
set -e
printf '%s\n' "$rc" > "$L/preparation-exit-code.txt"
scp -r "inference-host:$R/preparation-01" "$L/"
exit "$rc"
