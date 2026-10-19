#!/usr/bin/env bash
# CPU-only temporary tool build. No application source, host packages, or GPU mounts.
set -euo pipefail
test "$(hostname)" = inference-host
ROOT=/home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-w3-kernel-profile
IMAGE=intel/deep-learning-essentials@sha256:aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0
cd "$ROOT"
test ! -e tools
mkdir tools
exec > >(tee tool-build.log) 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > tool-build-exit-code.txt' EXIT
test "$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')" -gt 6291456
docker run --rm --pull=never --name bolt-unitrace-tool-build --cpus=6 --memory=8g \
  --mount type=tmpfs,destination=/tmp,tmpfs-size=2147483648 \
  -v "$ROOT/tools:/export:rw" "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install --no-upgrade -y -qq git build-essential cmake ninja-build python3
cd /tmp
git clone --depth 1 --branch pti-1.0.0 https://github.com/intel/pti-gpu.git
cd pti-gpu
test "$(git rev-parse HEAD)" = 887bba6e28ce84cc0d3813ef876e24add107c318
cmake -S tools/unitrace -B /tmp/unitrace-build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_WITH_MPI=0 -DBUILD_WITH_ITT=0 -DBUILD_WITH_XPTI=0 -DBUILD_WITH_OPENCL=0
cmake --build /tmp/unitrace-build -j6 --target unitrace unitrace_tool
cp /tmp/unitrace-build/unitrace /tmp/unitrace-build/libunitrace_tool.so /export/
sha256sum /export/unitrace /export/libunitrace_tool.so > /export/SHA256SUMS
/export/unitrace --help > /export/help.txt
ldd /export/unitrace > /export/dependencies.txt
ldd /export/libunitrace_tool.so >> /export/dependencies.txt
! grep -F "not found" /export/dependencies.txt
'
test "$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')" -gt 5242880
echo UNITRACE_TOOL_BUILD_PASSED_NO_GPU_ACTIONS
