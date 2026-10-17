#!/usr/bin/env bash
set -euo pipefail
test "$(hostname)" = inference-host
D=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-grouped-diag-r1
IMAGE=intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04
docker run --rm --name bolt-grouped-native-build --cpus=6 -v "$D:/src" \
  -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro -w /src "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ninja-build libcurl4-openssl-dev pkg-config
cmake -S /src -B /src/build-sycl -DLLAMA_BUILD_TESTS=ON
cmake --build /src/build-sycl --config Release -j6 --target test-grouped-iq2-sycl
'
