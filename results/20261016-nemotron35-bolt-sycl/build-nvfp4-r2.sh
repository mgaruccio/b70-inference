#!/usr/bin/env bash
# Host-only, CPU-only build; do not edit while running.
set -euo pipefail
R=/home/mike/b70-evals/20261016-nemotron35-bolt-sycl
D=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-r1
test "$(cat "$D/UPSTREAM_COMMIT")" = b6334e37e6ef286fc5091a41b6561b2113d915fb
(cd "$D" && git apply --reverse --check "$R/nvfp4-reorder-r1.patch")
docker run --rm --name bolt-nvfp4-r1-build -v "$D:/src" \
  -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro -w /src \
  intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04 bash -lc '
    set -eo pipefail
    source /opt/intel/oneapi/setvars.sh --force
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ninja-build libcurl4-openssl-dev pkg-config
    cmake -S /src -B /src/build-sycl -DLLAMA_BUILD_TESTS=ON
    cmake --build /src/build-sycl --config Release -j6 --target llama-server test-backend-ops llama-quantize
    sha256sum /src/build-sycl/bin/llama-server /src/build-sycl/bin/libggml-sycl.so.0 \
      /src/build-sycl/bin/test-backend-ops /src/build-sycl/bin/llama-quantize
  '
