#!/usr/bin/env bash
# Host-only pristine CPU quantizer. No GPU access; source stays separate.
set -euo pipefail
D=/home/mike/inference/src/llama.cpp-bolt-b6334e37-requant
test "$(cat "$D/UPSTREAM_COMMIT")" = b6334e37e6ef286fc5091a41b6561b2113d915fb
docker run --rm --name bolt-requant-cpu-build -v "$D:/src" -w /src \
  intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04 bash -lc '
    set -eo pipefail
    cmake -S /src -B /src/build-requant-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DGGML_SYCL=OFF -DGGML_VULKAN=OFF -DGGML_CUDA=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=ON \
      -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
    cmake --build /src/build-requant-cpu --config Release -j2 --target llama-quantize
    sha256sum /src/build-requant-cpu/bin/llama-quantize /src/build-requant-cpu/bin/libllama.so.0
  '
