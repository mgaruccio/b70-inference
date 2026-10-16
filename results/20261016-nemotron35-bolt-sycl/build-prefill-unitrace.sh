#!/usr/bin/env bash
# Isolated profiling tool, never installed into the host or accepted model source.
set -euo pipefail
test "$(hostname)" = inference-host
ROOT=/home/mike/b70-evals/20261016-nemotron35-bolt-sycl
OUT="$ROOT/prefill-unitrace-tool"
test ! -e "$OUT"
mkdir "$OUT"
exec > >(tee "$OUT/build.log") 2>&1
git clone --depth 1 --branch pti-1.0.0 https://github.com/intel/pti-gpu.git "$OUT/pti-gpu"
git -C "$OUT/pti-gpu" rev-parse HEAD | tee "$OUT/pti-commit.txt"
docker run --rm --name bolt-prefill-unitrace-build --label b70.experiment=bolt-fusion \
  -v "$OUT:/work" intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04 \
  bash -lc 'set -eo pipefail; cmake -S /work/pti-gpu/tools/unitrace -B /work/build -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MPI=OFF -DBUILD_WITH_ITT=0 -DBUILD_WITH_XPTI=0 -DBUILD_WITH_OPENCL=0; cmake --build /work/build -j 6; /work/build/unitrace --version; /work/build/unitrace --help'
