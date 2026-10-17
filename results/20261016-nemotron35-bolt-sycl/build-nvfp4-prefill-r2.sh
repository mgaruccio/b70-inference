#!/usr/bin/env bash
# CPU-only build on inference-host; accepted SSM source is read-only baseline.
set -euo pipefail
R=/home/mike/b70-evals/20261016-nemotron35-bolt-sycl
BASE=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1
D=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-prefill-pack-r2
IMAGE=intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04
test "$(hostname)" = inference-host
test "$(cat "$BASE/UPSTREAM_COMMIT")" = b6334e37e6ef286fc5091a41b6561b2113d915fb
(cd "$BASE" && git apply --reverse --check "$R/ssm-prefill-r1.patch" && git apply --reverse --check "$R/router-r1.patch")
test ! -e "$D"
cp -a --reflink=auto "$BASE" "$D"
(cd "$D" && git apply --check --whitespace=error "$R/nvfp4-prefill-r2.patch" && git apply --whitespace=error "$R/nvfp4-prefill-r2.patch")
sha256sum "$R/nvfp4-prefill-r2.patch"
docker run --rm --name bolt-nvfp4-prefill-r2-build --cpus=6 -v "$D:/src" \
  -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro -w /src "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ninja-build libcurl4-openssl-dev pkg-config
cmake -S /src -B /src/build-sycl -DLLAMA_BUILD_TESTS=ON
cmake --build /src/build-sycl --config Release -j6 --target llama-server test-backend-ops test-nvfp4-sycl-r2
'
docker run --rm --name bolt-nvfp4-prefill-r2-precheck -v "$D:/src:ro" \
  -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
export LD_LIBRARY_PATH=/src/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}
for f in /src/build-sycl/bin/llama-server /src/build-sycl/bin/libggml-sycl.so.0 /src/build-sycl/bin/test-backend-ops /src/build-sycl/bin/test-nvfp4-sycl-r2; do ldd "$f"; done | tee /tmp/nvfp4-prefill-dependencies.log
if grep -F "not found" /tmp/nvfp4-prefill-dependencies.log; then exit 1; fi
/src/build-sycl/bin/llama-server --version
sha256sum /opt/onednn/lib/libdnnl.so.3
echo BOLT_NVFP4_PREFILL_RUNTIME_PRECHECK_PASSED
' > "$R/nvfp4-prefill-r2-runtime-precheck.log" 2>&1
sha256sum "$D/build-sycl/bin/llama-server" "$D/build-sycl/bin/libggml-sycl.so.0" \
 "$D/build-sycl/bin/test-backend-ops" "$D/build-sycl/bin/test-nvfp4-sycl-r2" > "$R/nvfp4-prefill-r2-binaries.sha256"
