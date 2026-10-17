docker run --rm --name bolt-nvfp4-prefill-r1-build --cpus=6 -v "$D:/src" \
  -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro -w /src "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ninja-build libcurl4-openssl-dev pkg-config
cmake -S /src -B /src/build-sycl -DLLAMA_BUILD_TESTS=ON
cmake --build /src/build-sycl --config Release -j6 --target llama-server test-backend-ops test-nvfp4-sycl
'
docker run --rm --name bolt-nvfp4-prefill-r1-precheck -v "$D:/src:ro" \
  -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
export LD_LIBRARY_PATH=/src/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}
for f in /src/build-sycl/bin/llama-server /src/build-sycl/bin/libggml-sycl.so.0 /src/build-sycl/bin/test-backend-ops /src/build-sycl/bin/test-nvfp4-sycl; do ldd "$f"; done | tee /tmp/nvfp4-prefill-dependencies.log
if grep -F "not found" /tmp/nvfp4-prefill-dependencies.log; then exit 1; fi
/src/build-sycl/bin/llama-server --version
sha256sum /opt/onednn/lib/libdnnl.so.3
echo BOLT_NVFP4_PREFILL_RUNTIME_PRECHECK_PASSED
' > "$R/nvfp4-prefill-r1-runtime-precheck.log" 2>&1
sha256sum "$D/build-sycl/bin/llama-server" "$D/build-sycl/bin/libggml-sycl.so.0" \
 "$D/build-sycl/bin/test-backend-ops" "$D/build-sycl/bin/test-nvfp4-sycl" > "$R/nvfp4-prefill-r1-binaries.sha256"
