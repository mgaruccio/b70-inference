#!/usr/bin/env bash
set -euo pipefail
test "$(hostname)" = inference-host
R=/home/mike/b70-evals/20261017-nemotron35-bolt-prefill-grouped
OLD=/home/mike/b70-evals/20261016-nemotron35-bolt-sycl
BASE=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1
D=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-grouped-tail-r1
IMAGE=intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04
test "$(cat "$BASE/UPSTREAM_COMMIT")" = b6334e37e6ef286fc5091a41b6561b2113d915fb
(cd "$BASE" && git apply --reverse --check "$OLD/ssm-prefill-r1.patch" && git apply --reverse --check "$OLD/router-r1.patch")
test ! -e "$D"
cp -a --reflink=auto "$BASE" "$D"
for p in grouped-tail-r1.patch grouped-native.patch grouped-native-string-fix.patch grouped-native-candidate.patch; do
  (cd "$D" && patch --dry-run --fuzz=0 -p1 < "$R/$p" && git apply --check "$R/$p" && git apply "$R/$p")
  sha256sum "$R/$p"
done
cmp "$D/tests/test-grouped-iq2-sycl.cpp" "$R/test-grouped-iq2-sycl-candidate.cpp"
docker run --rm --name bolt-grouped-tail-r1-build --cpus=6 -v "$D:/src" \
  -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro -w /src "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ninja-build libcurl4-openssl-dev pkg-config
cmake -S /src -B /src/build-sycl -DLLAMA_BUILD_TESTS=ON
cmake --build /src/build-sycl --config Release -j6 --target llama-server test-backend-ops test-grouped-iq2-sycl
'
docker run --rm --name bolt-grouped-tail-r1-precheck -v "$D:/src:ro" \
  -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
export LD_LIBRARY_PATH=/src/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}
for f in /src/build-sycl/bin/llama-server /src/build-sycl/bin/libggml-sycl.so.0 /src/build-sycl/bin/test-backend-ops /src/build-sycl/bin/test-grouped-iq2-sycl; do ldd "$f"; done | tee /tmp/grouped-dependencies.log
if grep -F "not found" /tmp/grouped-dependencies.log; then exit 1; fi
/src/build-sycl/bin/llama-server --version
echo BOLT_GROUPED_TAIL_RUNTIME_PRECHECK_PASSED
' > "$R/tail-runtime-precheck.log" 2>&1
sha256sum "$D/build-sycl/bin/llama-server" "$D/build-sycl/bin/libggml-sycl.so.0" \
 "$D/build-sycl/bin/test-backend-ops" "$D/build-sycl/bin/test-grouped-iq2-sycl" > "$R/tail-binaries.sha256"
