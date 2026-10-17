#!/usr/bin/env bash
set -euo pipefail
test "$(hostname)" = inference-host
R=/home/mike/b70-evals/20261017-nemotron35-bolt-ssm-trace
BASE=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1
D=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-trace-r1
IMAGE=intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04
test "$(cat "$BASE/UPSTREAM_COMMIT")" = b6334e37e6ef286fc5091a41b6561b2113d915fb
(cd "$BASE" && git apply --check "$R/server-callback.patch")
sha256sum "$BASE/build-sycl/bin/llama-server" "$BASE/build-sycl/bin/libggml-sycl.so.0" "$BASE/build-sycl/bin/test-backend-ops" "$BASE/build-sycl/bin/libllama-server-impl.so" > "$R/accepted-before-build.sha256"
test ! -e "$D"
need=$(du -sk "$BASE" | cut -f1)
available=$(df -Pk "$(dirname "$D")" | awk 'NR==2 {print $4}')
test "$available" -gt "$((need + 2097152))"
cp -a --reflink=auto "$BASE" "$D"
(cd "$D" && patch --dry-run --fuzz=0 -p1 < "$R/server-callback.patch" && git apply --check "$R/server-callback.patch" && git apply "$R/server-callback.patch")
docker run --rm --name bolt-ssm-trace-r1-build --cpus=6 -v "$D:/src" \
    -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro -w /src "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ninja-build libcurl4-openssl-dev pkg-config
cmake -S /src -B /src/build-sycl -DLLAMA_BUILD_TESTS=ON
cmake --build /src/build-sycl --config Release -j6 --target llama-server
'
docker run --rm --name bolt-ssm-trace-r1-precheck -v "$D:/src:ro" \
    -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
export LD_LIBRARY_PATH=/src/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}
for f in /src/build-sycl/bin/llama-server /src/build-sycl/bin/libggml-sycl.so.0 /src/build-sycl/bin/test-backend-ops; do ldd "$f"; done | tee /tmp/dependencies.log
if grep -F "not found" /tmp/dependencies.log; then exit 1; fi
/src/build-sycl/bin/llama-server --version
echo BOLT_TENSOR_TRACE_RUNTIME_PRECHECK_PASSED
' > "$R/runtime-precheck.log" 2>&1
sha256sum -c "$R/accepted-before-build.sha256"
python3 -B - <<'PY' | tee "$R/post-build-verification.log"
import hashlib,json
from pathlib import Path
root=Path('/home/mike/b70-evals/20261017-nemotron35-bolt-ssm-trace')
a=Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1')
b=Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-trace-r1')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def sources(path):return {str(p.relative_to(path)):digest(p) for p in path.rglob('*') if p.is_file() and p.relative_to(path).parts[0]!='build-sycl'}
x,y=sources(a),sources(b)
changed=sorted(k for k in x.keys()|y.keys() if x.get(k)!=y.get(k))
(root/'source-diff.json').write_text(json.dumps({'changed':changed},indent=2)+'\n')
assert changed==['tools/server/server-context.cpp'],changed
names=['llama-server','libggml-sycl.so.0','test-backend-ops','libllama-server-impl.so']
x={n:digest(a/'build-sycl/bin'/n) for n in names}
y={n:digest(b/'build-sycl/bin'/n) for n in names}
assert x['libllama-server-impl.so']=='67ae7c8673110b52d11924677ad662141dc93f57d7c21d67a3f0d3e4489e09ad'
assert x['libllama-server-impl.so']!=y['libllama-server-impl.so']
assert all(x[n]==y[n] for n in names[:3])
(root/'candidate-binaries.json').write_text(json.dumps(y,indent=2)+'\n')
(root/'accepted-binaries.json').write_text(json.dumps(x,indent=2)+'\n')
print('BOLT_TENSOR_TRACE_SOURCE_AND_BINARY_SCOPE_VERIFIED')
PY
echo BOLT_TENSOR_TRACE_BUILD_PASSED
