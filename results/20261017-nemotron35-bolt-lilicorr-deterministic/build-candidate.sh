#!/usr/bin/env bash
set -euo pipefail
test "$(hostname)" = inference-host
R=/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-deterministic
REF=/home/mike/b70-evals/20261017-nemotron35-bolt-dnn-deterministic
ACCEPTED=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1
BASE=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-dnn-deterministic-r1
D=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-deterministic-r1
IMAGE=intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04
test "$(cat "$BASE/UPSTREAM_COMMIT")" = b6334e37e6ef286fc5091a41b6561b2113d915fb
(cd "$BASE" && git apply --reverse --check "$REF/dnn-deterministic.patch" && git apply --check "$R/outprod-layout-r1.patch" "$R/outprod-tests-r1.patch")
python3 -B - <<'PY'
import hashlib,json
from pathlib import Path
r=Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-deterministic')
ref=Path('/home/mike/b70-evals/20261017-nemotron35-bolt-dnn-deterministic')
for label,suffix,receipt in [('accepted','ssm-prefill-r1','accepted-binaries.json'),('baseline','dnn-deterministic-r1','candidate-binaries.json')]:
 p=Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-'+suffix)/'build-sycl/bin'
 expected=json.loads((ref/receipt).read_text())
 actual={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in expected}
 assert actual==expected,(label,actual)
 (r/(label+'-binaries.json')).write_text(json.dumps(actual,indent=2)+'\n')
 (r/(label+'-before-build.sha256')).write_text(''.join(f'{h}  {p/n}\n' for n,h in actual.items()))
PY
test ! -e "$D"
need=$(du -sk "$BASE" | cut -f1)
available=$(df -Pk "$(dirname "$D")" | awk 'NR==2 {print $4}')
test "$available" -gt "$((need + 2097152))"
cp -a --reflink=auto "$BASE" "$D"
for p in outprod-layout-r1.patch outprod-tests-r1.patch; do
    (cd "$D" && patch --dry-run --fuzz=0 -p1 < "$R/$p" && git apply --check "$R/$p" && git apply "$R/$p")
done
docker run --rm --name bolt-lilicorr-deterministic-r1-build --cpus=6 -v "$D:/src" \
    -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro -w /src "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ninja-build libcurl4-openssl-dev pkg-config
cmake -S /src -B /src/build-sycl -DLLAMA_BUILD_TESTS=ON
cmake --build /src/build-sycl --config Release -j6 --target llama-server test-backend-ops
'
docker run --rm --name bolt-lilicorr-deterministic-r1-precheck -v "$D:/src:ro" \
    -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
export LD_LIBRARY_PATH=/src/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}
for f in llama-server libllama-server-impl.so libggml-sycl.so.0 test-backend-ops; do ldd "/src/build-sycl/bin/$f"; done | tee /tmp/dependencies.log
if grep -F "not found" /tmp/dependencies.log; then exit 1; fi
/src/build-sycl/bin/llama-server --version
echo BOLT_LILICORR_DETERMINISTIC_RUNTIME_PRECHECK_PASSED
' > "$R/runtime-precheck.log" 2>&1
sha256sum -c "$R/accepted-before-build.sha256"
sha256sum -c "$R/baseline-before-build.sha256"
python3 -B - <<'PY' | tee "$R/post-build-verification.log"
import hashlib,json
from pathlib import Path
r=Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-deterministic')
a=Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-dnn-deterministic-r1')
b=Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-deterministic-r1')
prior=Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-outprod-r1')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sources(p):return {str(f.relative_to(p)):digest(f) for f in p.rglob('*') if f.is_file() and f.relative_to(p).parts[0]!='build-sycl'}
x,y=sources(a),sources(b)
changed=sorted(k for k in x.keys()|y.keys() if x.get(k)!=y.get(k))
expected=['ggml/src/ggml-sycl/ggml-sycl.cpp','ggml/src/ggml-sycl/outprod.cpp','ggml/src/ggml-sycl/outprod.hpp','tests/test-backend-ops.cpp']
assert changed==expected,changed
assert all((b/p).read_bytes()==(prior/p).read_bytes() for p in expected)
(r/'source-diff.json').write_text(json.dumps({'changed_relative_to_baseline':changed,'outprod_sources_match_prior':True},indent=2)+'\n')
x=json.loads((r/'baseline-binaries.json').read_text())
y={n:digest(b/'build-sycl/bin'/n) for n in x}
assert all(x[n]==y[n] for n in ('llama-server','libllama-server-impl.so'))
assert all(x[n]!=y[n] for n in ('libggml-sycl.so.0','test-backend-ops'))
(r/'candidate-binaries.json').write_text(json.dumps(y,indent=2)+'\n')
print('BOLT_LILICORR_DETERMINISTIC_BUILD_VERIFIED')
PY
