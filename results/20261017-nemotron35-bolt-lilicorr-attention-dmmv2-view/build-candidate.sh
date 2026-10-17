#!/usr/bin/env bash
# CPU-only build/dependency gate. Run on inference-host using the parent's background runner.
set -euo pipefail
test "$(hostname)" = inference-host
R=/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-attention-dmmv2-view
PARENT=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-attention-route-r1
D=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-attention-dmmv2-view-r1
IMAGE=intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04
cd "$R"
# Never overwrite an earlier build attempt or silently reuse a partial candidate.
test ! -e "$D"
mkdir "$R/build-attempt-r1"
exec > >(tee "$R/build-attempt-r1/build.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$R/build-attempt-r1/exit-code.txt"' EXIT
set -x
test "$(cat "$PARENT/UPSTREAM_COMMIT")" = b6334e37e6ef286fc5091a41b6561b2113d915fb
(cd "$PARENT" && patch --fuzz=0 --dry-run --batch --forward -p1 < "$R/attention-dmmv2.patch")
python3 -B - <<'PY'
import runpy
m=runpy.run_path('run-trace.py')
r=m['ROOT']
for label,source in [('accepted',m['ACCEPTED']),('parent',m['PARENT'])]:
    actual=m['binaries'](source)
    prior=m['load'](m['PARENT_RESULTS']/('accepted-binaries.json' if label=='accepted' else 'candidate-binaries.json'))
    assert set(prior)==set(m['ARTIFACTS'])
    assert all(actual[k]==v for k,v in prior.items()), label
    m['save'](r/(label+'-binaries.json'),actual)
    (r/(label+'-before-build.sha256')).write_text(''.join(f'{h}  {source}/build-sycl/bin/{name}\n' for name,h in actual.items()))
PY
need=$(du -sk "$PARENT" | cut -f1)
available=$(df -Pk "$(dirname "$D")" | awk 'NR==2 {print $4}')
test "$available" -gt "$((need + 4194304))"
cp -a --reflink=auto "$PARENT" "$D"
(cd "$D" && patch --fuzz=0 --batch --forward -p1 < "$R/attention-dmmv2.patch")
docker run --rm --name bolt-lilicorr-attention-dmmv2-view-r1-build --cpus=6 -v "$D:/src" \
    -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro -w /src "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install --no-upgrade -y -qq ninja-build libcurl4-openssl-dev pkg-config
cmake -S /src -B /src/build-sycl -DLLAMA_BUILD_TESTS=ON
cmake --build /src/build-sycl --config Release -j6 --target llama-server test-backend-ops
'
docker run --rm --network none --name bolt-lilicorr-attention-dmmv2-view-r1-precheck -v "$D:/src:ro" \
    -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
export LD_LIBRARY_PATH=/src/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}
for f in llama-server libllama-server-impl.so libggml-sycl.so.0 test-backend-ops libllama-common.so; do ldd "/src/build-sycl/bin/$f"; done | tee /tmp/dependencies.log
if grep -F "not found" /tmp/dependencies.log; then exit 1; fi
echo BOLT_VERIFY_DEPENDENCIES_PASSED
' > "$R/runtime-precheck.log" 2>&1
sha256sum -c "$R/accepted-before-build.sha256"
sha256sum -c "$R/parent-before-build.sha256"
python3 -B - <<'PY' | tee "$R/post-build-verification.log"
import hashlib,runpy
m=runpy.run_path('run-trace.py')
def sources(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and p.relative_to(root).parts[0]!='build-sycl'}
a,b=sources(m['PARENT']),sources(m['CANDIDATE'])
changed=sorted(k for k in a.keys()|b.keys() if a.get(k)!=b.get(k))
assert changed==m['SOURCE_SCOPE'],changed
m['save'](m['ROOT']/'source-diff.json',dict(changed_relative_to_parent=changed))
x,y=m['binaries'](m['PARENT']),m['binaries'](m['CANDIDATE'])
changed_artifacts=[n for n in m['ARTIFACTS'] if x[n]!=y[n]]
assert set(changed_artifacts)=={'libggml-sycl.so.0','test-backend-ops'},(changed_artifacts,x,y)
assert x['libggml-sycl.so.0']!=y['libggml-sycl.so.0'],(x,y)
assert x['test-backend-ops']!=y['test-backend-ops'],(x,y)
unchanged=['llama-server','libllama-server-impl.so','libllama-common.so']
assert all(x[n]==y[n] for n in unchanged),(x,y)
m['save'](m['ROOT']/'candidate-binaries.json',y)
print('BOLT_VERIFY_BUILD_PASSED')
PY
