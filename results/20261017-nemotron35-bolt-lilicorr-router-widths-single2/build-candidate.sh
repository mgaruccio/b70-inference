#!/usr/bin/env bash
# LEAD ONLY: isolated CPU build; inherited disposable-container bootstrap, no host changes.
set -euo pipefail
test "$(hostname)" = inference-host
R=/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-router-widths-single2
PARENT=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-single2-r1
D=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-widths-single2-r1
IMAGE=intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04
cd "$R"
test ! -e "$D"
mkdir "$R/build-attempt-r1"
exec > >(tee "$R/build-attempt-r1/build.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$R/build-attempt-r1/exit-code.txt"' EXIT
set -x
test "$(cat "$PARENT/UPSTREAM_COMMIT")" = b6334e37e6ef286fc5091a41b6561b2113d915fb
(cd "$PARENT" && git apply --check --verbose "$R/router-widths-single2.patch")
(cd "$PARENT" && patch --fuzz=0 --dry-run --batch --forward -p1 < "$R/router-widths-single2.patch")
python3 -B - <<'PY'
import hashlib, runpy
m = runpy.run_path('run-trace.py')
assert hashlib.sha256((m['PARENT']/m['SOURCE_SCOPE'][0]).read_bytes()).hexdigest() == '430c67f15a938601611cc5e73a3d25b07fb4fe35b688782833a875ebd522c920'
assert hashlib.sha256((m['PARENT']/m['SOURCE_SCOPE'][1]).read_bytes()).hexdigest() == 'f694ba3793ff6cd511fae0b3edfe92ef1037d5236a925ac940d484df5e2e5672'
protected = {}
for label, source in [('accepted', m['ACCEPTED']), ('parent', m['PARENT'])]:
    actual = m['binaries'](source)
    prior = m['load'](m['PARENT_RESULTS'] / ('accepted-binaries.json' if label == 'accepted' else 'candidate-binaries.json'))
    assert set(prior) == set(m['ARTIFACTS']) and actual == prior, label
    m['save'](m['ROOT']/(label+'-binaries.json'), actual)
    (m['ROOT']/(label+'-before-build.sha256')).write_text(''.join(f'{h}  {source}/build-sycl/bin/{name}\n' for name,h in actual.items()))
    protected[label] = m['source_hashes'](source)
assert m['binaries'](m['PARENT']) == m['PARENT_EXPECTED']
m['save'](m['ROOT']/'protected-source-hashes.json', protected)
PY
need=$(du -sk "$PARENT" | cut -f1)
available=$(df -Pk "$(dirname "$D")" | awk 'NR==2 {print $4}')
test "$available" -gt "$((need + 4194304))"
cp -a --reflink=auto "$PARENT" "$D"
(cd "$D" && patch --fuzz=0 --batch --forward -p1 < "$R/router-widths-single2.patch")
(cd "$D" && git apply --reverse --check --verbose "$R/router-widths-single2.patch")
# Reuse parent configuration/cache; only SYCL and test artifacts may change.
# Identical parent bootstrap in this disposable container ONLY. --no-upgrade can
# still update curl dependencies here; this is NOT zero container package changes.
docker run --rm --name bolt-lilicorr-router-widths-single2-r1-build --cpus=6 \
    -v "$D:/src" -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro \
    -w /src "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install --no-upgrade -y -qq ninja-build libcurl4-openssl-dev pkg-config
cmake --build /src/build-sycl --config Release -j6 --target ggml-sycl test-backend-ops
'
docker run --rm --network none --name bolt-lilicorr-router-widths-single2-r1-precheck -v "$D:/src:ro" \
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
import runpy
m = runpy.run_path('run-trace.py')
m['check_protected_sources']()
m['save'](m['ROOT']/'source-diff.json', dict(changed_relative_to_parent=m['SOURCE_SCOPE']))
x, y = m['binaries'](m['PARENT']), m['binaries'](m['CANDIDATE'])
assert {n for n in m['ARTIFACTS'] if x[n] != y[n]} == {'libggml-sycl.so.0', 'test-backend-ops'}, (x, y)
m['save'](m['ROOT']/'candidate-binaries.json', y)
print('BOLT_VERIFY_BUILD_PASSED')
PY
