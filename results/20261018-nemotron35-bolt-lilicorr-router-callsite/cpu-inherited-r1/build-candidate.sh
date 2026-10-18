#!/usr/bin/env bash
# LEAD ONLY, inference-host CPU build. No GPU mounts, host installs, or service changes.
set -euo pipefail
test "$(hostname)" = inference-host
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-verify-phase
PARENT=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-widths-single2-r1
CANDIDATE=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-verify-phase-r1
IMAGE=intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04
PREFIX=bolt-lilicorr-router-verify-phase-r1
cd "$ROOT"
test ! -e "$CANDIDATE" && test ! -L "$CANDIDATE"
# An existing attempt, including a failed one, is never reused or removed here.
mkdir "$ROOT/build-attempt-r1"
exec > >(tee "$ROOT/build-attempt-r1/build.log") 2>&1
finish() {
    rc=$?
    trap - EXIT
    set +e
    if test -f "$ROOT/protected-source-hashes.json"; then
        python3 -B - <<'PY'
import runpy
m = runpy.run_path('run-trace.py')
m['check_protected_sources'](candidate=False)
m['check_protected_binaries']()
print('BOLT_VERIFY_PROTECTED_AFTER_BUILD_ATTEMPT_PASSED')
PY
        if test "$?" -ne 0; then rc=1; fi
    fi
    printf '%s\n' "$rc" > "$ROOT/build-attempt-r1/exit-code.txt"
    exit "$rc"
}
trap finish EXIT
set -x
uname -a
docker version
# Require the existing image; never implicitly pull into the host's reserve.
docker image inspect "$IMAGE" > "$ROOT/build-attempt-r1/image.json"
for name in "$PREFIX-build" "$PREFIX-precheck"; do
    if docker container inspect "$name" >/dev/null 2>&1; then
        echo "container name collision: $name" >&2
        exit 1
    fi
done
test "$(cat "$PARENT/UPSTREAM_COMMIT")" = b6334e37e6ef286fc5091a41b6561b2113d915fb
test -f "$PARENT/build-sycl/CMakeCache.txt"
grep -Fx 'CMAKE_HOME_DIRECTORY:INTERNAL=/src' "$PARENT/build-sycl/CMakeCache.txt"
python3 -B - <<'PY'
import runpy
m = runpy.run_path('run-trace.py')
m['check_legacy_protection']()
m['check_consumed_sources'](m['PARENT'], 'parent')
protected = {}
for label, source, pins in m['PROTECTED']:
    actual = m['binaries'](source)
    assert set(actual) == set(m['ARTIFACTS'])
    assert {name: actual[name] for name in pins} == pins, label
    assert set(actual) - set(pins) == {'libllama.so.0.7.0'}
    m['save'](m['ROOT'] / (label + '-binaries.json'), actual)
    (m['ROOT'] / (label + '-before-build.sha256')).write_text(''.join(
        f'{h}  {source}/build-sycl/bin/{name}\n' for name, h in actual.items()))
    protected[label] = m['source_hashes'](source)
m['save'](m['ROOT'] / 'protected-source-hashes.json', protected)
PY
git -C "$PARENT" apply --check --verbose "$ROOT/router-verify-phase.patch"
patch --directory="$PARENT" --fuzz=0 --dry-run --batch --forward -p1 < "$ROOT/router-verify-phase.patch"
need=$(du -sk "$PARENT" | cut -f1)
available=$(df -Pk "$(dirname "$CANDIDATE")" | awk 'NR==2 {print $4}')
test "$available" -gt "$((need + 4194304))"
df -Pk "$ROOT" "$(dirname "$CANDIDATE")"
cp -a --reflink=auto "$PARENT" "$CANDIDATE"
patch --directory="$CANDIDATE" --fuzz=0 --batch --forward -p1 < "$ROOT/router-verify-phase.patch"
git -C "$CANDIDATE" apply --reverse --check --verbose "$ROOT/router-verify-phase.patch"
python3 -B - <<'PY'
import runpy
m = runpy.run_path('run-trace.py')
m['check_protected_sources']()
m['check_protected_binaries']()
PY
test "$(df -Pk "$CANDIDATE" | awk 'NR==2 {print $4}')" -gt 4194304
# Same disposable bootstrap as the parent. Installs only inside the container.
# /src matches the parent's CMake cache; no reconfiguration or CNR/env changes.
docker run --rm --name "$PREFIX-build" --cpus=6 \
    -v "$CANDIDATE:/src" -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro \
    -w /src "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install --no-upgrade -y -qq ninja-build libcurl4-openssl-dev pkg-config
cmake --build /src/build-sycl --config Release -j6 --target llama-server test-backend-ops
'
docker run --rm --network none --name "$PREFIX-precheck" -v "$CANDIDATE:/src:ro" \
    -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
export LD_LIBRARY_PATH=/src/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}
for f in llama-server libllama-server-impl.so libggml-sycl.so.0 test-backend-ops libllama-common.so libllama.so.0.7.0; do ldd "/src/build-sycl/bin/$f"; done | tee /tmp/dependencies.log
if grep -F "not found" /tmp/dependencies.log; then exit 1; fi
echo BOLT_VERIFY_DEPENDENCIES_PASSED
' > "$ROOT/runtime-precheck.log" 2>&1
for label in accepted parent columns8; do
    sha256sum -c "$ROOT/$label-before-build.sha256"
done
test "$(df -Pk "$CANDIDATE" | awk 'NR==2 {print $4}')" -gt 4194304
python3 -B - <<'PY' | tee "$ROOT/post-build-verification.log"
import runpy
m = runpy.run_path('run-trace.py')
m['check_protected_sources']()
m['check_protected_binaries']()
x, y = m['binaries'](m['PARENT']), m['binaries'](m['CANDIDATE'])
changed = sorted(n for n in m['ARTIFACTS'] if x[n] != y[n])
# The model graph lives in libllama, not just the SYCL backend. Dependent relinks
# may change other artifacts too; retain all six hashes rather than hide them.
assert {'libggml-sycl.so.0', 'test-backend-ops', 'libllama.so.0.7.0'} <= set(changed), (x, y)
m['save'](m['ROOT'] / 'source-diff.json', dict(changed_relative_to_parent=m['SOURCE_SCOPE']))
m['save'](m['ROOT'] / 'candidate-binaries.json', y)
print('Changed artifacts:', ', '.join(changed))
print('BOLT_VERIFY_BUILD_PASSED')
PY
