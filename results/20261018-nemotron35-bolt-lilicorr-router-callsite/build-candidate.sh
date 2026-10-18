#!/usr/bin/env bash
# LEAD ONLY, inference-host CPU build. No GPU mounts, host installs, or service changes.
# Derived from router-verify-phase/build-candidate.sh; do not run during preparation.
set -euo pipefail
test "$(hostname)" = inference-host
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-callsite
PHASE=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-verify-phase
PARENT=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-verify-phase-r1
CANDIDATE=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-callsite-r1
# Exact Intel 2026 image recorded by the phase build, not a refreshed tag.
IMAGE=intel/deep-learning-essentials@sha256:aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0
PREFIX=bolt-lilicorr-router-callsite-r1
# Retain the inherited 4 GiB reserve AND 1 GiB for subsequent evidence.
MIN_FREE_KIB=$((4194304 + 1048576))
cd "$ROOT"
test ! -e "$CANDIDATE" && test ! -L "$CANDIDATE"
# An existing attempt, including a failed one, is never reused or removed here.
mkdir "$ROOT/build-attempt-r1"
exec > >(tee "$ROOT/build-attempt-r1/build.log") 2>&1

check_protected() {
    python3 -B - "$PHASE" "$PARENT" <<'PY'
import json
import runpy
import sys
from pathlib import Path
phase, parent = map(Path, sys.argv[1:])
m = runpy.run_path(str(phase / 'run-trace.py'))
assert m['CANDIDATE'] == parent
# Existing source maps and all six captured binaries remain authoritative.
# All helpers below are read-only; never redirect their ROOT or rewrite old cases.
m['check_legacy_protection']()
m['check_protected_sources']()
m['check_protected_binaries']()
actual = m['binaries'](parent)
expected = m['load'](phase / 'candidate-binaries.json')
assert set(actual) == set(expected) == set(m['ARTIFACTS'])
assert actual == expected, 'verify-phase parent binaries changed'
for label, source in [(label, source) for label, source, _ in m['PROTECTED']] + [('verify-phase', parent)]:
    print(label, json.dumps(m['binaries'](source), sort_keys=True))
print('BOLT_VERIFY_CALLSITE_PROTECTED_PASSED')
PY
}

check_candidate_sources() {
    python3 -B "$ROOT/check-source.py" --candidate "$CANDIDATE"
    python3 -B - "$PHASE" "$ROOT" "$PARENT" "$CANDIDATE" <<'PY'
import runpy
import sys
from pathlib import Path
phase, root, parent, candidate = map(Path, sys.argv[1:])
m = runpy.run_path(str(phase / 'run-trace.py'))
x, y = m['source_hashes'](parent), m['source_hashes'](candidate)
changed = sorted(k for k in x.keys() | y.keys() if x.get(k) != y.get(k))
m['save'](root / 'source-diff.json', dict(changed_relative_to_parent=changed))
assert changed == ['src/models/nemotron-h.cpp'], changed
print('BOLT_VERIFY_CALLSITE_SOURCE_SCOPE_PASSED')
PY
}

finish() {
    rc=$?
    trap - EXIT
    set +e
    check_protected
    if test "$?" -ne 0; then rc=1; fi
    printf '%s\n' "$rc" > "$ROOT/build-attempt-r1/exit-code.txt"
    exit "$rc"
}
trap finish EXIT
set -x
uname -a
docker version
# Require the existing frozen image; never pull into the host's reserve.
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
check_protected
python3 -B "$ROOT/check-source.py" | tee "$ROOT/source-check-parent.log"
git -C "$PARENT" apply --check --verbose "$ROOT/router-callsite.patch"
patch --directory="$PARENT" --fuzz=0 --dry-run --batch --forward -p1 < "$ROOT/router-callsite.patch"
need=$(du -sk "$PARENT" | cut -f1)
available=$(df -Pk "$(dirname "$CANDIDATE")" | awk 'NR==2 {print $4}')
test "$available" -gt "$((need + MIN_FREE_KIB))"
test "$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')" -gt "$MIN_FREE_KIB"
df -Pk "$ROOT" "$(dirname "$CANDIDATE")"
cp -a --reflink=auto "$PARENT" "$CANDIDATE"
patch --directory="$CANDIDATE" --fuzz=0 --batch --forward -p1 < "$ROOT/router-callsite.patch"
git -C "$CANDIDATE" apply --reverse --check --verbose "$ROOT/router-callsite.patch"
check_candidate_sources | tee "$ROOT/source-check-candidate.log"
check_protected
test "$(df -Pk "$CANDIDATE" | awk 'NR==2 {print $4}')" -gt "$MIN_FREE_KIB"
# Same disposable bootstrap as the phase build. Installs only inside the container.
# /src matches the parent's CMake cache; no reconfiguration or CNR/env changes.
docker run --rm --pull=never --name "$PREFIX-build" --cpus=6 \
    -v "$CANDIDATE:/src" -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro \
    -w /src "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install --no-upgrade -y -qq ninja-build libcurl4-openssl-dev pkg-config
cmake --build /src/build-sycl --config Release -j6 --target llama-server test-backend-ops
'
docker run --rm --pull=never --network none --name "$PREFIX-precheck" -v "$CANDIDATE:/src:ro" \
    -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
export LD_LIBRARY_PATH=/src/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}
for f in llama-server libllama-server-impl.so libggml-sycl.so.0 test-backend-ops libllama-common.so libllama.so.0.7.0; do ldd "/src/build-sycl/bin/$f"; done | tee /tmp/dependencies.log
if grep -F "not found" /tmp/dependencies.log; then exit 1; fi
echo BOLT_VERIFY_CALLSITE_DEPENDENCIES_PASSED
' > "$ROOT/runtime-precheck.log" 2>&1
check_protected
test "$(df -Pk "$CANDIDATE" | awk 'NR==2 {print $4}')" -gt "$MIN_FREE_KIB"
test "$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')" -gt "$MIN_FREE_KIB"
check_candidate_sources
python3 -B - "$PHASE" "$ROOT" "$PARENT" "$CANDIDATE" <<'PY' | tee "$ROOT/post-build-verification.log"
import runpy
import sys
from pathlib import Path
phase, root, parent, candidate = map(Path, sys.argv[1:])
m = runpy.run_path(str(phase / 'run-trace.py'))
x, y = m['binaries'](parent), m['binaries'](candidate)
changed = sorted(n for n in m['ARTIFACTS'] if x[n] != y[n])
# Save observed six-artifact evidence even if the expected change scope fails.
m['save'](root / 'parent-binaries.json', x)
m['save'](root / 'candidate-binaries.json', y)
print('Changed artifacts:', ', '.join(changed), flush=True)
# The backend and native tests are inherited, not rebuilt into a new intervention.
# Do not copy guessed hashes or silently accept unexpected relinks.
assert changed == ['libllama.so.0.7.0'], (x, y)
print('BOLT_VERIFY_CALLSITE_BUILD_PASSED')
PY
