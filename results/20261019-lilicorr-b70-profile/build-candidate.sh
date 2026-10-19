#!/usr/bin/env bash
# LEAD ONLY. CPU build in the frozen SDK; no GPU mounts or host package installs.
set -euo pipefail
test "$(hostname)" = inference-host
ROOT=/home/mike/b70-evals/20261019-lilicorr-b70-profile
QUALIFIED=/home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-verify-singletons
PARENT=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2
CANDIDATE=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-b70-profile-r1
IMAGE=intel/deep-learning-essentials@sha256:aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0
PREFIX=bolt-lilicorr-b70-profile-r1
MIN_FREE_KIB=$((4194304 + 1048576))
cd "$ROOT"
test ! -e "$CANDIDATE" && test ! -L "$CANDIDATE"
mkdir "$ROOT/build-attempt-r1" # Fail closed on an earlier attempt, even a failed one.
exec > >(tee "$ROOT/build-attempt-r1/build.log") 2>&1

check_protected() {
    python3 -B - "$QUALIFIED" <<'PY'
import runpy, sys
from pathlib import Path
p = runpy.run_path(str(Path(sys.argv[1]) / 'public-gate.py'))
p['candidate_proof']()  # Includes protected ancestors, complete source maps, six binaries.
print('LILICORR_PROFILE_QUALIFIED_ANCESTRY_PASSED')
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
docker image inspect "$IMAGE" > "$ROOT/build-attempt-r1/image.json"
for name in "$PREFIX-build" "$PREFIX-precheck"; do
    if docker container inspect "$name" >/dev/null 2>&1; then
        echo "container name collision: $name" >&2; exit 1
    fi
done
check_protected
python3 -B "$ROOT/source-proof.py" > "$ROOT/build-attempt-r1/parent-source.json"
test -f "$PARENT/build-sycl/CMakeCache.txt"
grep -Fx 'CMAKE_HOME_DIRECTORY:INTERNAL=/src' "$PARENT/build-sycl/CMakeCache.txt"
# Parent is an exported tree without .git. GNU patch provides the zero-fuzz check.
patch --directory="$PARENT" --fuzz=0 --dry-run --batch --forward -p1 < "$ROOT/lilicorr-profile.patch"
need=$(du -sk "$PARENT" | cut -f1)
available=$(df -Pk "$(dirname "$CANDIDATE")" | awk 'NR==2 {print $4}')
test "$available" -gt "$((need + MIN_FREE_KIB))"
test "$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')" -gt "$MIN_FREE_KIB"
df -Pk "$ROOT" "$(dirname "$CANDIDATE")"
cp -a --reflink=auto "$PARENT" "$CANDIDATE"
patch --directory="$CANDIDATE" --fuzz=0 --batch --forward -p1 < "$ROOT/lilicorr-profile.patch"
patch --directory="$CANDIDATE" --fuzz=0 --dry-run --batch --reverse -p1 < "$ROOT/lilicorr-profile.patch"
python3 -B "$ROOT/source-proof.py" --candidate > "$ROOT/build-attempt-r1/candidate-source-before.json"
check_protected
# Keep /src for the inherited CMake cache; do not reconfigure or enable SYCL GRAPH.
docker run --rm --pull=never --name "$PREFIX-build" --cpus=6 \
    -v "$CANDIDATE:/src" -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro \
    -w /src "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install --no-upgrade -y -qq ninja-build libcurl4-openssl-dev pkg-config
cmake --build /src/build-sycl --config Release -j6 --target llama-server test-backend-ops
'
docker run --rm --pull=never --network none --name "$PREFIX-precheck" \
    -v "$CANDIDATE:/src:ro" -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro \
    "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
export LD_LIBRARY_PATH=/src/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}
for f in llama-server libllama-server-impl.so libggml-sycl.so.0 test-backend-ops libllama-common.so libllama.so.0.7.0; do ldd "/src/build-sycl/bin/$f"; done | tee /tmp/dependencies.log
if grep -F "not found" /tmp/dependencies.log; then exit 1; fi
echo LILICORR_PROFILE_DEPENDENCIES_PASSED
' > "$ROOT/runtime-precheck.log" 2>&1
check_protected
python3 -B "$ROOT/source-proof.py" --candidate > "$ROOT/build-attempt-r1/candidate-source-after.json"
cmp "$ROOT/build-attempt-r1/candidate-source-before.json" "$ROOT/build-attempt-r1/candidate-source-after.json"
test "$(df -Pk "$CANDIDATE" | awk 'NR==2 {print $4}')" -gt "$MIN_FREE_KIB"
test "$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')" -gt "$MIN_FREE_KIB"
python3 -B - "$ROOT" <<'PY' | tee "$ROOT/post-build-verification.log"
import json, runpy, sys
from pathlib import Path
root = Path(sys.argv[1])
s = runpy.run_path(str(root / 'source-proof.py'))
x, y = s['binaries'](s['PARENT']), s['binaries'](s['CANDIDATE'])
for name, data in [('parent-binaries.json', x), ('candidate-binaries.json', y),
                   ('source-diff.json', dict(changed_relative_to_parent=s['SOURCE_SCOPE']))]:
    (root / name).write_text(json.dumps(data, indent=2) + '\n')
changed = sorted(k for k in s['ARTIFACTS'] if x[k] != y[k])
print('Changed artifacts:', changed, flush=True)
assert changed == s['CHANGED_BINARIES'], 'Unexpected binary delta: retain evidence, do not loosen pins'
print('LILICORR_PROFILE_BUILD_PASSED')
PY
