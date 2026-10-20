#!/usr/bin/env bash
# LEAD ONLY. Fresh F32 R3 source; preserve R1/R2 and all original source/build bytes.
set -euo pipefail
test "$(hostname)" = inference-host
ROOT=/home/mike/b70-evals/20261020-lilicorr-device-embedding
ATTEMPT=$ROOT/build-attempt-r3
QUALIFIED=/home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-verify-singletons
PARENT=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2
CANDIDATE=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-device-embedding-r3
IMAGE=intel/deep-learning-essentials@sha256:aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0
PREFIX=bolt-lilicorr-device-embedding-r3
MIN_FREE_KIB=$((4194304 + 1048576))
cd "$ROOT"
test ! -e "$CANDIDATE" && test ! -L "$CANDIDATE"
mkdir "$ATTEMPT"
exec > >(tee "$ATTEMPT/build.log") 2>&1

check_seed_built() {
    python3 -B - "$QUALIFIED" <<'PY'
import json, runpy, sys
from pathlib import Path
p = runpy.run_path(str(Path(sys.argv[1]) / 'public-gate.py'))
print(json.dumps(p['candidate_proof'](), sort_keys=True, indent=2))
PY
}
finish() {
    rc=$?
    trap - EXIT
    set +e
    if test -s "$ATTEMPT/seed-built-before.json"; then
        check_seed_built > "$ATTEMPT/seed-built-after.json" &&
            cmp "$ATTEMPT/seed-built-before.json" "$ATTEMPT/seed-built-after.json" || rc=1
    fi
    if test -s "$ATTEMPT/candidate-source-before.json"; then
        python3 -B "$ROOT/source-proof.py" --candidate > "$ATTEMPT/candidate-source-after.json" &&
            cmp "$ATTEMPT/candidate-source-before.json" "$ATTEMPT/candidate-source-after.json" || rc=1
    fi
    printf '%s\n' "$rc" > "$ATTEMPT/exit-code.txt"
    exit "$rc"
}
trap finish EXIT
set -x
uname -a
docker version
docker image inspect "$IMAGE" > "$ATTEMPT/image.json"
for name in "$PREFIX-build" "$PREFIX-precheck"; do
    if docker container inspect "$name" >/dev/null 2>&1; then
        echo "container name collision: $name" >&2
        exit 1
    fi
done
check_seed_built > "$ATTEMPT/seed-built-before.json"
python3 -B "$ROOT/source-proof.py" > "$ATTEMPT/parent-source.json"
test -f "$PARENT/build-sycl/build.ninja"
grep -Fx 'CMAKE_HOME_DIRECTORY:INTERNAL=/src' "$PARENT/build-sycl/CMakeCache.txt"
patch --directory="$PARENT" --fuzz=0 --dry-run --batch --forward -p1 < "$ROOT/device-embedding-f32.patch"
need=$(du -sk "$PARENT" | cut -f1)
available=$(df -Pk "$(dirname "$CANDIDATE")" | awk 'NR==2 {print $4}')
test "$available" -gt "$((need + MIN_FREE_KIB))"
test "$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')" -gt "$MIN_FREE_KIB"
df -Pk "$ROOT" "$(dirname "$CANDIDATE")"
cp -a --reflink=auto "$PARENT" "$CANDIDATE"
python3 -B - "$ROOT" <<'PY'
import runpy, sys
from pathlib import Path
s = runpy.run_path(str(Path(sys.argv[1]) / 'source-proof.py'))
assert s['source_map'](s['PARENT']) == s['source_map'](s['CANDIDATE'])
assert s['binaries'](s['PARENT']) == s['binaries'](s['CANDIDATE'])
PY
patch --directory="$CANDIDATE" --fuzz=0 --batch --forward -p1 < "$ROOT/device-embedding-f32.patch"
patch --directory="$CANDIDATE" --fuzz=0 --dry-run --batch --reverse -p1 < "$ROOT/device-embedding-f32.patch"
python3 -B "$ROOT/source-proof.py" --candidate > "$ATTEMPT/candidate-source-before.json"

# Same frozen compiler/flags. Existing object rules only, then existing generated linker.
# No CMake regeneration, Ninja metadata repair, timestamp manipulation, or GPU mount.
docker run --rm --pull=never --name "$PREFIX-build" --cpus=6 \
    -v "$CANDIDATE:/src" -v "$PARENT:/qualified:ro" -v "$ROOT:/package:ro" \
    -v "$ATTEMPT:/build-evidence" -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro \
    -w /src/build-sycl "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
set -u
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install --no-upgrade -y -qq ninja-build libcurl4-openssl-dev pkg-config
python3 -B /package/build-module-r2.py
'

docker run --rm --pull=never --network none --name "$PREFIX-precheck" \
    -v "$CANDIDATE:/src:ro" -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro \
    "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
set -u
export LD_LIBRARY_PATH=/src/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}
for f in llama-server libllama-server-impl.so libggml-sycl.so.0 test-backend-ops libllama-common.so libllama.so.0.7.0; do ldd "/src/build-sycl/bin/$f"; done | tee /tmp/dependencies.log
if grep -F "not found" /tmp/dependencies.log; then exit 1; fi
echo LILICORR_PROFILE_DEPENDENCIES_PASSED
' > "$ATTEMPT/runtime-precheck.log" 2>&1
python3 -B "$ROOT/source-proof.py" --candidate > "$ATTEMPT/candidate-source-after.json"
cmp "$ATTEMPT/candidate-source-before.json" "$ATTEMPT/candidate-source-after.json"
check_seed_built > "$ATTEMPT/seed-built-after.json"
cmp "$ATTEMPT/seed-built-before.json" "$ATTEMPT/seed-built-after.json"
test "$(df -Pk "$CANDIDATE" | awk 'NR==2 {print $4}')" -gt "$MIN_FREE_KIB"
test "$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')" -gt "$MIN_FREE_KIB"
python3 -B - "$ROOT" "$ATTEMPT" <<'PY' | tee "$ATTEMPT/post-build-verification.log"
import json, runpy, sys
from pathlib import Path
root, attempt = map(Path, sys.argv[1:])
s = runpy.run_path(str(root / 'source-proof.py'))
x, y = s['binaries'](s['PARENT']), s['binaries'](s['CANDIDATE'])
for name, data in [('parent-binaries.json', x), ('candidate-binaries.json', y),
                   ('source-diff.json', dict(changed_relative_to_parent=s['SOURCE_SCOPE']))]:
    (attempt / name).write_text(json.dumps(data, indent=2) + '\n')
changed = sorted(k for k in s['ARTIFACTS'] if x[k] != y[k])
print('Changed artifacts:', changed, flush=True)
assert changed == s['CHANGED_BINARIES'], 'Unexpected binary delta: retain evidence, do not loosen pins'
print('LILICORR_PROFILE_BUILD_PASSED')
PY
