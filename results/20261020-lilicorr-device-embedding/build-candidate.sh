#!/usr/bin/env bash
# LEAD ONLY. CPU compilation in the frozen SDK; never mount a GPU or install on host.
set -euo pipefail
test "$(hostname)" = inference-host
ROOT=/home/mike/b70-evals/20261020-lilicorr-device-embedding
QUALIFIED=/home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-verify-singletons
PARENT=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2
SEED=$PARENT
CANDIDATE=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-device-embedding-r1
IMAGE=intel/deep-learning-essentials@sha256:aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0
PREFIX=bolt-lilicorr-device-embedding-r1
MIN_FREE_KIB=$((4194304 + 1048576))
cd "$ROOT"
test ! -e "$CANDIDATE" && test ! -L "$CANDIDATE"
mkdir "$ROOT/build-attempt-r1"
exec > >(tee "$ROOT/build-attempt-r1/build.log") 2>&1

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
    if test -s "$ROOT/build-attempt-r1/seed-built-before.json"; then
        check_seed_built > "$ROOT/build-attempt-r1/seed-built-after.json" &&
            cmp "$ROOT/build-attempt-r1/seed-built-before.json" "$ROOT/build-attempt-r1/seed-built-after.json" || rc=1
    fi
    if test -s "$ROOT/build-attempt-r1/candidate-source-before.json"; then
        python3 -B "$ROOT/source-proof.py" --candidate > "$ROOT/build-attempt-r1/candidate-source-after.json" &&
            cmp "$ROOT/build-attempt-r1/candidate-source-before.json" "$ROOT/build-attempt-r1/candidate-source-after.json" || rc=1
    fi
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
        echo "container name collision: $name" >&2
        exit 1
    fi
done
check_seed_built > "$ROOT/build-attempt-r1/seed-built-before.json"
python3 -B "$ROOT/source-proof.py" > "$ROOT/build-attempt-r1/parent-source.json"
test -f "$SEED/build-sycl/build.ninja"
grep -Fx 'CMAKE_HOME_DIRECTORY:INTERNAL=/src' "$SEED/build-sycl/CMakeCache.txt"
patch --directory="$PARENT" --fuzz=0 --dry-run --batch --forward -p1 < "$ROOT/device-embedding.patch"
need=$(du -sk "$SEED" | cut -f1)
available=$(df -Pk "$(dirname "$CANDIDATE")" | awk 'NR==2 {print $4}')
test "$available" -gt "$((need + MIN_FREE_KIB))"
test "$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')" -gt "$MIN_FREE_KIB"
df -Pk "$ROOT" "$(dirname "$CANDIDATE")"
cp -a --reflink=auto "$SEED" "$CANDIDATE"
# Original qualified source/build is the seed. No diagnostics ever enter the candidate.
python3 -B - "$ROOT" <<'PY'
import runpy, sys
from pathlib import Path
s = runpy.run_path(str(Path(sys.argv[1]) / 'source-proof.py'))
assert s['source_map'](s['PARENT']) == s['source_map'](s['CANDIDATE'])
assert s['binaries'](s['PARENT']) == s['binaries'](s['CANDIDATE'])
PY
patch --directory="$CANDIDATE" --fuzz=0 --batch --forward -p1 < "$ROOT/device-embedding.patch"
patch --directory="$CANDIDATE" --fuzz=0 --dry-run --batch --reverse -p1 < "$ROOT/device-embedding.patch"
python3 -B "$ROOT/source-proof.py" --candidate > "$ROOT/build-attempt-r1/candidate-source-before.json"

# Preserve /src cache/configuration, including the inherited compile-time graph option.
# Runtime graph policy stays the qualified launcher's; do not enable it or reconfigure.
docker run --rm --pull=never --name "$PREFIX-build" --cpus=6 \
    -v "$CANDIDATE:/src" -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro \
    -w /src "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install --no-upgrade -y -qq ninja-build libcurl4-openssl-dev pkg-config
cmake --build /src/build-sycl --config Release -j6 --target llama-server test-backend-ops -- -n | tee /tmp/build-plan.log
# Fail instead of regenerating configuration or compiling unrelated kernels/code.
if grep -E "Re-running CMake|regenerate-during-build" /tmp/build-plan.log; then
    echo UNEXPECTED_CONFIG_REGENERATION >&2
    exit 1
fi
if grep -E "Building .* object " /tmp/build-plan.log | grep -v "object src/CMakeFiles/llama.dir/"; then
    echo UNEXPECTED_COMPILE_SCOPE >&2
    exit 1
fi
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
python3 -B "$ROOT/source-proof.py" --candidate > "$ROOT/build-attempt-r1/candidate-source-after.json"
cmp "$ROOT/build-attempt-r1/candidate-source-before.json" "$ROOT/build-attempt-r1/candidate-source-after.json"
check_seed_built > "$ROOT/build-attempt-r1/seed-built-after.json"
cmp "$ROOT/build-attempt-r1/seed-built-before.json" "$ROOT/build-attempt-r1/seed-built-after.json"
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
# These exact marker names are required by the reused proof; they do not mean profiling is compiled.
print('LILICORR_PROFILE_BUILD_PASSED')
PY
