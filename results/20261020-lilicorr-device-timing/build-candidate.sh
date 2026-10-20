#!/usr/bin/env bash
# LEAD ONLY. Existing frozen SDK, three affected modules, no GPU mount.
set -euo pipefail
test "$(hostname)" = inference-host
ROOT=/home/mike/b70-evals/20261020-lilicorr-device-timing
ATTEMPT=$ROOT/build-attempt-r1
PARENT=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2
CANDIDATE=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-device-timing-r1
IMAGE=intel/deep-learning-essentials@sha256:aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0
PREFIX=bolt-lilicorr-device-timing-r1
MIN_FREE_KIB=$((4194304 + 1048576))
cd "$ROOT"
test ! -e "$CANDIDATE" && test ! -L "$CANDIDATE"
mkdir "$ATTEMPT"
exec > >(tee "$ATTEMPT/build.log") 2>&1

seed_binaries() {
    python3 -B - "$ROOT" <<'PY'
import json, runpy, sys
from pathlib import Path
s = runpy.run_path(str(Path(sys.argv[1]) / 'source-proof.py'))
print(json.dumps(s['binaries'](s['PARENT']), sort_keys=True, indent=2))
PY
}
finish() {
    rc=$?
    trap - EXIT
    set +e
    if test -s "$ATTEMPT/seed-built-before.json"; then
        seed_binaries > "$ATTEMPT/seed-built-after.json" &&
            cmp "$ATTEMPT/seed-built-before.json" "$ATTEMPT/seed-built-after.json" || rc=1
    fi
    if test -s "$ATTEMPT/candidate-source-before.json"; then
        python3 -B "$ROOT/source-proof.py" --candidate-sources > "$ATTEMPT/candidate-source-after.json" &&
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
seed_binaries > "$ATTEMPT/seed-built-before.json"
python3 -B "$ROOT/source-proof.py" --parent > "$ATTEMPT/parent-source.json"
# Do not accidentally build the superseded n8-only diagnostic checkpoint.
python3 -B - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
p = json.loads((Path(sys.argv[1]) / 'source-pins.json').read_text())
assert p['diagnostic']['common/speculative.cpp'] != p['composed_seed']['common/speculative.cpp'], 'injection-component extension required'
PY
test -f "$PARENT/build-sycl/build.ninja"
grep -Fx 'CMAKE_HOME_DIRECTORY:INTERNAL=/src' "$PARENT/build-sycl/CMakeCache.txt"
need=$(du -sk "$PARENT" | cut -f1)
available=$(df -Pk "$(dirname "$CANDIDATE")" | awk 'NR==2 {print $4}')
test "$available" -gt "$((need + MIN_FREE_KIB))"
test "$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')" -gt "$MIN_FREE_KIB"
cp -a --reflink=auto "$PARENT" "$CANDIDATE"
python3 -B - "$ROOT" <<'PY'
import runpy, sys
from pathlib import Path
s = runpy.run_path(str(Path(sys.argv[1]) / 'source-proof.py'))
assert s['source_map'](s['PARENT']) == s['source_map'](s['CANDIDATE'])
assert s['binaries'](s['PARENT']) == s['binaries'](s['CANDIDATE'])
PY
for delta in "$ROOT/../20261020-lilicorr-context-profile/lilicorr-profile.patch" \
             "$ROOT/../20261020-lilicorr-iq2-batch4/iq2-batch4.patch" "$ROOT/device-timing.patch"; do
    patch --directory="$CANDIDATE" --fuzz=0 --batch --forward -p1 < "$delta"
    patch --directory="$CANDIDATE" --fuzz=0 --dry-run --batch --reverse -p1 < "$delta"
done
python3 -B "$ROOT/source-proof.py" --candidate-sources > "$ATTEMPT/candidate-source-before.json"
# Same prerequisites as the qualified build, inside the disposable SDK only.
docker run --rm --pull=never --name "$PREFIX-build" --cpus=6 \
    -v "$CANDIDATE:/src" -v "$PARENT:/qualified:ro" -v "$ROOT:/package:ro" \
    -v "$ATTEMPT:/build-evidence" -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro \
    -w /src/build-sycl "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
set -u
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install --no-upgrade -y -qq ninja-build libcurl4-openssl-dev pkg-config
python3 -B /package/build-three-modules.py
'
docker run --rm --pull=never --network none --name "$PREFIX-precheck" \
    -v "$CANDIDATE:/src:ro" -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro \
    "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
set -u
export LD_LIBRARY_PATH=/src/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}
for f in llama-server libllama-server-impl.so libggml-sycl.so.0 test-backend-ops libllama-common.so libllama.so.0.7.0; do
    ldd "/src/build-sycl/bin/$f"
done | tee /tmp/dependencies.log
if grep -F "not found" /tmp/dependencies.log; then exit 1; fi
echo LILICORR_DEVICE_TIMING_DEPENDENCIES_PASSED
' > "$ATTEMPT/runtime-precheck.log" 2>&1
python3 -B "$ROOT/source-proof.py" --candidate-sources > "$ATTEMPT/candidate-source-after.json"
cmp "$ATTEMPT/candidate-source-before.json" "$ATTEMPT/candidate-source-after.json"
seed_binaries > "$ATTEMPT/seed-built-after.json"
cmp "$ATTEMPT/seed-built-before.json" "$ATTEMPT/seed-built-after.json"
python3 -B - "$ROOT" "$ATTEMPT" <<'PY' | tee "$ATTEMPT/post-build-verification.log"
import json, runpy, sys
from pathlib import Path
root, attempt = map(Path, sys.argv[1:])
s = runpy.run_path(str(root / 'source-proof.py'))
source = s['check_sources'](s['PARENT'], s['CANDIDATE'])
parent, candidate = s['binaries'](s['PARENT']), s['binaries'](s['CANDIDATE'])
for name, value in (('parent-binaries.json', parent), ('candidate-binaries.json', candidate),
                    ('source-diff.json', {'changed_relative_to_parent': source['changed_relative_to_parent']})):
    (attempt / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
changed = sorted(name for name in s['ARTIFACTS'] if parent[name] != candidate[name])
assert changed == s['CHANGED_BINARIES'], ('unexpected binary delta', changed)
print('Changed artifacts:', changed)
print(s['BUILD_MARKER'])
PY
