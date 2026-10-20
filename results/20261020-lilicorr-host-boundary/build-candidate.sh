#!/usr/bin/env bash
# LEAD ONLY. Rebuild only the frozen 105-object GGML-SYCL closure; no GPU.
set -euo pipefail
test "$(hostname)" = inference-host
ROOT=/home/mike/b70-evals/20261020-lilicorr-host-boundary
ATTEMPT=$ROOT/build-attempt-r1
PARENT=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2
SDKSEED=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-device-timing-r1
CANDIDATE=/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-host-boundary-r1
DEVICE_CASE=/home/mike/b70-evals/20261020-lilicorr-device-timing
IQ2_PACKAGE=/home/mike/b70-evals/20261020-lilicorr-iq2-batch4
IMAGE=intel/deep-learning-essentials@sha256:aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0
PREFIX=bolt-lilicorr-host-boundary-r1
MIN_FREE_KIB=$((4194304 + 1048576))

cd "$ROOT"
test ! -e "$CANDIDATE" && test ! -L "$CANDIDATE"
test ! -e "$ATTEMPT"
mkdir "$ATTEMPT"
exec > >(tee "$ATTEMPT/build.log") 2>&1

seed_source() {
    python3 -B "$ROOT/source-proof.py" --seed-sources
}
seed_binaries() {
    python3 -B - "$ROOT" <<'PY'
import json, runpy, sys
from pathlib import Path
s = runpy.run_path(str(Path(sys.argv[1]) / 'source-proof.py'))
print(json.dumps(s['binaries'](s['SEED']), sort_keys=True, indent=2))
PY
}
finish() {
    rc=$?
    trap - EXIT
    set +e
    if test -s "$ATTEMPT/seed-source-before.json"; then
        seed_source > "$ATTEMPT/seed-source-after.json" &&
            cmp "$ATTEMPT/seed-source-before.json" "$ATTEMPT/seed-source-after.json" || rc=1
    fi
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

# Validate the old completed device proof before copying its SDK seed.
python3 -B "$DEVICE_CASE/source-proof.py" --built > "$ATTEMPT/device-proof-before-copy.json"
python3 -B "$ROOT/source-proof.py" --parent > "$ATTEMPT/parent-source.json"
seed_source > "$ATTEMPT/seed-source-before.json"
seed_binaries > "$ATTEMPT/seed-built-before.json"
test -f "$SDKSEED/build-sycl/build.ninja"
grep -Fx 'CMAKE_HOME_DIRECTORY:INTERNAL=/src' "$SDKSEED/build-sycl/CMakeCache.txt"
need=$(du -sk "$SDKSEED" | cut -f1)
available=$(df -Pk "$(dirname "$CANDIDATE")" | awk 'NR==2 {print $4}')
test "$available" -gt "$((need + MIN_FREE_KIB))"
test "$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')" -gt "$MIN_FREE_KIB"
df -Pk "$ROOT" "$(dirname "$CANDIDATE")"

# The candidate starts from the completed device seed, not from original R2.
cp -a --reflink=auto "$SDKSEED" "$CANDIDATE"
python3 -B - "$ROOT" <<'PY'
import runpy, sys
from pathlib import Path
s = runpy.run_path(str(Path(sys.argv[1]) / 'source-proof.py'))
assert s['source_map'](s['SEED']) == s['source_map'](s['CANDIDATE'])
assert s['binaries'](s['SEED']) == s['binaries'](s['CANDIDATE'])
PY
patch --directory="$CANDIDATE" --fuzz=0 --batch --forward -p1 < "$ROOT/disable-device-profiling.patch"
patch --directory="$CANDIDATE" --fuzz=0 --dry-run --batch --reverse -p1 < "$ROOT/disable-device-profiling.patch"
python3 -B - "$ROOT" <<'PY'
import runpy, sys
from pathlib import Path
s = runpy.run_path(str(Path(sys.argv[1]) / 'source-proof.py'))
assert s['check_seed'](s['SEED'], s['CANDIDATE'])['changed_relative_to_seed'] == [s['HELPER']]
PY
python3 -B "$ROOT/source-proof.py" --candidate-sources > "$ATTEMPT/candidate-source-before.json"

# Same container-only prerequisites and frozen SDK setup as the passed IQ2 driver.
docker run --rm --pull=never --name "$PREFIX-build" --cpus=6 \
    -v "$CANDIDATE:/src" -v "$SDKSEED:/qualified:ro" -v "$IQ2_PACKAGE:/package:ro" \
    -v "$ATTEMPT:/build-evidence" -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro \
    -w /src/build-sycl "$IMAGE" bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
set -u
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install --no-upgrade -y -qq ninja-build libcurl4-openssl-dev pkg-config

python3 -B - /build-evidence/graph-before.json <<'PY'
import json, runpy, sys
from pathlib import Path
m = runpy.run_path("/package/build-module.py")
r = m["expected"]()
Path(sys.argv[1]).write_text(json.dumps(m["file_map"](m["BUILD"], r["graph_sha256"]), sort_keys=True, indent=2) + "\n")
PY

# LILICORR_OBJECT_CLEAN_BEGIN
python3 -B - /package/build-inputs.json.gz /src/build-sycl <<'PY'
import gzip
import json
import sys
from pathlib import Path

inputs, build = map(lambda value: Path(value).resolve(), sys.argv[1:])
reference = json.loads(gzip.decompress(inputs.read_bytes()))
objects = reference["objects"]
assert len(objects) == 105 and len(set(objects)) == 105
paths = []
for name in objects:
    path = (build / name).resolve(strict=False)
    assert path.is_relative_to(build)
    assert path.name.endswith(".o") and path.is_file()
    paths.append(path)
assert len(paths) == 105 and len({str(path) for path in paths}) == 105
for path in paths:
    path.unlink()
assert all(not path.exists() for path in paths)
PY
# LILICORR_OBJECT_CLEAN_END

python3 -B /package/build-module.py

python3 -B - /build-evidence/graph-after.json <<'PY'
import json, runpy, sys
from pathlib import Path
m = runpy.run_path("/package/build-module.py")
r = m["expected"]()
Path(sys.argv[1]).write_text(json.dumps(m["file_map"](m["BUILD"], r["graph_sha256"]), sort_keys=True, indent=2) + "\n")
PY
cmp /build-evidence/graph-before.json /build-evidence/graph-after.json
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
echo LILICORR_HOST_BOUNDARY_DEPENDENCIES_PASSED
' > "$ATTEMPT/runtime-precheck.log" 2>&1

python3 -B "$ROOT/source-proof.py" --candidate-sources > "$ATTEMPT/candidate-source-after.json"
cmp "$ATTEMPT/candidate-source-before.json" "$ATTEMPT/candidate-source-after.json"
seed_source > "$ATTEMPT/seed-source-after.json"
cmp "$ATTEMPT/seed-source-before.json" "$ATTEMPT/seed-source-after.json"
seed_binaries > "$ATTEMPT/seed-built-after.json"
cmp "$ATTEMPT/seed-built-before.json" "$ATTEMPT/seed-built-after.json"

python3 -B - "$ROOT" "$ATTEMPT" <<'PY' | tee "$ATTEMPT/post-build-verification.log"
import json, runpy, sys
from pathlib import Path
root, attempt = map(Path, sys.argv[1:])
s = runpy.run_path(str(root / 'source-proof.py'))
source = s['check_sources'](s['PARENT'], s['CANDIDATE'])
seed_source = s['check_seed'](s['SEED'], s['CANDIDATE'])
parent, seed, candidate = (s['binaries'](path) for path in (s['PARENT'], s['SEED'], s['CANDIDATE']))
for name, value in (
    ('parent-binaries.json', parent), ('seed-built-before.json', seed),
    ('seed-built-after.json', seed), ('candidate-binaries.json', candidate),
    ('source-diff.json', {'changed_relative_to_parent': source['changed_relative_to_parent']}),
    ('seed-source-diff.json', {'changed_relative_to_seed': seed_source['changed_relative_to_seed']}),
):
    (attempt / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
changed_seed = sorted(name for name in s['ARTIFACTS'] if seed[name] != candidate[name])
changed_parent = sorted(name for name in s['ARTIFACTS'] if parent[name] != candidate[name])
assert changed_seed == s['SEED_CHANGED_BINARIES'], ('unexpected seed artifact delta', changed_seed)
assert changed_parent == s['CHANGED_BINARIES'], ('unexpected original artifact delta', changed_parent)
pins = s['load_pins']()
expected_delta = sorted(name for name in pins['diagnostic'] if pins['parent'].get(name) != pins['diagnostic'][name])
assert source['changed_relative_to_parent'] == expected_delta
assert seed_source['changed_relative_to_seed'] == [s['HELPER']]
print('Changed artifacts vs SDK seed:', changed_seed)
print('Changed artifacts vs original R2:', changed_parent)
print(s['BUILD_MARKER'])
PY
