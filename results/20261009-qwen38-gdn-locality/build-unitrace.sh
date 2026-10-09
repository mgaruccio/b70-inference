#!/usr/bin/env bash
# Build Intel PTI unitrace in an isolated, CPU-only container.
#
# This is an opt-in diagnostic prerequisite.  It never receives /dev/dri, does
# not install packages or drivers, and writes only the fresh directories below
# this campaign directory.  The serving image is pinned to the same digest as
# the normal B70 run; the cached oneAPI compiler is mounted read-only.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SOURCE_SHA=887bba6e28ce84cc0d3813ef876e24add107c318
SOURCE_URL=https://github.com/intel/pti-gpu.git
IMAGE=${UNITRACE_BUILD_IMAGE:-vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f}
COMPILER_ROOT=${UNITRACE_COMPILER_ROOT:-/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-gemm-catalog/compiler}
BUILD_NAME=${UNITRACE_BUILD_CONTAINER:-b70-gdn-locality-unitrace-build}
TIMEOUT_SECONDS=${UNITRACE_BUILD_TIMEOUT_SECONDS:-1800}
CPUS=${UNITRACE_BUILD_CPUS:-8}
MEMORY=${UNITRACE_BUILD_MEMORY:-24g}

SOURCE_DIR="$SCRIPT_DIR/unitrace-src"
BUILD_DIR="$SCRIPT_DIR/unitrace-build"
INSTALL_DIR="$SCRIPT_DIR/unitrace-install"

for path in "$SOURCE_DIR" "$BUILD_DIR" "$INSTALL_DIR"; do
  if [[ -e "$path" || -L "$path" ]]; then
    echo "refusing non-fresh unitrace path: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$COMPILER_ROOT" ]]; then
  echo "cached compiler is missing: $COMPILER_ROOT" >&2
  exit 2
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "pinned build image is not installed: $IMAGE" >&2
  exit 2
fi
if docker ps -a --format '{{.Names}}' | grep -Fxq "$BUILD_NAME"; then
  echo "owned build container name is not fresh: $BUILD_NAME" >&2
  exit 2
fi

mkdir -p "$BUILD_DIR" "$INSTALL_DIR"
git clone --filter=blob:none --no-checkout "$SOURCE_URL" "$SOURCE_DIR"
git -C "$SOURCE_DIR" checkout --detach "$SOURCE_SHA"
actual=$(git -C "$SOURCE_DIR" rev-parse HEAD)
[[ "$actual" == "$SOURCE_SHA" ]] || { echo "source SHA mismatch: $actual" >&2; exit 2; }

# The exact 887bba source exposes BUILD_WITH_MPI, BUILD_WITH_ITT,
# BUILD_WITH_XPTI, BUILD_WITH_OPENCL and mandatory Level Zero support.  It does
# not expose BUILD_WITH_OMP or BUILD_WITH_PERFETTO; do not pass invented flags.
# ITT/XPTI/OpenCL/MPI are disabled because this run needs Level Zero kernel and
# device tracing only.  --start-paused/--resume is unitrace session control,
# not an in-graph PyTorch event marker.
set +e
timeout --signal=TERM --kill-after=30s "${TIMEOUT_SECONDS}s" docker run --rm \
  --name "$BUILD_NAME" \
  --network=none \
  --cpus="$CPUS" \
  --memory="$MEMORY" \
  --pids-limit=512 \
  -v "$SOURCE_DIR:/src:ro" \
  -v "$BUILD_DIR:/build" \
  -v "$INSTALL_DIR:/install" \
  -v "$COMPILER_ROOT:/opt/intel/oneapi/compiler:ro" \
  --entrypoint bash "$IMAGE" -lc '
    set -euo pipefail
    test -x /opt/intel/oneapi/compiler/bin/icpx
    test -f /opt/intel/oneapi/compiler/env/vars.sh
    source /opt/intel/oneapi/compiler/env/vars.sh
    command -v cmake >/dev/null
    command -v ninja >/dev/null
    cmake -S /src/tools/unitrace -B /build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/install \
      -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/bin/icx \
      -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/bin/icpx \
      -DBUILD_WITH_MPI=0 \
      -DBUILD_WITH_ITT=0 \
      -DBUILD_WITH_XPTI=0 \
      -DBUILD_WITH_OPENCL=0 \
      -DBUILD_WITH_L0=1
    cmake --build /build --parallel 8
    cmake --install /build
  '
status=$?
set -e
if [[ "$status" -ne 0 ]]; then
  echo "unitrace build failed (status=$status); no serving run is authorized" >&2
  exit "$status"
fi
test -x "$INSTALL_DIR/bin/unitrace"
printf 'unitrace source=%s\nunitrace install=%s\n' "$SOURCE_SHA" "$INSTALL_DIR"
