#!/usr/bin/env bash
# Local command; all compiler execution occurs in disposable remote containers.
set -euo pipefail
L=$(cd -- "$(dirname -- "$0")" && pwd)
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-gemm-catalog
CELL=${1:?Pass a fresh build-NN directory}
[[ $CELL =~ ^build-[0-9]+$ ]]
ssh inference-host "bash -lc 'mkdir $R/$CELL'"
scp "$L/build.sh" "$L/patch-catalog.py" "$L/CMakeLists.txt" "$L/binding.cpp" "$L/probe.py" "inference-host:$R/"
set +e
ssh inference-host "bash -s -- '$CELL'" <<'REMOTE'
set -euo pipefail
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-gemm-catalog
O=$R/$1
exec > >(tee "$O/driver.log") 2>&1
set -x
cp "$R"/{build.sh,patch-catalog.py,CMakeLists.txt,binding.cpp,probe.py} "$O/"
date -Is
trap 'docker rm -f b70-gemm-catalog-build >/dev/null 2>&1 || true' EXIT
set +e
timeout --signal=TERM --kill-after=30s 90m docker run --pull=never --rm \
 --name b70-gemm-catalog-build --cpus=12 --memory=24g --memory-swap=24g \
 -v "$R:/experiment" -v "$R/compiler:/opt/intel/oneapi/compiler/2026.0:ro" \
 -e BUILD_CELL="$1" --entrypoint /bin/bash \
 vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f -lc '
set -euxo pipefail
C=/opt/intel/oneapi/compiler/2026.0
export PATH="$C/bin:/opt/venv/bin:$PATH"
export LD_LIBRARY_PATH="/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib:$C/lib:${LD_LIBRARY_PATH:-}"
export CC="$C/bin/icx" CXX="$C/bin/icpx" SYCL_ROOT="$C"
O=/experiment/$BUILD_CELL
icpx --version
git config --global --add safe.directory /experiment/oneDNN
python -P /experiment/patch-catalog.py /experiment/oneDNN > "$O/onednn.patch"
git -C /experiment/oneDNN apply --check "$O/onednn.patch"
git -C /experiment/oneDNN apply "$O/onednn.patch"
cmake -S /experiment/oneDNN -B /experiment/onednn-build -G Ninja \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/experiment/onednn-install \
 -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
 -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
 -DDNNL_LIBRARY_TYPE=STATIC -DDNNL_CPU_RUNTIME=NONE -DDNNL_GPU_RUNTIME=SYCL \
 -DDNNL_BUILD_TESTS=OFF -DDNNL_BUILD_EXAMPLES=OFF -DONEDNN_BUILD_GRAPH=OFF \
 -DDNNL_ENABLE_PRIMITIVE=ALL -DDNNL_ENABLE_PRIMITIVE_GPU_ISA=XE2 \
 -DDNNL_ENABLE_PRIMITIVE_CACHE=ON -DDNNL_DEV_MODE=OFF
cp /experiment/onednn-build/CMakeCache.txt "$O/onednn-CMakeCache.txt"
cmake --build /experiment/onednn-build --parallel 8
cmake --install /experiment/onednn-build
cmake -S /experiment -B /experiment/binding-build -G Ninja \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX" \
 -DNATIVE_SOURCE_DIR=/experiment/native -DONEDNN_INSTALL_PREFIX=/experiment/onednn-install \
 -DCMAKE_PREFIX_PATH="/experiment/onednn-install;/opt/venv/lib/python3.12/site-packages/torch/share/cmake"
cp /experiment/binding-build/CMakeCache.txt "$O/binding-CMakeCache.txt"
cmake --build /experiment/binding-build --parallel 2
find /experiment/binding-build -maxdepth 2 -name "*.so" -exec sha256sum {} \;
'
rc=$?
set -e
printf '%s\n' "$rc" > "$O/exit-code.txt"
docker rm -f b70-gemm-catalog-build >/dev/null 2>&1 || true
date -Is
exit "$rc"
REMOTE
rc=$?
set -e
scp -r "inference-host:$R/$CELL" "$L/"
exit "$rc"
