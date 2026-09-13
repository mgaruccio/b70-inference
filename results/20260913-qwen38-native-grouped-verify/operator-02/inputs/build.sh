#!/usr/bin/env bash
# Lead-run ON inference-host. CPU-only disposable build, never a GPU container.
set -euo pipefail
R=$(cd -- "$(dirname -- "$0")" && pwd -P)
EXPECTED=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-native-grouped-verify
A=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-gemm-catalog
[[ $(hostname -s) == inference-host && $R == "$EXPECTED" ]]
CELL=${1:?Pass a fresh build-NN}
[[ $CELL =~ ^build-[0-9]+$ ]]
NAME=b70-grouped-verify-$CELL
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "Refusing to reuse existing container $NAME" >&2
  exit 1
fi
O=$R/$CELL
mkdir "$O"
exec > >(tee "$O/driver.log") 2>&1
finish() {
  rc=$?
  trap - EXIT
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  printf '%s\n' "$rc" > "$O/exit-code.txt"
  date -Is
  exit "$rc"
}
trap finish EXIT
set -x
date -Is
mkdir "$O/inputs"
cp "$R"/{build.sh,run-probe.sh,CMakeLists.txt,binding.cpp,patch-native.py,grouped_verify.py,probe.py,README.md,check-grouped-split-k.py} "$O/inputs/"
docker image inspect vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
# Fetch pinned source into this build cell; never lazy-fetch into shared assets.
timeout --signal=TERM --kill-after=30s 90m docker run --pull=never --rm \
  --name "$NAME" --cpus=12 --memory=24g --memory-swap=24g \
  -v "$O:/build-cell" \
  -v "$A/compiler:/opt/intel/oneapi/compiler/2026.0:ro" \
  --entrypoint /bin/bash \
  vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f -lc '
set -euxo pipefail
O=/build-cell
C=/opt/intel/oneapi/compiler/2026.0
export PATH="$C/bin:/opt/venv/bin:$PATH"
export LD_LIBRARY_PATH="/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib:$C/lib:${LD_LIBRARY_PATH:-}"
export CC="$C/bin/icx" CXX="$C/bin/icpx" SYCL_ROOT="$C"
export GIT_NO_LAZY_FETCH=1
NATIVE=1796aa8bc8db4ac68d9cd19636cef88f3af81d2b
TLA=cd763790ad2f74d7294435ecf77682bac0062c3a
[[ ${#NATIVE} == 40 && ${#TLA} == 40 ]]
mkdir "$O/native" "$O/tla"
curl --fail --location --retry 2 --max-time 300 \
  "https://codeload.github.com/vllm-project/vllm-xpu-kernels/tar.gz/$NATIVE" -o "$O/native.tar.gz"
tar -xzf "$O/native.tar.gz" -C "$O/native" --strip-components=1 \
  "vllm-xpu-kernels-$NATIVE/csrc/xpu/attn/xe_2" \
  "vllm-xpu-kernels-$NATIVE/csrc/xpu/attn/paged_kv_utils.h" \
  "vllm-xpu-kernels-$NATIVE/LICENSE" "vllm-xpu-kernels-$NATIVE/CMakeLists.txt"
cp "$O/native/CMakeLists.txt" "$O/native-CMakeLists.txt"
sha256sum "$O/native.tar.gz"
grep -F "\"$TLA\"" "$O/native-CMakeLists.txt"
curl --fail --location --retry 2 --max-time 300 \
  "https://codeload.github.com/intel/sycl-tla/tar.gz/$TLA" -o "$O/sycl-tla.tar.gz"
tar -xzf "$O/sycl-tla.tar.gz" -C "$O/tla" --strip-components=1 \
  "sycl-tla-$TLA/include" "sycl-tla-$TLA/tools/util/include" \
  "sycl-tla-$TLA/applications/flash_attention_v2" "sycl-tla-$TLA/LICENSE.txt"
python -P "$O/inputs/patch-native.py" "$O/native" > "$O/native.patch"
git -C "$O/native" apply --check "$O/native.patch"
git -C "$O/native" apply "$O/native.patch"
icpx --version
cmake --version
cmake -S "$O/inputs" -B "$O/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX" \
  -DNATIVE_SOURCE_DIR="$O/native" -DTLA_SOURCE_DIR="$O/tla" \
  -DCMAKE_PREFIX_PATH=/opt/venv/lib/python3.12/site-packages/torch/share/cmake \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$O/build" --parallel 2
sha256sum "$O/build/libb70_grouped_verify.so"
nm -D --defined-only "$O/build/libb70_grouped_verify.so" > "$O/exported-symbols.txt"
'
