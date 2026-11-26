#!/usr/bin/env bash
# b70-dev, CPU-only AOT compiles. Run from verify-kernels; csrc/ and scripts/
# populated from plugin main @7884405. Never install over the production .so.
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null
set -u
cd "$(dirname "$0")"
root=$PWD
stage=$HOME/inkling-vllm-xpu/sparse-69444
D=$HOME/flint-vllm-spike/driver-26.40
export PATH="$HOME/ck-inkling/bin:$PATH" LD_LIBRARY_PATH="$D/usr/lib/x86_64-linux-gnu:$D/usr/local/lib:${LD_LIBRARY_PATH:-}"
export SYCL_TLA_DIR=$HOME/ck-inkling/sycl-tla VLLM_XPU_KERNELS_DIR=$HOME/ck-inkling/vllm-xpu-kernels PYTHON=$stage/.venv/bin/python
for dist in "$@"; do
    case $dist in 3|6|9) tree=$root/pf$dist ;; static) tree=$root/static ;; *) exit 2 ;; esac
    mkdir -p "$tree"/{build,csrc,scripts,plugin}
    cp -a "$stage/plugin/vllm_inkling2_xpu" "$tree/plugin/"
    cp -a "$stage"/plugin/vllm_inkling2_xpu-*.dist-info "$tree/plugin/" # vLLM entry point registration
    test ! -L "$tree/plugin/vllm_inkling2_xpu/ops"
    ln -sfnT "$stage/sparse" "$tree/sparse"
    cp scripts/build-cutlass-xe2.sh "$tree/scripts/"
    if [[ $dist == static ]]; then
        patch -s -o "$tree/csrc/nvfp4_xe2_ext.cpp" csrc/nvfp4_xe2_ext.cpp <static-tiles.patch
    else
        grep -qF 'constexpr int prefetch_dist = 6;' csrc/nvfp4_xe2_ext.cpp
        sed "s/constexpr int prefetch_dist = 6;/constexpr int prefetch_dist = $dist;/" csrc/nvfp4_xe2_ext.cpp >"$tree/csrc/nvfp4_xe2_ext.cpp"
    fi
    rm -f "$tree"/plugin/vllm_inkling2_xpu/ops/_inkling_nvfp4_xe2*.so
    export PYTHONPATH="$tree/plugin:$stage/sparse:$HOME/flint-vllm-spike/serve" BUILD_DIR=$tree/build
    "$PYTHON" -Bc 'import importlib.util, pathlib, sys; assert pathlib.Path(importlib.util.find_spec("vllm_inkling2_xpu").origin).resolve().parent == pathlib.Path(sys.argv[1])' "$tree/plugin/vllm_inkling2_xpu"
    bash "$tree/scripts/build-cutlass-xe2.sh" >"$tree/build.log" 2>&1 || { tail -30 "$tree/build.log"; exit 1; }
    sha256sum "$tree"/plugin/vllm_inkling2_xpu/ops/_inkling*.so
 done
