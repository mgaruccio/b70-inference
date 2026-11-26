#!/usr/bin/env bash
# Guest: place plugin tree from git archive 1d10e1c under layout/ first.
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null
set -u
case ${1:-layout} in layout|dequant) cd "$(dirname "$0")/${1:-layout}" ;; *) exit 2 ;; esac
stage=$HOME/inkling-vllm-xpu/sparse-69444
D=$HOME/flint-vllm-spike/driver-26.40
export PATH="$HOME/ck-inkling/bin:$PATH" LD_LIBRARY_PATH="$D/usr/lib/x86_64-linux-gnu:$D/usr/local/lib:${LD_LIBRARY_PATH:-}"
export SYCL_TLA_DIR=$HOME/ck-inkling/sycl-tla VLLM_XPU_KERNELS_DIR=$HOME/ck-inkling/vllm-xpu-kernels PYTHON=$stage/.venv/bin/python
ln -sfnT "$stage/sparse" sparse
ln -sfnT src plugin
cp -a "$stage"/plugin/*.dist-info src/
export PYTHONPATH="$PWD/plugin:$stage/sparse:$HOME/flint-vllm-spike/serve" BUILD_DIR="$PWD/build"
mkdir -p "$BUILD_DIR"
"$PYTHON" -Bc 'import importlib.util, pathlib; assert pathlib.Path(importlib.util.find_spec("vllm_inkling2_xpu").origin).resolve().parent == pathlib.Path("src/vllm_inkling2_xpu").resolve()'
bash scripts/build-cutlass-xe2.sh >build.log 2>&1 || { tail -30 build.log; exit 1; }
sha256sum src/vllm_inkling2_xpu/ops/_inkling*.so
