#!/bin/bash
# Build SergiioB's _inkling_nvfp4_xe2 (cookbook f10f667) for the hc-ck1 tree, same flags as build-nvfp4-cutlass-xe2.sh.
set -eo pipefail
. /opt/intel/oneapi/setvars.sh --force >/dev/null; set -u
D=$HOME/flint-vllm-spike/driver-26.40; export PATH=$HOME/ck-inkling/bin:$PATH LD_LIBRARY_PATH=$D/usr/lib/x86_64-linux-gnu:$D/usr/local/lib:${LD_LIBRARY_PATH:-}
PY=$HOME/flint-vllm-spike/.venv/bin/python
CB=$HOME/ck-inkling/cookbook/patches/inkling2-mini-vllm-xpu
OPS=$HOME/inkling-vllm-xpu/src/hc-ck1/vllm/models/inkling2/nvidia/ops
PY_INC=$($PY -c 'import sysconfig; print(sysconfig.get_path("include"))')
EXT=$($PY -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
TORCH_DIR=$($PY -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).resolve().parent)' 2>/dev/null)
ABI=$($PY -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))' 2>/dev/null)
mkdir -p $HOME/ck-inkling/build
time icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg_g31 -options -ze-opt-large-register-file" \
  -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate -O3 -std=c++20 -fPIC -shared -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED -DTORCH_EXTENSION_NAME=_inkling_nvfp4_xe2 \
  -D_GLIBCXX_USE_CXX11_ABI=$ABI -I$PY_INC -I$TORCH_DIR/include -I$TORCH_DIR/include/torch/csrc/api/include \
  -I$HOME/ck-inkling/sycl-tla/include -I$HOME/ck-inkling/sycl-tla/tools/util/include -I$HOME/ck-inkling/sycl-tla/applications -I$HOME/ck-inkling/vllm-xpu-kernels/csrc/xpu/cutlass_kernels -I$HOME/ck-inkling/vllm-xpu-kernels/csrc -I$HOME/ck-inkling/vllm-xpu-kernels \
  $CB/nvfp4_xe2_ext.cpp -L$TORCH_DIR/lib -Wl,-rpath,$TORCH_DIR/lib -ltorch -ltorch_cpu -ltorch_python -lc10 -lc10_xpu -ltorch_xpu \
  -o $HOME/ck-inkling/build/_inkling_nvfp4_xe2$EXT
cp $HOME/ck-inkling/build/_inkling_nvfp4_xe2$EXT $OPS/
ls -la $OPS/_inkling_nvfp4_xe2$EXT
