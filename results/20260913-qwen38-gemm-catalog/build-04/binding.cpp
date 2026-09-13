#include <torch/library.h>

#include "ops.h"  // Original vllm-xpu-kernels v0.1.12 csrc/xpu/ops.h.

// Exact upstream seven-argument schema and function; no numerical wrapper.
// https://raw.githubusercontent.com/vllm-project/vllm-xpu-kernels/v0.1.12/csrc/xpu/torch_bindings.cpp
// Register only this isolated namespace. Leave the original _xpu_C loaded.
// Load libb70_gemm_catalog.so with torch.ops.load_library(path); no PyInit export
// or torch/extension.h (Python/pybind dependency) is needed.
TORCH_LIBRARY(b70_gemm_catalog, ops) {
  ops.def(
      "int4_gemm_w4a16(Tensor A, Tensor B, Tensor? bias, Tensor B_scale, "
      "Tensor B_zp, int group_size, Tensor? g_idx) -> Tensor");
  ops.impl("int4_gemm_w4a16", torch::kXPU, &int4_gemm_w4a16);
}
