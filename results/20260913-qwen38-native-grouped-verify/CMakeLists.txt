cmake_minimum_required(VERSION 3.26)
project(b70_grouped_verify LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS "2026.0" OR
   NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS "2026.1")
  message(FATAL_ERROR "Use Linux and the pinned Intel icpx 2026.0 toolchain")
endif()
set(NATIVE_SOURCE_DIR "" CACHE PATH "Experiment-owned patched native headers")
set(TLA_SOURCE_DIR "" CACHE PATH "Pinned SYCL-TLA headers/applications")
if(NOT EXISTS "${NATIVE_SOURCE_DIR}/csrc/xpu/attn/xe_2/paged_decode.hpp" OR
   NOT EXISTS "${TLA_SOURCE_DIR}/applications/flash_attention_v2/collective/fmha_fusion.hpp")
  message(FATAL_ERROR "Set NATIVE_SOURCE_DIR and TLA_SOURCE_DIR")
endif()
find_package(Torch REQUIRED CONFIG)
if(NOT Torch_VERSION VERSION_EQUAL "2.13.0")
  message(FATAL_ERROR "Use the pinned Torch 2.13.0+xpu installation")
endif()
find_library(B70_TORCH_XPU NAMES torch_xpu
  PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(B70_C10_XPU NAMES c10_xpu
  PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)

# Header-only native/SYCL-TLA use: no upstream CMake, oneDNN, pybind, or
# additional decode/prefill policy instantiations. Match native Xe2 flags.
add_library(b70_grouped_verify SHARED binding.cpp)
set_target_properties(b70_grouped_verify PROPERTIES
  CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO
  POSITION_INDEPENDENT_CODE YES
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
target_include_directories(b70_grouped_verify PRIVATE
  "${NATIVE_SOURCE_DIR}/csrc"
  "${NATIVE_SOURCE_DIR}/csrc/xpu/attn/xe_2"
  "${TLA_SOURCE_DIR}/include" "${TLA_SOURCE_DIR}/tools/util/include"
  "${TLA_SOURCE_DIR}/applications" ${TORCH_INCLUDE_DIRS})
separate_arguments(B70_TORCH_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_definitions(b70_grouped_verify PRIVATE
  CUTLASS_ENABLE_HEADERS_ONLY CUTLASS_ENABLE_SYCL SYCL_INTEL_TARGET
  CUTLASS_VERSIONS_GENERATED VLLM_XPU_ENABLE_XE2)
target_compile_options(b70_grouped_verify PRIVATE ${B70_TORCH_FLAGS}
  -fsycl -fsycl-targets=spir64_gen -fno-sycl-instrument-device-code)
target_link_libraries(b70_grouped_verify PRIVATE ${TORCH_LIBRARIES}
  "${B70_TORCH_XPU}" "${B70_C10_XPU}" ze_loader)
target_link_options(b70_grouped_verify PRIVATE
  -fsycl -fsycl-targets=spir64_gen -fsycl-max-parallel-link-jobs=2
  -flink-huge-device-code
  -Xspirv-translator
  -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate
  -Xsycl-target-backend=spir64_gen
  "-device bmg-g31-a0 -internal_options -cl-intel-256-GRF-per-thread"
  "LINKER:--exclude-libs,ALL" "LINKER:--no-undefined")
