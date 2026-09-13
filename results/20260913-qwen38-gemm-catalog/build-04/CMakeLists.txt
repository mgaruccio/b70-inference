cmake_minimum_required(VERSION 3.26)
project(b70_gemm_catalog LANGUAGES CXX)

# Isolated Linux/ELF experiment: configure with Intel icpx 2026.0 and the
# Torch 2.13.0+xpu prefix; no Python interpreter, pybind, or native top-level build.
# Example:
# cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
#   -DCMAKE_CXX_COMPILER=icpx -DNATIVE_SOURCE_DIR=/src/vllm-xpu-kernels \
#   -DONEDNN_INSTALL_PREFIX=/opt/b70-onednn -DCMAKE_PREFIX_PATH=/path/to/torch
# cmake --build build --target b70_gemm_catalog
# oneDNN must be the patched 80afa71049cd69a3df32adcccb623b12cd7baa22 build:
# STATIC, PIC, Release, CPU_RUNTIME=NONE, GPU_RUNTIME=SYCL, Xe2 only,
# tests/examples off, DNNL_DEV_MODE off. Its build/install is lead-owned.
# Upstream target/flag references inspected 2026-09-13:
# https://raw.githubusercontent.com/uxlfoundation/oneDNN/80afa71049cd69a3df32adcccb623b12cd7baa22/src/CMakeLists.txt
# https://raw.githubusercontent.com/pytorch/pytorch/main/cmake/TorchConfig.cmake.in
# https://sourceware.org/binutils/docs/ld/Options.html (--exclude-libs: hidden on ELF)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "This isolated experiment requires Linux/ELF")
endif()
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS "2026.0" OR
   NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS "2026.1")
  message(FATAL_ERROR "Use the pinned Intel icpx 2026.0 toolchain")
endif()

set(NATIVE_SOURCE_DIR "" CACHE PATH "Unmodified vllm-xpu-kernels v0.1.12 checkout")
set(ONEDNN_INSTALL_PREFIX "" CACHE PATH "Patched static oneDNN install prefix")
if(NOT IS_DIRECTORY "${NATIVE_SOURCE_DIR}/csrc/xpu/onednn" OR
   NOT IS_DIRECTORY "${ONEDNN_INSTALL_PREFIX}")
  message(FATAL_ERROR "Set NATIVE_SOURCE_DIR and ONEDNN_INSTALL_PREFIX")
endif()

find_package(Torch REQUIRED CONFIG)
if(NOT Torch_VERSION VERSION_EQUAL "2.13.0")
  message(FATAL_ERROR "Use the pinned Torch 2.13.0+xpu installation")
endif()
find_library(B70_TORCH_XPU_LIBRARY NAMES torch_xpu
  PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(B70_C10_XPU_LIBRARY NAMES c10_xpu
  PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
find_package(dnnl REQUIRED CONFIG PATHS "${ONEDNN_INSTALL_PREFIX}" NO_DEFAULT_PATH)
get_target_property(B70_DNNL_TYPE DNNL::dnnl TYPE)
if(NOT B70_DNNL_TYPE STREQUAL "STATIC_LIBRARY" OR
   NOT DNNL_CPU_RUNTIME STREQUAL "NONE" OR NOT DNNL_GPU_RUNTIME STREQUAL "SYCL")
  message(FATAL_ERROR "oneDNN must be STATIC with CPU_RUNTIME=NONE and GPU_RUNTIME=SYCL")
endif()

add_library(b70_gemm_catalog SHARED
  binding.cpp
  "${NATIVE_SOURCE_DIR}/csrc/xpu/onednn/onednn_matmul.cpp"
  "${NATIVE_SOURCE_DIR}/csrc/xpu/onednn/onednn_runtime.cpp")
set_target_properties(b70_gemm_catalog PROPERTIES
  CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO
  POSITION_INDEPENDENT_CODE YES
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
target_include_directories(b70_gemm_catalog PRIVATE
  "${NATIVE_SOURCE_DIR}/csrc/xpu"
  "${NATIVE_SOURCE_DIR}/csrc/xpu/onednn"
  ${TORCH_INCLUDE_DIRS})
separate_arguments(B70_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(b70_gemm_catalog PRIVATE -fsycl ${B70_TORCH_CXX_FLAGS})
# Resolve our oneDNN references from the private archive before Torch's DSOs.
# Hidden native symbols also prevent binding to the already-loaded _xpu_C.
target_link_libraries(b70_gemm_catalog PRIVATE DNNL::dnnl ${TORCH_LIBRARIES}
  "${B70_TORCH_XPU_LIBRARY}" "${B70_C10_XPU_LIBRARY}")
target_link_options(b70_gemm_catalog PRIVATE -fsycl
  "LINKER:--exclude-libs,ALL" "LINKER:--no-undefined")
