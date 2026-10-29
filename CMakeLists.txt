cmake_minimum_required(VERSION 3.26)
project(int4-gemv-cuda VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 89)  # RTX 3090, RTX 4090
endif()
# nvcc otherwise compiles host code with the default c++ on PATH, which may be older than CXX.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif()
enable_language(CUDA)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.6)
  message(FATAL_ERROR "nvcc ${CMAKE_CUDA_COMPILER_VERSION} (${CMAKE_CUDA_COMPILER}) is too old; "
                      "torch needs CUDA 12.6 or newer. Set CUDACXX to a newer nvcc and "
                      "reconfigure with --fresh.")
endif()
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Torch's CMake config reads its arch list from here; without it, it probes the local GPU.
list(TRANSFORM CMAKE_CUDA_ARCHITECTURES REPLACE "^([0-9]+)([0-9])$" "\\1.\\2"
     OUTPUT_VARIABLE torch_cuda_arch_list)
set(ENV{TORCH_CUDA_ARCH_LIST} "${torch_cuda_arch_list}")

include(cmake/torch.cmake)

option(INT4GEMV_PTXAS_VERBOSE "Print registers and spills per kernel at build time" OFF)

Python_add_library(int4gemv_ops MODULE WITH_SOABI
  csrc/bindings.cpp
  csrc/checks.cpp
  csrc/sequential.cpp
  csrc/naive.cu
  csrc/coalesced.cu
  csrc/vectorized.cu
  csrc/load_only.cu
)
target_include_directories(int4gemv_ops PRIVATE csrc)
target_link_libraries(int4gemv_ops PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_compile_definitions(int4gemv_ops PRIVATE TORCH_EXTENSION_NAME=int4gemv_ops
                                                ${TORCH_PYBIND11_ABI})
separate_arguments(torch_cxx_flags UNIX_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(int4gemv_ops PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${torch_cxx_flags}>)
if(INT4GEMV_PTXAS_VERBOSE)
  target_compile_options(int4gemv_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
endif()
set_target_properties(int4gemv_ops PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)
