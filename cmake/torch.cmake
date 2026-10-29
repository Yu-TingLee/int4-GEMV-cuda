# Python from the uv venv, and the LibTorch bundled with its pip torch.
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE torch_cmake_prefix
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE torch_import_result)
if(NOT torch_import_result EQUAL 0)
  message(FATAL_ERROR
    "torch is not installed in ${Python_EXECUTABLE}. Run `uv sync` on the GPU machine.")
endif()
list(APPEND CMAKE_PREFIX_PATH "${torch_cmake_prefix}")
find_package(Torch REQUIRED)

find_library(TORCH_PYTHON_LIBRARY torch_python
  PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)

# Python extensions must carry torch's pybind11 ABI tags to exchange types with it.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "import torch._C as c; print(';'.join(f'{k}={getattr(c, \"_PYBIND11_\" + k)}' for k in ('COMPILER_TYPE', 'STDLIB', 'BUILD_ABI') if getattr(c, '_PYBIND11_' + k, None)))"
  OUTPUT_VARIABLE TORCH_PYBIND11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
list(TRANSFORM TORCH_PYBIND11_ABI REPLACE "^([A-Z_]+)=(.*)$" "PYBIND11_\\1=\"\\2\"")
