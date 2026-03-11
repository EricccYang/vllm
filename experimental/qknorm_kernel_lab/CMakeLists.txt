cmake_minimum_required(VERSION 3.18)
project(qknorm_kernel_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Default to H100 to reduce compile time for this isolated lab.
  set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY
             NAMES torch_python libtorch_python
             PATHS "${TORCH_INSTALL_PREFIX}/lib")
if(NOT TORCH_PYTHON_LIBRARY)
  message(FATAL_ERROR "libtorch_python not found under TORCH_INSTALL_PREFIX.")
endif()

set(LAB_NAME qknorm_rope_lab)

if(NOT DEFINED LAB_KERNEL_SOURCE)
  set(LAB_KERNEL_SOURCE "src/fused_qknorm_rope_improve_kernel.cu")
endif()
message(STATUS "Lab kernel source: ${LAB_KERNEL_SOURCE}")

set(LAB_SOURCES
    src/qknorm_rope_lab_binding.cpp
    ${LAB_KERNEL_SOURCE}
)

add_library(${LAB_NAME} MODULE ${LAB_SOURCES})
target_include_directories(${LAB_NAME} PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../../csrc
)
target_link_libraries(${LAB_NAME} PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(${LAB_NAME} PRIVATE ${TORCH_PYTHON_LIBRARY})
target_compile_definitions(${LAB_NAME} PRIVATE TORCH_EXTENSION_NAME=${LAB_NAME})
target_compile_options(${LAB_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr>
)

set_target_properties(${LAB_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
