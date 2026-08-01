cmake_minimum_required(VERSION 3.18)
project(cpp_api_pipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GGML_CUDA ON CACHE BOOL "Enable CUDA backend" FORCE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture: 72=Xavier, 87=Orin, 110=Thor
# Override with: cmake .. -DCMAKE_CUDA_ARCHITECTURES=72
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA architecture" FORCE)
endif()

add_subdirectory(llama.cpp)

find_package(CUDAToolkit REQUIRED)

add_executable(cpp_api_pipeline cpp_api_llama.cpp)

target_include_directories(cpp_api_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(cpp_api_pipeline PRIVATE llama dl pthread CUDA::cudart)

target_compile_definitions(cpp_api_pipeline PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:GGML_CUDA>)
