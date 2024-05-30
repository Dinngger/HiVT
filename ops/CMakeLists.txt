cmake_minimum_required(VERSION 3.9.0)
project(cuda_test LANGUAGES C CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")  # -G

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/dinger/.local/lib/python3.8/site-packages/torch/share/cmake")
find_package(Torch)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

list(APPEND TEST_INCLUDE_DIRECTORIES "/home/dinger/.local/lib/python3.8/site-packages/torch/include/torch/csrc/api/include")
list(APPEND TEST_INCLUDE_DIRECTORIES "/home/dinger/.local/lib/python3.8/site-packages/torch/include")
list(APPEND TEST_INCLUDE_DIRECTORIES "/usr/include/python3.8")

add_executable(cuda_test cuda_test.cc flash_gat_single.cu)
target_include_directories(cuda_test PUBLIC ${TEST_INCLUDE_DIRECTORIES})
target_link_libraries(cuda_test ${TORCH_LIBRARIES})
