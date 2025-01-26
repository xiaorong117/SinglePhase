cmake_minimum_required(VERSION 3.22)
project(myproject LANGUAGES C CXX Fortran CUDA)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

add_subdirectory(src bin)