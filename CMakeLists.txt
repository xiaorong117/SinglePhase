cmake_minimum_required(VERSION 3.22)
project(myproject LANGUAGES C CXX Fortran CUDA)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

# set(CMAKE_C_COMPILER mpicc)
# set(CMAKE_CXX_COMPILER mpicxx)
# set(CMAKE_Fortran_COMPILER mpif90)

add_subdirectory(src bin)