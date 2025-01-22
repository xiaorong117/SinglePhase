#Current work path
set(CURRENT_WORK_PATH /home/rong/桌面/Mycode/shale-gas)
set(LIB_PATH /home/rong/桌面/Mycode/lib)

#寻找添加Eigen头文件目录 
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

#SuiteSq
#-------------------------------------------------------------------------------
# find library dependencies
#-------------------------------------------------------------------------------

# look for all SuiteSparse packages:
find_package ( SuiteSparse_config 7.8.2 REQUIRED )
find_package ( AMD 3.3.3 REQUIRED )
find_package ( BTF 2.3.2 REQUIRED )
find_package ( CAMD 3.3.3 REQUIRED )
find_package ( CCOLAMD 3.3.4 REQUIRED )
find_package ( CHOLMOD 5.3.0 REQUIRED )
find_package ( COLAMD 3.3.4 REQUIRED )
find_package ( CXSparse 4.4.1 REQUIRED )
find_package ( GraphBLAS 9.3.1 )
find_package ( KLU 2.3.4 REQUIRED )
find_package ( KLU_CHOLMOD 2.3.4 REQUIRED )
find_package ( LDL 3.3.2 REQUIRED )
find_package ( LAGraph 1.1.4 )
find_package ( SuiteSparse_Mongoose 3.3.4 REQUIRED )
find_package ( ParU 0.3.0 REQUIRED )
find_package ( RBio 4.3.3 REQUIRED )
find_package ( SPEX 3.2.1 REQUIRED )    # requires GMP and MPFR
find_package ( SPQR 4.3.4 REQUIRED )
find_package ( UMFPACK 6.3.4 REQUIRED )

# for GMP and MPFR
find_package ( MPFR 4.0.2 REQUIRED )    # from SPEX/cmake_modules
find_package ( GMP 6.1.2 REQUIRED )     # from SPEX/cmake_modules

#寻找添加CUDA的头文件目录 寻找库目录
find_package(CUDA REQUIRED)
include_directories(/home/rong/桌面/Mycode/lib/cuda-samples-master/Common)
include_directories(${CUDA_INCLUDE_DIRS})
find_library(CUBLAS_LIBRARY cublas /usr/local/cuda-12.1/targets/x86_64-linux/lib)
find_library(CUSPARSE_LIBRARY cusparse /usr/local/cuda-12.1/targets/x86_64-linux/lib)
find_library(CUDART_LIBRARY cudart /usr/local/cuda-12.1/targets/x86_64-linux/lib)
find_library(CUSOLVER_LIBRARY cusolver /usr/local/cuda-12.1/targets/x86_64-linux/lib)

#fadbad++
include_directories(/home/rong/桌面/Mycode/lib/FADBAD++-2.1/FADBAD++)

#Mumps
# find_path(MUMPS_INCLUDE_DIR dmumps_c.h PATHS ${LIB_PATH}/mumps-main/mumps/5.7.1/include)
# find_library(MUMPS_LIBRARIES NAMES dmumps mumps_common metis pord esmumps scotch scotcherr PATHS ${LIB_PATH}/mumps-main/mumps/5.7.1/lib /usr/lib ${LIB_PATH}/mumps-main/mumps/5.7.1/PORD/lib/)
# find_library(BLAS_LIBRARIES NAMES scalapack-openmpi lapack blas pthread)
if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU|Intel")
  add_compile_options($<$<COMPILE_LANGUAGE:C>:-Werror-implicit-function-declaration>)
elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
  add_link_options($<$<COMPILE_LANGUAGE:C>:/NODEFAULTLIB:MSVCRT>)
endif()

find_package(MUMPS CONFIG REQUIRED)
message(STATUS "MUMPS_DIR: ${MUMPS_DIR}")


#Add exe
# add_executable(${PROJECT_NAME} ${CURRENT_WORK_PATH}/src/noncom_Eigen_CUDA_AMGX.cpp ${CURRENT_WORK_PATH}/src/bicgstab_API.cpp ${CURRENT_WORK_PATH}/src/cg_API.cpp ${CURRENT_WORK_PATH}/src/mmio_wrapper.cpp  ${CURRENT_WORK_PATH}/src/mmio.c)  
# add_executable(${PROJECT_NAME} ${CURRENT_WORK_PATH}/src/cuSolverSp_LinearSolver.cpp ${CURRENT_WORK_PATH}/src/mmio_wrapper.cpp  ${CURRENT_WORK_PATH}/src/mmio.c)  
# add_executable(${PROJECT_NAME} ${CURRENT_WORK_PATH}/src/superlu.cpp ${CURRENT_WORK_PATH}/src/noncom_Eigen_CUDA_AMGX.cpp ${CURRENT_WORK_PATH}/src/bicgstab_API.cpp ${CURRENT_WORK_PATH}/src/cg_API.cpp ${CURRENT_WORK_PATH}/src/mmio_wrapper.cpp  ${CURRENT_WORK_PATH}/src/mmio.c)
# add_executable(${PROJECT_NAME} ${CURRENT_WORK_PATH}/src/superlu.cpp)
# add_executable(${PROJECT_NAME} ${CURRENT_WORK_PATH}/src/dfgmr.c ${CURRENT_WORK_PATH}/src/ditersol.c)
# add_executable(${PROJECT_NAME} ${CURRENT_WORK_PATH}/src/noncom_Eigen_CUDA_AMGX.cpp ${CURRENT_WORK_PATH}/src/bicgstab_API.cpp ${CURRENT_WORK_PATH}/src/cg_API.cpp ${CURRENT_WORK_PATH}/src/mmio_wrapper.cpp  ${CURRENT_WORK_PATH}/src/mmio.c ${CURRENT_WORK_PATH}/src/superlu.cpp ${CURRENT_WORK_PATH}/src/dfgmr.c ${CURRENT_WORK_PATH}/src/ditersol.c)
# add_executable(${PROJECT_NAME} ${CURRENT_WORK_PATH}/src/main.cu ${CURRENT_WORK_PATH}/src/sequential.c  ${CURRENT_WORK_PATH}/src/helper.c)
add_executable(${PROJECT_NAME} ${CURRENT_WORK_PATH}/src/workstation.cpp)

#MPI AND OPENMP
find_package(MPI COMPONENTS CXX REQUIRED)
target_compile_options(${PROJECT_NAME}
  PUBLIC
    ${MPI_C_COMPILE_FLAGS}
  )

target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${MPI_C_INCLUDE_PATH}                         
  )

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    ${MPI_C_LIBRARIES}
  )


# 链接OpenMP库
find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(${PROJECT_NAME}
  PUBLIC
    ${OpenMP_CXX_FLAGS}
  )
set_target_properties(${PROJECT_NAME}
  PROPERTIES
    LINK_FLAGS ${OpenMP_CXX_FLAGS}
  )
#openacc
find_package (OpenACC)
set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenACC_CXX_FLAGS}" )

# 链接库
target_include_directories(${PROJECT_NAME} PRIVATE ${CURRENT_WORK_PATH}/src)
target_include_directories(${PROJECT_NAME} PRIVATE /home/rong/桌面/Mycode/lib/AMGX/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${MUMPS_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${MPI_INCLUDE_PATH})



target_link_libraries(${PROJECT_NAME} PUBLIC superlu)
target_link_libraries(${PROJECT_NAME} PUBLIC metis)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CUSPARSE_LIBRARY} ${CUBLAS_LIBRARY} ${CUDART_LIBRARY} ${CUSOLVER_LIBRARY})
target_link_libraries(${PROJECT_NAME} PUBLIC amgxsh)
target_link_libraries(${PROJECT_NAME} PUBLIC gsl gslcblas)
target_link_libraries(${PROJECT_NAME} PUBLIC ${MUMPS_LIBRARIES} ${BLAS_LIBRARIES} )
target_link_libraries(${PROJECT_NAME} PRIVATE MUMPS::MUMPS "$<$<BOOL:${MPI_CXX_FOUND}>:MPI::MPI_CXX>")

target_link_libraries (${PROJECT_NAME} PUBLIC SuiteSparse::AMD)
target_link_libraries (${PROJECT_NAME} PUBLIC SuiteSparse::BTF)
target_link_libraries (${PROJECT_NAME} PUBLIC SuiteSparse::CAMD)
target_link_libraries (${PROJECT_NAME} PUBLIC SuiteSparse::CCOLAMD)
target_link_libraries (${PROJECT_NAME} PUBLIC SuiteSparse::CHOLMOD)
target_link_libraries (${PROJECT_NAME} PUBLIC SuiteSparse::COLAMD)
target_link_libraries (${PROJECT_NAME} PUBLIC SuiteSparse::CXSparse)
target_link_libraries (${PROJECT_NAME} PUBLIC SuiteSparse::GraphBLAS)
target_link_libraries (${PROJECT_NAME} PUBLIC SuiteSparse::KLU) 
target_link_libraries (${PROJECT_NAME} PUBLIC SuiteSparse::KLU_CHOLMOD)
target_link_libraries (${PROJECT_NAME} PUBLIC SuiteSparse::LAGraph)
target_link_libraries (${PROJECT_NAME} PUBLIC SuiteSparse::LDL)
target_link_libraries (${PROJECT_NAME} PUBLIC SuiteSparse::Mongoose)
target_link_libraries (${PROJECT_NAME} PUBLIC SuiteSparse::ParU)
target_link_libraries (${PROJECT_NAME} PUBLIC SuiteSparse::RBio)
target_link_libraries (${PROJECT_NAME} PUBLIC SuiteSparse::SPEX)
target_link_libraries (${PROJECT_NAME} PUBLIC SuiteSparse::SPQR)
target_link_libraries (${PROJECT_NAME} PUBLIC SuiteSparse::SuiteSparseConfig)
target_link_libraries (${PROJECT_NAME} PUBLIC SuiteSparse::UMFPACK)
target_link_libraries (${PROJECT_NAME} PUBLIC m)
target_link_libraries (${PROJECT_NAME} PUBLIC ${BLAS_LIBRARIES})
target_link_libraries (${PROJECT_NAME} PUBLIC ${LAPACK_LIBRARIES})
target_link_libraries (${PROJECT_NAME} PUBLIC ${MPFR_LIBRARIES})
target_link_libraries (${PROJECT_NAME} PUBLIC ${GMP_LIBRARIES})
#copy
set(TARGET_DIRECTORY "${CURRENT_WORK_PATH}/")
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${TARGET_DIRECTORY}
    COMMENT "Copying ${PROJECT_NAME} to ${TARGET_DIRECTORY}"
)