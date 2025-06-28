#Current work path
set(CURRENT_WORK_PATH /home/rong/Mycode/SinglePhase)
set(LIB_PATH /home/rong/桌面/Mycode/lib)

#寻找添加Eigen头文件目录 
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

#寻找添加CUDA的头文件目录 寻找库目录
find_package(CUDA REQUIRED)
include_directories(/home/rong/Mycode/lib/cuda-samples-master/Common)
include_directories(${CUDA_INCLUDE_DIRS})
find_library(CUBLAS_LIBRARY cublas /usr/local/cuda-12.9/targets/x86_64-linux/lib)
find_library(CUSPARSE_LIBRARY cusparse /usr/local/cuda-12.9/targets/x86_64-linux/lib)
find_library(CUDART_LIBRARY cudart /usr/local/cuda-12.9/targets/x86_64-linux/lib)
find_library(CUSOLVER_LIBRARY cusolver /usr/local/cuda-12.9/targets/x86_64-linux/lib)
# find_library(NVTX_LIB nvToolsExt /usr/local/cuda-12.1/targets/x86_64-linux/lib)
#fadbad++
include_directories(/home/rong/桌面/Mycode/lib/FADBAD++-2.1/FADBAD++)
# add_executable
add_executable(${PROJECT_NAME} ${CURRENT_WORK_PATH}/src/Solver_openmp.cpp)
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

# 链接库
target_include_directories(${PROJECT_NAME} PRIVATE ${CURRENT_WORK_PATH}/src)
target_include_directories(${PROJECT_NAME} PRIVATE /home/rong/Mycode/lib/AMGX/include)
# target_link_libraries(${PROJECT_NAME} PUBLIC -static-libstdc++ -static-libgcc)
# target_link_libraries(${PROJECT_NAME} PUBLIC "/home/rong/Mycode/lib/AMGX/build/libamgx.a")
# target_link_libraries(${PROJECT_NAME} PUBLIC  "/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcusolver_static.a")
# target_link_libraries(${PROJECT_NAME} PUBLIC  "/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcublas_static.a")
# target_link_libraries(${PROJECT_NAME} PUBLIC  "/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcublasLt_static.a")
# target_link_libraries(${PROJECT_NAME} PUBLIC  "/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcudart_static.a")
# target_link_libraries(${PROJECT_NAME} PUBLIC  "/usr/local/cuda-12.9/targets/x86_64-linux/lib/libculibos.a")
# target_link_libraries(${PROJECT_NAME} PUBLIC  "/usr/local/cuda-12.9/targets/x86_64-linux/lib/libcusparse_static.a")
# target_link_libraries(${PROJECT_NAME} PUBLIC  "/home/rong/Mycode/lib/libnvToolsExt.so.1.0.0")
# target_link_libraries(${PROJECT_NAME} PUBLIC "/home/rong/Mycode/lib/gsl/lib/libgsl.a" "/home/rong/Mycode/lib/gsl/lib/libgslcblas.a")

target_link_libraries(${PROJECT_NAME} PUBLIC  ${CUSOLVER_LIBRARY} ${CUDART_LIBRARY} ${CUBLAS_LIBRARY} ${CUSPARSE_LIBRARY})
target_link_libraries(${PROJECT_NAME} PUBLIC "/home/rong/Mycode/lib/AMGX/build/libamgxsh.so")
target_link_libraries(${PROJECT_NAME} PUBLIC gsl gslcblas)


# copy
set(TARGET_DIRECTORY "${CURRENT_WORK_PATH}/")
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${TARGET_DIRECTORY}
    COMMENT "Copying ${PROJECT_NAME} to ${TARGET_DIRECTORY}"
)