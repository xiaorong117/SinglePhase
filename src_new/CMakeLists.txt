#Current work path
set(CURRENT_WORK_PATH /home/rong/Mycode/SinglePhase)
set(LIB_PATH /home/rong/桌面/Mycode/lib)

# add_library(gpu_solver_single STATIC ${CURRENT_WORK_PATH}/src/GPU.cpp)
# find_package(OpenMP REQUIRED)
# target_link_libraries(gpu_solver_single PRIVATE OpenMP::OpenMP_CXX)
# target_compile_options(gpu_solver_single
#   PUBLIC
#     ${OpenMP_CXX_FLAGS}
#   )
# set_target_properties(gpu_solver_single
#   PROPERTIES
#     LINK_FLAGS ${OpenMP_CXX_FLAGS}
#   )
# find_package(CUDA REQUIRED)
# target_include_directories(gpu_solver_single PUBLIC
#     /home/rong/Mycode/SinglePhase/src
#     /home/rong/Mycode/lib/cuda-samples-master/Common
#     /home/rong/Mycode/lib/AMGX/include
#     ${CUDA_INCLUDE_DIRS}
# )
# # # 静态连接AMGX库
# target_link_libraries(gpu_solver_single PRIVATE "/home/rong/Mycode/lib/AMGX/build/libamgx.a")
# # # 静态连接CUDA库
# target_link_libraries(gpu_solver_single PRIVATE  "/usr/local/cuda-12.6/targets/x86_64-linux/lib/libcusolver_static.a")
# target_link_libraries(gpu_solver_single PRIVATE  "/usr/local/cuda-12.6/targets/x86_64-linux/lib/libcublas_static.a")
# target_link_libraries(gpu_solver_single PRIVATE  "/usr/local/cuda-12.6/targets/x86_64-linux/lib/libcublasLt_static.a")
# target_link_libraries(gpu_solver_single PRIVATE  "/usr/local/cuda-12.6/targets/x86_64-linux/lib/libcudart_static.a")
# target_link_libraries(gpu_solver_single PRIVATE  "/usr/local/cuda-12.6/targets/x86_64-linux/lib/libculibos.a")
# target_link_libraries(gpu_solver_single PRIVATE  "/usr/local/cuda-12.6/targets/x86_64-linux/lib/libcusparse_static.a")
# target_link_libraries(gpu_solver_single PRIVATE  "/home/rong/Mycode/lib/libnvToolsExt.so.1.0.0")

add_executable(${PROJECT_NAME} ${CURRENT_WORK_PATH}/src/Solver_openmp_ad.cpp)

# 添加头文件路径
target_include_directories(${PROJECT_NAME} PRIVATE /home/rong/Mycode/lib/AMGX/include)
target_include_directories(${PROJECT_NAME} PRIVATE /home/rong/Mycode/lib/cuda-samples-master/Common)
target_include_directories(${PROJECT_NAME} PRIVATE /usr/local/cuda-12.6/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${CURRENT_WORK_PATH}/src)

# 链接库
#寻找添加Eigen头文件目录 
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# # 链接OpenMP库
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

#寻找添加CUDA的头文件目录 寻找库目录
find_package(CUDA REQUIRED)
include_directories(/home/rong/Mycode/lib/cuda-samples-master/Common)
include_directories(${CUDA_INCLUDE_DIRS})
find_library(CUBLAS_LIBRARY cublas /usr/local/cuda-12.6/targets/x86_64-linux/lib)
find_library(CUSPARSE_LIBRARY cusparse /usr/local/cuda-12.6/targets/x86_64-linux/lib)
find_library(CUDART_LIBRARY cudart /usr/local/cuda-12.6/targets/x86_64-linux/lib)
find_library(CUSOLVER_LIBRARY cusolver /usr/local/cuda-12.6/targets/x86_64-linux/lib)
find_library(NVTX_LIB nvToolsExt /usr/local/cuda-12.1/targets/x86_64-linux/lib)
target_link_libraries(${PROJECT_NAME} PUBLIC  ${CUSOLVER_LIBRARY} ${CUDART_LIBRARY} ${CUBLAS_LIBRARY} ${CUSPARSE_LIBRARY})

# 连接AMGX库和头文件
target_include_directories(${PROJECT_NAME} PRIVATE /home/rong/Mycode/lib/AMGX/include)
target_link_libraries(${PROJECT_NAME} PUBLIC "/home/rong/Mycode/lib/AMGX/build/libamgxsh.so")


# 连接GLS数学库
target_link_libraries(${PROJECT_NAME} PUBLIC gsl gslcblas)

# 连接fadbad++
include_directories(/home/rong/Mycode/lib/FADBAD++-2.1/FADBAD++/)


#### 静态连接库 #######
# CUDA 头文件
# find_package(CUDA REQUIRED)
# include_directories(/home/rong/Mycode/lib/cuda-samples-master/Common)
# include_directories(${CUDA_INCLUDE_DIRS})
# 静态连接AMGX库
# target_link_libraries(${PROJECT_NAME} PRIVATE "/home/rong/Mycode/lib/AMGX/build/libamgx.a")
# 静态连接CUDA库
# target_link_libraries(${PROJECT_NAME} PRIVATE  "/usr/local/cuda-12.6/targets/x86_64-linux/lib/libcusolver_static.a")
# target_link_libraries(${PROJECT_NAME} PRIVATE  "/usr/local/cuda-12.6/targets/x86_64-linux/lib/libcublas_static.a")
# target_link_libraries(${PROJECT_NAME} PRIVATE  "/usr/local/cuda-12.6/targets/x86_64-linux/lib/libcublasLt_static.a")
# target_link_libraries(${PROJECT_NAME} PRIVATE  "/usr/local/cuda-12.6/targets/x86_64-linux/lib/libcudart_static.a")
# target_link_libraries(${PROJECT_NAME} PRIVATE  "/usr/local/cuda-12.6/targets/x86_64-linux/lib/libculibos.a")
# target_link_libraries(${PROJECT_NAME} PRIVATE  "/usr/local/cuda-12.6/targets/x86_64-linux/lib/libcusparse_static.a")

# 动态连接 libnvToolsExt.so.1.0.0 这个库没有静态版本
# target_link_libraries(${PROJECT_NAME} PRIVATE  "/home/rong/Mycode/lib/libnvToolsExt.so.1.0.0")
# 静态连接GSL库
# target_link_libraries(${PROJECT_NAME} PRIVATE "/home/rong/Mycode/lib/gsl/lib/libgsl.a" "/home/rong/Mycode/lib/gsl/lib/libgslcblas.a")
#### 静态连接库 #######

# copy
set(TARGET_DIRECTORY "${CURRENT_WORK_PATH}/")
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${TARGET_DIRECTORY}
    COMMENT "Copying ${PROJECT_NAME} to ${TARGET_DIRECTORY}"
)