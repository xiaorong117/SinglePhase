#include "Eigen/Core"
#include "Eigen/Eigen"
#include "Eigen/IterativeLinearSolvers"
#include <array>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <math.h>
#include <numeric> // accumulate
#include <omp.h>
#include <set> // STL set
#include <sstream>
#include <string>

#include <dirent.h>
#include <filesystem>
#include <sys/types.h>
#include <unistd.h> // 函数所在头文件
// For gsl
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_poly.h>
#include <gsl/gsl_roots.h>

// For AMGX
#include <amgx_c.h>
#include <amgx_config.h>

/* Using updated (v2) interfaces to cublas usparseSparseToDense*/
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

// Utilities and system includes
#include <helper_cuda.h> // helper function CUDA error checking and initialization
#include <helper_functions.h> // helper for shared functions common to CUDA Samples

const char *sSDKname = "conjugateGradient";
using namespace std;
using namespace std::chrono;
const double CLOCKS_PER_SECOND = ((clock_t)1000);

// 应用ILU预条件子的函数（求解L和U系统）
void apply_ilu_preconditioner(
    cusparseHandle_t cusparseHandle, cusparseMatDescr_t descrL,
    cusparseMatDescr_t descrU, csrsv2Info_t infoL, csrsv2Info_t infoU, int n,
    int nnz, const int *d_csr_offsets, const int *d_csr_columns,
    const double *d_csr_values_ILU, const double *r, double *z, void *bufferL,
    void *bufferU, double *d_y) {
  double alpha = 1.0;

  // 第一步：求解 L*y = r (下三角系统)
  cusparseStatus_t status = cusparseDcsrsv2_solve(
      cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, &alpha, descrL,
      d_csr_values_ILU, d_csr_offsets, d_csr_columns, infoL,
      r,   // 输入向量 r
      d_y, // 输出向量 y
      CUSPARSE_SOLVE_POLICY_USE_LEVEL, bufferL);

  if (status != CUSPARSE_STATUS_SUCCESS) {
    std::cerr << "cusparseL solve failed!" << std::endl;
    exit(1);
  }

  // 第二步：求解 U*z = y (上三角系统)
  status = cusparseDcsrsv2_solve(
      cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz, &alpha, descrU,
      d_csr_values_ILU, d_csr_offsets, d_csr_columns, infoU,
      d_y, // 输入向量 y
      z,   // 输出向量 z
      CUSPARSE_SOLVE_POLICY_USE_LEVEL, bufferU);

  if (status != CUSPARSE_STATUS_SUCCESS) {
    std::cerr << "cusparseU solve failed!" << std::endl;
    exit(1);
  }
}

int conjugateGradient_solver(int iters_, double tol_) {
  /* 5x5 对称正定矩阵示例 */
  array<int, 6> row = {0, 3, 6, 9, 12, 15}; // 行偏移
  array<int, 15> col = {
      0, 1, 4, // 第0行
      0, 1, 2, // 第1行
      1, 2, 3, // 第2行
      2, 3, 4, // 第3行
      0, 3, 4  // 第4行
  };
  array<double, 15> va = {
      4.0,  -1.0, -1.0, // 第0行
      -1.0, 4.0,  -1.0, // 第1行
      -1.0, 4.0,  -1.0, // 第2行
      -1.0, 4.0,  -1.0, // 第3行
      -1.0, -1.0, 4.0   // 第4行
  };
  array<double, 5> B = {1.0, 2.0, 3.0, 4.0, 5.0}; // 右侧向量

  int op = row.size() - 1; // 矩阵行数
  int mp = 0;

  int NA = va.size();
  int *rows_offsets, *columns;
  double *values, *A;

  int *ia, *ja;
  int *d_csr_offsets, *d_csr_columns;
  double *d_csr_values, *d_csr_values_ILU; // 添加用于ILU的矩阵值存储

  ia = new int[op + mp + 1];
  ja = new int[NA];
  A = new double[NA];

  for (int i = 0; i < row.size(); i++) {
    ia[i] = row[i];
  }
  for (int i = 0; i < col.size(); i++) {
    ja[i] = col[i];
  }
  for (int i = 0; i < va.size(); i++) {
    A[i] = va[i];
  }

  rows_offsets = ia;
  columns = ja;
  values = A;

  // 打印矩阵信息
  cout << "Testing with 5x5 symmetric positive definite matrix" << endl;
  cout << "Matrix structure:" << endl;
  for (int i = 0; i < op; i++) {
    cout << "Row " << i << ": ";
    for (int j = ia[i]; j < ia[i + 1]; j++) {
      cout << "(" << ja[j] << ", " << values[j] << ") ";
    }
    cout << endl;
  }
  cout << "Right-hand side vector: ";
  for (int i = 0; i < op; i++) {
    cout << B[i] << " ";
  }
  cout << endl;

  clock_t startTime, endTime;
  startTime = clock();

  int N = op + mp;
  int nnz = ia[op + mp];
  const double tol = tol_;
  const int max_iter = iters_;
  cout << "max_iter:" << max_iter << endl;
  cout << "Matrix size: " << N << "x" << N << endl;
  cout << "Non-zero elements: " << nnz << endl;
  double rz_new, rz_old, rr, r1, dot_pAp, dot_rr;

  double *x;
  double *rhs;
  double *d_x;
  double *d_r, *d_p, *d_Ax, *d_z;
  int k;
  double alpha, beta;

  x = (double *)malloc(N * sizeof(double));
  rhs = (double *)malloc(N * sizeof(double));
  for (size_t i = 0; i < N; i++) {
    rhs[i] = B[i];
    cout << "B[" << i << "] = " << rhs[i] << endl;
  }

  for (int i = 0; i < N; i++) {
    x[i] = 0.0;
  }

  /* Get handle to the CUBLAS context */
  cublasHandle_t cublasHandle = 0;
  cublasStatus_t cublasStatus;
  cublasStatus = cublasCreate(&cublasHandle);
  checkCudaErrors(cublasStatus);

  /* Get handle to the CUSPARSE context */
  cusparseHandle_t cusparseHandle = 0;
  checkCudaErrors(cusparseCreate(&cusparseHandle));

  checkCudaErrors(cudaMalloc((void **)&d_csr_columns, nnz * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_csr_offsets, (N + 1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **)&d_csr_values, nnz * sizeof(double)));
  // 为ILU矩阵值分配内存
  checkCudaErrors(cudaMalloc((void **)&d_csr_values_ILU, nnz * sizeof(double)));

  checkCudaErrors(cudaMalloc((void **)&d_x, N * sizeof(double)));
  checkCudaErrors(cudaMalloc((void **)&d_r, N * sizeof(double)));
  checkCudaErrors(cudaMalloc((void **)&d_p, N * sizeof(double)));
  checkCudaErrors(cudaMalloc((void **)&d_Ax, N * sizeof(double)));
  checkCudaErrors(cudaMalloc((void **)&d_z, N * sizeof(double))); // z向量
  checkCudaErrors(cudaMalloc((void **)&d_y, N * sizeof(double))); // y中间向量

  /* Initialize problem data */
  checkCudaErrors(cudaMemcpy(d_csr_columns, columns, nnz * sizeof(int),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csr_offsets, rows_offsets, (N + 1) * sizeof(int),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csr_values, values, nnz * sizeof(double),
                             cudaMemcpyHostToDevice));
  // 复制矩阵值到ILU的存储空间
  checkCudaErrors(cudaMemcpy(d_csr_values_ILU, values, nnz * sizeof(double),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_r, rhs, N * sizeof(double), cudaMemcpyHostToDevice));

  // ================= ILU(0) 预条件子计算 =================
  cout << "Computing ILU(0) preconditioner..." << endl;

  // 1. 创建矩阵描述符
  cusparseMatDescr_t descrA = 0;
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

  // 2. 为ILU分解创建信息结构体
  csrilu02Info_t info_ilu = 0;
  cusparseCreateCsrilu02Info(&info_ilu);

  // 3. 计算ILU(0)所需的缓冲区大小
  int bufferSize_ilu = 0;
  cusparseDcsrilu02_bufferSize(cusparseHandle, N, nnz, descrA, d_csr_values_ILU,
                               d_csr_offsets, d_csr_columns, info_ilu,
                               &bufferSize_ilu);

  // 4. 分配缓冲区
  void *buffer_ilu = 0;
  cudaMalloc(&buffer_ilu, bufferSize_ilu);

  // 5. 执行ILU(0)分析
  int structural_zero, numerical_zero;
  cusparseDcsrilu02_analysis(cusparseHandle, N, nnz, descrA, d_csr_values_ILU,
                             d_csr_offsets, d_csr_columns, info_ilu,
                             CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer_ilu);

  // 检查分析结果
  cusparseStatus_t status;
  status =
      cusparseXcsrilu02_zeroPivot(cusparseHandle, info_ilu, &numerical_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
    printf("A(%d,%d) 是缺失的结构零\n", numerical_zero, numerical_zero);
    exit(EXIT_FAILURE);
  }

  // 6. 执行ILU(0)数值分解
  cusparseDcsrilu02(cusparseHandle, N, nnz, descrA, d_csr_values_ILU,
                    d_csr_offsets, d_csr_columns, info_ilu,
                    CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer_ilu);

  // 检查分解结果
  status =
      cusparseXcsrilu02_zeroPivot(cusparseHandle, info_ilu, &numerical_zero);
  if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
    printf("U(%d,%d) 为零\n", numerical_zero, numerical_zero);
    exit(EXIT_FAILURE);
  }

  // 7. 为三角求解器创建描述符和信息结构体
  // 创建L和U的描述符
  cusparseMatDescr_t descrL = 0;
  cusparseCreateMatDescr(&descrL);
  cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER); // 下三角矩阵
  cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT);  // 单位对角

  cusparseMatDescr_t descrU = 0;
  cusparseCreateMatDescr(&descrU);
  cusparseSetMatType(descrU, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrU, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER); // 上三角矩阵
  cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT); // 非单位对角

  // 创建三角求解器信息结构体
  csrsv2Info_t infoL = 0;
  csrsv2Info_t infoU = 0;
  cusparseCreateCsrsv2Info(&infoL);
  cusparseCreateCsrsv2Info(&infoU);

  // 8. 为下三角求解器分析
  int bufferSizeL = 0;
  cusparseDcsrsv2_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             N, nnz, descrL, d_csr_values_ILU, d_csr_offsets,
                             d_csr_columns, infoL, &bufferSizeL);

  void *bufferL = 0;
  cudaMalloc(&bufferL, bufferSizeL);

  cusparseDcsrsv2_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N,
                           nnz, descrL, d_csr_values_ILU, d_csr_offsets,
                           d_csr_columns, infoL,
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, bufferL);

  // 9. 为上三角求解器分析
  int bufferSizeU = 0;
  cusparseDcsrsv2_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             N, nnz, descrU, d_csr_values_ILU, d_csr_offsets,
                             d_csr_columns, infoU, &bufferSizeU);

  void *bufferU = 0;
  cudaMalloc(&bufferU, bufferSizeU);

  cusparseDcsrsv2_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N,
                           nnz, descrU, d_csr_values_ILU, d_csr_offsets,
                           d_csr_columns, infoU,
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, bufferU);
  // ======================================================

  double one = 1.0;
  double zero = 0.0;
  double minus_one = -1.0;

  /* Wrap raw data into cuSPARSE generic API objects */
  cusparseSpMatDescr_t matA = NULL;
  checkCudaErrors(cusparseCreateCsr(&matA, N, N, nnz, d_csr_offsets,
                                    d_csr_columns, d_csr_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  cusparseDnVecDescr_t vecx = NULL;
  checkCudaErrors(cusparseCreateDnVec(&vecx, N, d_x, CUDA_R_64F));
  cusparseDnVecDescr_t vecp = NULL;
  checkCudaErrors(cusparseCreateDnVec(&vecp, N, d_p, CUDA_R_64F));
  cusparseDnVecDescr_t vecAx = NULL;
  checkCudaErrors(cusparseCreateDnVec(&vecAx, N, d_Ax, CUDA_R_64F));

  /* Allocate workspace for cuSPARSE */
  size_t bufferSize = 0;
  checkCudaErrors(cusparseSpMV_bufferSize(
      cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecx, &zero,
      vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  void *buffer = NULL;
  checkCudaErrors(cudaMalloc(&buffer, bufferSize));

  /* Begin CG */
  cout << "Initial SpMV..." << endl;
  checkCudaErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &one, matA, vecx, &zero, vecAx, CUDA_R_64F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));
  cudaDeviceSynchronize();

  // 计算初始残差 r = b - Ax
  cout << "Computing initial residual..." << endl;
  checkCudaErrors(cublasDaxpy(cublasHandle, N, &minus_one, d_Ax, 1, d_r, 1));

  // 应用ILU预条件子
  cout << "Applying ILU(0) preconditioner..." << endl;
  apply_ilu_preconditioner(cusparseHandle, descrL, descrU, infoL, infoU, N, nnz,
                           d_csr_offsets, d_csr_columns, d_csr_values_ILU, d_r,
                           d_z, bufferL, bufferU, d_y);
  cudaDeviceSynchronize();

  // 计算 rz = r·z
  checkCudaErrors(cublasDdot(cublasHandle, N, d_r, 1, d_z, 1, &rz_old));
  cout << "Initial rz_old: " << rz_old << endl;

  // 计算 rr = r·r (用于收敛判断)
  checkCudaErrors(cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &rr));
  cout << "Initial residual norm: " << sqrt(rr) << " (rr = " << rr << ")"
       << endl;

  // 设置初始搜索方向 p = z
  checkCudaErrors(cublasDcopy(cublasHandle, N, d_z, 1, d_p, 1));

  r1 = rr; // 当前残差范数平方
  k = 0;

  double RESI = 0;
  while (sqrt(r1) > tol * sqrt(rr) && k < max_iter) {
    // 计算 Ap = A * p
    checkCudaErrors(cusparseSpMV(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecp,
        &zero, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    cudaDeviceSynchronize();

    // 计算 p·Ap
    checkCudaErrors(cublasDdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot_pAp));
    cout << "Iter " << k << " dot_pAp: " << dot_pAp << endl;

    // 保护除以零
    if (abs(dot_pAp) < 1e-15) {
      if (dot_pAp < 0)
        dot_pAp = -1e-15;
      else
        dot_pAp = 1e-15;
    }

    alpha = rz_old / dot_pAp;

    // 更新解: x = x + alpha * p
    checkCudaErrors(cublasDaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1));

    // 更新残差: r = r - alpha * Ap
    double neg_alpha = -alpha;
    checkCudaErrors(cublasDaxpy(cublasHandle, N, &neg_alpha, d_Ax, 1, d_r, 1));

    // 应用ILU预条件子
    apply_ilu_preconditioner(cusparseHandle, descrL, descrU, infoL, infoU, N,
                             nnz, d_csr_offsets, d_csr_columns,
                             d_csr_values_ILU, d_r, d_z, bufferL, bufferU, d_y);
    cudaDeviceSynchronize();

    // 计算新的点积 rz_new = r·z
    checkCudaErrors(cublasDdot(cublasHandle, N, d_r, 1, d_z, 1, &rz_new));

    // 计算残差范数 r·r
    checkCudaErrors(cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));

    beta = rz_new / rz_old;

    // 更新搜索方向: p = z + beta * p
    checkCudaErrors(cublasDscal(cublasHandle, N, &beta, d_p, 1));
    checkCudaErrors(cublasDaxpy(cublasHandle, N, &one, d_z, 1, d_p, 1));

    rz_old = rz_new;
    k++;

    RESI = sqrt(r1) / sqrt(rr);
    if (k % 1 == 0) { // 每次迭代都输出
      printf("PCG (ILU) iteration:%3d\tresidual:%e\n", k, RESI);
    }
  }
  printf("Final PCG (ILU) iteration:%3d\tresidual:%e\n", k, RESI);

  endTime = clock();
  cout << "PCG SOLVE COSTS:"
       << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s"
       << endl;

  // 将结果复制回主机
  checkCudaErrors(
      cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost));

  // 输出结果
  cout << "Solution vector:" << endl;
  for (int i = 0; i < N; i++) {
    cout << "x[" << i << "] = " << x[i] << endl;
  }

  // 释放内存
  cudaFree(d_z);
  cudaFree(d_y);

  // 释放ILU相关资源
  if (buffer_ilu)
    cudaFree(buffer_ilu);
  if (bufferL)
    cudaFree(bufferL);
  if (bufferU)
    cudaFree(bufferU);

  if (descrA)
    cusparseDestroyMatDescr(descrA);
  if (descrL)
    cusparseDestroyMatDescr(descrL);
  if (descrU)
    cusparseDestroyMatDescr(descrU);
  if (info_ilu)
    cusparseDestroyCsrilu02Info(info_ilu);
  if (infoL)
    cusparseDestroyCsrsv2Info(infoL);
  if (infoU)
    cusparseDestroyCsrsv2Info(infoU);
  cudaFree(d_csr_values_ILU);

  cusparseDestroy(cusparseHandle);
  cublasDestroy(cublasHandle);
  if (matA) {
    checkCudaErrors(cusparseDestroySpMat(matA));
  }
  if (vecx) {
    checkCudaErrors(cusparseDestroyDnVec(vecx));
  }
  if (vecAx) {
    checkCudaErrors(cusparseDestroyDnVec(vecAx));
  }
  if (vecp) {
    checkCudaErrors(cusparseDestroyDnVec(vecp));
  }

  free(x);
  free(rhs);
  cudaFree(d_x);
  cudaFree(d_r);
  cudaFree(d_p);
  cudaFree(d_Ax);
  cudaFree(buffer);
  cudaFree(d_csr_offsets);
  cudaFree(d_csr_columns);
  cudaFree(d_csr_values);

  return 0;
}

int main() {
  int iters = 1000;
  double tol = 1e-10;

  cout << "iters:" << iters << endl;
  cout << "tol:" << tol << endl;

  conjugateGradient_solver(iters, tol);

  return 0;
}