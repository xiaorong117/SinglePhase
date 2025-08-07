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

// 预条件子内核函数
__global__ void apply_jacobi_preconditioner(int n, const double *M_inv,
                                            const double *r, double *z) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    z[idx] = r[idx] * M_inv[idx]; // 正确计算: z = r * M^{-1}
  }
}

const char *sSDKname = "conjugateGradient";
using namespace std;
using namespace std::chrono;
const double CLOCKS_PER_SECOND = ((clock_t)1000);

int conjugateGradient_solver(int iters_, double tol_) {
  /* 5x5 对称正定矩阵示例 */
  // 矩阵结构：5点离散拉普拉斯算子
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
  double *d_csr_values, *d_M_values;

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

  // 矩阵的内存空间CSR
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
  double *d_r, *d_p, *d_Ax, *d_z, *d_M_inv;
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
  checkCudaErrors(cudaMalloc((void **)&d_x, N * sizeof(double)));
  checkCudaErrors(cudaMalloc((void **)&d_r, N * sizeof(double)));
  checkCudaErrors(cudaMalloc((void **)&d_p, N * sizeof(double)));
  checkCudaErrors(cudaMalloc((void **)&d_Ax, N * sizeof(double)));

  // 添加预条件子内存分配
  checkCudaErrors(cudaMalloc((void **)&d_z, N * sizeof(double)));
  checkCudaErrors(cudaMalloc((void **)&d_M_inv, N * sizeof(double)));

  /* Initialize problem data */
  checkCudaErrors(cudaMemcpy(d_csr_columns, columns, nnz * sizeof(int),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csr_offsets, rows_offsets, (N + 1) * sizeof(int),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csr_values, values, nnz * sizeof(double),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(
      cudaMemcpy(d_r, rhs, N * sizeof(double), cudaMemcpyHostToDevice));

  // ================= 预条件子计算 =================
  cout << "Computing Jacobi preconditioner..." << endl;
  double *M_inv = new double[N];

  // 改进的对角线提取
  for (int i = 0; i < N; i++) {
    int start = ia[i];
    int end = ia[i + 1];
    bool diagonal_found = false;

    for (int j = start; j < end; j++) {
      if (ja[j] == i) {
        diagonal_found = true;
        if (values[j] == 0) {
          cout << "Warning: Zero diagonal element at row " << i << endl;
          M_inv[i] = 1.0;
        } else {
          M_inv[i] = 1.0 / values[j];
        }
        break;
      }
    }

    if (!diagonal_found) {
      cout << "Warning: No diagonal element found for row " << i << endl;
      M_inv[i] = 1.0;
    }
    cout << "M_inv[" << i << "] = " << M_inv[i] << endl;
  }

  checkCudaErrors(
      cudaMemcpy(d_M_inv, M_inv, N * sizeof(double), cudaMemcpyHostToDevice));
  // ===============================================

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

  // ======== 关键修复：预条件子应用（正确参数顺序） ========
  cout << "Applying Jacobi preconditioner..." << endl;
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  // 正确参数顺序：N, d_M_inv, d_r, d_z
  apply_jacobi_preconditioner<<<numBlocks, blockSize>>>(N, d_M_inv, d_r, d_z);

  // 添加CUDA内核错误检查
  cudaError_t kernelErr = cudaGetLastError();
  if (kernelErr != cudaSuccess) {
    cout << "CUDA kernel error: " << cudaGetErrorString(kernelErr) << endl;
  }
  cudaDeviceSynchronize();
  // ===============================================

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

    // 再次应用预条件子（正确参数顺序）
    apply_jacobi_preconditioner<<<numBlocks, blockSize>>>(N, d_M_inv, d_r, d_z);
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
      printf("PCG iteration:%3d\tresidual:%e\n", k, RESI);
    }
  }
  printf("Final PCG iteration:%3d\tresidual:%e\n", k, RESI);

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
  delete[] M_inv;
  cudaFree(d_z);
  cudaFree(d_M_inv);

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
  int iters = 10000;
  double tol = 1e-20;

  cout << "iters:" << iters << endl;
  cout << "tol:" << tol << endl;

  conjugateGradient_solver(iters, tol);

  return 0;
}