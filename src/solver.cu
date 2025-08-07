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
#include <helper_cuda.h>      // helper function CUDA error checking and initialization
#include <helper_functions.h> // helper for shared functions common to CUDA Samples
// extern "C" {
//   #include "mkl.h"
//   }
const char *sSDKname = "conjugateGradient";
using namespace std;
using namespace std::chrono;
const double CLOCKS_PER_SECOND = ((clock_t)1000);

int conjugateGradient_solver(int iters_, double tol_)
{
  /*debug_rong*/
  array<int, 4> row = {0, 2, 5, 7};
  array<int, 7> col = {0, 1, 0, 1, 2, 1, 2};
  array<double, 7> va = {4, 1, 1, 3, 1, 1, 2};
  array<double, 3> B = {1, 2, 3};

  int op = row.size() - 1;
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

  for (int i = 0; i < row.size(); i++)
  {
    ia[i] = row[i];
  }
  for (int i = 0; i < col.size(); i++)
  {
    ja[i] = col[i];
  }
  for (int i = 0; i < va.size(); i++)
  {
    A[i] = va[i];
  }

  // 矩阵的内存空间CSR
  rows_offsets = ia;
  columns = ja;
  values = A;
  /*debug_rong*/

  clock_t startTime, endTime;
  startTime = clock();

  int N = op + mp;
  int nnz = ia[op + mp];
  const double tol = tol_;
  const int max_iter = iters_;
  cout << "max_iter:" << max_iter << endl;
  double a, b, na, r0, r1, rr;

  double *x;
  double *rhs;
  double *d_x, dot;
  double *d_r, *d_p, *d_Ax;
  int k;
  double alpha, beta, alpham1;

  x = (double *)malloc(N * sizeof(double));
  rhs = (double *)malloc(N * sizeof(double));
  for (size_t i = 0; i < N; i++)
  {
    rhs[i] = B[i];
  }

  for (int i = 0; i < N; i++)
  {
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

  /* Initialize problem data */
  cudaMemcpy(d_csr_columns, columns, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csr_offsets, rows_offsets, (N + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_csr_values, values, nnz * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_r, rhs, N * sizeof(double), cudaMemcpyHostToDevice);

  alpha = 1.0;
  alpham1 = -1.0;
  beta = 0.0;
  r0 = 0.;

  /* Allocate workspace for cuSPARSE */
  size_t bufferSize = 0;
  checkCudaErrors(cusparseSpMV_bufferSize(
      cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
      &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  void *buffer = NULL;
  checkCudaErrors(cudaMalloc(&buffer, bufferSize));

  /* Begin CG */
  checkCudaErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecx, &beta, vecAx, CUDA_R_64F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));
  checkCudaErrors(cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &rr));
  checkCudaErrors(cublasDaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1));
  checkCudaErrors(cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));

  k = 1;

  int MYK = 0;
  double RESI = 0;
  while (r1 > tol * tol * rr && k <= max_iter)
  {
    if (k > 1)
    {
      b = r1 / r0;
      cublasStatus = cublasDscal(cublasHandle, N, &b, d_p, 1);
      cublasStatus = cublasDaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
    }
    else
    {
      cublasStatus = cublasDcopy(cublasHandle, N, d_r, 1, d_p, 1);
    }

    checkCudaErrors(cusparseSpMV(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp,
        &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    cublasStatus = cublasDdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);

    a = r1 / dot;

    cublasStatus = cublasDaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
    na = -a;
    cublasStatus = cublasDaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

    r0 = r1;
    cublasStatus = cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
    cudaDeviceSynchronize();
    MYK = k;
    RESI = sqrt(r1) / sqrt(rr);
    // 修改开始：每1000步输出一次，包括第一步
    if (k == 1 || k % 10000 == 0)
    {
      printf("iteration:%3d\nresidual:%e\n", MYK, RESI);
    }
    // 修改结束
    k++;
  }
  printf("iteration:%3d\nresidual:%e\n", MYK, RESI);
  endTime = clock();
  cout << "CUDA SOLVE COSTS:"
       << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s"
       << endl;

  fstream Cuda_data, Cuda_data_execel;
  // Cuda_data.open("Cuda_data.txt", ios::out | ios::app | ios::ate);
  // Cuda_data << "iteration:" << MYK << "\t"
  // 		  << "residual:" << RESI << "\t"
  // 		  << "Cuda_time:" << (double)(endTime - startTime) /
  // CLOCKS_PER_SECOND / 1000 << "s"
  // 		  << "\t";
  // Cuda_data.close();

  // Cuda_data_execel.open("Cuda_data_execel.txt", ios::out | ios::app |
  // ios::ate); Cuda_data_execel << "iteration:" << MYK << "\n"
  // 				 << "residual:" << RESI << "\n"
  // 				 << "Cuda_time:" << (double)(endTime -
  // startTime) / CLOCKS_PER_SECOND / 1000 << "s"
  // 				 << "\n";
  // Cuda_data_execel.close();

  cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);

  ofstream out("answer2.txt");
  for (int i = 0; i < op + mp; i++)
  {
    out << "x[" << i << "] = " << x[i] << endl;
  }
  out.close();

  cusparseDestroy(cusparseHandle);
  cublasDestroy(cublasHandle);
  if (matA)
  {
    checkCudaErrors(cusparseDestroySpMat(matA));
  }
  if (vecx)
  {
    checkCudaErrors(cusparseDestroyDnVec(vecx));
  }
  if (vecAx)
  {
    checkCudaErrors(cusparseDestroyDnVec(vecAx));
  }
  if (vecp)
  {
    checkCudaErrors(cusparseDestroyDnVec(vecp));
  }

  free(x);
  free(rhs);
  cudaFree(d_x);
  cudaFree(d_r);
  cudaFree(d_p);
  cudaFree(d_Ax);

  return 0;
}

int main()
{
  int iters = 1000000;
  double tol = 1e-12;

  cout << "iters:" << iters << endl;
  cout << "tol:" << tol << endl;

  conjugateGradient_solver(iters, tol);

  return 0;
}