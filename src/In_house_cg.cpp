#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSparseToDense
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include "cusolverSp.h"

// Utilities and system includes
#include <helper_cuda.h>      // helper function CUDA error checking and initialization
#include <helper_functions.h> // helper for shared functions common to CUDA Samples
#include "helper_cusolver.h"
#include "iostream"

using namespace std;
#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    }

#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("cuSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            return EXIT_FAILURE;                                           \
        }                                                                  \
    }

#define CHECK_CUBLAS(func)                                          \
    {                                                               \
        cublasStatus_t status = (func);                             \
        if (status != CUBLAS_STATUS_SUCCESS)                        \
        {                                                           \
            printf("CUBLAS API failed at line %d with error: %d\n", \
                   __LINE__, status);                               \
            return EXIT_FAILURE;                                    \
        }                                                           \
    }

#if defined(NDEBUG)
#define PRINT_INFO(var)
#else
#define PRINT_INFO(var) printf("  " #var ": %f\n", var);
#endif

void conjugateGradient_solver(int iters_, double tol_, int *rows, int *columns_ptr, double *avalues, int n, double *rhsb)
{
    // 矩阵的内存空间CSR
    int *rows_offsets = rows;
    int *columns = columns_ptr;
    double *values = avalues;
    int *d_csr_offsets, *d_csr_columns;
    double *d_csr_values;
    clock_t startTime, endTime;
    startTime = clock();

    int N = n;
    int nnz = rows[n];
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
    rhs = rhsb;

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
    checkCudaErrors(cusparseCreateCsr(&matA, N, N, nnz, d_csr_offsets, d_csr_columns, d_csr_values,
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
    cudaMemcpy(d_csr_offsets, rows_offsets, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_values, values, nnz * sizeof(double), cudaMemcpyHostToDevice);
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
        printf("iteration:%3d\nresidual:%e\n", MYK, RESI);
        k++;
    }
    printf("iteration:%3d\nresidual:%e\n", MYK, RESI);
    endTime = clock();

    cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(rhsb, x, N * sizeof(double), cudaMemcpyHostToHost);
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
    // free(rhs);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ax);
}