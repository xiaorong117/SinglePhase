#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>  // fopen
#include <stdlib.h> // EXIT_FAILURE
#include <string.h> // strtok
#include <assert.h>
# pragma once
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("cuSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("CUBLAS API failed at line %d with error: %d\n",                \
               __LINE__, status);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#if defined(NDEBUG)
#   define PRINT_INFO(var)
#else
#   define PRINT_INFO(var) printf("  " #var ": %f\n", var);
#endif

typedef struct VecStruct {
    cusparseDnVecDescr_t vec;
    double*              ptr;
} Vec;
void make_test_matrix(int * n_out,
                      int **row_offsets_out, 
                      int **columns_out, 
                      double **values_out);
int API_BICGSTAB(int maxIter,double toler,int*    rows,int*    columns,double* values, int m,double* rhs);
int gpu_BiCGStab(cublasHandle_t       cublasHandle,
                 cusparseHandle_t     cusparseHandle,
                 int                  m,
                 cusparseSpMatDescr_t matA,
                 cusparseSpMatDescr_t matM_lower,
                 cusparseSpMatDescr_t matM_upper,
                 Vec                  d_B,
                 Vec                  d_X,
                 Vec                  d_R0,
                 Vec                  d_R,
                 Vec                  d_P,
                 Vec                  d_P_aux,
                 Vec                  d_S,
                 Vec                  d_S_aux,
                 Vec                  d_V,
                 Vec                  d_T,
                 Vec                  d_tmp,
                 void*                d_bufferMV,
                 int                  maxIterations,
                 double               tolerance,
                 double * solution
                 );


int API_UNPBICGSTAB(int maxIter, double toler, int *rows, int *columns, double *values, int n, double *rhs);
int gpu_UNPBiCGStab(cublasHandle_t       cublasHandle,
                 cusparseHandle_t     cusparseHandle,
                 int                  m,
                 cusparseSpMatDescr_t matA,
                 Vec                  d_B,
                 Vec                  d_X,
                 Vec                  d_R0,
                 Vec                  d_R,
                 Vec                  d_P,
                 Vec                  d_P_aux,
                 Vec                  d_S,
                 Vec                  d_S_aux,
                 Vec                  d_V,
                 Vec                  d_T,
                 Vec                  d_tmp,
                 void*                d_bufferMV,
                 int                  maxIterations,
                 double               tolerance,
                 double * solution
                 );