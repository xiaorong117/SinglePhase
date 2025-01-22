#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>  // fopen
#include <stdlib.h> // EXIT_FAILURE
#include <string.h> // strtok
#include <assert.h>
# pragma once
#include "bicgstab.hpp"


void make_laplace_matrix(int * n_out,
                         int **row_offsets_out, 
                         int **columns_out, 
                         double **values_out);

int gpu_CG(cublasHandle_t       cublasHandle,
           cusparseHandle_t     cusparseHandle,
           int                  m,
           cusparseSpMatDescr_t matA,
           cusparseSpMatDescr_t matL,
           Vec                  d_B,
           Vec                  d_X,
           Vec                  d_R,
           Vec                  d_R_aux,
           Vec                  d_P,
           Vec                  d_T,
           Vec                  d_tmp,
           void*                d_bufferMV,
           int                  maxIterations,
           double               tolerance,
           double *             solution
           );


int cg_API(int maxIter, double toler, int *rows, int *columns, double *values, int n, double *rhs);
