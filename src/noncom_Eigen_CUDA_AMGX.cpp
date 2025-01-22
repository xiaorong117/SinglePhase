#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <ctime>
#include <vector>
#include <cstring>
#include <cstdio>

// FOR Eigen
#include "Eigen/Eigen"
#include "Eigen/IterativeLinearSolvers"

// For AMGX
#include <amgx_c.h>
#include <amgx_config.h>

/* Using updated (v2) interfaces to cublas */
#include "device_launch_parameters.h"
#include "driver_types.h"
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>		  // cusparseSparseToDense
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include "cusolverSp.h"

// Utilities and system includes
#include <helper_cuda.h>	  // helper function CUDA error checking and initialization
#include <helper_functions.h> // helper for shared functions common to CUDA Samples
#include "helper_cusolver.h"
#include "bicgstab.hpp"
#include "cg.hpp"
const char *sSDKname = "conjugateGradient";
using namespace std;

#include "superlu_config.h"
///////相比1.0版本，边界接触面分为6种类型来进行调和平均
// CLOCKS_PER_SECOND这个常量表示每一秒（per second）有多少个时钟计时单元
const double CLOCKS_PER_SECOND = ((clock_t)1000);
int super_API(double *csra_ptr, int_t *csra_rowptr, int_t *csra_col_ptr, int number_rows, int number_nzeros, double *rhs_ptr, int number_rhs);

#ifdef __cplusplus
extern "C" {
#endif
int ditersol(double *csra_ptr, int_t *csra_rowptr, int_t *csra_col_ptr, int number_rows, int number_nzeros, double *rhs_ptr);
#ifdef __cplusplus
}
#endif


// 常量设置
double pi = 3.1415927;
double gas_vis = 1.4e-3;	   // 粘度
double porosity = 0.31;		   // 孔隙率
double ko = 12e-21;			   // 微孔达西渗透率  5e-15
double inlet_pre = 100;		   // 进口压力
double outlet_pre = 0;		   // 出口压力
double voxel_size = 8e-9;	   // 像素尺寸，单位m 5.345e-6
double outlet_element_n = 200; // 模型的在各个方向的像素数量，本模拟所用模型为正方形

// const int pn = 3467404;
// const int tn = 13945096;
// const int inlet = 189, outlet = 190, m_inlet = 15380, m_outlet = 13997, op = 7979, mp = 3429669;

// const int pn = 748092;
// const int tn = 2918583;
// const int inlet = 106, op = 3843, outlet = 112, m_inlet = 6644, mp = 730834, m_outlet = 6553;

const int pn = 575147;
const int tn = 2133069;
const int inlet=22,	outlet=54,	m_inlet=2919,	m_outlet=4754,	op=1826,	mp=565572;

// const int pn = 812675;
// const int tn = 3320281;
// const int inlet=144,	outlet=146,	m_inlet=6193,	m_outlet=5965,	op=3991,	mp=796236;

// const int pn = 1923201;
// const int tn = 5602823;
// const int inlet = 40, op = 468, outlet = 31, m_inlet = 12731, mp = 1901511, m_outlet = 8420;

const int macro_n = inlet + op + outlet;
const int micro_n = m_inlet + mp + m_outlet;
const int para_macro = inlet + outlet + m_inlet;
const int NA = (tn - inlet - outlet - m_inlet - m_outlet) * 2 + (op + mp);

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
			printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
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

template <typename T_ELEM>
int loadMMSparseMatrix(char *filename, char elem_type, bool csrFormat, int *m,
					   int *n, int *nnz, T_ELEM **aVal, int **aRowInd,
					   int **aColInd, int extendSymMatrix);

void UsageSP(void)
{
	printf("<options>\n");
	printf("-h          : display this help\n");
	printf("-R=<name>   : choose a linear solver\n");
	printf("              chol (cholesky factorization), this is default\n");
	printf("              qr   (QR factorization)\n");
	printf("              lu   (LU factorization)\n");
	printf("-P=<name>    : choose a reordering\n");
	printf("              symrcm (Reverse Cuthill-McKee)\n");
	printf("              symamd (Approximate Minimum Degree)\n");
	printf("              metis  (nested dissection)\n");
	printf("-file=<filename> : filename containing a matrix in MM format\n");
	printf("-device=<device_id> : <device_id> if want to run on specific GPU\n");

	exit(0);
}

void parseCommandLineArguments(int argc, char *argv[], struct testOpts &opts)
{
	memset(&opts, 0, sizeof(opts));

	if (checkCmdLineFlag(argc, (const char **)argv, "-h"))
	{
		UsageSP();
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "R"))
	{
		char *solverType = NULL;
		getCmdLineArgumentString(argc, (const char **)argv, "R", &solverType);

		if (solverType)
		{
			if ((STRCASECMP(solverType, "chol") != 0) &&
				(STRCASECMP(solverType, "lu") != 0) &&
				(STRCASECMP(solverType, "qr") != 0))
			{
				printf("\nIncorrect argument passed to -R option\n");
				UsageSP();
			}
			else
			{
				opts.testFunc = solverType;
			}
		}
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "P"))
	{
		char *reorderType = NULL;
		getCmdLineArgumentString(argc, (const char **)argv, "P", &reorderType);

		if (reorderType)
		{
			if ((STRCASECMP(reorderType, "symrcm") != 0) &&
				(STRCASECMP(reorderType, "symamd") != 0) &&
				(STRCASECMP(reorderType, "metis") != 0))
			{
				printf("\nIncorrect argument passed to -P option\n");
				UsageSP();
			}
			else
			{
				opts.reorder = reorderType;
			}
		}
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "file"))
	{
		char *fileName = 0;
		getCmdLineArgumentString(argc, (const char **)argv, "file", &fileName);

		if (fileName)
		{
			opts.sparse_mat_filename = fileName;
		}
		else
		{
			printf("\nIncorrect filename passed to -file \n ");
			UsageSP();
		}
	}
}

int fun(int argc, char *argv[], int *csrRowPtrA, int *csrColIndA, double *csrValA, int m, int nnz, struct testOpts &opt, int base, double *B, double *X)
{
	struct testOpts opts = opt;
	cusolverSpHandle_t handle = NULL;
	cusparseHandle_t cusparseHandle = NULL; /* used in residual evaluation */
	cudaStream_t stream = NULL;
	cusparseMatDescr_t descrA = NULL;

	int rowsA = m;	  /* number of rows of A */
	int colsA = m;	  /* number of columns of A */
	int nnzA = nnz;	  /* number of nonzeros of A */
	int baseA = base; /* base index in CSR format */

	/* CSR(A) from I/O */
	int *h_csrRowPtrA = csrRowPtrA;
	int *h_csrColIndA = csrColIndA;
	double *h_csrValA = csrValA;

	double *h_z = NULL;	 /* z = B \ (Q*b) */
	double *h_x = NULL;	 /* x = A \ b */
	double *h_b = NULL;	 /* b = ones(n,1) */
	double *h_Qb = NULL; /* Q*b */
	double *h_r = NULL;	 /* r = b - A*x */

	int *h_Q = NULL; /* <int> n */
					 /* reorder to reduce zero fill-in */
					 /* Q = symrcm(A) or Q = symamd(A) */
	/* B = Q*A*Q' or B = A(Q,Q) by MATLAB notation */
	int *h_csrRowPtrB = NULL; /* <int> n+1 */
	int *h_csrColIndB = NULL; /* <int> nnzA */
	double *h_csrValB = NULL; /* <double> nnzA */
	int *h_mapBfromA = NULL;  /* <int> nnzA */

	size_t size_perm = 0;
	void *buffer_cpu = NULL; /* working space for permutation: B = Q*A*Q^T */

	/* device copy of A: used in residual evaluation */
	int *d_csrRowPtrA = NULL;
	int *d_csrColIndA = NULL;
	double *d_csrValA = NULL;

	/* device copy of B: used in B*z = Q*b */
	int *d_csrRowPtrB = NULL;
	int *d_csrColIndB = NULL;
	double *d_csrValB = NULL;

	int *d_Q = NULL;	 /* device copy of h_Q */
	double *d_z = NULL;	 /* z = B \ Q*b */
	double *d_x = NULL;	 /* x = A \ b */
	double *d_b = NULL;	 /* a copy of h_b */
	double *d_Qb = NULL; /* a copy of h_Qb */
	double *d_r = NULL;	 /* r = b - A*x */

	double tol = 1.e-12;
	const int reorder = 0; /* no reordering */
	int singularity = 0;   /* -1 if A is invertible under tol. */

	/* the constants are used in residual evaluation, r = b - A*x */
	const double minus_one = -1.0;
	const double one = 1.0;

	double b_inf = 0.0;
	double x_inf = 0.0;
	double r_inf = 0.0;
	double A_inf = 0.0;
	int errors = 0;
	int issym = 0;

	double start, stop;
	double time_solve_cpu;
	double time_solve_gpu;

	// parseCommandLineArguments(argc, argv, opts);

	// if (NULL == opts.testFunc) {
	//   opts.testFunc =
	//       "chol"; /* By default running Cholesky as NO solver selected with -R
	//                  option. */
	// }

	// findCudaDevice(argc, (const char **)argv);

	// if (opts.sparse_mat_filename == NULL) {
	//   opts.sparse_mat_filename = sdkFindFilePath("lap2D_5pt_n100.mtx", argv[0]);
	//   if (opts.sparse_mat_filename != NULL)
	//     printf("Using default input file [%s]\n", opts.sparse_mat_filename);
	//   else
	//     printf("Could not find lap2D_5pt_n100.mtx\n");
	// } else {
	//   printf("Using input file [%s]\n", opts.sparse_mat_filename);
	// }

	// printf("step 1: read matrix market format\n");

	// if (opts.sparse_mat_filename == NULL) {
	//   fprintf(stderr, "Error: input matrix is not provided\n");
	//   return EXIT_FAILURE;
	// }

	// if (loadMMSparseMatrix<double>(opts.sparse_mat_filename, 'd', true, &rowsA,
	//                                &colsA, &nnzA, &h_csrValA, &h_csrRowPtrA,
	//                                &h_csrColIndA, true)) {
	//   exit(EXIT_FAILURE);
	// }
	// baseA = h_csrRowPtrA[0]; // baseA = {0,1}
	// printf("sparse matrix A is %d x %d with %d nonzeros, base=%d\n", rowsA, colsA,
	//        nnzA, baseA);

	// if (rowsA != colsA)
	// {
	//   fprintf(stderr, "Error: only support square matrix\n");
	//   return 1;
	// }

	checkCudaErrors(cusolverSpCreate(&handle));
	checkCudaErrors(cusparseCreate(&cusparseHandle));

	checkCudaErrors(cudaStreamCreate(&stream));
	/* bind stream to cusparse and cusolver*/
	checkCudaErrors(cusolverSpSetStream(handle, stream));
	checkCudaErrors(cusparseSetStream(cusparseHandle, stream));

	/* configure matrix descriptor*/
	checkCudaErrors(cusparseCreateMatDescr(&descrA));
	checkCudaErrors(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	if (baseA)
	{
		checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));
	}
	else
	{
		checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
	}

	h_z = (double *)malloc(sizeof(double) * colsA);
	h_x = X;
	h_b = B;
	h_Qb = (double *)malloc(sizeof(double) * rowsA);
	h_r = (double *)malloc(sizeof(double) * rowsA);

	h_Q = (int *)malloc(sizeof(int) * colsA);
	h_csrRowPtrB = (int *)malloc(sizeof(int) * (rowsA + 1));
	h_csrColIndB = (int *)malloc(sizeof(int) * nnzA);
	h_csrValB = (double *)malloc(sizeof(double) * nnzA);
	h_mapBfromA = (int *)malloc(sizeof(int) * nnzA);

	assert(NULL != h_z);
	assert(NULL != h_x);
	assert(NULL != h_b);
	assert(NULL != h_Qb);
	assert(NULL != h_r);
	assert(NULL != h_Q);
	assert(NULL != h_csrRowPtrB);
	assert(NULL != h_csrColIndB);
	assert(NULL != h_csrValB);
	assert(NULL != h_mapBfromA);

	checkCudaErrors(
		cudaMalloc((void **)&d_csrRowPtrA, sizeof(int) * (rowsA + 1)));
	checkCudaErrors(cudaMalloc((void **)&d_csrColIndA, sizeof(int) * nnzA));
	checkCudaErrors(cudaMalloc((void **)&d_csrValA, sizeof(double) * nnzA));
	checkCudaErrors(
		cudaMalloc((void **)&d_csrRowPtrB, sizeof(int) * (rowsA + 1)));
	checkCudaErrors(cudaMalloc((void **)&d_csrColIndB, sizeof(int) * nnzA));
	checkCudaErrors(cudaMalloc((void **)&d_csrValB, sizeof(double) * nnzA));
	checkCudaErrors(cudaMalloc((void **)&d_Q, sizeof(int) * colsA));
	checkCudaErrors(cudaMalloc((void **)&d_z, sizeof(double) * colsA));
	checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(double) * colsA));
	checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(double) * rowsA));
	checkCudaErrors(cudaMalloc((void **)&d_Qb, sizeof(double) * rowsA));
	checkCudaErrors(cudaMalloc((void **)&d_r, sizeof(double) * rowsA));

	/* verify if A has symmetric pattern or not */
	checkCudaErrors(cusolverSpXcsrissymHost(handle, rowsA, nnzA, descrA,
											h_csrRowPtrA, h_csrRowPtrA + 1,
											h_csrColIndA, &issym));

	if (0 == strcmp(opts.testFunc, "chol"))
	{
		if (!issym)
		{
			printf("Error: A has no symmetric pattern, please use LU or QR \n");
			exit(EXIT_FAILURE);
		}
	}

	printf("step 2: reorder the matrix A to minimize zero fill-in\n");
	printf(
		"        if the user choose a reordering by -P=symrcm, -P=symamd or "
		"-P=metis\n");

	if (NULL != opts.reorder)
	{
		if (0 == strcmp(opts.reorder, "symrcm"))
		{
			printf("step 2.1: Q = symrcm(A) \n");
			checkCudaErrors(cusolverSpXcsrsymrcmHost(
				handle, rowsA, nnzA, descrA, h_csrRowPtrA, h_csrColIndA, h_Q));
		}
		else if (0 == strcmp(opts.reorder, "symamd"))
		{
			printf("step 2.1: Q = symamd(A) \n");
			checkCudaErrors(cusolverSpXcsrsymamdHost(
				handle, rowsA, nnzA, descrA, h_csrRowPtrA, h_csrColIndA, h_Q));
		}
		else if (0 == strcmp(opts.reorder, "metis"))
		{
			printf("step 2.1: Q = metis(A) \n");
			checkCudaErrors(cusolverSpXcsrmetisndHost(handle, rowsA, nnzA, descrA,
													  h_csrRowPtrA, h_csrColIndA,
													  NULL, /* default setting. */
													  h_Q));
		}
		else
		{
			fprintf(stderr, "Error: %s is unknown reordering\n", opts.reorder);
			return 1;
		}
	}
	else
	{
		printf("step 2.1: no reordering is chosen, Q = 0:n-1 \n");
		for (int j = 0; j < rowsA; j++)
		{
			h_Q[j] = j;
		}
	}

	printf("step 2.2: B = A(Q,Q) \n");

	memcpy(h_csrRowPtrB, h_csrRowPtrA, sizeof(int) * (rowsA + 1));
	memcpy(h_csrColIndB, h_csrColIndA, sizeof(int) * nnzA);

	checkCudaErrors(cusolverSpXcsrperm_bufferSizeHost(
		handle, rowsA, colsA, nnzA, descrA, h_csrRowPtrB, h_csrColIndB, h_Q, h_Q,
		&size_perm));

	if (buffer_cpu)
	{
		free(buffer_cpu);
	}
	buffer_cpu = (void *)malloc(sizeof(char) * size_perm);
	assert(NULL != buffer_cpu);

	/* h_mapBfromA = Identity */
	for (int j = 0; j < nnzA; j++)
	{
		h_mapBfromA[j] = j;
	}
	checkCudaErrors(cusolverSpXcsrpermHost(handle, rowsA, colsA, nnzA, descrA,
										   h_csrRowPtrB, h_csrColIndB, h_Q, h_Q,
										   h_mapBfromA, buffer_cpu));

	/* B = A( mapBfromA ) */
	for (int j = 0; j < nnzA; j++)
	{
		h_csrValB[j] = h_csrValA[h_mapBfromA[j]];
	}

	// printf("step 3: b(j) = 1 + j/n \n");
	// for (int row = 0; row < rowsA; row++)
	// {
	// 	h_b[row] = 1.0 + ((double)row) / ((double)rowsA);
	// }

	/* h_Qb = b(Q) */
	for (int row = 0; row < rowsA; row++)
	{
		h_Qb[row] = h_b[h_Q[row]];
	}

	printf("step 4: prepare data on device\n");
	checkCudaErrors(cudaMemcpyAsync(d_csrRowPtrA, h_csrRowPtrA,
									sizeof(int) * (rowsA + 1),
									cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_csrColIndA, h_csrColIndA,
									sizeof(int) * nnzA, cudaMemcpyHostToDevice,
									stream));
	checkCudaErrors(cudaMemcpyAsync(d_csrValA, h_csrValA, sizeof(double) * nnzA,
									cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_csrRowPtrB, h_csrRowPtrB,
									sizeof(int) * (rowsA + 1),
									cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_csrColIndB, h_csrColIndB,
									sizeof(int) * nnzA, cudaMemcpyHostToDevice,
									stream));
	checkCudaErrors(cudaMemcpyAsync(d_csrValB, h_csrValB, sizeof(double) * nnzA,
									cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_b, h_b, sizeof(double) * rowsA,
									cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_Qb, h_Qb, sizeof(double) * rowsA,
									cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_Q, h_Q, sizeof(int) * rowsA,
									cudaMemcpyHostToDevice, stream));

	printf("step 5: solve A*x = b on CPU \n");
	start = second();

	/* solve B*z = Q*b */
	if (0 == strcmp(opts.testFunc, "chol"))
	{
		checkCudaErrors(cusolverSpDcsrlsvcholHost(
			handle, rowsA, nnzA, descrA, h_csrValB, h_csrRowPtrB, h_csrColIndB,
			h_Qb, tol, reorder, h_z, &singularity));
	}
	else if (0 == strcmp(opts.testFunc, "lu"))
	{
		checkCudaErrors(cusolverSpDcsrlsvluHost(
			handle, rowsA, nnzA, descrA, h_csrValB, h_csrRowPtrB, h_csrColIndB,
			h_Qb, tol, reorder, h_z, &singularity));
	}
	else if (0 == strcmp(opts.testFunc, "qr"))
	{
		checkCudaErrors(cusolverSpDcsrlsvqrHost(
			handle, rowsA, nnzA, descrA, h_csrValB, h_csrRowPtrB, h_csrColIndB,
			h_Qb, tol, reorder, h_z, &singularity));
	}
	else
	{
		fprintf(stderr, "Error: %s is unknown function\n", opts.testFunc);
		return 1;
	}

	/* Q*x = z */
	for (int row = 0; row < rowsA; row++)
	{
		h_x[h_Q[row]] = h_z[row];
	}

	if (0 <= singularity)
	{
		printf("WARNING: the matrix is singular at row %d under tol (%E)\n",
			   singularity, tol);
	}

	stop = second();
	time_solve_cpu = stop - start;

	printf("step 6: evaluate residual r = b - A*x (result on CPU)\n");
	checkCudaErrors(cudaMemcpyAsync(d_r, d_b, sizeof(double) * rowsA,
									cudaMemcpyDeviceToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_x, h_x, sizeof(double) * colsA,
									cudaMemcpyHostToDevice, stream));

	/* Wrap raw data into cuSPARSE generic API objects */
	cusparseSpMatDescr_t matA = NULL;
	if (baseA)
	{
		checkCudaErrors(cusparseCreateCsr(&matA, rowsA, colsA, nnzA, d_csrRowPtrA,
										  d_csrColIndA, d_csrValA,
										  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
										  CUSPARSE_INDEX_BASE_ONE, CUDA_R_64F));
	}
	else
	{
		checkCudaErrors(cusparseCreateCsr(&matA, rowsA, colsA, nnzA, d_csrRowPtrA,
										  d_csrColIndA, d_csrValA,
										  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
										  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
	}

	cusparseDnVecDescr_t vecx = NULL;
	checkCudaErrors(cusparseCreateDnVec(&vecx, colsA, d_x, CUDA_R_64F));
	cusparseDnVecDescr_t vecAx = NULL;
	checkCudaErrors(cusparseCreateDnVec(&vecAx, rowsA, d_r, CUDA_R_64F));

	/* Allocate workspace for cuSPARSE */
	size_t bufferSize = 0;
	checkCudaErrors(cusparseSpMV_bufferSize(
		cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one, matA, vecx,
		&one, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
	void *buffer = NULL;
	checkCudaErrors(cudaMalloc(&buffer, bufferSize));

	checkCudaErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
								 &minus_one, matA, vecx, &one, vecAx, CUDA_R_64F,
								 CUSPARSE_SPMV_ALG_DEFAULT, buffer));

	checkCudaErrors(cudaMemcpyAsync(h_r, d_r, sizeof(double) * rowsA,
									cudaMemcpyDeviceToHost, stream));
	/* wait until h_r is ready */
	checkCudaErrors(cudaDeviceSynchronize());

	b_inf = vec_norminf(rowsA, h_b);
	x_inf = vec_norminf(colsA, h_x);
	r_inf = vec_norminf(rowsA, h_r);
	A_inf = csr_mat_norminf(rowsA, colsA, nnzA, descrA, h_csrValA, h_csrRowPtrA,
							h_csrColIndA);

	// printf("(CPU) |b - A*x| = %E \n", r_inf);
	// printf("(CPU) |A| = %E \n", A_inf);
	// printf("(CPU) |x| = %E \n", x_inf);
	// printf("(CPU) |b| = %E \n", b_inf);
	// printf("(CPU) |b - A*x|/(|A|*|x| + |b|) = %E \n",
	// 	   r_inf / (A_inf * x_inf + b_inf));

	printf("step 7: solve A*x = b on GPU\n");
	start = second();

	/* solve B*z = Q*b */
	if (0 == strcmp(opts.testFunc, "chol"))
	{
		checkCudaErrors(cusolverSpDcsrlsvchol(
			handle, rowsA, nnzA, descrA, d_csrValB, d_csrRowPtrB, d_csrColIndB,
			d_Qb, tol, reorder, d_z, &singularity));
	}
	else if (0 == strcmp(opts.testFunc, "lu"))
	{
		printf("WARNING: no LU available on GPU \n");
	}
	else if (0 == strcmp(opts.testFunc, "qr"))
	{
		checkCudaErrors(cusolverSpDcsrlsvqr(handle, rowsA, nnzA, descrA, d_csrValB,
											d_csrRowPtrB, d_csrColIndB, d_Qb, tol,
											reorder, d_z, &singularity));
	}
	else
	{
		fprintf(stderr, "Error: %s is unknow function\n", opts.testFunc);
		return 1;
	}
	checkCudaErrors(cudaDeviceSynchronize());
	if (0 <= singularity)
	{
		printf("WARNING: the matrix is singular at row %d under tol (%E)\n",
			   singularity, tol);
	}
	/* Q*x = z */
	cusparseSpVecDescr_t vecz = NULL;
	checkCudaErrors(cusparseCreateSpVec(&vecz, colsA, rowsA, d_Q, d_z, CUSPARSE_INDEX_32I,
										CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
	checkCudaErrors(cusparseScatter(cusparseHandle, vecz, vecx));
	checkCudaErrors(cusparseDestroySpVec(vecz));

	checkCudaErrors(cudaDeviceSynchronize());

	stop = second();
	time_solve_gpu = stop - start;

	printf("step 8: evaluate residual r = b - A*x (result on GPU)\n");
	checkCudaErrors(cudaMemcpyAsync(d_r, d_b, sizeof(double) * rowsA,
									cudaMemcpyDeviceToDevice, stream));

	checkCudaErrors(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
								 &minus_one, matA, vecx, &one, vecAx, CUDA_R_64F,
								 CUSPARSE_SPMV_ALG_DEFAULT, buffer));

	checkCudaErrors(cudaMemcpyAsync(h_x, d_x, sizeof(double) * colsA,
									cudaMemcpyDeviceToHost, stream));
	checkCudaErrors(cudaMemcpyAsync(h_r, d_r, sizeof(double) * rowsA,
									cudaMemcpyDeviceToHost, stream));
	/* wait until h_x and h_r are ready */
	checkCudaErrors(cudaDeviceSynchronize());

	b_inf = vec_norminf(rowsA, h_b);
	x_inf = vec_norminf(colsA, h_x);
	r_inf = vec_norminf(rowsA, h_r);

	if (0 != strcmp(opts.testFunc, "lu"))
	{
		// only cholesky and qr have GPU version
		printf("(GPU) |b - A*x| = %E \n", r_inf);
		printf("(GPU) |A| = %E \n", A_inf);
		printf("(GPU) |x| = %E \n", x_inf);
		printf("(GPU) |b| = %E \n", b_inf);
		printf("(GPU) |b - A*x|/(|A|*|x| + |b|) = %E \n",
			   r_inf / (A_inf * x_inf + b_inf));
	}

	fprintf(stdout, "timing %s: CPU = %10.6f sec , GPU = %10.6f sec\n",
			opts.testFunc, time_solve_cpu, time_solve_gpu);

	if (0 != strcmp(opts.testFunc, "lu"))
	{
		printf("show last 10 elements of solution vector (GPU) \n");
		printf("consistent result for different reordering and solver \n");
		for (int j = rowsA - 10; j < rowsA; j++)
		{
			printf("x[%d] = %E\n", j, h_x[j]);
		}
	}

	if (handle)
	{
		checkCudaErrors(cusolverSpDestroy(handle));
	}
	if (cusparseHandle)
	{
		checkCudaErrors(cusparseDestroy(cusparseHandle));
	}
	if (stream)
	{
		checkCudaErrors(cudaStreamDestroy(stream));
	}
	if (descrA)
	{
		checkCudaErrors(cusparseDestroyMatDescr(descrA));
	}
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

	// if (h_csrValA)
	// {
	// 	free(h_csrValA);
	// }
	// if (h_csrRowPtrA)
	// {
	// 	free(h_csrRowPtrA);
	// }
	// if (h_csrColIndA)
	// {
	// 	free(h_csrColIndA);
	// }
	if (h_z)
	{
		free(h_z);
	}
	// if (h_x)
	// {
	// 	free(h_x);
	// }
	// if (h_b)
	// {
	// 	free(h_b);
	// }
	if (h_Qb)
	{
		free(h_Qb);
	}
	if (h_r)
	{
		free(h_r);
	}

	if (h_Q)
	{
		free(h_Q);
	}

	if (h_csrRowPtrB)
	{
		free(h_csrRowPtrB);
	}
	if (h_csrColIndB)
	{
		free(h_csrColIndB);
	}
	if (h_csrValB)
	{
		free(h_csrValB);
	}
	if (h_mapBfromA)
	{
		free(h_mapBfromA);
	}

	if (buffer_cpu)
	{
		free(buffer_cpu);
	}

	if (buffer)
	{
		checkCudaErrors(cudaFree(buffer));
	}
	if (d_csrValA)
	{
		checkCudaErrors(cudaFree(d_csrValA));
	}
	if (d_csrRowPtrA)
	{
		checkCudaErrors(cudaFree(d_csrRowPtrA));
	}
	if (d_csrColIndA)
	{
		checkCudaErrors(cudaFree(d_csrColIndA));
	}
	if (d_csrValB)
	{
		checkCudaErrors(cudaFree(d_csrValB));
	}
	if (d_csrRowPtrB)
	{
		checkCudaErrors(cudaFree(d_csrRowPtrB));
	}
	if (d_csrColIndB)
	{
		checkCudaErrors(cudaFree(d_csrColIndB));
	}
	if (d_Q)
	{
		checkCudaErrors(cudaFree(d_Q));
	}
	if (d_z)
	{
		checkCudaErrors(cudaFree(d_z));
	}
	if (d_x)
	{
		checkCudaErrors(cudaFree(d_x));
	}
	if (d_b)
	{
		checkCudaErrors(cudaFree(d_b));
	}
	if (d_Qb)
	{
		checkCudaErrors(cudaFree(d_Qb));
	}
	if (d_r)
	{
		checkCudaErrors(cudaFree(d_r));
	}

	return 0;
}

double getmax_2(double a, double b)
{
	return a > b ? a : b;
}

double getmax_3(double a, double b, double c)
{
	double temp = getmax_2(a, b);
	temp = getmax_2(temp, c);
	return temp;
}

struct pore
{
	double X;
	double Y;
	double Z;
	double Radiu;
	int Half_coord;
	int half_accum;
	int full_coord;
	int full_accum;
	int type;
	double pressure;
	double volume;
	double km;
};

struct throat
{
	int ID_1;
	int ID_2;
	int n_direction;
	double Radiu;
	double Length;
	double Conductivity;
	double center_x;
	double center_y;
	double center_z;
};

struct throatmerge
{
	int ID_1;
	int ID_2;
	double Radiu;
	double Conductivity;
};

class PNMsolver // ������
{
public:
	double *X, *B;
	// 求解的时间变量
	int *ia, *ja;
	double *a;
	// 申请孔喉的动态存储空间
	pore *Pb;
	throat *Tb_in;
	throatmerge *Tb;
	int *label_outlet;
	double error;
	// 矩阵的内存空间CSR
	int *rows_offsets, *columns;
	double *values;
	// 矩阵的显存空间CSR
	int *d_csr_offsets, *d_csr_columns;
	double *d_csr_values, *d_M_values;

	PNMsolver();							  // ��̬����洢��
	void Poredateinput();					  // ��϶���ݵ��뺯������
	void Throatdateinput();					  // �׺����ݵ��뺯������
	void para_cal();						  // �������ȵ���ز�������
	void PressureMatrix();					  // ѹ������
	void EigenSolve(int iters_, double tol_); // �����������
	int conjugateGradient_solver(int iters_, double tol_);
	void PCG(int iters_, double tol_);
	void AMGXsolver();
	void solver1(PNMsolver &Berea, int iters_, double tol_);
	void solver2(PNMsolver &Berea, int iters_, double tol_); // ˲̬��ɢ�����������
	void solver3();
	void solver4(PNMsolver &Berea, int iters_, double tol_);
	void solver5(PNMsolver &Berea, int iters_, double tol_);
	void solver6(PNMsolver &Berea, int iters_, double tol_);
	void solver7(PNMsolver &Berea, int iters_, double tol_);
	void solver8(PNMsolver &Berea);
	void solver9(PNMsolver &Berea);
	void output(); // ���VTK�ļ�
	void MMout();
	double permeability1();
	double permeability2();
	double permeability3();
	~PNMsolver() // �����������ͷŶ�̬�洢
	{
		delete[] X, B;
		// delete[] ia, ja, a;
		delete[] label_outlet, Pb, Tb_in, Tb;
	}
};

PNMsolver::PNMsolver()
{
	X = new double[op + mp];
	B = new double[op + mp];

	ia = new int[op + mp + 1];
	ja = new int[NA];
	a = new double[NA];

	Pb = new pore[pn];
	Tb_in = new throat[2 * tn];
	Tb = new throatmerge[2 * tn];
	label_outlet = new int[outlet + m_outlet];

	for (int i = 0; i < op + mp; i++)
	{
		B[i] = 0;
	}
	// ��ʼӦ������ֵ(�����������洢��������һ����Ӧ����)
	for (int i = 0; i < pn; i++)
	{
		Pb[i].pressure = inlet_pre;
	}
	for (int i = macro_n - outlet; i < macro_n; i++)
	{
		Pb[i].pressure = outlet_pre;
	}
	for (int i = pn - m_outlet; i < pn; i++)
	{
		Pb[i].pressure = outlet_pre;
	}
}

struct matrix
{
	int i;
	int j;
	double val;
};

bool my_compare(struct matrix a, struct matrix b)
{
	if (a.j != b.j)
		return a.j < b.j; // 第一级比较
	else
	{
		return a.i < b.i; // 如果第一级相同，比较第二级
	}
}

void PNMsolver::MMout()
{
	ofstream mmout("PNM.mtx");
	// 将CSR格式转换为COO格式
	for (size_t i = 0; i < op + mp + 1; i++)
	{
		ia[i] += 1;
	}

	for (size_t i = 0; i < ia[op + mp] - 1; i++)
	{
		ja[i] += 1;
	}

	// csr2coo row domain
	vector<matrix> mat(ia[op + mp] - 1);
	int count{0};
	for (int i = 0; i < op + mp; i++)
	{
		for (int j = ia[i] - 1; j < ia[i + 1] - 1; j++)
		{
			mat[count].i = i + 1;
			mat[count].j = ja[j];
			mat[count].val = a[i];
			count++;
		}
	}
	// column domain by muliti-level-sort
	sort(mat.begin(), mat.end(), my_compare);

	mmout << "\%\%MatrixMarket matrix coordinate real general" << endl;
	mmout << "\%test" << endl;
	mmout << op + mp << " " << op + mp << " " << ia[op + mp] - 1 << endl;
	for (size_t i = 0; i < mat.size(); i++)
	{
		mmout << mat[i].i << " " << mat[i].j << " " << mat[i].val << endl;
	}
	mmout.close();

	ofstream bout("b.txt");
	for (size_t i = 0; i < op + mp; i++)
	{
		bout << B[i] << endl;
	}
	bout.close();
}

void PNMsolver::Poredateinput()
{
	clock_t startTime, endTime;
	startTime = clock();

	ifstream porefile("full_pore_re3_z.txt", ios::in);
	if (!porefile.is_open())
	{
		cout << " can not open poredate" << endl;
	}
	for (int i = 0; i < pn; i++)
	{
		double waste{0};
		porefile >> Pb[i].X >> Pb[i].Y >> Pb[i].Z >> waste >> Pb[i].Radiu >> Pb[i].Half_coord >> Pb[i].half_accum >> Pb[i].type >> Pb[i].full_coord >> Pb[i].full_accum >> waste >> waste >> Pb[i].km;
	}
	porefile.close();
	for (int i = 0; i < pn; i++)
	{
		Pb[i].X = voxel_size * Pb[i].X;
		Pb[i].Y = voxel_size * Pb[i].Y;
		Pb[i].Z = voxel_size * Pb[i].Z;
		Pb[i].Radiu = voxel_size * Pb[i].Radiu;
	}
	endTime = clock();
	cout << "poredata time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << endl;
	// ��֤��������
	/*for (int i = 0;i < pn;i++)
	{
		cout << pore_EqRadius[i] << endl;
	}*/
}

void PNMsolver::Throatdateinput()
{
	clock_t startTime, endTime;
	startTime = clock();
	// date input
	ifstream throatfile("200_RS2Bfull_throat_z.txt", ios::in);
	if (!throatfile.is_open())
	{
		cout << " can not open throatfile" << endl;
	}

	for (int i = 0; i < 2 * tn; i++)
	{
		throatfile >> Tb_in[i].ID_1 >> Tb_in[i].ID_2 >> Tb_in[i].Radiu >> Tb_in[i].center_x >> Tb_in[i].center_y >> Tb_in[i].center_z >> Tb_in[i].n_direction >> Tb_in[i].Length;
	}
	throatfile.close();

	// ��ʼ���ṹ����Ԫ��
	for (int i = 0; i < 2 * tn; i++)
	{
		if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n)
		{
			Tb_in[i].Radiu = voxel_size * Tb_in[i].Radiu; // pnm部分为喉道的半径
		}
		else
		{
			Tb_in[i].Radiu = voxel_size * voxel_size * Tb_in[i].Radiu; // Darcy区的为接触面积
		}
		Tb_in[i].Length = voxel_size * Tb_in[i].Length;
		Tb_in[i].center_x = voxel_size * Tb_in[i].center_x;
		Tb_in[i].center_y = voxel_size * Tb_in[i].center_y;
		Tb_in[i].center_z = voxel_size * Tb_in[i].center_z;
	}
	endTime = clock();
	cout << "throat time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << endl;
	// for (int i = 0;i < tn;i++)
	// {
	// 	cout << Tb_in[i].Radiu <<endl;
	// }
}

void PNMsolver::para_cal()
{
	// 计算孔隙的体积
	for (int i = 0; i < pn; i++)
	{
		if (Pb[i].type == 0)
		{
			Pb[i].volume = 4 * pi * pow(Pb[i].Radiu, 3) / 3; // 孔隙网络单元
		}
		else if (Pb[i].type == 1)
		{
			Pb[i].volume = pow(Pb[i].Radiu, 3); // 正方形微孔单元
		}
		else
		{
			Pb[i].volume = pow(Pb[i].Radiu, 3) / 2; // 2×2×1、1×2×2和2×1×2的微孔网格
		}
	}

	// 水力传导系数计算
	double temp1 = 0, temp2 = 0, angle1 = 0, angle2 = 0, length1 = 0, length2 = 0; // 两点流量计算中的临时存储变量

	for (int i = 0; i < 2 * tn; i++)
	{
		if (Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 < macro_n)
		{
			if (Tb_in[i].Length <= 0) // 剔除可能存在的负喉道长度
			{
				Tb_in[i].Length = 0.5 * voxel_size;
			}
			Tb_in[i].Conductivity = pi * pow(Tb_in[i].Radiu, 4) / (8 * gas_vis * Tb_in[i].Length);
		}
		else if ((Tb_in[i].ID_1 < macro_n && Tb_in[i].ID_2 >= macro_n))
		{
			temp1 = pi * pow(Pb[Tb_in[i].ID_1].Radiu, 3) / (8 * gas_vis);
			length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
			if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
			{
				angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
			}
			else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
			{
				angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
			}
			else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
			{
				angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
			}
			temp2 = abs(Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle2 / (gas_vis * length2));
			/*if (Pb[Tb_in[i].ID_2].type == 1)
			{
				temp2 = ko * Tb_in[i].Radiu / (gas_vis * Pb[Tb_in[i].ID_2].Radiu * 0.5);
			}
			else
			{
				temp2 = ko * Tb_in[i].Radiu / (gas_vis * voxel_size * 0.5);
			}*/
			Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
		}
		else if ((Tb_in[i].ID_1 >= macro_n && Tb_in[i].ID_2 < macro_n))
		{
			temp1 = pi * pow(Pb[Tb_in[i].ID_2].Radiu, 3) / (8 * gas_vis);
			length2 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
			if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
			{
				angle2 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length2;
			}
			else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
			{
				angle2 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length2;
			}
			else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
			{
				angle2 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length2;
			}
			temp2 = abs(Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle2 / (gas_vis * length2));
			/*if (Pb[Tb_in[i].ID_2].type == 1)
			{
				temp2 = ko * Tb_in[i].Radiu / (gas_vis * Pb[Tb_in[i].ID_2].Radiu * 0.5);
			}
			else
			{
				temp2 = ko * Tb_in[i].Radiu / (gas_vis * voxel_size * 0.5);
			}*/
			Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
		}
		else if (Tb_in[i].ID_1 < macro_n + m_inlet || Tb_in[i].ID_1 >= pn - m_outlet || Tb_in[i].ID_2 < macro_n + m_inlet || Tb_in[i].ID_2 >= pn - m_outlet)
		{
			Tb_in[i].Conductivity = ko * Tb_in[i].Radiu / (gas_vis * Tb_in[i].Length);
		}
		else
		{
			length1 = sqrt(pow(Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z, 2));
			length2 = sqrt(pow(Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x, 2) + pow(Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y, 2) + pow(Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z, 2));
			if (Tb_in[i].n_direction == 0 || Tb_in[i].n_direction == 1)
			{
				angle1 = (Pb[Tb_in[i].ID_1].X - Tb_in[i].center_x) / length1;
				angle2 = (Pb[Tb_in[i].ID_2].X - Tb_in[i].center_x) / length2;
			}
			else if (Tb_in[i].n_direction == 2 || Tb_in[i].n_direction == 3)
			{
				angle1 = (Pb[Tb_in[i].ID_1].Y - Tb_in[i].center_y) / length1;
				angle2 = (Pb[Tb_in[i].ID_2].Y - Tb_in[i].center_y) / length2;
			}
			else if (Tb_in[i].n_direction == 4 || Tb_in[i].n_direction == 5)
			{
				angle1 = (Pb[Tb_in[i].ID_1].Z - Tb_in[i].center_z) / length1;
				angle2 = (Pb[Tb_in[i].ID_2].Z - Tb_in[i].center_z) / length2;
			}
			temp1 = abs(Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle1 / (gas_vis * length1));
			temp2 = abs(Pb[Tb_in[i].ID_1].km * Tb_in[i].Radiu * angle2 / (gas_vis * length2));
			Tb_in[i].Conductivity = temp1 * temp2 / (temp1 + temp2);
			// cout << temp1 << "\t" << temp2 <<"\t"<< Tb_in[i].Conductivity << endl;
		}
	}

	// merge throat
	int label = 0;
	Tb[0].ID_1 = Tb_in[0].ID_1;
	Tb[0].ID_2 = Tb_in[0].ID_2;
	Tb[0].Radiu = Tb_in[0].Radiu;
	Tb[0].Conductivity = Tb_in[0].Conductivity;
	for (int i = 1; i < 2 * tn; i++)
	{
		if (Tb[label].ID_1 == Tb_in[i].ID_1 && Tb[label].ID_2 == Tb_in[i].ID_2)
		{
			Tb[label].Conductivity += Tb_in[i].Conductivity;
		}
		else
		{
			label++;
			Tb[label].ID_1 = Tb_in[i].ID_1;
			Tb[label].ID_2 = Tb_in[i].ID_2;
			Tb[label].Radiu = Tb_in[i].Radiu;
			Tb[label].Conductivity = Tb_in[i].Conductivity;
		}
	}
	// full_coord
	for (int i = 0; i < pn; i++)
	{
		Pb[i].full_coord = 0;
		Pb[i].full_accum = 0;
	}

	for (int i = 0; i <= label; i++)
	{
		Pb[Tb[i].ID_1].full_coord += 1;
	}

	// full_accum
	Pb[0].full_accum = Pb[0].full_coord;
	for (int i = 1; i < pn; i++)
	{
		Pb[i].full_accum = Pb[i - 1].full_accum + Pb[i].full_coord;
	}
}

void PNMsolver::PressureMatrix()
{
	clock_t startTime, endTime;
	startTime = clock();
	for (int i = 0; i < mp + op; i++)
	{
		B[i] = 0;
	}
	for (int i = 0; i < NA; i++)
	{
		ja[i] = 0;
		a[i] = 0;
	}

	int num;	  // 每行第一个非0参数的累计编号
	int num1 = 0; // 矩阵中每行的非0数据数量
	int temp;	  // 确定对角线前面的数据数量
	int temp1;
	ia[0] = 1;
	/* -------------------------------------------------------------------------------------  */
	/* 大孔组装 */
	/* -------------------------------------------------------------------------------------  */
	for (int i = inlet; i < op + inlet; i++)
	{
		temp = 0, temp1 = 0;
		num = ia[i - inlet];

		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if (Tb[j].ID_2 < inlet) // 进口
			{
				B[Tb[j].ID_1 - inlet] = Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure;
			}
			else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n) // 出口
			{
				B[Tb[j].ID_1 - inlet] = Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure;
			}
			else
			{
				if (Tb[j].ID_1 > Tb[j].ID_2)
				{
					temp++; // 矩阵每行对角线值前面的非0值数量
				}
				num1++; // 除对角线值外矩阵每行非0值数量
			}
		}
		num1 += 1; // 加上对角线的非0值

		ia[i - inlet + 1] = num1 + 1;		// 前i行累计的非零值数量，其中1为ia[0]的值
		ja[num + temp - 1] = i - inlet + 1; // 第i行对角线的值的位置

		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if (Tb[j].ID_2 < inlet)
			{
				a[ia[i - inlet] + temp - 1] += Tb[j].Conductivity; // 对角线
			}
			else if (Tb[j].ID_2 >= op + inlet && Tb[j].ID_2 < macro_n)
			{
				a[ia[i - inlet] + temp - 1] += Tb[j].Conductivity; // 对角线
			}
			else
			{
				if (temp1 < temp)
				{
					a[ia[i - inlet] + temp - 1] += Tb[j].Conductivity; // 对角线
					a[num - 1] += -Tb[j].Conductivity;

					if (Tb[j].ID_2 < macro_n)
					{
						ja[num - 1] = Tb[j].ID_2 - inlet + 1; // 下三角值的列位置																					 //下三角值的列位置
					}
					else
					{
						ja[num - 1] = Tb[j].ID_2 - para_macro + 1; // 下三角值的列位置																				 //下三角值的列位置
					}
					num++;
					temp1++;
				}
				else
				{
					a[ia[i - inlet] + temp - 1] += Tb[j].Conductivity; // 对角线
					a[num] += -Tb[j].Conductivity;

					if (Tb[j].ID_2 < macro_n)
					{
						ja[num] = Tb[j].ID_2 - inlet + 1; // 下三角值的列位置																									 //下三角值的列位置
					}
					else
					{
						ja[num] = Tb[j].ID_2 - para_macro + 1; // 下三角值的列位置																									 //下三角值的列位置
					}
					num++;
				}
			}
		}
	}
	/* -------------------------------------------------------------------------------------  */
	/* 微孔组装 */
	/* -------------------------------------------------------------------------------------  */
	for (int i = macro_n + m_inlet; i < pn - m_outlet; i++)
	{
		temp = 0, temp1 = 0;
		num = ia[i - para_macro];

		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if (Tb[j].ID_2 >= macro_n && Tb[j].ID_2 < macro_n + m_inlet) // 微孔进口边界
			{
				B[Tb[j].ID_1 - para_macro] += Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure;
			}
			else if (Tb[j].ID_2 >= pn - m_outlet) // 微孔出口边界
			{
				B[Tb[j].ID_1 - para_macro] += Tb[j].Conductivity * Pb[Tb[j].ID_2].pressure;
			}
			else
			{
				if (Tb[j].ID_1 > Tb[j].ID_2)
				{
					temp++; // 矩阵每行对角线值前面的非0值数量
				}
				num1++; // 除对角线值外矩阵每行非0值数量
			}
		}
		num1 += 1; // 加上对角线的非0值
		/*cout << num1 << "\t" << full_coord[i] << endl;*/
		ia[i - para_macro + 1] = num1 + 1; // 前i行累计的非零值数量，其中1为ia[0]的值

		ja[num + temp - 1] = i - para_macro + 1; // 第i行对角线的值的位置

		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			if (Tb[j].ID_2 >= macro_n && Tb[j].ID_2 < macro_n + m_inlet)
			{
				a[ia[i - para_macro] + temp - 1] += Tb[j].Conductivity; // 对角线
			}
			else if (Tb[j].ID_2 >= pn - m_outlet)
			{
				a[ia[i - para_macro] + temp - 1] += Tb[j].Conductivity; // 对角线
			}
			else
			{
				if (temp1 < temp)
				{
					a[ia[i - para_macro] + temp - 1] += Tb[j].Conductivity;
					a[num - 1] += -Tb[j].Conductivity;

					if (Tb[j].ID_2 < macro_n)
					{
						ja[num - 1] = Tb[j].ID_2 - inlet + 1; // 下三角值的列位置
					}
					else
					{
						ja[num - 1] = Tb[j].ID_2 - para_macro + 1; // 下三角值的列位置
					}
					num++;
					temp1++;
				}
				else
				{
					a[ia[i - para_macro] + temp - 1] += Tb[j].Conductivity;
					a[num] += -Tb[j].Conductivity;
					if (Tb[j].ID_2 < macro_n)
					{
						ja[num] = Tb[j].ID_2 - inlet + 1; // 下三角值的列位置
					}
					else
					{
						ja[num] = Tb[j].ID_2 - para_macro + 1; // 下三角值的列位置
					}
					num++;
				}
			}
			// cout<<j<<"\t"<<Tb_in[j].ID_2 <<"conduct"<<Tb_in[j].Conductivity<<endl;
		}
	}

	/* -------------------------输出参数--------------------------- */
	// for (int i = 0;i < op+mp+1;i++)
	// {
	// 	cout << "ia(" << i << ") = " << ia[i] << endl;
	// }

	// for (int i = 0;i < 72;i++)
	// {

	// 	cout << "ja(" << i << ") = " << ja[i] <<"\t\t" <<"a("<<i<<") = "<<a[i] << endl;

	// }
	for (int i = 0; i < op + mp + 1; i++)
	{
		ia[i] = ia[i] - 1;
	}
	for (int i = 0; i < ia[op + mp]; i++)
	{
		ja[i] = ja[i] - 1;
	}
	endTime = clock();
	cout << "Martix end:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << endl;
}

void PNMsolver::EigenSolve(int iters_, double tol_)
{
	clock_t startTime, endTime;
	startTime = clock();
	// create the Eigen sparse matrix
	using namespace Eigen;
	SparseMatrix<double> A0(op + mp, op + mp);
	A0.reserve(ia[op + mp]);
	vector<Triplet<double>> triplets;
	triplets.reserve(ia[op + mp]);
	// fill the matrix
	for (int i = 0; i < op + mp; i++)
	{
		// cout << "i = " << i << endl;
		for (int j = ia[i]; j < ia[i + 1]; j++)
		{
			if (a[j] != 0)
			{
				triplets.push_back(Triplet<double>(i, ja[j], a[j]));
			}
		}
	}
	A0.setFromTriplets(triplets.begin(), triplets.end());

	BiCGSTAB<SparseMatrix<double>> solver;
	solver.setTolerance(tol_);
	solver.setMaxIterations(iters_);
	solver.compute(A0);
	if (solver.info() != Eigen::Success)
	{
		std::cout << "failed!" << std::endl;
	}
	else
	{
		std::cout << "solver.compute(A0) success." << std::endl;
	}
	VectorXd B0(op + mp, 1);
	for (int i = 0; i < op + mp; i++)
	{
		B0[i] = B[i];
	}
	VectorXd x = solver.solve(B0);

	ofstream out("answer1.txt");
	for (int i = 0; i < op + mp; i++)
	{
		out << "x[" << i << "] = " << x[i] << endl;
	}
	out.close();

	// 更新应力场
	for (int i = inlet; i < inlet + op; i++)
	{
		Pb[i].pressure = x[i - inlet];
	}
	for (int i = op; i < op + mp; i++)
	{
		Pb[i + para_macro].pressure = x[i];
	}

	if (solver.info() != Eigen::Success)
	{
		std::cout << "Solver failed to converge!" << std::endl;
	}
	double residual = (A0 * x - B0).norm() / B0.norm();
	std::cout << "Residual: " << residual << std::endl;
	std::cout << "#iterations:" << solver.iterations() << std::endl;
	endTime = clock();
	cout << "Eigen time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << endl;

	fstream Eigen_data, Eigen_data_execel;
	Eigen_data.open("Eigen_data.txt", ios::out | ios::app | ios::ate);
	Eigen_data << "iteration:" << solver.iterations() << "\t"
			   << "residual:" << residual << "\t"
			   << "Eigen_time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s"
			   << "\t";
	Eigen_data.close();

	Eigen_data_execel.open("Eigen_data_execel.txt", ios::out | ios::app | ios::ate);
	Eigen_data_execel << "iteration:" << solver.iterations() << "\n"
					  << "residual:" << residual << "\n"
					  << "Eigen_time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s"
					  << "\n";
	Eigen_data_execel.close();
}

int PNMsolver::conjugateGradient_solver(int iters_, double tol_)
{
	// 矩阵的内存空间CSR
	rows_offsets = ia;
	columns = ja;
	values = a;

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
	cout << "CUDA SOLVE COSTS:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << endl;

	fstream Cuda_data, Cuda_data_execel;
	Cuda_data.open("Cuda_data.txt", ios::out | ios::app | ios::ate);
	Cuda_data << "iteration:" << MYK << "\t"
			  << "residual:" << RESI << "\t"
			  << "Cuda_time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s"
			  << "\t";
	Cuda_data.close();

	Cuda_data_execel.open("Cuda_data_execel.txt", ios::out | ios::app | ios::ate);
	Cuda_data_execel << "iteration:" << MYK << "\n"
					 << "residual:" << RESI << "\n"
					 << "Cuda_time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s"
					 << "\n";
	Cuda_data_execel.close();

	cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);

	ofstream out("answer2.txt");
	for (int i = 0; i < op + mp; i++)
	{
		out << "x[" << i << "] = " << x[i] << endl;
	}
	out.close();

	// 更新应力场
	for (int i = inlet; i < inlet + op; i++)
	{
		Pb[i].pressure = x[i - inlet];
	}
	for (int i = op; i < op + mp; i++)
	{
		Pb[i + para_macro].pressure = x[i];
	}

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

void PNMsolver::PCG(int iters_, double tol_)
{
	// 矩阵的内存空间CSR
	rows_offsets = ia;
	columns = ja;
	values = a;

	clock_t startTime, endTime;
	startTime = clock();

	int N = op + mp;
	int nnz = ia[op + mp];
	const double tol = tol_;
	const int max_iter = iters_;
	cout << "max_iter:" << max_iter << endl;

	double *x, *rhs;
	double r0, r1, alpha, beta;
	double *d_x;
	double *d_zm1, *d_zm2, *d_rm2;
	double *d_r, *d_p, *d_omega, *d_y;
	double *d_valsILU0;
	double rsum, diff, err = 0.0;
	double qaerr1, qaerr2 = 0.0;
	double dot, numerator, denominator, nalpha;
	const double floatone = 1.0;
	const double floatzero = 0.0;

	int nErrors = 0;

	x = (double *)malloc(sizeof(double) * N);
	rhs = (double *)malloc(sizeof(double) * N);

	for (int i = 0; i < N; i++)
	{
		rhs[i] = B[i]; // Initialize RHS
		x[i] = 0.0;	   // Initial solution approximation
	}

	/* Create CUBLAS context */
	cublasHandle_t cublasHandle = NULL;
	checkCudaErrors(cublasCreate(&cublasHandle));

	/* Create CUSPARSE context */
	cusparseHandle_t cusparseHandle = NULL;
	checkCudaErrors(cusparseCreate(&cusparseHandle));

	/* Description of the A matrix */
	cusparseMatDescr_t descr = 0;
	checkCudaErrors(cusparseCreateMatDescr(&descr));
	checkCudaErrors(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
	checkCudaErrors(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

	/* Allocate required memory */
	checkCudaErrors(cudaMalloc((void **)&d_csr_columns, nnz * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_csr_offsets, (N + 1) * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_csr_values, nnz * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_x, N * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_y, N * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_r, N * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_p, N * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_omega, N * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_valsILU0, nnz * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_zm1, (N) * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_zm2, (N) * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_rm2, (N) * sizeof(double)));

	/* Wrap raw data into cuSPARSE generic API objects */
	cusparseDnVecDescr_t vecp = NULL, vecX = NULL, vecY = NULL, vecR = NULL, vecZM1 = NULL;
	checkCudaErrors(cusparseCreateDnVec(&vecp, N, d_p, CUDA_R_64F));
	checkCudaErrors(cusparseCreateDnVec(&vecX, N, d_x, CUDA_R_64F));
	checkCudaErrors(cusparseCreateDnVec(&vecY, N, d_y, CUDA_R_64F));
	checkCudaErrors(cusparseCreateDnVec(&vecR, N, d_r, CUDA_R_64F));
	checkCudaErrors(cusparseCreateDnVec(&vecZM1, N, d_zm1, CUDA_R_64F));
	cusparseDnVecDescr_t vecomega = NULL;
	checkCudaErrors(cusparseCreateDnVec(&vecomega, N, d_omega, CUDA_R_64F));

	/* Initialize problem data */
	checkCudaErrors(cudaMemcpy(
		d_csr_columns, columns, nnz * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(
		d_csr_offsets, rows_offsets, (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(
		d_csr_values, values, nnz * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(
		d_x, x, N * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(
		d_r, rhs, N * sizeof(double), cudaMemcpyHostToDevice));

	cusparseSpMatDescr_t matA = NULL;
	cusparseSpMatDescr_t matM_lower, matM_upper;
	cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
	cusparseDiagType_t diag_unit = CUSPARSE_DIAG_TYPE_UNIT;
	cusparseFillMode_t fill_upper = CUSPARSE_FILL_MODE_UPPER;
	cusparseDiagType_t diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;

	checkCudaErrors(cusparseCreateCsr(
		&matA, N, N, nnz, d_csr_offsets, d_csr_columns, d_csr_values, CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

	/* Copy A data to ILU(0) vals as input*/
	checkCudaErrors(cudaMemcpy(
		d_valsILU0, d_csr_values, nnz * sizeof(double), cudaMemcpyDeviceToDevice));

	// Lower Part
	checkCudaErrors(cusparseCreateCsr(&matM_lower, N, N, nnz, d_csr_offsets, d_csr_columns, d_valsILU0,
									  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
									  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

	checkCudaErrors(cusparseSpMatSetAttribute(matM_lower,
											  CUSPARSE_SPMAT_FILL_MODE,
											  &fill_lower, sizeof(fill_lower)));
	checkCudaErrors(cusparseSpMatSetAttribute(matM_lower,
											  CUSPARSE_SPMAT_DIAG_TYPE,
											  &diag_unit, sizeof(diag_unit)));
	// M_upper
	checkCudaErrors(cusparseCreateCsr(&matM_upper, N, N, nnz, d_csr_offsets, d_csr_columns, d_valsILU0,
									  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
									  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
	checkCudaErrors(cusparseSpMatSetAttribute(matM_upper,
											  CUSPARSE_SPMAT_FILL_MODE,
											  &fill_upper, sizeof(fill_upper)));
	checkCudaErrors(cusparseSpMatSetAttribute(matM_upper,
											  CUSPARSE_SPMAT_DIAG_TYPE,
											  &diag_non_unit,
											  sizeof(diag_non_unit)));

	/* Create ILU(0) info object */
	int bufferSizeLU = 0;
	size_t bufferSizeMV, bufferSizeL, bufferSizeU;
	void *d_bufferLU, *d_bufferMV, *d_bufferL, *d_bufferU;
	cusparseSpSVDescr_t spsvDescrL, spsvDescrU;
	cusparseMatDescr_t matLU;
	csrilu02Info_t infoILU = NULL;

	checkCudaErrors(cusparseCreateCsrilu02Info(&infoILU));
	checkCudaErrors(cusparseCreateMatDescr(&matLU));
	checkCudaErrors(cusparseSetMatType(matLU, CUSPARSE_MATRIX_TYPE_GENERAL));
	checkCudaErrors(cusparseSetMatIndexBase(matLU, CUSPARSE_INDEX_BASE_ZERO));

	/* Allocate workspace for cuSPARSE */
	checkCudaErrors(cusparseSpMV_bufferSize(
		cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matA,
		vecp, &floatzero, vecomega, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
		&bufferSizeMV));
	checkCudaErrors(cudaMalloc(&d_bufferMV, bufferSizeMV));

	checkCudaErrors(cusparseDcsrilu02_bufferSize(
		cusparseHandle, N, nnz, matLU, d_csr_values, d_csr_offsets, d_csr_columns, infoILU, &bufferSizeLU));
	checkCudaErrors(cudaMalloc(&d_bufferLU, bufferSizeLU));

	checkCudaErrors(cusparseSpSV_createDescr(&spsvDescrL));
	checkCudaErrors(cusparseSpSV_bufferSize(
		cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matM_lower, vecR, vecX, CUDA_R_64F,
		CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL));
	checkCudaErrors(cudaMalloc(&d_bufferL, bufferSizeL));

	checkCudaErrors(cusparseSpSV_createDescr(&spsvDescrU));
	checkCudaErrors(cusparseSpSV_bufferSize(
		cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matM_upper, vecR, vecX, CUDA_R_64F,
		CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, &bufferSizeU));
	checkCudaErrors(cudaMalloc(&d_bufferU, bufferSizeU));

	/* Conjugate gradient without preconditioning.
	   ------------------------------------------

	   Follows the description by Golub & Van Loan,
	   "Matrix Computations 3rd ed.", Section 10.2.6  */

	// printf("Convergence of CG without preconditioning: \n");

	int k;
	// k = 0;
	// r0 = 0;
	double rr;
	// checkCudaErrors(cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &rr));
	// checkCudaErrors(cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));

	// while (r1 > tol * tol * rr && k <= max_iter)
	// {
	// 	k++;

	// 	if (k == 1)
	// 	{
	// 		checkCudaErrors(cublasDcopy(cublasHandle, N, d_r, 1, d_p, 1));
	// 	}
	// 	else
	// 	{
	// 		beta = r1 / r0;
	// 		checkCudaErrors(cublasDscal(cublasHandle, N, &beta, d_p, 1));
	// 		checkCudaErrors(cublasDaxpy(
	// 			cublasHandle, N, &floatone, d_r, 1, d_p, 1));
	// 	}

	// 	checkCudaErrors(cusparseSpMV(
	// 		cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matA,
	// 		vecp, &floatzero, vecomega, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
	// 		d_bufferMV));
	// 	checkCudaErrors(cublasDdot(cublasHandle, N, d_p, 1, d_omega, 1, &dot));
	// 	alpha = r1 / dot;
	// 	checkCudaErrors(cublasDaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1));
	// 	nalpha = -alpha;
	// 	checkCudaErrors(cublasDaxpy(
	// 		cublasHandle, N, &nalpha, d_omega, 1, d_r, 1));
	// 	r0 = r1;
	// 	checkCudaErrors(cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));
	// }

	// printf("  iteration = %3d, residual = %e \n", k, sqrt(r1) / sqrt(rr));

	// checkCudaErrors(cudaMemcpy(
	// 	x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost));

	// /* check result */
	// err = 0.0;

	// for (int i = 0; i < N; i++)
	// {
	// 	rsum = 0.0;

	// 	for (int j = rows_offsets[i]; j < rows_offsets[i + 1]; j++)
	// 	{
	// 		rsum += values[j] * x[columns[j]];
	// 	}

	// 	diff = fabs(rsum - rhs[i]);

	// 	if (diff > err)
	// 	{
	// 		err = diff;
	// 	}
	// }

	// printf("  Convergence Test: %s \n", (k <= max_iter) ? "OK" : "FAIL");
	// nErrors += (k > max_iter) ? 1 : 0;
	// qaerr1 = err;

	// if (0)
	// {
	// 	// output result in matlab-style array
	// 	int n = (int)sqrt((double)N);
	// 	printf("a = [  ");

	// 	for (int iy = 0; iy < n; iy++)
	// 	{
	// 		for (int ix = 0; ix < n; ix++)
	// 		{
	// 			printf(" %f ", x[iy * n + ix]);
	// 		}

	// 		if (iy == n - 1)
	// 		{
	// 			printf(" ]");
	// 		}

	// 		printf("\n");
	// 	}
	// }

	/* Preconditioned Conjugate Gradient using ILU.
	   --------------------------------------------
	   Follows the description by Golub & Van Loan,
	   "Matrix Computations 3rd ed.", Algorithm 10.3.1  */

	printf("\nConvergence of CG using ILU(0) preconditioning: \n");

	/* Perform analysis for ILU(0) */
	checkCudaErrors(cusparseDcsrilu02_analysis(
		cusparseHandle, N, nnz, descr, d_valsILU0, d_csr_offsets, d_csr_columns, infoILU,
		CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU));

	/* generate the ILU(0) factors */
	checkCudaErrors(cusparseDcsrilu02(
		cusparseHandle, N, nnz, matLU, d_valsILU0, d_csr_offsets, d_csr_columns, infoILU,
		CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU));

	/* perform triangular solve analysis */
	checkCudaErrors(cusparseSpSV_analysis(
		cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone,
		matM_lower, vecR, vecX, CUDA_R_64F,
		CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_bufferL));

	checkCudaErrors(cusparseSpSV_analysis(
		cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone,
		matM_upper, vecR, vecX, CUDA_R_64F,
		CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, d_bufferU));

	/* reset the initial guess of the solution to zero */
	for (int i = 0; i < N; i++)
	{
		x[i] = 0.0;
	}
	checkCudaErrors(cudaMemcpy(
		d_r, rhs, N * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(
		d_x, x, N * sizeof(double), cudaMemcpyHostToDevice));

	k = 0;
	checkCudaErrors(cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));
	checkCudaErrors(cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &rr));
	while (r1 > tol * tol * rr && k <= max_iter)
	{
		// preconditioner application: d_zm1 = U^-1 L^-1 d_r
		checkCudaErrors(cusparseSpSV_solve(cusparseHandle,
										   CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone,
										   matM_lower, vecR, vecY, CUDA_R_64F,
										   CUSPARSE_SPSV_ALG_DEFAULT,
										   spsvDescrL));

		checkCudaErrors(cusparseSpSV_solve(cusparseHandle,
										   CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matM_upper,
										   vecY, vecZM1,
										   CUDA_R_64F,
										   CUSPARSE_SPSV_ALG_DEFAULT,
										   spsvDescrU));
		k++;

		if (k == 1)
		{
			checkCudaErrors(cublasDcopy(cublasHandle, N, d_zm1, 1, d_p, 1));
		}
		else
		{
			checkCudaErrors(cublasDdot(
				cublasHandle, N, d_r, 1, d_zm1, 1, &numerator));
			checkCudaErrors(cublasDdot(
				cublasHandle, N, d_rm2, 1, d_zm2, 1, &denominator));
			beta = numerator / denominator;
			checkCudaErrors(cublasDscal(cublasHandle, N, &beta, d_p, 1));
			checkCudaErrors(cublasDaxpy(
				cublasHandle, N, &floatone, d_zm1, 1, d_p, 1));
		}

		checkCudaErrors(cusparseSpMV(
			cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &floatone, matA,
			vecp, &floatzero, vecomega, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
			d_bufferMV));
		checkCudaErrors(cublasDdot(
			cublasHandle, N, d_r, 1, d_zm1, 1, &numerator));
		checkCudaErrors(cublasDdot(
			cublasHandle, N, d_p, 1, d_omega, 1, &denominator));
		alpha = numerator / denominator;
		checkCudaErrors(cublasDaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1));
		checkCudaErrors(cublasDcopy(cublasHandle, N, d_r, 1, d_rm2, 1));
		checkCudaErrors(cublasDcopy(cublasHandle, N, d_zm1, 1, d_zm2, 1));
		nalpha = -alpha;
		checkCudaErrors(cublasDaxpy(
			cublasHandle, N, &nalpha, d_omega, 1, d_r, 1));
		checkCudaErrors(cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1));
	}

	printf("  iteration = %3d, residual = %e \n", k, sqrt(r1) / sqrt(rr));

	checkCudaErrors(cudaMemcpy(
		x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost));

	/* check result */
	err = 0.0;

	for (int i = 0; i < N; i++)
	{
		rsum = 0.0;

		for (int j = rows_offsets[i]; j < rows_offsets[i + 1]; j++)
		{
			rsum += values[j] * x[columns[j]];
		}

		diff = fabs(rsum - rhs[i]);

		if (diff > err)
		{
			err = diff;
		}
	}

	printf("  Convergence Test: %s \n", (k <= max_iter) ? "OK" : "FAIL");
	nErrors += (k > max_iter) ? 1 : 0;
	qaerr2 = err;

	// 更新应力场
	for (int i = inlet; i < inlet + op; i++)
	{
		Pb[i].pressure = x[i - inlet];
	}
	for (int i = op; i < op + mp; i++)
	{
		Pb[i + para_macro].pressure = x[i];
	}
	/* Destroy descriptors */
	checkCudaErrors(cusparseDestroyCsrilu02Info(infoILU));
	checkCudaErrors(cusparseDestroyMatDescr(matLU));
	checkCudaErrors(cusparseSpSV_destroyDescr(spsvDescrL));
	checkCudaErrors(cusparseSpSV_destroyDescr(spsvDescrU));
	checkCudaErrors(cusparseDestroySpMat(matM_lower));
	checkCudaErrors(cusparseDestroySpMat(matM_upper));
	checkCudaErrors(cusparseDestroySpMat(matA));
	checkCudaErrors(cusparseDestroyDnVec(vecp));
	checkCudaErrors(cusparseDestroyDnVec(vecomega));
	checkCudaErrors(cusparseDestroyDnVec(vecR));
	checkCudaErrors(cusparseDestroyDnVec(vecX));
	checkCudaErrors(cusparseDestroyDnVec(vecY));
	checkCudaErrors(cusparseDestroyDnVec(vecZM1));

	/* Destroy contexts */
	checkCudaErrors(cusparseDestroy(cusparseHandle));
	checkCudaErrors(cublasDestroy(cublasHandle));

	/* Free device memory */

	free(x);
	free(rhs);
	checkCudaErrors(cudaFree(d_bufferMV));
	checkCudaErrors(cudaFree(d_bufferLU));
	checkCudaErrors(cudaFree(d_bufferL));
	checkCudaErrors(cudaFree(d_bufferU));
	checkCudaErrors(cudaFree(d_csr_offsets));
	checkCudaErrors(cudaFree(d_csr_columns));
	checkCudaErrors(cudaFree(d_csr_values));
	checkCudaErrors(cudaFree(d_x));
	checkCudaErrors(cudaFree(d_y));
	checkCudaErrors(cudaFree(d_r));
	checkCudaErrors(cudaFree(d_p));
	checkCudaErrors(cudaFree(d_omega));
	checkCudaErrors(cudaFree(d_valsILU0));
	checkCudaErrors(cudaFree(d_zm1));
	checkCudaErrors(cudaFree(d_zm2));
	checkCudaErrors(cudaFree(d_rm2));

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	// cudaDeviceReset();

	// printf("\n");
	// printf("Test Summary:\n");
	// printf("   Counted total of %d errors\n", nErrors);
	// printf("   qaerr1 = %f qaerr2 = %f\n\n", fabs(qaerr1), fabs(qaerr2));
	// exit((nErrors == 0 && fabs(qaerr1) < 1e-5 && fabs(qaerr2) < 1e-5
	// 		  ? EXIT_SUCCESS
	// 		  : EXIT_FAILURE));
}

void PNMsolver::AMGXsolver()
{
	// begin AMGX init
	AMGX_initialize();

	AMGX_config_handle config;
	AMGX_config_create_from_file(&config, "/home/rong/桌面/Mycode/lib/AMGX/build/configs/core/1.json");

	AMGX_resources_handle rsrc;
	AMGX_resources_create_simple(&rsrc, config);

	AMGX_solver_handle solver;
	AMGX_matrix_handle A_amgx;
	AMGX_vector_handle b_amgx;
	AMGX_vector_handle solution_amgx;

	AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, config);
	AMGX_matrix_create(&A_amgx, rsrc, AMGX_mode_dDDI);
	AMGX_vector_create(&b_amgx, rsrc, AMGX_mode_dDDI);
	AMGX_vector_create(&solution_amgx, rsrc, AMGX_mode_dDDI);
	// end AMGX init

	int n_amgx = op + mp;
	int nnz_amgx = ia[op + mp];

	AMGX_matrix_upload_all(A_amgx, n_amgx, nnz_amgx, 1, 1, ia, ja, a, 0);
	AMGX_vector_upload(b_amgx, n_amgx, 1, B);
	AMGX_vector_set_zero(solution_amgx, n_amgx, 1);
	AMGX_solver_setup(solver, A_amgx);

	// ************ begin AMGX solver ************

	AMGX_solver_solve_with_0_initial_guess(solver, b_amgx, solution_amgx);
	AMGX_vector_download(solution_amgx, X);
	ofstream out("answer3.txt");
	for (int i = 0; i < op + mp; i++)
	{
		out << "x[" << i << "] = " << X[i] << endl;
	}
	out.close();

	// 更新应力场
	for (int i = inlet; i < inlet + op; i++)
	{
		Pb[i].pressure = X[i - inlet];
	}
	for (int i = op; i < op + mp; i++)
	{
		Pb[i + para_macro].pressure = X[i];
	}
	// ************ end AMGX solver ************

	AMGX_solver_destroy(solver);
	AMGX_vector_destroy(b_amgx);
	AMGX_vector_destroy(solution_amgx);
	AMGX_matrix_destroy(A_amgx);
	AMGX_resources_destroy(rsrc);
	AMGX_finalize();
}

void PNMsolver::output()
{
	clock_t startTime, endTime;
	startTime = clock();
	ostringstream name;
	name << "result_noncom"
		 << ".vtk";
	ofstream outfile(name.str().c_str());
	outfile << "# vtk DataFile Version 2.0" << endl;
	outfile << "output.vtk" << endl;
	outfile << "ASCII " << endl;
	outfile << "DATASET POLYDATA " << endl;
	outfile << "POINTS " << pn << " float" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].X << "\t" << Pb[i].Y << "\t" << Pb[i].Z << endl;
	}
	// ����׺�������Ϣ
	outfile << "LINES"
			<< "\t" << tn << "\t" << 3 * Pb[pn - 1].full_accum << endl;
	for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	{
		outfile << 2 << "\t" << Tb[i].ID_1 << "\t" << Tb[i].ID_2 << endl;
	}
	// ���������Ϣ
	outfile << "POINT_DATA "
			<< "\t" << pn << endl;
	outfile << "SCALARS size_pb double 1" << endl;
	outfile << "LOOKUP_TABLE table1" << endl;
	for (int i = 0; i < pn; i++)
	{
		if (i < macro_n)
		{
			outfile << Pb[i].Radiu * 2 << "\t";
		}
		else
		{
			outfile << Pb[i].Radiu << "\t";
		}
	}
	outfile << endl;
	// ��������Ϣ
	outfile << "SCALARS NUMBER double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << i << endl;
	}
	// ���ѹ������Ϣ
	outfile << "SCALARS Pressure double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].pressure << endl;
	}
	// �����������Ϣ
	outfile << "SCALARS pb_type double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < pn; i++)
	{
		outfile << Pb[i].type << endl;
	}
	// ��������Ϣ
	outfile << "CELL_DATA"
			<< "\t" << Pb[pn - 1].full_accum << endl;
	outfile << "SCALARS throat_EqRadius double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	{

		outfile << Tb[i].Radiu << "\t";
	}
	outfile << "SCALARS conductivity double 1" << endl;
	outfile << "LOOKUP_TABLE table2" << endl;
	for (int i = 0; i < Pb[pn - 1].full_accum; i++)
	{

		outfile << Tb[i].Conductivity << "\t";
	}
	outfile.close();
	endTime = clock();
	cout << "output end:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << endl;
}

void PNMsolver::solver1(PNMsolver &Berea, int iters_, double tol_)
{
	clock_t startTime, endTime;
	startTime = clock();

	Poredateinput();
	Throatdateinput();
	para_cal();
	PressureMatrix();
	EigenSolve(iters_, tol_);
	Berea.permeability1();
	// output();

	endTime = clock();
	cout << "all time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << endl;

	fstream Eigen_data, Eigen_data_execel;
	Eigen_data.open("Eigen_data.txt", ios::out | ios::app | ios::ate);
	Eigen_data << '\t' << "all time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << endl;
	Eigen_data.close();
	Eigen_data_execel.open("Eigen_data_execel.txt", ios::out | ios::app | ios::ate);
	Eigen_data_execel << "all time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << '\n'
					  << endl;
	Eigen_data_execel.close();
}

void PNMsolver::solver2(PNMsolver &Berea, int iters_, double tol_)
{
	clock_t startTime, endTime;
	startTime = clock();

	Poredateinput();
	Throatdateinput();
	para_cal();
	PressureMatrix();
	// MMout();
	conjugateGradient_solver(iters_, tol_);
	Berea.permeability2();
	output();

	endTime = clock();
	cout << "all time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << endl;

	fstream Cuda_data, Cuda_data_execel;
	Cuda_data.open("Cuda_data.txt", ios::out | ios::app | ios::ate);
	Cuda_data << '\t' << "all time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << endl;
	Cuda_data.close();
	Cuda_data_execel.open("Cuda_data_execel.txt", ios::out | ios::app | ios::ate);
	Cuda_data_execel << "all time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << '\n'
					 << endl;
	Cuda_data_execel.close();
}

void PNMsolver::solver3()
{
	clock_t startTime, endTime;
	startTime = clock();
	Poredateinput();
	Throatdateinput();
	para_cal();
	PressureMatrix();
	AMGXsolver();
	// output();
	endTime = clock();
	cout << "all time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << endl;

	fstream AMGX_data, AMGX_data_execel;
	AMGX_data.open("AMGX_data.txt", ios::out | ios::app | ios::ate);
	AMGX_data << '\t' << "all time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << endl;
	AMGX_data.close();
	AMGX_data_execel.open("AMGX_data_execel.txt", ios::out | ios::app | ios::ate);
	AMGX_data_execel << "all time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << '\n'
					 << endl;
	AMGX_data_execel.close();
}

void PNMsolver::solver4(PNMsolver &Berea, int iters_, double tol_)
{
	clock_t startTime, endTime;
	startTime = clock();

	Poredateinput();
	Throatdateinput();
	para_cal();
	PressureMatrix();
	PCG(iters_, tol_);
	Berea.permeability2();
	output();

	endTime = clock();
	cout << "all time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << endl;
	fstream Cuda_data, Cuda_data_execel;
	Cuda_data.open("Cuda_data.txt", ios::out | ios::app | ios::ate);
	Cuda_data << '\t' << "all time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << endl;
	Cuda_data.close();
	Cuda_data_execel.open("Cuda_data_execel.txt", ios::out | ios::app | ios::ate);
	Cuda_data_execel << "all time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << '\n'
					 << endl;
	Cuda_data_execel.close();
}

void PNMsolver::solver5(PNMsolver &Berea, int iters_, double tol_)
{
	clock_t startTime, endTime;
	startTime = clock();

	Poredateinput();
	Throatdateinput();
	para_cal();
	PressureMatrix();
	API_BICGSTAB(iters_, tol_, ia, ja, a, op + mp, B);
	// API_UNPBICGSTAB(iters_, tol_, ia, ja, a, op + mp, B);

	// 更新应力场
	for (int i = inlet; i < inlet + op; i++)
	{
		Pb[i].pressure = B[i - inlet];
	}
	for (int i = op; i < op + mp; i++)
	{
		Pb[i + para_macro].pressure = B[i];
	}

	Berea.permeability2();
	output();

	endTime = clock();
	cout << "all time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << endl;
	fstream Cuda_data, Cuda_data_execel;
	Cuda_data.open("Cuda_data.txt", ios::out | ios::app | ios::ate);
	Cuda_data << '\t' << "all time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << endl;
	Cuda_data.close();
	Cuda_data_execel.open("Cuda_data_execel.txt", ios::out | ios::app | ios::ate);
	Cuda_data_execel << "all time:" << (double)(endTime - startTime) / CLOCKS_PER_SECOND / 1000 << "s" << '\n'
					 << endl;
	Cuda_data_execel.close();
}

void PNMsolver::solver6(PNMsolver &Berea, int iters_, double tol_)
{
	clock_t startTime, endTime;
	startTime = clock();

	Poredateinput();
	Throatdateinput();
	para_cal();
	PressureMatrix();
	cg_API(iters_, tol_, ia, ja, a, op + mp, B);

	// 更新应力场
	for (int i = inlet; i < inlet + op; i++)
	{
		Pb[i].pressure = B[i - inlet];
	}
	for (int i = op; i < op + mp; i++)
	{
		Pb[i + para_macro].pressure = B[i];
	}

	Berea.permeability2();
	output();
}

void PNMsolver::solver8(PNMsolver &Berea)
{
	clock_t startTime, endTime;
	startTime = clock();

	Poredateinput();
	Throatdateinput();
	para_cal();
	PressureMatrix();
	int nrhs = 1;
	super_API(a, ia, ja, op + mp, ia[op + mp], B, nrhs);

	// 更新应力场
	for (int i = inlet; i < inlet + op; i++)
	{
		Pb[i].pressure = B[i - inlet];
	}
	for (int i = op; i < op + mp; i++)
	{
		Pb[i + para_macro].pressure = B[i];
	}
	Berea.permeability2();
	output();
}

void PNMsolver::solver9(PNMsolver &Berea)
{
	clock_t startTime, endTime;
	startTime = clock();

	Poredateinput();
	Throatdateinput();
	para_cal();
	PressureMatrix();
	int nrhs = 1;
	ditersol(a, ia, ja, op + mp, ia[op + mp], B);

	// 更新应力场
	for (int i = inlet; i < inlet + op; i++)
	{
		Pb[i].pressure = B[i - inlet];
	}
	for (int i = op; i < op + mp; i++)
	{
		Pb[i + para_macro].pressure = B[i];
	}
	Berea.permeability2();
	output();
}

double PNMsolver::permeability1()
{
	double TOTAL_FLOW = 0;
	double P = 0;
	for (int i = macro_n - outlet; i < macro_n; i++)
	{
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			TOTAL_FLOW += abs((Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * Tb[j].Conductivity);
		}
	}

	for (int i = pn - m_outlet; i < pn; i++)
	{
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			TOTAL_FLOW += abs((Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * Tb[j].Conductivity);
		}
	}

	// ģ��Ϊ������
	P = TOTAL_FLOW * gas_vis / (voxel_size * outlet_element_n * (inlet_pre - outlet_pre));
	cout << "渗透率:" << P << endl;

	fstream Eigen_data, Eigen_data_execel;
	Eigen_data.open("Eigen_data.txt", ios::out | ios::app | ios::ate);
	Eigen_data << "permeability:" << P;
	Eigen_data.close();

	Eigen_data_execel.open("Eigen_data_execel.txt", ios::out | ios::app | ios::ate);
	Eigen_data_execel << "permeability:" << P << "\n";
	Eigen_data_execel.close();
	return P;
}
double PNMsolver::permeability2()
{
	double TOTAL_FLOW = 0;
	double P = 0;
	for (int i = macro_n - outlet; i < macro_n; i++)
	{
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			TOTAL_FLOW += abs((Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * Tb[j].Conductivity);
		}
	}

	for (int i = pn - m_outlet; i < pn; i++)
	{
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			TOTAL_FLOW += abs((Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * Tb[j].Conductivity);
		}
	}

	// ģ��Ϊ������
	P = TOTAL_FLOW * gas_vis / (voxel_size * outlet_element_n * (inlet_pre - outlet_pre));
	cout << "渗透率:" << P << endl;

	fstream Cuda_data, Cuda_data_execel;
	Cuda_data.open("Cuda_data.txt", ios::out | ios::app | ios::ate);
	Cuda_data << "permeability:" << P;
	Cuda_data.close();

	Cuda_data_execel.open("Cuda_data_execel.txt", ios::out | ios::app | ios::ate);
	Cuda_data_execel << "permeability:" << P << "\n";
	Cuda_data_execel.close();
	return P;
}
double PNMsolver::permeability3()
{
	double TOTAL_FLOW = 0;
	double P = 0;
	for (int i = macro_n - outlet; i < macro_n; i++)
	{
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			TOTAL_FLOW += abs((Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * Tb[j].Conductivity);
		}
	}

	for (int i = pn - m_outlet; i < pn; i++)
	{
		for (int j = Pb[i].full_accum - Pb[i].full_coord; j < Pb[i].full_accum; j++)
		{
			TOTAL_FLOW += abs((Pb[Tb[j].ID_1].pressure - Pb[Tb[j].ID_2].pressure) * Tb[j].Conductivity);
		}
	}

	// ģ��Ϊ������
	P = TOTAL_FLOW * gas_vis / (voxel_size * outlet_element_n * (inlet_pre - outlet_pre));
	cout << "渗透率:" << P << endl;

	fstream AMGX_data, AMGX_data_execel;
	AMGX_data.open("AMGX_data.txt", ios::out | ios::app | ios::ate);
	AMGX_data << "permeability:" << P;
	AMGX_data.close();

	AMGX_data_execel.open("AMGX_data_execel.txt", ios::out | ios::app | ios::ate);
	AMGX_data_execel << "permeability:" << P << "\n";
	AMGX_data_execel.close();
	return P;
}

int main(int argc, char *argv[])
 {
	int iters_eigen{1000};
	double tol{1e-10};

	// {
	// 	PNMsolver Berea;
	// 	Berea.solver1(Berea, iters_eigen, tol);
	// 	cout << "*****************************Eigen CALCULATION END*****************************" << endl;
	// }

	// cg
	// {
	// 	int iters_cuda{100000};
	// 	PNMsolver Berea;
	// 	Berea.solver2(Berea, iters_cuda, tol);
	// 	cout << "*****************************CUDA CALCULATION END*****************************" << endl;
	// }

	// pcg ilu0
	// {
	// 	int iters_cuda{10000};
	// 	PNMsolver Berea;
	// 	Berea.solver4(Berea, iters_cuda, tol);
	// 	cout << "*****************************CUDA CALCULATION END*****************************" << endl;
	// }

	// pcg icho  single
	// {
	// 	int iters_cuda{10000};
	// 	PNMsolver Berea;
	// 	Berea.solver6(Berea, iters_cuda, tol);
	// 	cout << "*****************************CUDA CALCULATION END*****************************" << endl;
	// }

	// bicgstab  pbicgstab  single
	// {
	// 	int iters_cuda{1000};
	// 	PNMsolver Berea;
	// 	Berea.solver5(Berea, iters_cuda, tol);
	// 	cout << "*****************************CUDA CALCULATION END*****************************" << endl;
	// }

	// amgx
	//  {
	//  	ofstream out("Permeability3.txt");
	//  	PNMsolver Berea;
	//  	Berea.solver3();
	//  	double permeabilty = Berea.permeability3();
	//  	out << ko << "\t" << permeabilty << endl;
	//  	out.close();
	//  	cout << "*****************************CALCULATION END*****************************" << endl;
	//  }

	// superlu
	// {
	// 	ofstream out("Permeability3.txt");
	// 	PNMsolver Berea;
	// 	Berea.solver8(Berea);
	// 	double permeabilty = Berea.permeability3();
	// 	out << ko << "\t" << permeabilty << endl;
	// 	out.close();
	// 	cout << "*****************************CALCULATION END*****************************" << endl;
	// }

	//superlu preconditoned grmes
	// {
	// 	ofstream out("Permeability3.txt");
	// 	PNMsolver Berea;
	// 	Berea.solver9(Berea);
	// 	double permeabilty = Berea.permeability3();
	// 	out << ko << "\t" << permeabilty << endl;
	// 	out.close();
	// 	cout << "*****************************CALCULATION END*****************************" << endl;
	// }

	return 0;
}
