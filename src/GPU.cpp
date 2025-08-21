#include "GPU.hpp"

// For AMGX
#include <amgx_c.h>
#include <amgx_config.h>

/* Using updated (v2) interfaces to cublas usparseSparseToDense*/
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

// Utilities and system includes
#include <helper_cuda.h>             // helper function CUDA error checking and initialization
#include <helper_functions.h>        // helper for shared functions common to CUDA Samples

struct GPUObjectsImpl {
  AMGX_config_handle config;
  AMGX_resources_handle rsrc;
  AMGX_solver_handle solver;
  AMGX_matrix_handle A_amgx;
  AMGX_vector_handle b_amgx;
  AMGX_vector_handle solution_amgx;
  int n_amgx;
  int nnz_amgx;
};

GPUObjects GPU_init(int* _ia, int* _ja, double* _a, double* _B, double* _Geta_X, int _op_plus_mp) {
  GPUObjects obj;
  obj.impl = new GPUObjectsImpl;

  const char* config_string = R"###({
      "config_version": 2,
      "solver": {
          "preconditioner": {
              "error_scaling": 0,
              "print_grid_stats": 1,
              "max_uncolored_percentage": 0.05,
              "algorithm": "AGGREGATION",
              "solver": "AMG",
              "smoother": "MULTICOLOR_DILU",
              "presweeps": 0,
              "selector": "SIZE_2",
              "coarse_solver": "NOSOLVER",
              "max_iters": 1,
              "postsweeps": 3,
              "min_coarse_rows": 32,
              "relaxation_factor": 0.75,
              "scope": "amg",
              "max_levels": 100,
              "matrix_coloring_scheme": "PARALLEL_GREEDY",
              "cycle": "V"
          },
          "use_scalar_norm": 1,
          "solver": "FGMRES",
          "print_solve_stats": 1,
          "obtain_timings": 1,
          "max_iters": 1000,
          "monitor_residual": 1,
          "gmres_n_restart": 10,
          "convergence": "RELATIVE_INI_CORE",
          "scope": "main",
          "tolerance": 1e-10,
          "norm": "L2"
      }
  })###";

  // begin AMGX initialization
  AMGX_initialize();

  AMGX_config_create(&obj.impl->config, config_string);

  AMGX_resources_create_simple(&obj.impl->rsrc, obj.impl->config);

  AMGX_solver_create(&obj.impl->solver, obj.impl->rsrc, AMGX_mode_dDDI, obj.impl->config);
  AMGX_matrix_create(&obj.impl->A_amgx, obj.impl->rsrc, AMGX_mode_dDDI);
  AMGX_vector_create(&obj.impl->b_amgx, obj.impl->rsrc, AMGX_mode_dDDI);
  AMGX_vector_create(&obj.impl->solution_amgx, obj.impl->rsrc, AMGX_mode_dDDI);

  obj.impl->n_amgx = _op_plus_mp;
  obj.impl->nnz_amgx = _ia[_op_plus_mp];
  AMGX_pin_memory(_ia, (obj.impl->n_amgx + 1) * sizeof(int));
  AMGX_pin_memory(_ja, obj.impl->nnz_amgx * sizeof(int));
  AMGX_pin_memory(_a, obj.impl->nnz_amgx * sizeof(double));
  AMGX_pin_memory(_B, sizeof(double) * obj.impl->n_amgx);
  AMGX_pin_memory(_Geta_X, sizeof(double) * obj.impl->n_amgx);

  return obj;
};

void GPU_solveX(int& _Icount, int* ia, int* ja, double* a, double* B, double* geta_X, const GPUObjects& _obj) {
  if (_Icount == 0) {
    AMGX_matrix_upload_all(_obj.impl->A_amgx, _obj.impl->n_amgx, _obj.impl->nnz_amgx, 1, 1, ia, ja, a, 0);
    AMGX_solver_setup(_obj.impl->solver, _obj.impl->A_amgx);
    _Icount += 1;
  } else {
    AMGX_matrix_replace_coefficients(_obj.impl->A_amgx, _obj.impl->n_amgx, _obj.impl->nnz_amgx, a, 0);
  }
  AMGX_vector_upload(_obj.impl->b_amgx, _obj.impl->n_amgx, 1, B);
  AMGX_vector_set_zero(_obj.impl->solution_amgx, _obj.impl->n_amgx, 1);
  AMGX_solver_solve_with_0_initial_guess(_obj.impl->solver, _obj.impl->b_amgx, _obj.impl->solution_amgx);
  AMGX_vector_download(_obj.impl->solution_amgx, geta_X);
};

void GPU_release(int* _ia, int* _ja, double* _a, double* _B, double* _Geta_X, GPUObjects& _obj) {
  AMGX_unpin_memory(_ia);
  AMGX_unpin_memory(_ja);
  AMGX_unpin_memory(_a);
  AMGX_unpin_memory(_B);
  AMGX_unpin_memory(_Geta_X);

  AMGX_solver_destroy(_obj.impl->solver);
  AMGX_vector_destroy(_obj.impl->b_amgx);
  AMGX_vector_destroy(_obj.impl->solution_amgx);
  AMGX_matrix_destroy(_obj.impl->A_amgx);
  AMGX_resources_destroy(_obj.impl->rsrc);
  AMGX_config_destroy(_obj.impl->config);
  AMGX_finalize();

  delete _obj.impl;
  _obj.impl = nullptr;
};
