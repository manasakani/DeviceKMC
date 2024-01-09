#pragma once
#include <string> 
#include <omp.h>

#include "utils.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <iostream>
#include <mpi.h>

namespace own_test{

void solve_cg_mpi(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution,
    double *starting_guess_h,
    int nnz,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    int *steps_taken,
    double *time_taken);

} // namespace own_test
