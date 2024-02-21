#pragma once
#include <string> 
#include <omp.h>
#include "../utils.h"
#include "utils_cg.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <iostream>
#include <mpi.h>
#include "cudaerrchk.h"
#include "dist_objects.h"
#include <unistd.h>  

namespace iterative_solver{

template <void (*distributed_mv)(Distributed_matrix&, Distributed_vector&, cusparseDnVecDescr_t&, cudaStream_t&, cusparseHandle_t&)>
void conjugate_gradient(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    int *steps_taken,
    double *time_taken);

} // namespace iterative_solver

