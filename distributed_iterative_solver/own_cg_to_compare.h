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
#include "cudaerrchk.h"
#include "distributed_objects.h"
#include <unistd.h>  

namespace own_test{

void solve_cg_allgather_mpi(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    int *steps_taken,
    double *time_taken);

void solve_cg_allgatherv_mpi(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    int *steps_taken,
    double *time_taken);

void solve_cg_rma_fetch_whole(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    int *steps_taken,
    double *time_taken);

void solve_cg_nonblocking_point_to_point(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    int *steps_taken,
    double *time_taken);

void solve_cg_nonblocking_point_to_point_fetch_specific(
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

void solve_cg(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution_h,
    double *starting_guess_h,
    int nnz,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    int *steps_taken,
    double *time_taken);

} // namespace own_test

