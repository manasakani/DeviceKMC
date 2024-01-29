#pragma once
#include <omp.h>
#include <mpi.h>
#include <iostream>

#include <petscerror.h>
#include <petscksp.h>
#include <petscvec.h>
#include <petscdevice.h> 

#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_utilities.h>
#include <HYPRE_krylov.h>

#include <ginkgo/ginkgo.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

namespace lib_to_compare
{


int solve_petsc(
    int rank,
    double *data_local,
    int *row_ptr_local,
    int *col_indices_local,
    double *rhs,
    double *reference_solution,
    int row_start_index,
    int rows_per_rank, 
    int matrix_size,
    int max_iterations,
    KSPType solver_type,
    PCType preconditioner,
    double relative_tolerance,
    double absolute_tolerance,
    double divergence_tolerance,
    int *iterations,
    double *time_taken);


void solve_hypre(
    double *data_local,
    int *row_ptr_local,
    int *col_indices_local,
    double *rhs_local,
    double *reference_solution,
    int row_start_index,
    int row_end_index,
    int rows_per_rank,
    int max_iterations,
    double relative_tolerance,
    double absolute_tolerance,
    int *iterations,
    double *time_taken);

void solve_ginkgo(
    double *data_local,
    int *row_ptr_local,
    int *col_indices_local,
    double *rhs,
    double *reference_solution,
    int matrix_size,
    int max_iterations,
    double relative_tolerance,
    int *iterations,
    double *time_taken);

} // namespace lib_to_compare