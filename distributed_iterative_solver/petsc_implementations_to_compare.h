#include <omp.h>
#include <mpi.h>
#include <iostream>

#include <petscksp.h>
#include <petscvec.h>
#include <petscdevice.h> 

int cpu_solve(
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
    double *time_taken,
    bool *correct_solution
);

int gpu_solve(
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
    double *time_taken,
    bool *correct_solution
);

