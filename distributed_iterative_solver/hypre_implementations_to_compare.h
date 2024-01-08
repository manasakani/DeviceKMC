#pragma once
#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_utilities.h>
#include <HYPRE_krylov.h>
#include <iostream>
#include <cuda_runtime.h>
#include <mpi.h>
#include <cmath>

namespace hypre_test {

void gpu_solve(
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
    HYPRE_MemoryLocation MEMORY_LOCATION,
    int *iterations,
    double *time_taken);


}