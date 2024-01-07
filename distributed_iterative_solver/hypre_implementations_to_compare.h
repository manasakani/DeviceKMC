#pragma once
#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_utilities.h>
#include <HYPRE_krylov.h>

namespace hypre_test {

void gpu_solve(
    double *data_local,
    int *row_ptr_local,
    int *col_indices_local,
    double *rhs,
    double *reference_solution,
    int matrix_size,
    int max_iterations,
    double relative_tolerance,
    int *iterations,
    double *time_taken,
    bool *correct_solution);


}