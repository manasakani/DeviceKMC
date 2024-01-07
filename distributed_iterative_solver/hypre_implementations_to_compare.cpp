#include "hypre_implementations_to_compare.h"

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
    HYPRE_MemoryLocation MEMORY_LOCATION,
    int *iterations,
    double *time_taken,
    bool *correct_solution)
{


}


}