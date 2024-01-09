#pragma once
#include <ginkgo/ginkgo.hpp>
#include <cuda_runtime.h>
#include <mpi.h>

#include <cuda.h>

namespace gko_test {

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

} // namespace gko_test