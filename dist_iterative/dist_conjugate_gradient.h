#pragma once
#include <string> 
#include <omp.h>
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

template <void (*distributed_spmv)(Distributed_matrix&, Distributed_vector&, cusparseDnVecDescr_t&, cudaStream_t&, cusparseHandle_t&)>
void conjugate_gradient(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);

} // namespace iterative_solver

