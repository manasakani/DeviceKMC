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

template <void (*distributed_spmv)(Distributed_matrix&, Distributed_vector&, cusparseDnVecDescr_t&, cudaStream_t&, cusparseHandle_t&)>
void conjugate_gradient_jacobi(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);

template <void (*distributed_spmv)(Distributed_matrix&, Distributed_vector&, cusparseDnVecDescr_t&, cudaStream_t&, cusparseHandle_t&)>
void conjugate_gradient_split(
    int *subblock_indices_local_d,
    double *A_subblock_local_d,
    int subblock_size,
    int *count_subblock_h,
    int *displ_subblock_h,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);

template <void (*distributed_spmv)(Distributed_matrix&, Distributed_vector&, cusparseDnVecDescr_t&, cudaStream_t&, cusparseHandle_t&)>
void conjugate_gradient_jacobi_split(
    int *subblock_indices_d,
    double *A_subblock_local_d,
    int subblock_size,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);

} // namespace iterative_solver

