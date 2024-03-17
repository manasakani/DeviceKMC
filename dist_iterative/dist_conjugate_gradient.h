#pragma once
#include <string> 
#include <omp.h>
#include "utils_cg.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <hipsparse.h>
#include <hipblas.h>
#include <iostream>
#include <mpi.h>
#include "cudaerrchk.h"
#include "dist_objects.h"
#include <unistd.h>  

namespace iterative_solver{

template <void (*distributed_spmv)(Distributed_matrix&, Distributed_vector&, hipsparseDnVecDescr_t&, hipStream_t&, hipsparseHandle_t&)>
void conjugate_gradient(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);

template <void (*distributed_spmv)(Distributed_matrix&, Distributed_vector&, hipsparseDnVecDescr_t&, hipStream_t&, hipsparseHandle_t&)>
void conjugate_gradient_jacobi(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);

template <void (*distributed_spmv_split)
    (Distributed_subblock &,
    Distributed_matrix &,    
    double *,
    double *,
    Distributed_vector &,
    double *,
    hipsparseDnVecDescr_t &,
    double *,
    hipStream_t &,
    hipsparseHandle_t &,
    hipblasHandle_t &)>
void conjugate_gradient_split(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);

template <void (*distributed_spmv_split)
    (Distributed_subblock &,
    Distributed_matrix &,    
    double *,
    double *,
    Distributed_vector &,
    double *,
    hipsparseDnVecDescr_t &,
    double *,
    hipStream_t &,
    hipsparseHandle_t &,
    hipblasHandle_t &)>
void conjugate_gradient_jacobi_split(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);

template <void (*distributed_spmv_split_sparse)
    (Distributed_subblock_sparse &,
    Distributed_matrix &,    
    double *,
    double *,
    rocsparse_dnvec_descr &,
    Distributed_vector &,
    double *,
    rocsparse_dnvec_descr &,
    hipsparseDnVecDescr_t &,
    double *,
    hipStream_t &,
    hipsparseHandle_t &,
    rocsparse_handle &)>
void conjugate_gradient_jacobi_split_sparse(
    Distributed_subblock_sparse &A_subblock,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);

} // namespace iterative_solver