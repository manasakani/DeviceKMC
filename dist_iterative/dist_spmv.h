
#pragma once
#include <cuda_runtime.h>
#include <cusparse.h>
#include <mpi.h>
#include "cudaerrchk.h"
#include "dist_objects.h"
#include "utils_cg.h"

namespace dspmv{


void gpu_packing(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    cusparseDnVecDescr_t &vecAp_local,
    cudaStream_t &default_stream,
    cusparseHandle_t &default_cusparseHandle);

void gpu_packing_cam(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    cusparseDnVecDescr_t &vecAp_local,
    cudaStream_t &default_stream,
    cusparseHandle_t &default_cusparseHandle);

} // namespace dspmv

namespace dspmv_split{

void spmm_split1(
    int *subblock_indices_local_d,
    double *A_subblock_local_d,
    int subblock_size,
    int *count_subblock_h,
    int *displ_subblock_h,
    double *p_subblock_d,
    double *p_subblock_h,
    double *Ap_subblock_d,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    cusparseDnVecDescr_t &vecAp_local,
    double *Ap_local_d,
    cudaStream_t &default_stream,
    cusparseHandle_t &default_cusparseHandle,
    cublasHandle_t &default_cublasHandle);

} // namespace dspmv_split