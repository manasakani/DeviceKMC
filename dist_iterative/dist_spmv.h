
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
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    cusparseDnVecDescr_t &vecAp_local,
    double *Ap_local_d,
    cudaStream_t &default_stream,
    cusparseHandle_t &default_cusparseHandle,
    cublasHandle_t &default_cublasHandle);

void spmm_split2(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    cusparseDnVecDescr_t &vecAp_local,
    double *Ap_local_d,
    cudaStream_t &default_stream,
    cusparseHandle_t &default_cusparseHandle,
    cublasHandle_t &default_cublasHandle);

} // namespace dspmv_split