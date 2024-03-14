
#pragma once
#include <hip/hip_runtime.h>
#include <hipsparse.h>
#include <mpi.h>
#include "cudaerrchk.h"
#include "dist_objects.h"
#include "utils_cg.h"
#include <pthread.h>
#include "rocsparse.h"

namespace dspmv{


void gpu_packing(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    hipsparseDnVecDescr_t &vecAp_local,
    hipStream_t &default_stream,
    hipsparseHandle_t &default_cusparseHandle);

void gpu_packing_cam(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    hipsparseDnVecDescr_t &vecAp_local,
    hipStream_t &default_stream,
    hipsparseHandle_t &default_cusparseHandle);

} // namespace dspmv

namespace dspmv_split{

void spmm_split1(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    hipsparseDnVecDescr_t &vecAp_local,
    double *Ap_local_d,
    hipStream_t &default_stream,
    hipsparseHandle_t &default_cusparseHandle,
    hipblasHandle_t &default_cublasHandle);

void spmm_split2(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    hipsparseDnVecDescr_t &vecAp_local,
    double *Ap_local_d,
    hipStream_t &default_stream,
    hipsparseHandle_t &default_cusparseHandle,
    hipblasHandle_t &default_cublasHandle);

void spmm_split3(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    hipsparseDnVecDescr_t &vecAp_local,
    double *Ap_local_d,
    hipStream_t &default_stream,
    hipsparseHandle_t &default_cusparseHandle,
    hipblasHandle_t &default_cublasHandle);

void spmm_split4(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    hipsparseDnVecDescr_t &vecAp_local,
    double *Ap_local_d,
    hipStream_t &default_stream,
    hipsparseHandle_t &default_cusparseHandle,
    hipblasHandle_t &default_cublasHandle);

void spmm_split5(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    hipsparseDnVecDescr_t &vecAp_local,
    double *Ap_local_d,
    hipStream_t &default_stream,
    hipsparseHandle_t &default_cusparseHandle,
    hipblasHandle_t &default_cublasHandle);

void spmm_split6(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    hipsparseDnVecDescr_t &vecAp_local,
    double *Ap_local_d,
    hipStream_t &default_stream,
    hipsparseHandle_t &default_cusparseHandle,
    hipblasHandle_t &default_cublasHandle);

} // namespace dspmv_split


namespace dspmv_split_sparse{

void spmm_split_sparse1(
    Distributed_subblock_sparse &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    hipsparseDnVecDescr_t &vecp_subblock,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    hipsparseDnVecDescr_t &vecAp_subblock,
    hipsparseDnVecDescr_t &vecAp_local,
    double *Ap_local_d,
    hipStream_t &default_stream,
    hipsparseHandle_t &default_cusparseHandle);

void spmm_split_sparse2(
    Distributed_subblock_sparse &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    hipsparseDnVecDescr_t &vecp_subblock,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    hipsparseDnVecDescr_t &vecAp_subblock,
    hipsparseDnVecDescr_t &vecAp_local,
    double *Ap_local_d,
    hipStream_t &default_stream,
    hipsparseHandle_t &default_cusparseHandle);

void spmm_split_sparse3(
    Distributed_subblock_sparse &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    hipsparseDnVecDescr_t &vecp_subblock,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    hipsparseDnVecDescr_t &vecAp_subblock,
    hipsparseDnVecDescr_t &vecAp_local,
    double *Ap_local_d,
    hipStream_t &default_stream,
    hipsparseHandle_t &default_cusparseHandle);

void spmm_split_sparse4(
    Distributed_subblock_sparse &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    hipsparseDnVecDescr_t &vecp_subblock,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    hipsparseDnVecDescr_t &vecAp_subblock,
    hipsparseDnVecDescr_t &vecAp_local,
    double *Ap_local_d,
    hipStream_t &default_stream,
    hipsparseHandle_t &default_cusparseHandle);

void spmm_split_sparse5(
    Distributed_subblock_sparse &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    hipsparseDnVecDescr_t &vecp_subblock,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    hipsparseDnVecDescr_t &vecAp_subblock,
    hipsparseDnVecDescr_t &vecAp_local,
    double *Ap_local_d,
    hipStream_t &default_stream,
    hipsparseHandle_t &default_cusparseHandle);

} // namespace dspmv_split_sparse