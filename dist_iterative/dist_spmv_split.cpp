#include "dist_spmv.h"

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
    cublasHandle_t &default_cublasHandle)
{
    int rank = A_distributed.rank;
    int size = A_distributed.size;

    MPI_Request send_subblock_requests[size-1];
    MPI_Request recv_subblock_requests[size-1];
    double alpha = 1.0;
    double beta = 0.0;

    // pack dense sublblock p
    pack_gpu(p_subblock_d + displ_subblock_h[rank],
        p_distributed.vec_d[0],
        subblock_indices_local_d,
        count_subblock_h[rank],
        default_stream);

    if(size > 1){
        cudaErrchk(cudaMemcpy(p_subblock_h + displ_subblock_h[rank],
            p_subblock_d + displ_subblock_h[rank],
            count_subblock_h[rank] * sizeof(double), cudaMemcpyDeviceToHost));
        for(int i = 0; i < size-1; i++){
            int dest = (rank + 1 + i) % size;
            MPI_Isend(p_subblock_h + displ_subblock_h[rank], count_subblock_h[rank],
                MPI_DOUBLE, dest, dest, A_distributed.comm, &send_subblock_requests[i]);
        }
        for(int i = 0; i < size-1; i++){
            int source = (rank + 1 + i) % size;
            MPI_Irecv(p_subblock_h + displ_subblock_h[source], count_subblock_h[source],
                MPI_DOUBLE, source, rank, A_distributed.comm, &recv_subblock_requests[i]);
        }
    }

    // ak = rk^T * rk / pk^T * A * pk
    // has to be done for k=0 if x0 != 0
    dspmv::gpu_packing(
        A_distributed,
        p_distributed,
        vecAp_local,
        default_stream,
        default_cusparseHandle
    );
    if(size > 1){
        MPI_Waitall(size-1, recv_subblock_requests, MPI_STATUSES_IGNORE);
        MPI_Waitall(size-1, send_subblock_requests, MPI_STATUSES_IGNORE);
        // recv whole vector
        cudaErrchk(cudaMemcpyAsync(p_subblock_d,
            p_subblock_h, subblock_size * sizeof(double),
            cudaMemcpyHostToDevice, default_stream));
    }

    cublasErrchk(cublasDgemv(
        default_cublasHandle,
        CUBLAS_OP_N,
        count_subblock_h[rank], subblock_size,
        &alpha,
        A_subblock_local_d, count_subblock_h[rank],
        p_subblock_d, 1,
        &beta,
        Ap_subblock_d, 1
    ));
    // unpack and add it to Ap
    unpack_add(
        Ap_local_d,
        Ap_subblock_d,
        subblock_indices_local_d,
        count_subblock_h[rank],
        default_stream
    );
}

} // namespace dspmv_split