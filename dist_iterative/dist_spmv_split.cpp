#include "dist_spmv.h"

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
    cublasHandle_t &default_cublasHandle)
{
    // Isend Irecv subblock
    // sparse part
    //gemv

    int rank = A_distributed.rank;
    int size = A_distributed.size;

    MPI_Request send_subblock_requests[size-1];
    MPI_Request recv_subblock_requests[size-1];
    double alpha = 1.0;
    double beta = 0.0;

    // pack dense sublblock p
    pack_gpu(p_subblock_d + A_subblock.displ_subblock_h[rank],
        p_distributed.vec_d[0],
        A_subblock.subblock_indices_local_d,
        A_subblock.count_subblock_h[rank],
        default_stream);

    if(size > 1){
        cudaErrchk(cudaMemcpy(p_subblock_h + A_subblock.displ_subblock_h[rank],
            p_subblock_d + A_subblock.displ_subblock_h[rank],
            A_subblock.count_subblock_h[rank] * sizeof(double), cudaMemcpyDeviceToHost));
        for(int i = 0; i < size-1; i++){
            int dest = (rank + 1 + i) % size;
            MPI_Isend(p_subblock_h + A_subblock.displ_subblock_h[rank], A_subblock.count_subblock_h[rank],
                MPI_DOUBLE, dest, dest, A_distributed.comm, &send_subblock_requests[i]);
        }
        for(int i = 0; i < size-1; i++){
            int source = (rank + 1 + i) % size;
            MPI_Irecv(p_subblock_h + A_subblock.displ_subblock_h[source], A_subblock.count_subblock_h[source],
                MPI_DOUBLE, source, rank, A_distributed.comm, &recv_subblock_requests[i]);
        }
    }

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
            p_subblock_h, A_subblock.subblock_size * sizeof(double),
            cudaMemcpyHostToDevice, default_stream));
    }

    cublasErrchk(cublasDgemv(
        default_cublasHandle,
        CUBLAS_OP_N,
        A_subblock.count_subblock_h[rank], A_subblock.subblock_size,
        &alpha,
        A_subblock.A_subblock_local_d, A_subblock.count_subblock_h[rank],
        p_subblock_d, 1,
        &beta,
        Ap_subblock_d, 1
    ));
    // unpack and add it to Ap
    unpack_add(
        Ap_local_d,
        Ap_subblock_d,
        A_subblock.subblock_indices_local_d,
        A_subblock.count_subblock_h[rank],
        default_stream
    );
}


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
    cublasHandle_t &default_cublasHandle)
{
    int rank = A_distributed.rank;
    int size = A_distributed.size;

    MPI_Request send_subblock_requests[size-1];
    MPI_Request recv_subblock_requests[size-1];
    double alpha = 1.0;
    double beta = 0.0;


    // pack dense sublblock p
    pack_gpu(p_subblock_d + A_subblock.displ_subblock_h[rank],
        p_distributed.vec_d[0],
        A_subblock.subblock_indices_local_d,
        A_subblock.count_subblock_h[rank],
        default_stream);

    if(size > 1){
        cudaErrchk(cudaMemcpyAsync(p_subblock_h + A_subblock.displ_subblock_h[rank],
            p_subblock_d + A_subblock.displ_subblock_h[rank],
            A_subblock.count_subblock_h[rank] * sizeof(double), cudaMemcpyDeviceToHost, default_stream));

        for(int i = 0; i < size-1; i++){
            int dest = (rank + 1 + i) % size;
            MPI_Isend(p_subblock_h + A_subblock.displ_subblock_h[rank], A_subblock.count_subblock_h[rank],
                MPI_DOUBLE, dest, dest, A_distributed.comm, &send_subblock_requests[i]);
        }
        for(int i = 0; i < size-1; i++){
            int source = (rank + 1 + i) % size;
            MPI_Irecv(p_subblock_h + A_subblock.displ_subblock_h[source], A_subblock.count_subblock_h[source],
                MPI_DOUBLE, source, rank, A_distributed.comm, &recv_subblock_requests[i]);
        }
    }

    // post all send requests
    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        pack_gpu(A_distributed.send_buffer_d[i], p_distributed.vec_d[0],
            A_distributed.rows_per_neighbour_d[i], A_distributed.nnz_rows_per_neighbour[i], A_distributed.streams_send[i]);

        cudaErrchk(cudaMemcpyAsync(A_distributed.send_buffer_h[i], A_distributed.send_buffer_d[i],
            A_distributed.nnz_rows_per_neighbour[i] * sizeof(double), cudaMemcpyDeviceToHost, A_distributed.streams_send[i]));
    
        cudaErrchk(cudaEventRecord(A_distributed.events_send[i], A_distributed.streams_send[i]));
    }

    if(size > 1){
        cudaErrchk(cudaStreamSynchronize(default_stream));
        for(int i = 0; i < size-1; i++){
            int dest = (rank + 1 + i) % size;
            MPI_Isend(p_subblock_h + A_subblock.displ_subblock_h[rank], A_subblock.count_subblock_h[rank],
                MPI_DOUBLE, dest, dest, A_distributed.comm, &send_subblock_requests[i]);
        }
        for(int i = 0; i < size-1; i++){
            int source = (rank + 1 + i) % size;
            MPI_Irecv(p_subblock_h + A_subblock.displ_subblock_h[source], A_subblock.count_subblock_h[source],
                MPI_DOUBLE, source, rank, A_distributed.comm, &recv_subblock_requests[i]);
        }
    }


    
    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        int send_idx = p_distributed.neighbours[i];
        int send_tag = std::abs(send_idx-A_distributed.rank);

        cudaErrchk(cudaEventSynchronize(A_distributed.events_send[i]));

        MPI_Isend(A_distributed.send_buffer_h[i], A_distributed.nnz_rows_per_neighbour[i],
            MPI_DOUBLE, send_idx, send_tag, A_distributed.comm, &A_distributed.send_requests[i]);
    }

    for(int i = 0; i < A_distributed.number_of_neighbours; i++){
        // loop over neighbors
        if(i < A_distributed.number_of_neighbours-1){
            int recv_idx = p_distributed.neighbours[i+1];
            int recv_tag = std::abs(recv_idx-A_distributed.rank);
            MPI_Irecv(A_distributed.recv_buffer_h[i+1], A_distributed.nnz_cols_per_neighbour[i+1],
                MPI_DOUBLE, recv_idx, recv_tag, A_distributed.comm, &A_distributed.recv_requests[i+1]);
        }

        // calc A*p
        if(i > 0){
            cudaErrchk(cudaStreamWaitEvent(default_stream, A_distributed.events_recv[i], 0));
        }
        if(i > 0){
            cusparseErrchk(cusparseSpMV(
                default_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                A_distributed.descriptors[i], p_distributed.descriptors[i],
                &alpha, vecAp_local, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, A_distributed.buffer_d[i]));
        }
        else{
            cusparseErrchk(cusparseSpMV(
                default_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                A_distributed.descriptors[i], p_distributed.descriptors[i],
                &beta, vecAp_local, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, A_distributed.buffer_d[i]));
        }

        if(i < A_distributed.number_of_neighbours-1){
            MPI_Wait(&A_distributed.recv_requests[i+1], MPI_STATUS_IGNORE);

            cudaErrchk(cudaMemcpyAsync(A_distributed.recv_buffer_d[i+1], A_distributed.recv_buffer_h[i+1],
                A_distributed.nnz_cols_per_neighbour[i+1] * sizeof(double), cudaMemcpyHostToDevice, A_distributed.streams_recv[i+1]));

            unpack_gpu(p_distributed.vec_d[i+1], A_distributed.recv_buffer_d[i+1],
                A_distributed.cols_per_neighbour_d[i+1], A_distributed.nnz_cols_per_neighbour[i+1], A_distributed.streams_recv[i+1]);
            cudaErrchk(cudaEventRecord(A_distributed.events_recv[i+1], A_distributed.streams_recv[i+1]));

        }
        
    }

    if(size > 1){
        MPI_Waitall(A_distributed.number_of_neighbours-1, &A_distributed.send_requests[1], MPI_STATUSES_IGNORE);
        MPI_Waitall(size-1, recv_subblock_requests, MPI_STATUSES_IGNORE);
        MPI_Waitall(size-1, send_subblock_requests, MPI_STATUSES_IGNORE);
        // recv whole vector
        cudaErrchk(cudaMemcpyAsync(p_subblock_d,
            p_subblock_h, A_subblock.subblock_size * sizeof(double),
            cudaMemcpyHostToDevice, default_stream));
    }

    cublasErrchk(cublasDgemv(
        default_cublasHandle,
        CUBLAS_OP_N,
        A_subblock.count_subblock_h[rank], A_subblock.subblock_size,
        &alpha,
        A_subblock.A_subblock_local_d, A_subblock.count_subblock_h[rank],
        p_subblock_d, 1,
        &beta,
        Ap_subblock_d, 1
    ));
    // unpack and add it to Ap
    unpack_add(
        Ap_local_d,
        Ap_subblock_d,
        A_subblock.subblock_indices_local_d,
        A_subblock.count_subblock_h[rank],
        default_stream
    );
}

} // namespace dspmv_split