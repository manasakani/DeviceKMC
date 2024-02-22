#include "dist_spmv.h"


namespace dspmv{


void gpu_packing(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    cusparseDnVecDescr_t &vecAp_local,
    cudaStream_t &default_stream,
    cusparseHandle_t &default_cusparseHandle)
{

    double alpha = 1.0;
    double beta = 0.0;

    // pinned memory
    // streams_recv
    // stream_send
    cudaEvent_t event_default_finished;
    cudaErrchk(cudaEventCreateWithFlags(&event_default_finished, cudaEventDisableTiming));
    cudaErrchk(cudaEventRecord(event_default_finished, default_stream));


    // post all send requests
    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        cudaErrchk(cudaStreamWaitEvent(A_distributed.streams_send[i], event_default_finished, 0));
        pack_gpu(A_distributed.send_buffer_d[i], p_distributed.vec_d[0],
            A_distributed.rows_per_neighbour_d[i], A_distributed.nnz_rows_per_neighbour[i], A_distributed.streams_send[i]);

        cudaErrchk(cudaMemcpyAsync(A_distributed.send_buffer_h[i], A_distributed.send_buffer_d[i],
            A_distributed.nnz_rows_per_neighbour[i] * sizeof(double), cudaMemcpyDeviceToHost, A_distributed.streams_send[i]));
    
        cudaErrchk(cudaEventRecord(A_distributed.events_send[i], A_distributed.streams_send[i]));
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
    MPI_Waitall(A_distributed.number_of_neighbours-1, &A_distributed.send_requests[1], MPI_STATUSES_IGNORE);

    cudaErrchk(cudaEventDestroy(event_default_finished));
}


void gpu_packing_cam(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    cusparseDnVecDescr_t &vecAp_local,
    cudaStream_t &default_stream,
    cusparseHandle_t &default_cusparseHandle)
{

    double alpha = 1.0;
    double beta = 0.0;

    // pinned memory
    // streams_recv
    // stream_send
    // cuda aware mpi
    cudaEvent_t event_default_finished;
    cudaErrchk(cudaEventCreateWithFlags(&event_default_finished, cudaEventDisableTiming));
    cudaErrchk(cudaEventRecord(event_default_finished, default_stream));


    // post all send requests
    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        cudaErrchk(cudaStreamWaitEvent(A_distributed.streams_send[i], event_default_finished, 0));
        pack_gpu(A_distributed.send_buffer_d[i], p_distributed.vec_d[0],
            A_distributed.rows_per_neighbour_d[i], A_distributed.nnz_rows_per_neighbour[i], A_distributed.streams_send[i]);

        cudaErrchk(cudaEventRecord(A_distributed.events_send[i], A_distributed.streams_send[i]));
    }
    

    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        int send_idx = p_distributed.neighbours[i];
        int send_tag = std::abs(send_idx-A_distributed.rank);

        cudaErrchk(cudaEventSynchronize(A_distributed.events_send[i]));

        MPI_Isend(A_distributed.send_buffer_d[i], A_distributed.nnz_rows_per_neighbour[i],
            MPI_DOUBLE, send_idx, send_tag, A_distributed.comm, &A_distributed.send_requests[i]);
    }

    for(int i = 0; i < A_distributed.number_of_neighbours; i++){
        // loop over neighbors
        if(i < A_distributed.number_of_neighbours-1){
            int recv_idx = p_distributed.neighbours[i+1];
            int recv_tag = std::abs(recv_idx-A_distributed.rank);
            MPI_Irecv(A_distributed.recv_buffer_d[i+1], A_distributed.nnz_cols_per_neighbour[i+1],
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

            unpack_gpu(p_distributed.vec_d[i+1], A_distributed.recv_buffer_d[i+1],
                A_distributed.cols_per_neighbour_d[i+1], A_distributed.nnz_cols_per_neighbour[i+1], A_distributed.streams_recv[i+1]);
            cudaErrchk(cudaEventRecord(A_distributed.events_recv[i+1], A_distributed.streams_recv[i+1]));

        }
        
    }
    MPI_Waitall(A_distributed.number_of_neighbours-1, &A_distributed.send_requests[1], MPI_STATUSES_IGNORE);

    cudaErrchk(cudaEventDestroy(event_default_finished));
}

} // namespace dspmv