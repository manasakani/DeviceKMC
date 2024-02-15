#include "mv_own_implementations.h"

namespace own_mv
{

void distributed_mv_point_to_point1(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    cusparseDnVecDescr_t &vecAp_local,
    cudaStream_t &default_stream,
    cusparseHandle_t &default_cusparseHandle)
{

    double alpha = 1.0;
    double beta = 0.0;
    // pinned memory
    if(A_distributed.size > 1){
        cudaErrchk(cudaMemcpy(p_distributed.vec_h[0], p_distributed.vec_d[0],
            A_distributed.rows_this_rank * sizeof(double), cudaMemcpyDeviceToHost));
    }


    // post all send requests
    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        int send_idx = p_distributed.neighbours[i];
        int send_tag = std::abs(send_idx - A_distributed.rank);
        MPI_Isend(p_distributed.vec_h[0], p_distributed.rows_this_rank,
                    MPI_DOUBLE, send_idx, send_tag, A_distributed.comm, &A_distributed.send_requests[i]);
    }

    for(int i = 0; i < A_distributed.number_of_neighbours; i++){
        // loop over neighbors
        if(i < A_distributed.number_of_neighbours-1){
            int recv_idx = p_distributed.neighbours[i+1];
            int recv_tag = std::abs(recv_idx - A_distributed.rank);
            MPI_Irecv(p_distributed.vec_h[i+1], p_distributed.counts[recv_idx],
                        MPI_DOUBLE, recv_idx, recv_tag, A_distributed.comm, &A_distributed.recv_requests[i+1]);
        }

        // calc A*p
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
            int neighbour_idx = p_distributed.neighbours[i+1];
            cudaErrchk(cudaMemcpyAsync(p_distributed.vec_d[i+1], p_distributed.vec_h[i+1], p_distributed.counts[neighbour_idx] * sizeof(double), cudaMemcpyHostToDevice, default_stream));


        }
        
    }
    MPI_Waitall(A_distributed.number_of_neighbours-1, &A_distributed.send_requests[1], MPI_STATUSES_IGNORE);


}

void distributed_mv_point_to_point2(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    cusparseDnVecDescr_t &vecAp_local,
    cudaStream_t &default_stream,
    cusparseHandle_t &default_cusparseHandle)
{

    double alpha = 1.0;
    double beta = 0.0;

    // pinned memory
    // cudaaware mpi
    cudaErrchk(cudaStreamSynchronize(default_stream));

    // post all send requests
    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        int send_idx = p_distributed.neighbours[i];
        int send_tag = std::abs(send_idx - A_distributed.rank);
        MPI_Isend(p_distributed.vec_d[0], p_distributed.rows_this_rank,
                    MPI_DOUBLE, send_idx, send_tag, A_distributed.comm,
                    &A_distributed.send_requests[i]);
    }

    for(int i = 0; i < A_distributed.number_of_neighbours; i++){
        // loop over neighbors
        if(i < A_distributed.number_of_neighbours-1){
            int recv_idx = p_distributed.neighbours[i+1];
            int recv_tag = std::abs(recv_idx - A_distributed.rank);
            MPI_Irecv(p_distributed.vec_d[i+1], p_distributed.counts[recv_idx],
                        MPI_DOUBLE, recv_idx, recv_tag, A_distributed.comm, &A_distributed.recv_requests[i+1]);
        }

        // calc A*p
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
        }
        
    }
    MPI_Waitall(A_distributed.number_of_neighbours-1, &A_distributed.send_requests[1], MPI_STATUSES_IGNORE);


}


void distributed_mv_point_to_point3(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    cusparseDnVecDescr_t &vecAp_local,
    cudaStream_t &default_stream,
    cusparseHandle_t &default_cusparseHandle)
{

    double alpha = 1.0;
    double beta = 0.0;

    // pinned memory
    // streams
    if(A_distributed.size > 1){
        cudaErrchk(cudaMemcpy(p_distributed.vec_h[0], p_distributed.vec_d[0],
            A_distributed.rows_this_rank * sizeof(double), cudaMemcpyDeviceToHost));
    }
    
    // post all send requests
    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        int send_idx = p_distributed.neighbours[i];
        int send_tag = std::abs(send_idx - A_distributed.rank);
        MPI_Isend(p_distributed.vec_h[0], p_distributed.rows_this_rank,
                    MPI_DOUBLE, send_idx, send_tag, A_distributed.comm, &A_distributed.send_requests[i]);
    }

    for(int i = 0; i < A_distributed.number_of_neighbours; i++){
        // loop over neighbors
        if(i < A_distributed.number_of_neighbours-1){
            int recv_idx = p_distributed.neighbours[i+1];
            int recv_tag = std::abs(recv_idx - A_distributed.rank);
            MPI_Irecv(p_distributed.vec_h[i+1], p_distributed.counts[recv_idx],
                        MPI_DOUBLE, recv_idx, recv_tag, A_distributed.comm, &A_distributed.recv_requests[i+1]);
        }

        if(i > 0){
            cudaErrchk(cudaStreamWaitEvent(default_stream, A_distributed.events_recv[i], 0));
        }

        // calc A*p
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
            int neighbour_idx = p_distributed.neighbours[i+1];
            cudaErrchk(cudaMemcpyAsync(p_distributed.vec_d[i+1], p_distributed.vec_h[i+1],
            p_distributed.counts[neighbour_idx] * sizeof(double),
            cudaMemcpyHostToDevice, A_distributed.streams_recv[i+1]));
            cudaErrchk(cudaEventRecord(A_distributed.events_recv[i+1], A_distributed.streams_recv[i+1]));


        }
        
    }
    MPI_Waitall(A_distributed.number_of_neighbours-1, &A_distributed.send_requests[1], MPI_STATUSES_IGNORE);


}

} // namespace own_mv