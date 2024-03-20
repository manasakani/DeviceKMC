#include "dist_spmv.h"


namespace dspmv{


void gpu_packing(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    rocsparse_dnvec_descr &vecAp_local,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle)
{

    double alpha = 1.0;
    double beta = 0.0;

    // pinned memory
    // streams_recv
    // stream_send
    cudaErrchk(hipEventRecord(A_distributed.event_default_finished, default_stream));

    // post all send requests
    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        cudaErrchk(hipStreamWaitEvent(A_distributed.streams_send[i], A_distributed.event_default_finished, 0));
        pack_gpu(A_distributed.send_buffer_d[i], p_distributed.vec_d[0],
            A_distributed.rows_per_neighbour_d[i], A_distributed.nnz_rows_per_neighbour[i], A_distributed.streams_send[i]);

        cudaErrchk(hipMemcpyAsync(A_distributed.send_buffer_h[i], A_distributed.send_buffer_d[i],
            A_distributed.nnz_rows_per_neighbour[i] * sizeof(double), hipMemcpyDeviceToHost, A_distributed.streams_send[i]));
    
        cudaErrchk(hipEventRecord(A_distributed.events_send[i], A_distributed.streams_send[i]));
    }
    
    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        int send_idx = p_distributed.neighbours[i];
        int send_tag = std::abs(send_idx-A_distributed.rank);

        cudaErrchk(hipEventSynchronize(A_distributed.events_send[i]));

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
            cudaErrchk(hipStreamWaitEvent(default_stream, A_distributed.events_recv[i], 0));
        }
        if(i > 0){
            // cusparseErrchk(hipsparseSpMV(
            //     default_cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            //     A_distributed.descriptors[i], p_distributed.descriptors[i],
            //     &alpha, vecAp_local, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, A_distributed.buffer_d[i]));

            rocsparse_spmv(
                default_rocsparseHandle, rocsparse_operation_none, &alpha,
                A_distributed.descriptors[i], p_distributed.descriptors[i],
                &alpha, vecAp_local, rocsparse_datatype_f64_r,
                A_distributed.algo,
                &A_distributed.buffer_size[i],
                A_distributed.buffer_d[i]);

        }
        else{
            // cusparseErrchk(hipsparseSpMV(
            //     default_cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            //     A_distributed.descriptors[i], p_distributed.descriptors[i],
            //     &beta, vecAp_local, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, A_distributed.buffer_d[i]));

            rocsparse_spmv(
                default_rocsparseHandle, rocsparse_operation_none, &alpha,
                A_distributed.descriptors[i], p_distributed.descriptors[i],
                &beta, vecAp_local, rocsparse_datatype_f64_r,
                A_distributed.algo,
                &A_distributed.buffer_size[i],
                A_distributed.buffer_d[i]);
        }

        if(i < A_distributed.number_of_neighbours-1){
            MPI_Wait(&A_distributed.recv_requests[i+1], MPI_STATUS_IGNORE);

            cudaErrchk(hipMemcpyAsync(A_distributed.recv_buffer_d[i+1], A_distributed.recv_buffer_h[i+1],
                A_distributed.nnz_cols_per_neighbour[i+1] * sizeof(double), hipMemcpyHostToDevice, A_distributed.streams_recv[i+1]));

            unpack_gpu(p_distributed.vec_d[i+1], A_distributed.recv_buffer_d[i+1],
                A_distributed.cols_per_neighbour_d[i+1], A_distributed.nnz_cols_per_neighbour[i+1], A_distributed.streams_recv[i+1]);
            cudaErrchk(hipEventRecord(A_distributed.events_recv[i+1], A_distributed.streams_recv[i+1]));

        }
        
    }
    MPI_Waitall(A_distributed.number_of_neighbours-1, &A_distributed.send_requests[1], MPI_STATUSES_IGNORE);

}


void gpu_packing_cam(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    rocsparse_dnvec_descr &vecAp_local,
    hipStream_t &default_stream,
    rocsparse_handle &default_rocsparseHandle)
{

    double alpha = 1.0;
    double beta = 0.0;

    // pinned memory
    // streams_recv
    // stream_send
    // cuda aware mpi
    cudaErrchk(hipEventRecord(A_distributed.event_default_finished, default_stream));

    // post all send requests
    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        cudaErrchk(hipStreamWaitEvent(A_distributed.streams_send[i], A_distributed.event_default_finished, 0));
        pack_gpu(A_distributed.send_buffer_d[i], p_distributed.vec_d[0],
            A_distributed.rows_per_neighbour_d[i], A_distributed.nnz_rows_per_neighbour[i], A_distributed.streams_send[i]);

        cudaErrchk(hipEventRecord(A_distributed.events_send[i], A_distributed.streams_send[i]));
    }
    

    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        int send_idx = p_distributed.neighbours[i];
        int send_tag = std::abs(send_idx-A_distributed.rank);

        cudaErrchk(hipEventSynchronize(A_distributed.events_send[i]));

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
            cudaErrchk(hipStreamWaitEvent(default_stream, A_distributed.events_recv[i], 0));
        }
        if(i > 0){
            // cusparseErrchk(hipsparseSpMV(
            //     default_cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            //     A_distributed.descriptors[i], p_distributed.descriptors[i],
            //     &alpha, vecAp_local, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, A_distributed.buffer_d[i]));

            rocsparse_spmv(
                default_rocsparseHandle, rocsparse_operation_none, &alpha,
                A_distributed.descriptors[i], p_distributed.descriptors[i],
                &alpha, vecAp_local, rocsparse_datatype_f64_r,
                A_distributed.algo,
                &A_distributed.buffer_size[i],
                A_distributed.buffer_d[i]);

        }
        else{
            // cusparseErrchk(hipsparseSpMV(
            //     default_cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            //     A_distributed.descriptors[i], p_distributed.descriptors[i],
            //     &beta, vecAp_local, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, A_distributed.buffer_d[i]));

            rocsparse_spmv(
                default_rocsparseHandle, rocsparse_operation_none, &alpha,
                A_distributed.descriptors[i], p_distributed.descriptors[i],
                &beta, vecAp_local, rocsparse_datatype_f64_r,
                A_distributed.algo,
                &A_distributed.buffer_size[i],
                A_distributed.buffer_d[i]);

        }

        if(i < A_distributed.number_of_neighbours-1){
            MPI_Wait(&A_distributed.recv_requests[i+1], MPI_STATUS_IGNORE);

            unpack_gpu(p_distributed.vec_d[i+1], A_distributed.recv_buffer_d[i+1],
                A_distributed.cols_per_neighbour_d[i+1], A_distributed.nnz_cols_per_neighbour[i+1], A_distributed.streams_recv[i+1]);
            cudaErrchk(hipEventRecord(A_distributed.events_recv[i+1], A_distributed.streams_recv[i+1]));

        }
        
    }
    MPI_Waitall(A_distributed.number_of_neighbours-1, &A_distributed.send_requests[1], MPI_STATUSES_IGNORE);

}

} // namespace dspmv