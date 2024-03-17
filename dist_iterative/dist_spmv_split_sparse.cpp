#include "dist_spmv.h"

namespace dspmv_split_sparse{

void spmm_split_sparse1(
    Distributed_subblock_sparse &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    rocsparse_dnvec_descr &vecp_subblock,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    rocsparse_dnvec_descr &vecAp_subblock,
    hipsparseDnVecDescr_t &vecAp_local,
    double *Ap_local_d,
    hipStream_t &default_stream,
    hipsparseHandle_t &default_cusparseHandle,
    rocsparse_handle &default_rocsparseHandle)
{
    // Isend Irecv subblock
    // sparse part
    //gemv

    int rank = A_distributed.rank;
    int size = A_distributed.size;

    double alpha = 1.0;
    double beta = 0.0;

    // pack dense sublblock p
    pack_gpu(p_subblock_d + A_subblock.displ_subblock_h[rank],
        p_distributed.vec_d[0],
        A_subblock.subblock_indices_local_d,
        A_subblock.count_subblock_h[rank],
        default_stream);

    if(size > 1){
        cudaErrchk(hipMemcpy(p_subblock_h + A_subblock.displ_subblock_h[rank],
            p_subblock_d + A_subblock.displ_subblock_h[rank],
            A_subblock.count_subblock_h[rank] * sizeof(double), hipMemcpyDeviceToHost));
        for(int i = 0; i < size-1; i++){
            int dest = (rank + 1 + i) % size;
            MPI_Isend(p_subblock_h + A_subblock.displ_subblock_h[rank], A_subblock.count_subblock_h[rank],
                MPI_DOUBLE, dest, dest, A_distributed.comm, &A_subblock.send_subblock_requests[i]);
        }
        for(int i = 0; i < size-1; i++){
            int source = (rank + 1 + i) % size;
            MPI_Irecv(p_subblock_h + A_subblock.displ_subblock_h[source], A_subblock.count_subblock_h[source],
                MPI_DOUBLE, source, rank, A_distributed.comm, &A_subblock.recv_subblock_requests[i]);
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
        MPI_Waitall(size-1, A_subblock.recv_subblock_requests, MPI_STATUSES_IGNORE);
        MPI_Waitall(size-1, A_subblock.send_subblock_requests, MPI_STATUSES_IGNORE);
        // recv whole vector
        cudaErrchk(hipMemcpyAsync(p_subblock_d,
            p_subblock_h, A_subblock.subblock_size * sizeof(double),
            hipMemcpyHostToDevice, default_stream));
    }

    // cusparseErrchk(hipsparseSpMV(
    //     default_cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
    //     *A_subblock.descriptor, vecp_subblock,
    //     &beta, vecAp_subblock, HIP_R_64F, HIPSPARSE_CSRMV_ALG2, A_subblock.buffer_d));

    rocsparse_spmv(
        default_rocsparseHandle, rocsparse_operation_none, &alpha,
        *A_subblock.descriptor, vecp_subblock,
        &beta, vecAp_subblock, rocsparse_datatype_f64_r,
        A_subblock.algo,
        A_subblock.buffersize,
        A_subblock.buffer_d);


    // unpack and add it to Ap
    unpack_add(
        Ap_local_d,
        Ap_subblock_d,
        A_subblock.subblock_indices_local_d,
        A_subblock.count_subblock_h[rank],
        default_stream
    );
}


void spmm_split_sparse2(
    Distributed_subblock_sparse &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    rocsparse_dnvec_descr &vecp_subblock,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    rocsparse_dnvec_descr &vecAp_subblock,
    hipsparseDnVecDescr_t &vecAp_local,
    double *Ap_local_d,
    hipStream_t &default_stream,
    hipsparseHandle_t &default_cusparseHandle,
    rocsparse_handle &default_rocsparseHandle)
{
    int rank = A_distributed.rank;
    int size = A_distributed.size;

    double alpha = 1.0;
    double beta = 0.0;

    cudaErrchk(hipEventRecord(A_distributed.event_default_finished, default_stream));

    // pack dense sublblock p
    pack_gpu(p_subblock_d + A_subblock.displ_subblock_h[rank],
        p_distributed.vec_d[0],
        A_subblock.subblock_indices_local_d,
        A_subblock.count_subblock_h[rank],
        default_stream);

    if(size > 1){
        cudaErrchk(hipMemcpyAsync(p_subblock_h + A_subblock.displ_subblock_h[rank],
            p_subblock_d + A_subblock.displ_subblock_h[rank],
            A_subblock.count_subblock_h[rank] * sizeof(double), hipMemcpyDeviceToHost, default_stream));
    }

    // post all send requests
    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        cudaErrchk(hipStreamWaitEvent(A_distributed.streams_send[i], A_distributed.event_default_finished, 0));
        pack_gpu(A_distributed.send_buffer_d[i], p_distributed.vec_d[0],
            A_distributed.rows_per_neighbour_d[i], A_distributed.nnz_rows_per_neighbour[i], A_distributed.streams_send[i]);

        cudaErrchk(hipMemcpyAsync(A_distributed.send_buffer_h[i], A_distributed.send_buffer_d[i],
            A_distributed.nnz_rows_per_neighbour[i] * sizeof(double), hipMemcpyDeviceToHost, A_distributed.streams_send[i]));
    
        cudaErrchk(hipEventRecord(A_distributed.events_send[i], A_distributed.streams_send[i]));
    }

    if(size > 1){
        cudaErrchk(hipStreamSynchronize(default_stream));
        for(int i = 0; i < size-1; i++){
            int dest = (rank + 1 + i) % size;
            MPI_Isend(p_subblock_h + A_subblock.displ_subblock_h[rank], A_subblock.count_subblock_h[rank],
                MPI_DOUBLE, dest, dest, A_distributed.comm, &A_subblock.send_subblock_requests[i]);
        }
        for(int i = 0; i < size-1; i++){
            int source = (rank + 1 + i) % size;
            MPI_Irecv(p_subblock_h + A_subblock.displ_subblock_h[source], A_subblock.count_subblock_h[source],
                MPI_DOUBLE, source, rank, A_distributed.comm, &A_subblock.recv_subblock_requests[i]);
        }
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
            cusparseErrchk(hipsparseSpMV(
                default_cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                A_distributed.descriptors[i], p_distributed.descriptors[i],
                &alpha, vecAp_local, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, A_distributed.buffer_d[i]));
        }
        else{
            cusparseErrchk(hipsparseSpMV(
                default_cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                A_distributed.descriptors[i], p_distributed.descriptors[i],
                &beta, vecAp_local, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, A_distributed.buffer_d[i]));
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

    if(size > 1){
        MPI_Waitall(A_distributed.number_of_neighbours-1, &A_distributed.send_requests[1], MPI_STATUSES_IGNORE);
        MPI_Waitall(size-1, A_subblock.recv_subblock_requests, MPI_STATUSES_IGNORE);
        MPI_Waitall(size-1, A_subblock.send_subblock_requests, MPI_STATUSES_IGNORE);
        // recv whole vector
        cudaErrchk(hipMemcpyAsync(p_subblock_d,
            p_subblock_h, A_subblock.subblock_size * sizeof(double),
            hipMemcpyHostToDevice, default_stream));
    }

    // cusparseErrchk(hipsparseSpMV(
    //     default_cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
    //     *A_subblock.descriptor, vecp_subblock,
    //     &beta, vecAp_subblock, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, A_subblock.buffer_d));

    rocsparse_spmv(
        default_rocsparseHandle, rocsparse_operation_none, &alpha,
        *A_subblock.descriptor, vecp_subblock,
        &beta, vecAp_subblock, rocsparse_datatype_f64_r,
        A_subblock.algo,
        A_subblock.buffersize,
        A_subblock.buffer_d);

    // unpack and add it to Ap
    unpack_add(
        Ap_local_d,
        Ap_subblock_d,
        A_subblock.subblock_indices_local_d,
        A_subblock.count_subblock_h[rank],
        default_stream
    );
}


void spmm_split_sparse3(
    Distributed_subblock_sparse &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    rocsparse_dnvec_descr &vecp_subblock,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    rocsparse_dnvec_descr &vecAp_subblock,
    hipsparseDnVecDescr_t &vecAp_local,
    double *Ap_local_d,
    hipStream_t &default_stream,
    hipsparseHandle_t &default_cusparseHandle,
    rocsparse_handle &default_rocsparseHandle)
{
    // Isend Irecv subblock
    // sparse part
    //gemv

    int rank = A_distributed.rank;
    int size = A_distributed.size;

    double alpha = 1.0;
    double beta = 0.0;
    // pack dense sublblock p
    pack_gpu(p_subblock_d + A_subblock.displ_subblock_h[rank],
        p_distributed.vec_d[0],
        A_subblock.subblock_indices_local_d,
        A_subblock.count_subblock_h[rank],
        default_stream);

    #pragma omp parallel sections num_threads(2)
    {
    #pragma omp section
    {
        if(size > 1){
            cudaErrchk(hipMemcpy(p_subblock_h + A_subblock.displ_subblock_h[rank],
                p_subblock_d + A_subblock.displ_subblock_h[rank],
                A_subblock.count_subblock_h[rank] * sizeof(double), hipMemcpyDeviceToHost));

            MPI_Allgatherv(MPI_IN_PLACE, A_subblock.count_subblock_h[rank],
                MPI_DOUBLE,
                p_subblock_h,
                A_subblock.count_subblock_h,
                A_subblock.displ_subblock_h,
                MPI_DOUBLE, A_distributed.comm);
            // recv whole vector
            cudaErrchk(hipMemcpyAsync(p_subblock_d,
                p_subblock_h, A_subblock.subblock_size * sizeof(double),
                hipMemcpyHostToDevice, default_stream));                    
        }
        // cusparseErrchk(hipsparseSpMV(
        //     default_cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
        //     *A_subblock.descriptor, vecp_subblock,
        //     &beta, vecAp_subblock, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, A_subblock.buffer_d));

        rocsparse_spmv(
            default_rocsparseHandle, rocsparse_operation_none, &alpha,
            *A_subblock.descriptor, vecp_subblock,
            &beta, vecAp_subblock, rocsparse_datatype_f64_r,
            A_subblock.algo,
            A_subblock.buffersize,
            A_subblock.buffer_d);
    }
    #pragma omp section
    {
        dspmv::gpu_packing(
            A_distributed,
            p_distributed,
            vecAp_local,
            default_stream,
            default_cusparseHandle
        );        
    }
    }
    // unpack and add it to Ap
    unpack_add(
        Ap_local_d,
        Ap_subblock_d,
        A_subblock.subblock_indices_local_d,
        A_subblock.count_subblock_h[rank],
        default_stream
    );
}


void spmm_split_sparse4(
    Distributed_subblock_sparse &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    rocsparse_dnvec_descr &vecp_subblock,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    rocsparse_dnvec_descr &vecAp_subblock,
    hipsparseDnVecDescr_t &vecAp_local,
    double *Ap_local_d,
    hipStream_t &default_stream,
    hipsparseHandle_t &default_cusparseHandle,
    rocsparse_handle &default_rocsparseHandle)
{
    // Isend Irecv subblock
    // sparse part
    //gemv

    int rank = A_distributed.rank;
    int size = A_distributed.size;

    double alpha = 1.0;
    double beta = 0.0;
    // pack dense sublblock p
    pack_gpu(p_subblock_d + A_subblock.displ_subblock_h[rank],
        p_distributed.vec_d[0],
        A_subblock.subblock_indices_local_d,
        A_subblock.count_subblock_h[rank],
        default_stream);
    if(size > 1){
        cudaErrchk(hipMemcpy(p_subblock_h + A_subblock.displ_subblock_h[rank],
            p_subblock_d + A_subblock.displ_subblock_h[rank],
            A_subblock.count_subblock_h[rank] * sizeof(double), hipMemcpyDeviceToHost));
    }

    #pragma omp parallel sections num_threads(2)
    {
    #pragma omp section
    {
        if(size > 1){
            MPI_Allgatherv(MPI_IN_PLACE, A_subblock.count_subblock_h[rank],
                MPI_DOUBLE,
                p_subblock_h,
                A_subblock.count_subblock_h,
                A_subblock.displ_subblock_h,
                MPI_DOUBLE, A_distributed.comm);
            // recv whole vector
            cudaErrchk(hipMemcpyAsync(p_subblock_d,
                p_subblock_h, A_subblock.subblock_size * sizeof(double),
                hipMemcpyHostToDevice, default_stream));                    
        }

    }
    #pragma omp section
    {
        dspmv::gpu_packing(
            A_distributed,
            p_distributed,
            vecAp_local,
            default_stream,
            default_cusparseHandle
        );        
    }
    }

    // cusparseErrchk(hipsparseSpMV(
    //     default_cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
    //     *A_subblock.descriptor, vecp_subblock,
    //     &beta, vecAp_subblock, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, A_subblock.buffer_d));
    
    rocsparse_spmv(
        default_rocsparseHandle, rocsparse_operation_none, &alpha,
        *A_subblock.descriptor, vecp_subblock,
        &beta, vecAp_subblock, rocsparse_datatype_f64_r,
        A_subblock.algo,
        A_subblock.buffersize,
        A_subblock.buffer_d);

    // unpack and add it to Ap
    unpack_add(
        Ap_local_d,
        Ap_subblock_d,
        A_subblock.subblock_indices_local_d,
        A_subblock.count_subblock_h[rank],
        default_stream
    );
}

struct thread_input{
    double *p_subblock_d;
    double *p_subblock_h;
    Distributed_subblock_sparse *A_subblock;
    MPI_Comm *comm;
    hipStream_t *default_stream;
    int rank;
    int size;
};

void *thread_fnc(void *arg){
    thread_input *input = (thread_input *)arg;
    cudaErrchk(hipMemcpy(input->p_subblock_h + input->A_subblock->displ_subblock_h[input->rank],
        input->p_subblock_d + input->A_subblock->displ_subblock_h[input->rank],
        input->A_subblock->count_subblock_h[input->rank] * sizeof(double), hipMemcpyDeviceToHost));

    MPI_Allgatherv(MPI_IN_PLACE, input->A_subblock->count_subblock_h[input->rank],
        MPI_DOUBLE,
        input->p_subblock_h,
        input->A_subblock->count_subblock_h,
        input->A_subblock->displ_subblock_h,
        MPI_DOUBLE, *input->comm);
    
    cudaErrchk(hipMemcpyAsync(input->p_subblock_d,
        input->p_subblock_h, input->A_subblock->subblock_size * sizeof(double),
        hipMemcpyHostToDevice, *input->default_stream));

    pthread_exit(NULL);
}

void spmm_split_sparse5(
    Distributed_subblock_sparse &A_subblock,
    Distributed_matrix &A_distributed,    
    double *p_subblock_d,
    double *p_subblock_h,
    rocsparse_dnvec_descr &vecp_subblock,
    Distributed_vector &p_distributed,
    double *Ap_subblock_d,
    rocsparse_dnvec_descr &vecAp_subblock,
    hipsparseDnVecDescr_t &vecAp_local,
    double *Ap_local_d,
    hipStream_t &default_stream,
    hipsparseHandle_t &default_cusparseHandle,
    rocsparse_handle &default_rocsparseHandle)
{
    // Isend Irecv subblock
    // sparse part
    //gemv

    int rank = A_distributed.rank;
    int size = A_distributed.size;

    double alpha = 1.0;
    double beta = 0.0;
    pthread_t allgatherv_thread;

    // pack dense sublblock p
    pack_gpu(p_subblock_d + A_subblock.displ_subblock_h[rank],
        p_distributed.vec_d[0],
        A_subblock.subblock_indices_local_d,
        A_subblock.count_subblock_h[rank],
        default_stream);



    if(size > 1){
        thread_input input;
        input.p_subblock_d = p_subblock_d;
        input.p_subblock_h = p_subblock_h;
        input.A_subblock = &A_subblock;
        input.comm = &A_distributed.comm;
        input.default_stream = &default_stream;
        input.rank = rank;
        input.size = size;

        pthread_create(&allgatherv_thread, NULL, thread_fnc, &input);
    }


    dspmv::gpu_packing(
        A_distributed,
        p_distributed,
        vecAp_local,
        default_stream,
        default_cusparseHandle
    ); 

    if(size > 1){
        pthread_join(allgatherv_thread, NULL);
    }

    // cusparseErrchk(hipsparseSpMV(
    //     default_cusparseHandle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
    //     *A_subblock.descriptor, vecp_subblock,
    //     &beta, vecAp_subblock, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, A_subblock.buffer_d));

    rocsparse_spmv(
        default_rocsparseHandle, rocsparse_operation_none, &alpha,
        *A_subblock.descriptor, vecp_subblock,
        &beta, vecAp_subblock, rocsparse_datatype_f64_r,
        A_subblock.algo,
        A_subblock.buffersize,
        A_subblock.buffer_d);
            
    // unpack and add it to Ap
    unpack_add(
        Ap_local_d,
        Ap_subblock_d,
        A_subblock.subblock_indices_local_d,
        A_subblock.count_subblock_h[rank],
        default_stream
    );
}

} // namespace dspmv_split_sparse