#include "own_cg_to_compare.h"


namespace own_test{

void solve_cg_nonblocking_point_to_point(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    int *steps_taken,
    double *time_taken)
{
    MPI_Barrier(comm);
    std::printf("CG starts\n");

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);


    // prepare for allgatherv
    int counts[size];
    int displacements[size];
    int rows_per_rank = matrix_size / size;    
    split_matrix(matrix_size, size, counts, displacements);

    int row_start_index = displacements[rank];
    rows_per_rank = counts[rank];




    double *p_local_h = new double[rows_per_rank];
    double *p_h = new double[matrix_size];
    int *row_indptr_local_h = new int[rows_per_rank+1];
    double *r_local_h = new double[rows_per_rank];
    for (int i = 0; i < rows_per_rank+1; ++i) {
        row_indptr_local_h[i] = row_indptr_h[i+row_start_index] - row_indptr_h[row_start_index];
    }
    for (int i = 0; i < rows_per_rank; ++i) {
        r_local_h[i] = r_h[i+row_start_index];
        p_local_h[i] = starting_guess_h[i+row_start_index];
    }

    int nnz_local = row_indptr_local_h[rows_per_rank];

    int *col_indices_local_h = new int[nnz_local];
    double *data_local_h = new double[nnz_local];

    for (int i = 0; i < nnz_local; ++i) {
        col_indices_local_h[i] = col_indices_h[i+row_indptr_h[row_start_index]];
        data_local_h[i] = data_h[i+row_indptr_h[row_start_index]];
    }


    // initialize cuda
    cudaStream_t stream = NULL;
    cublasHandle_t cublasHandle = 0;
    cublasErrchk(cublasCreate(&cublasHandle));
    
    cusparseHandle_t cusparseHandle = 0;
    cusparseErrchk(cusparseCreate(&cusparseHandle));    


    cudaErrchk(cudaStreamCreate(&stream));
    cusparseErrchk(cusparseSetStream(cusparseHandle, stream));
    cublasErrchk(cublasSetStream(cublasHandle, stream));


    cusparseSpMatDescr_t matA_local = NULL;


    double a, b, na;
    double alpha, beta, alpham1, r0, r_norm2;
    size_t bufferSize = 0;
    void *buffer = NULL;

    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.0;

    double *r_norm2_d = NULL;
    double *dot_d = NULL;
    cudaErrchk(cudaMalloc((void**)&r_norm2_d, sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&dot_d, sizeof(double)));

    //allocate memory on device
    int *row_indptr_local_d = NULL;
    int *col_indices_local_d = NULL;
    double *data_local_d = NULL;
    double *r_local_d = NULL;
    double *x_local_d = NULL;
    double *p_d = NULL;
    double dot_h;
    double *starting_guess_local_h = starting_guess_h + row_start_index;


    cudaErrchk(cudaMalloc((void **)&p_d, matrix_size * sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&row_indptr_local_d, (rows_per_rank+1)*sizeof(int)));
    cudaErrchk(cudaMalloc((void**)&col_indices_local_d, nnz_local*sizeof(int)));
    cudaErrchk(cudaMalloc((void**)&data_local_d, nnz_local*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&r_local_d, rows_per_rank*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&x_local_d, rows_per_rank*sizeof(double)));


    cudaErrchk(cudaMemcpy(row_indptr_local_d, row_indptr_local_h, (rows_per_rank+1)*sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(col_indices_local_d, col_indices_local_h, nnz_local*sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(data_local_d, data_local_h, nnz_local*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(r_local_d, r_local_h, rows_per_rank*sizeof(double), cudaMemcpyHostToDevice));
    //copy data to device
    // starting guess for x
    cudaErrchk(cudaMemcpy(x_local_d, starting_guess_h + row_start_index, rows_per_rank * sizeof(double), cudaMemcpyHostToDevice));


    /* Wrap raw data into cuSPARSE generic API objects */
    cusparseErrchk(cusparseCreateCsr(&matA_local, rows_per_rank, matrix_size,
                                        nnz_local, row_indptr_local_d, col_indices_local_d, data_local_d,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));



    cusparseDnVecDescr_t vecp = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecp, matrix_size, p_d, CUDA_R_64F));

    double *p_local_d = NULL;
    double *Ax_local_d = NULL;
    cudaErrchk(cudaMalloc((void **)&p_local_d, rows_per_rank * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&Ax_local_d, rows_per_rank * sizeof(double)));
    cudaErrchk(cudaMemset(Ax_local_d, 0, rows_per_rank * sizeof(double)));

    cusparseDnVecDescr_t vecAp_local = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecAp_local, rows_per_rank, Ax_local_d, CUDA_R_64F));




    //figure out extra amount of memory needed
    cusparseErrchk(cusparseSpMV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA_local, vecp,
        &beta, vecAp_local, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    cudaErrchk(cudaMalloc(&buffer, bufferSize));


    // create distributed matrix
    Distributed_matrix A_distributed(
        matrix_size,
        nnz_local,
        counts,
        displacements,
        col_indices_local_h,
        row_indptr_local_h,
        data_local_h,
        MPI_COMM_WORLD,
        cusparseHandle
    );

    Distributed_vector p_distributed(
        matrix_size,
        counts,
        displacements,
        A_distributed.number_of_neighbours,
        A_distributed.neighbours,
        MPI_COMM_WORLD,
        cusparseHandle
    );


    //begin CG
    std::printf("CG starts\n");
    cudaErrchk(cudaStreamSynchronize(stream));
    cudaErrchk(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    time_taken[0] = -omp_get_wtime();

    // norm of rhs for convergence check
    double norm2_rhs = 0;
    cublasErrchk(cublasDdot(cublasHandle, rows_per_rank, r_local_d, 1, r_local_d, 1, &norm2_rhs));
    //allreduce
    MPI_Allreduce(MPI_IN_PLACE, &norm2_rhs, 1, MPI_DOUBLE, MPI_SUM, comm);


    // MPI_Allgatherv(p_local_h, rows_per_rank, MPI_DOUBLE, p_h, counts, displacements, MPI_DOUBLE, comm);

    cudaErrchk(cudaMemcpy(p_distributed.vec_d[0], starting_guess_local_h,
        p_distributed.counts[rank] * sizeof(double), cudaMemcpyHostToDevice));
    std::memcpy(p_distributed.vec_h[0], starting_guess_local_h,
        p_distributed.counts[rank] * sizeof(double));


    MPI_Request send_requests[p_distributed.number_of_neighbours-1];
    MPI_Request recv_requests[p_distributed.number_of_neighbours-1];

    // this stuff only works due to the symmetry of the matrix
    // i.e. a rank knows which other ranks needs its data
    // without symmetry, this would be more complicated

    // post all send requests
    for(int i = 1; i < p_distributed.number_of_neighbours; i++){
        int send_idx = p_distributed.neighbours[p_distributed.number_of_neighbours-i];
        int send_tag = std::abs(send_idx-rank);
        MPI_Isend(p_distributed.vec_h[0], p_distributed.rows_this_rank,
            MPI_DOUBLE, send_idx, send_tag, comm, &send_requests[i-1]);
    }

    //MPI_Waitall(p_distributed.number_of_neighbours-1, recv_requests, MPI_STATUSES_IGNORE);

    for(int i = 0; i < A_distributed.number_of_neighbours; i++){
        // loop over neighbors
        if(i < A_distributed.number_of_neighbours-1){
            int recv_idx = p_distributed.neighbours[i+1];
            int recv_tag = std::abs(recv_idx-rank);
            MPI_Irecv(p_distributed.vec_h[i+1], p_distributed.counts[recv_idx],
                MPI_DOUBLE, recv_idx, recv_tag, comm, &recv_requests[i]);
        }

        //memcpy
        int neighbour_idx = p_distributed.neighbours[i];
        cudaErrchk(cudaMemcpyAsync(p_distributed.vec_d[i], p_distributed.vec_h[i], p_distributed.counts[neighbour_idx] * sizeof(double), cudaMemcpyHostToDevice, stream));
        // calc A*x0
        cusparseErrchk(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            A_distributed.descriptors[i], p_distributed.descriptors[i],
            &alpha, vecAp_local, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));

        if(i < A_distributed.number_of_neighbours-1){
            MPI_Wait(&recv_requests[i], MPI_STATUS_IGNORE); 
        }
        
    }
    MPI_Waitall(p_distributed.number_of_neighbours-1, send_requests, MPI_STATUSES_IGNORE);

    // cal residual r0 = b - A*x0
    // r_norm2 = r0*r0
    cublasErrchk(cublasDaxpy(cublasHandle, rows_per_rank, &alpham1, Ax_local_d, 1, r_local_d, 1));
    cublasErrchk(cublasDdot(cublasHandle, rows_per_rank, r_local_d, 1, r_local_d, 1, r_norm2_d));
    //memcpy
    cudaErrchk(cudaMemcpy(&r_norm2, r_norm2_d, sizeof(double), cudaMemcpyDeviceToHost));
    //allreduce
    MPI_Allreduce(MPI_IN_PLACE, &r_norm2, 1, MPI_DOUBLE, MPI_SUM, comm);

    int k = 1;
    while (r_norm2/norm2_rhs > relative_tolerance * relative_tolerance && k <= max_iterations) {
        if(k > 1){
            // pk+1 = rk+1 + b*pk
            b = r_norm2 / r0;
            cublasErrchk(cublasDscal(cublasHandle, rows_per_rank, &b, p_local_d, 1));
            cublasErrchk(cublasDaxpy(cublasHandle, rows_per_rank, &alpha, r_local_d, 1, p_local_d, 1)); 
        }
        else {
            // p0 = r0
            cublasErrchk(cublasDcopy(cublasHandle, rows_per_rank, r_local_d, 1, p_local_d, 1));
        }


        // ak = rk^T * rk / pk^T * A * pk
        // has to be done for k=0 if x0 != 0
        //memcpy
        //allgather
        //memcpy
        cudaErrchk(cudaMemcpy(p_distributed.vec_h[0], p_local_d, rows_per_rank * sizeof(double), cudaMemcpyDeviceToHost));
        // result is accumulated in Ax_local_d
        cudaErrchk(cudaMemset(Ax_local_d, 0, rows_per_rank * sizeof(double)));


        MPI_Allgatherv(p_distributed.vec_h[0], rows_per_rank, MPI_DOUBLE, p_h, counts, displacements, MPI_DOUBLE, comm);


        // post all send requests
        for(int i = 1; i < p_distributed.number_of_neighbours; i++){
            int send_idx = p_distributed.neighbours[p_distributed.number_of_neighbours-i];
            int send_tag = std::abs(send_idx-rank);
            MPI_Isend(p_distributed.vec_h[0], p_distributed.rows_this_rank,
                MPI_DOUBLE, send_idx, send_tag, comm, &send_requests[i-1]);
        }


        for(int i = 0; i < A_distributed.number_of_neighbours; i++){
            // loop over neighbors
            if(i < A_distributed.number_of_neighbours-1){
                int recv_idx = p_distributed.neighbours[i+1];
                int recv_tag = std::abs(recv_idx-rank);
                MPI_Irecv(p_distributed.vec_h[i+1], p_distributed.counts[recv_idx],
                    MPI_DOUBLE, recv_idx, recv_tag, comm, &recv_requests[i]);
            }

            //memcpy
            int neighbour_idx = p_distributed.neighbours[i];
            cudaErrchk(cudaMemcpyAsync(p_distributed.vec_d[i], p_distributed.vec_h[i], p_distributed.counts[neighbour_idx] * sizeof(double), cudaMemcpyHostToDevice, stream));
            // calc A*x0
            cusparseErrchk(cusparseSpMV(
                cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                A_distributed.descriptors[i], p_distributed.descriptors[i],
                &alpha, vecAp_local, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));

            if(i < A_distributed.number_of_neighbours-1){
                MPI_Wait(&recv_requests[i], MPI_STATUS_IGNORE); 
            }
            
        }
        MPI_Waitall(p_distributed.number_of_neighbours-1, send_requests, MPI_STATUSES_IGNORE);
        // sleep(0.1);
        // for(int j = 0; j < size-1; j++){
        //     if(j == rank){
        //         std::cout << rank << " " << " reference " << std::endl;
        //         for(int i = 0; i < 3; i++){
        //             std::cout << i << " " << (p_h + displacements[rank+1])[i] << std::endl;
        //         }    
        //         std::cout << rank << " " << " test " << std::endl;
        //         for(int i = 0; i < 3; i++){
        //             std::cout << i << " " << (p_distributed.vec_h[1])[i] << std::endl;
        //         }       
        //     }
        //     sleep(0.1);
        //     MPI_Barrier(comm);
        // }


        cublasErrchk(cublasDdot(cublasHandle, rows_per_rank, p_local_d, 1, Ax_local_d, 1, dot_d));
        //memcpy
        cudaErrchk(cudaMemcpy(&dot_h, dot_d, sizeof(double), cudaMemcpyDeviceToHost));
        //allreduce        
        MPI_Allreduce(MPI_IN_PLACE, &dot_h, 1, MPI_DOUBLE, MPI_SUM, comm);

        a = r_norm2 / dot_h;

        // xk+1 = xk + ak * pk
        cublasErrchk(cublasDaxpy(cublasHandle, rows_per_rank, &a, p_local_d, 1, x_local_d, 1));

        // rk+1 = rk - ak * A * pk
        na = -a;
        cublasErrchk(cublasDaxpy(cublasHandle, rows_per_rank, &na, Ax_local_d, 1, r_local_d, 1));
        r0 = r_norm2;

        // r_norm2 = r0*r0
        cublasErrchk(cublasDdot(cublasHandle, rows_per_rank, r_local_d, 1, r_local_d, 1, r_norm2_d));
        //memcpy
        cudaErrchk(cudaMemcpy(&r_norm2, r_norm2_d, sizeof(double), cudaMemcpyDeviceToHost));
        //allreduce
        MPI_Allreduce(MPI_IN_PLACE, &r_norm2, 1, MPI_DOUBLE, MPI_SUM, comm);
        // std::cout << r_norm2 << std::endl;
        cudaErrchk(cudaStreamSynchronize(stream));

        k++;
    }

    steps_taken[0] = k;
    if(rank == 0){
        std::printf("iteration = %3d, residual = %e\n", k, sqrt(r_norm2));
    }


    //end CG
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaStreamSynchronize(stream));
    time_taken[0] += omp_get_wtime();

    std::cout << "rank " << rank << " time_taken[0] " << time_taken[0] << std::endl;

    //copy solution to host
    cudaErrchk(cudaMemcpy(r_local_h, x_local_d, rows_per_rank * sizeof(double), cudaMemcpyDeviceToHost));


    double difference = 0;
    double sum_ref = 0;
    for (int i = 0; i < rows_per_rank; ++i) {
        difference += std::sqrt( (r_local_h[i] - reference_solution[i+row_start_index]) * (r_local_h[i] - reference_solution[i+row_start_index]) );
        sum_ref += std::sqrt( (reference_solution[i+row_start_index]) * (reference_solution[i+row_start_index]) );
    }
    MPI_Allreduce(MPI_IN_PLACE, &difference, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &sum_ref, 1, MPI_DOUBLE, MPI_SUM, comm);
    if(rank == 0){
        std::cout << "difference/sum_ref " << difference/sum_ref << std::endl;
    }

    cusparseErrchk(cusparseDestroy(cusparseHandle));
    cublasErrchk(cublasDestroy(cublasHandle));
    cudaErrchk(cudaStreamDestroy(stream));
    cusparseErrchk(cusparseDestroyDnVec(vecp));
    cusparseErrchk(cusparseDestroySpMat(matA_local));
    cusparseErrchk(cusparseDestroyDnVec(vecAp_local));
    cudaErrchk(cudaFree(buffer));
    cudaErrchk(cudaFree(p_d));
    cudaErrchk(cudaFree(row_indptr_local_d));
    cudaErrchk(cudaFree(col_indices_local_d));
    cudaErrchk(cudaFree(data_local_d));
    cudaErrchk(cudaFree(r_local_d));
    cudaErrchk(cudaFree(x_local_d));
    cudaErrchk(cudaFree(p_local_d));
    cudaErrchk(cudaFree(Ax_local_d));

    cudaErrchk(cudaFree(r_norm2_d));
    cudaErrchk(cudaFree(dot_d));

    delete[] row_indptr_local_h;
    delete[] col_indices_local_h;
    delete[] data_local_h;
    delete[] r_local_h;
    delete[] p_local_h;
    delete[] p_h;

}


} // namespace own_test