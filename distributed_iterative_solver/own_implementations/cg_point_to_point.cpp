#include "cg_own_implementations.h"

namespace own_test
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
    cudaErrchk(cudaMemcpy(p_distributed.vec_h[0], p_distributed.vec_d[0],
        rows_per_rank * sizeof(double), cudaMemcpyDeviceToHost));
    
    // post all send requests
    for(int i = 1; i < A_distributed.number_of_neighbours; i++){
        int send_idx = p_distributed.neighbours[i];
        int send_tag = std::abs(send_idx - A_distributed.rank);
        MPI_Isend(p_distributed.vec_h[0], p_distributed.rows_this_rank,
                    MPI_DOUBLE, send_idx, send_tag, A_distributed.comm, &send_requests[i]);
    }

    for(int i = 0; i < A_distributed.number_of_neighbours; i++){
        // loop over neighbors
        if(i < A_distributed.number_of_neighbours-1){
            int recv_idx = p_distributed.neighbours[i + 1];
            int recv_tag = std::abs(recv_idx - A_distributed.rank);
            MPI_Irecv(p_distributed.vec_h[i + 1], p_distributed.counts[recv_idx],
                        MPI_DOUBLE, recv_idx, recv_tag, A_distributed.comm, &recv_requests[i]);
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

            MPI_Wait(&A_distributed.recv_requests[i], MPI_STATUS_IGNORE);
            int neighbour_idx = p_distributed.neighbours[i + 1];
            cudaErrchk(cudaMemcpyAsync(p_distributed.vec_d[i + 1], p_distributed.vec_h[i + 1], p_distributed.counts[neighbour_idx] * sizeof(double), cudaMemcpyHostToDevice, default_stream));


        }
        
    }
    MPI_Waitall(A_distributed.number_of_neighbours-1, &A_distributed.send_requests[1], MPI_STATUSES_IGNORE);


}

void solve_cg_nonblocking_point_to_point1(
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

    // cudaHostMalloc instead of new

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

    int *row_indptr_local_h = new int[rows_per_rank + 1];
    double *r_local_h = new double[rows_per_rank];
    for (int i = 0; i < rows_per_rank + 1; ++i)
    {
        row_indptr_local_h[i] = row_indptr_h[i + row_start_index] - row_indptr_h[row_start_index];
    }
    for (int i = 0; i < rows_per_rank; ++i)
    {
        r_local_h[i] = r_h[i + row_start_index];
    }

    int nnz_local = row_indptr_local_h[rows_per_rank];

    int *col_indices_local_h = new int[nnz_local];
    double *data_local_h = new double[nnz_local];

    for (int i = 0; i < nnz_local; ++i)
    {
        col_indices_local_h[i] = col_indices_h[i + row_indptr_h[row_start_index]];
        data_local_h[i] = data_h[i + row_indptr_h[row_start_index]];
    }

    // initialize cuda
    cudaStream_t default_stream = NULL;
    cublasHandle_t default_cublasHandle = 0;
    cublasErrchk(cublasCreate(&default_cublasHandle));

    cusparseHandle_t default_cusparseHandle = 0;
    cusparseErrchk(cusparseCreate(&default_cusparseHandle));

    cudaErrchk(cudaStreamCreate(&default_stream));
    cusparseErrchk(cusparseSetStream(default_cusparseHandle, default_stream));
    cublasErrchk(cublasSetStream(default_cublasHandle, default_stream));

    double a, b, na;
    double alpha, alpham1, r0;
    double *r_norm2_h;
    double *dot_h;
    cudaErrchk(cudaMallocHost((void **)&r_norm2_h, sizeof(double)));
    cudaErrchk(cudaMallocHost((void **)&dot_h, sizeof(double)));

    alpha = 1.0;
    alpham1 = -1.0;
    r0 = 0.0;

    double *r_norm2_d = NULL;
    double *dot_d = NULL;
    cudaErrchk(cudaMalloc((void **)&r_norm2_d, sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&dot_d, sizeof(double)));

    // allocate memory on device
    int *row_indptr_local_d = NULL;
    int *col_indices_local_d = NULL;
    double *data_local_d = NULL;
    double *r_local_d = NULL;
    double *x_local_d = NULL;

    double *starting_guess_local_h = starting_guess_h + row_start_index;

    cudaErrchk(cudaMalloc((void **)&row_indptr_local_d, (rows_per_rank + 1) * sizeof(int)));
    cudaErrchk(cudaMalloc((void **)&col_indices_local_d, nnz_local * sizeof(int)));
    cudaErrchk(cudaMalloc((void **)&data_local_d, nnz_local * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&r_local_d, rows_per_rank * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&x_local_d, rows_per_rank * sizeof(double)));

    cudaErrchk(cudaMemcpy(row_indptr_local_d, row_indptr_local_h, (rows_per_rank + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(col_indices_local_d, col_indices_local_h, nnz_local * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(data_local_d, data_local_h, nnz_local * sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(r_local_d, r_local_h, rows_per_rank * sizeof(double), cudaMemcpyHostToDevice));
    // copy data to device
    //  starting guess for x
    cudaErrchk(cudaMemcpy(x_local_d, starting_guess_local_h, rows_per_rank * sizeof(double), cudaMemcpyHostToDevice));

    double *Ap_local_d = NULL;
    cudaErrchk(cudaMalloc((void **)&Ap_local_d, rows_per_rank * sizeof(double)));
    cudaErrchk(cudaMemset(Ap_local_d, 0, rows_per_rank * sizeof(double)));

    cusparseDnVecDescr_t vecAp_local = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecAp_local, rows_per_rank, Ap_local_d, CUDA_R_64F));

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
        default_cusparseHandle);

    Distributed_vector p_distributed(
        matrix_size,
        counts,
        displacements,
        A_distributed.number_of_neighbours,
        A_distributed.neighbours,
        MPI_COMM_WORLD,
        default_cusparseHandle);

    MPI_Request send_requests[A_distributed.number_of_neighbours - 1];
    MPI_Request recv_requests[A_distributed.number_of_neighbours - 1];

    // begin CG
    std::printf("CG starts\n");
    cudaErrchk(cudaStreamSynchronize(default_stream));
    cudaErrchk(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    time_taken[0] = -omp_get_wtime();

    // norm of rhs for convergence check
    double norm2_rhs = 0;
    cublasErrchk(cublasDdot(default_cublasHandle, rows_per_rank, r_local_d, 1, r_local_d, 1, &norm2_rhs));
    // allreduce
    MPI_Allreduce(MPI_IN_PLACE, &norm2_rhs, 1, MPI_DOUBLE, MPI_SUM, comm);

    cudaErrchk(cudaMemcpy(p_distributed.vec_d[0], starting_guess_local_h,
                            p_distributed.counts[rank] * sizeof(double), cudaMemcpyHostToDevice));
    std::memcpy(p_distributed.vec_h[0], starting_guess_local_h,
                p_distributed.counts[rank] * sizeof(double));

    // this stuff only works due to the symmetry of the matrix
    // i.e. a rank knows which other ranks needs its data
    // without symmetry, this would be more complicated
    // Ax0
    distributed_mv_point_to_point1(
        A_distributed,
        p_distributed,
        vecAp_local,
        default_stream,
        default_cusparseHandle
    );
    // cal residual r0 = b - A*x0
    // r_norm2_h = r0*r0
    cublasErrchk(cublasDaxpy(default_cublasHandle, rows_per_rank, &alpham1, Ap_local_d, 1, r_local_d, 1));
    cublasErrchk(cublasDdot(default_cublasHandle, rows_per_rank, r_local_d, 1, r_local_d, 1, r_norm2_d));
    // memcpy
    cudaErrchk(cudaMemcpy(r_norm2_h, r_norm2_d, sizeof(double), cudaMemcpyDeviceToHost));
    // allreduce
    MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);

    int k = 1;
    while (r_norm2_h[0] / norm2_rhs > relative_tolerance * relative_tolerance && k <= max_iterations)
    {
        if (k > 1)
        {
            // pk+1 = rk+1 + b*pk
            b = r_norm2_h[0] / r0;
            cublasErrchk(cublasDscal(default_cublasHandle, rows_per_rank, &b, p_distributed.vec_d[0], 1));
            cublasErrchk(cublasDaxpy(default_cublasHandle, rows_per_rank, &alpha, r_local_d, 1, p_distributed.vec_d[0], 1));
        }
        else
        {
            // p0 = r0
            cublasErrchk(cublasDcopy(default_cublasHandle, rows_per_rank, r_local_d, 1, p_distributed.vec_d[0], 1));
        }

        // ak = rk^T * rk / pk^T * A * pk
        // has to be done for k=0 if x0 != 0
        // memcpy
        // allgather
        // memcpy
        distributed_mv_point_to_point1(
            A_distributed,
            p_distributed,
            vecAp_local,
            default_stream,
            default_cusparseHandle
        );

        cublasErrchk(cublasDdot(default_cublasHandle, rows_per_rank, p_distributed.vec_d[0], 1, Ap_local_d, 1, dot_d));
        // memcpy
        cudaErrchk(cudaMemcpy(dot_h, dot_d, sizeof(double), cudaMemcpyDeviceToHost));
        // allreduce
        MPI_Allreduce(MPI_IN_PLACE, dot_h, 1, MPI_DOUBLE, MPI_SUM, comm);

        a = r_norm2_h[0] / dot_h[0];

        // xk+1 = xk + ak * pk
        cublasErrchk(cublasDaxpy(default_cublasHandle, rows_per_rank, &a, p_distributed.vec_d[0], 1, x_local_d, 1));

        // rk+1 = rk - ak * A * pk
        na = -a;
        cublasErrchk(cublasDaxpy(default_cublasHandle, rows_per_rank, &na, Ap_local_d, 1, r_local_d, 1));
        r0 = r_norm2_h[0];

        // r_norm2_h = r0*r0
        cublasErrchk(cublasDdot(default_cublasHandle, rows_per_rank, r_local_d, 1, r_local_d, 1, r_norm2_d));
        // memcpy
        cudaErrchk(cudaMemcpy(r_norm2_h, r_norm2_d, sizeof(double), cudaMemcpyDeviceToHost));
        // allreduce
        MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);
        k++;
    }

    // end CG
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaStreamSynchronize(default_stream));
    time_taken[0] += omp_get_wtime();

    steps_taken[0] = k;
    if (rank == 0)
    {
        std::printf("iteration = %3d, residual = %e\n", k, sqrt(r_norm2_h[0]));
    }

    std::cout << "rank " << rank << " time_taken[0] " << time_taken[0] << std::endl;

    // copy solution to host
    cudaErrchk(cudaMemcpy(r_local_h, x_local_d, rows_per_rank * sizeof(double), cudaMemcpyDeviceToHost));

    double difference = 0;
    double sum_ref = 0;
    for (int i = 0; i < rows_per_rank; ++i)
    {
        difference += std::sqrt((r_local_h[i] - reference_solution[i + row_start_index]) * (r_local_h[i] - reference_solution[i + row_start_index]));
        sum_ref += std::sqrt((reference_solution[i + row_start_index]) * (reference_solution[i + row_start_index]));
    }
    MPI_Allreduce(MPI_IN_PLACE, &difference, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &sum_ref, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (rank == 0)
    {
        std::cout << "difference/sum_ref " << difference / sum_ref << std::endl;
    }

    cusparseErrchk(cusparseDestroy(default_cusparseHandle));
    cublasErrchk(cublasDestroy(default_cublasHandle));
    cudaErrchk(cudaStreamDestroy(default_stream));
    cusparseErrchk(cusparseDestroyDnVec(vecAp_local));
    cudaErrchk(cudaFree(row_indptr_local_d));
    cudaErrchk(cudaFree(col_indices_local_d));
    cudaErrchk(cudaFree(data_local_d));
    cudaErrchk(cudaFree(r_local_d));
    cudaErrchk(cudaFree(x_local_d));
    cudaErrchk(cudaFree(Ap_local_d));

    cudaErrchk(cudaFree(r_norm2_d));
    cudaErrchk(cudaFree(dot_d));

    delete[] row_indptr_local_h;
    delete[] col_indices_local_h;
    delete[] data_local_h;
    delete[] r_local_h;
    cudaErrchk(cudaFreeHost(r_norm2_h));
    cudaErrchk(cudaFreeHost(dot_h));
}

void solve_cg_nonblocking_point_to_point2(
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

    // cudaHostMalloc instead of new
    // no manual copy ofdot_h and r_norm2_h

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

    int *row_indptr_local_h = new int[rows_per_rank + 1];
    double *r_local_h = new double[rows_per_rank];
    for (int i = 0; i < rows_per_rank + 1; ++i)
    {
        row_indptr_local_h[i] = row_indptr_h[i + row_start_index] - row_indptr_h[row_start_index];
    }
    for (int i = 0; i < rows_per_rank; ++i)
    {
        r_local_h[i] = r_h[i + row_start_index];
    }

    int nnz_local = row_indptr_local_h[rows_per_rank];

    int *col_indices_local_h = new int[nnz_local];
    double *data_local_h = new double[nnz_local];

    for (int i = 0; i < nnz_local; ++i)
    {
        col_indices_local_h[i] = col_indices_h[i + row_indptr_h[row_start_index]];
        data_local_h[i] = data_h[i + row_indptr_h[row_start_index]];
    }

    // initialize cuda
    cudaStream_t default_stream = NULL;
    cublasHandle_t default_cublasHandle = 0;
    cublasErrchk(cublasCreate(&default_cublasHandle));

    cusparseHandle_t default_cusparseHandle = 0;
    cusparseErrchk(cusparseCreate(&default_cusparseHandle));

    cudaErrchk(cudaStreamCreate(&default_stream));
    cusparseErrchk(cusparseSetStream(default_cusparseHandle, default_stream));
    cublasErrchk(cublasSetStream(default_cublasHandle, default_stream));

    double a, b, na;
    double alpha, alpham1, r0;
    double *r_norm2_h;
    double *dot_h;
    cudaErrchk(cudaMallocHost((void **)&r_norm2_h, sizeof(double)));
    cudaErrchk(cudaMallocHost((void **)&dot_h, sizeof(double)));

    alpha = 1.0;
    alpham1 = -1.0;
    r0 = 0.0;

    double *r_norm2_d = NULL;
    double *dot_d = NULL;
    cudaErrchk(cudaMalloc((void **)&r_norm2_d, sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&dot_d, sizeof(double)));

    // allocate memory on device
    int *row_indptr_local_d = NULL;
    int *col_indices_local_d = NULL;
    double *data_local_d = NULL;
    double *r_local_d = NULL;
    double *x_local_d = NULL;

    double *starting_guess_local_h = starting_guess_h + row_start_index;

    cudaErrchk(cudaMalloc((void **)&row_indptr_local_d, (rows_per_rank + 1) * sizeof(int)));
    cudaErrchk(cudaMalloc((void **)&col_indices_local_d, nnz_local * sizeof(int)));
    cudaErrchk(cudaMalloc((void **)&data_local_d, nnz_local * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&r_local_d, rows_per_rank * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&x_local_d, rows_per_rank * sizeof(double)));

    cudaErrchk(cudaMemcpy(row_indptr_local_d, row_indptr_local_h, (rows_per_rank + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(col_indices_local_d, col_indices_local_h, nnz_local * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(data_local_d, data_local_h, nnz_local * sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(r_local_d, r_local_h, rows_per_rank * sizeof(double), cudaMemcpyHostToDevice));
    // copy data to device
    //  starting guess for x
    cudaErrchk(cudaMemcpy(x_local_d, starting_guess_local_h, rows_per_rank * sizeof(double), cudaMemcpyHostToDevice));

    double *Ap_local_d = NULL;
    cudaErrchk(cudaMalloc((void **)&Ap_local_d, rows_per_rank * sizeof(double)));
    cudaErrchk(cudaMemset(Ap_local_d, 0, rows_per_rank * sizeof(double)));

    cusparseDnVecDescr_t vecAp_local = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecAp_local, rows_per_rank, Ap_local_d, CUDA_R_64F));

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
        default_cusparseHandle);

    Distributed_vector p_distributed(
        matrix_size,
        counts,
        displacements,
        A_distributed.number_of_neighbours,
        A_distributed.neighbours,
        MPI_COMM_WORLD,
        default_cusparseHandle);

    MPI_Request send_requests[A_distributed.number_of_neighbours - 1];
    MPI_Request recv_requests[A_distributed.number_of_neighbours - 1];

    // begin CG
    std::printf("CG starts\n");
    cudaErrchk(cudaStreamSynchronize(default_stream));
    cudaErrchk(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    time_taken[0] = -omp_get_wtime();

    // norm of rhs for convergence check
    double norm2_rhs = 0;
    cublasErrchk(cublasDdot(default_cublasHandle, rows_per_rank, r_local_d, 1, r_local_d, 1, &norm2_rhs));
    // allreduce
    MPI_Allreduce(MPI_IN_PLACE, &norm2_rhs, 1, MPI_DOUBLE, MPI_SUM, comm);

    cudaErrchk(cudaMemcpy(p_distributed.vec_d[0], starting_guess_local_h,
                            p_distributed.counts[rank] * sizeof(double), cudaMemcpyHostToDevice));
    std::memcpy(p_distributed.vec_h[0], starting_guess_local_h,
                p_distributed.counts[rank] * sizeof(double));

    // this stuff only works due to the symmetry of the matrix
    // i.e. a rank knows which other ranks needs its data
    // without symmetry, this would be more complicated

    // post all send requests
    for (int i = 1; i < A_distributed.number_of_neighbours; i++)
    {
        int send_idx = p_distributed.neighbours[A_distributed.number_of_neighbours - i];
        int send_tag = std::abs(send_idx - rank);
        MPI_Isend(p_distributed.vec_h[0], p_distributed.rows_this_rank,
                    MPI_DOUBLE, send_idx, send_tag, comm, &send_requests[i - 1]);
    }

    for (int i = 0; i < A_distributed.number_of_neighbours; i++)
    {
        // loop over neighbors
        if (i < A_distributed.number_of_neighbours - 1)
        {
            int recv_idx = p_distributed.neighbours[i + 1];
            int recv_tag = std::abs(recv_idx - rank);
            MPI_Irecv(p_distributed.vec_h[i + 1], p_distributed.counts[recv_idx],
                        MPI_DOUBLE, recv_idx, recv_tag, comm, &recv_requests[i]);
        }

        // calc A*x0
        cusparseErrchk(cusparseSpMV(
            default_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            A_distributed.descriptors[i], p_distributed.descriptors[i],
            &alpha, vecAp_local, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, A_distributed.buffer_d[i]));

        if (i < A_distributed.number_of_neighbours - 1)
        {
            MPI_Wait(&recv_requests[i], MPI_STATUS_IGNORE);
            int neighbour_idx = p_distributed.neighbours[i + 1];
            cudaErrchk(cudaMemcpyAsync(p_distributed.vec_d[i + 1], p_distributed.vec_h[i + 1], p_distributed.counts[neighbour_idx] * sizeof(double), cudaMemcpyHostToDevice, default_stream));
        }
    }
    MPI_Waitall(A_distributed.number_of_neighbours - 1, send_requests, MPI_STATUSES_IGNORE);

    // cal residual r0 = b - A*x0
    // r_norm2_h = r0*r0
    cublasErrchk(cublasDaxpy(default_cublasHandle, rows_per_rank, &alpham1, Ap_local_d, 1, r_local_d, 1));
    cublasErrchk(cublasDdot(default_cublasHandle, rows_per_rank, r_local_d, 1, r_local_d, 1, r_norm2_h));
    // allreduce
    MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);

    int k = 1;
    while (r_norm2_h[0] / norm2_rhs > relative_tolerance * relative_tolerance && k <= max_iterations)
    {
        if (k > 1)
        {
            // pk+1 = rk+1 + b*pk
            b = r_norm2_h[0] / r0;
            cublasErrchk(cublasDscal(default_cublasHandle, rows_per_rank, &b, p_distributed.vec_d[0], 1));
            cublasErrchk(cublasDaxpy(default_cublasHandle, rows_per_rank, &alpha, r_local_d, 1, p_distributed.vec_d[0], 1));
        }
        else
        {
            // p0 = r0
            cublasErrchk(cublasDcopy(default_cublasHandle, rows_per_rank, r_local_d, 1, p_distributed.vec_d[0], 1));
        }

        // ak = rk^T * rk / pk^T * A * pk
        // has to be done for k=0 if x0 != 0
        // memcpy
        // allgather
        // memcpy
        cudaErrchk(cudaMemcpy(p_distributed.vec_h[0], p_distributed.vec_d[0], rows_per_rank * sizeof(double), cudaMemcpyDeviceToHost));
        // result is accumulated in Ap_local_d
        cudaErrchk(cudaMemset(Ap_local_d, 0, rows_per_rank * sizeof(double)));

        // post all send requests
        for (int i = 1; i < A_distributed.number_of_neighbours; i++)
        {
            int send_idx = p_distributed.neighbours[A_distributed.number_of_neighbours - i];
            int send_tag = std::abs(send_idx - rank);
            MPI_Isend(p_distributed.vec_h[0], p_distributed.rows_this_rank,
                        MPI_DOUBLE, send_idx, send_tag, comm, &send_requests[i - 1]);
        }

        for (int i = 0; i < A_distributed.number_of_neighbours; i++)
        {
            // loop over neighbors
            if (i < A_distributed.number_of_neighbours - 1)
            {
                int recv_idx = p_distributed.neighbours[i + 1];
                int recv_tag = std::abs(recv_idx - rank);
                MPI_Irecv(p_distributed.vec_h[i + 1], p_distributed.counts[recv_idx],
                            MPI_DOUBLE, recv_idx, recv_tag, comm, &recv_requests[i]);
            }

            // calc A*x0
            cusparseErrchk(cusparseSpMV(
                default_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                A_distributed.descriptors[i], p_distributed.descriptors[i],
                &alpha, vecAp_local, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, A_distributed.buffer_d[i]));

            if (i < A_distributed.number_of_neighbours - 1)
            {
                MPI_Wait(&recv_requests[i], MPI_STATUS_IGNORE);
                int neighbour_idx = p_distributed.neighbours[i + 1];
                cudaErrchk(cudaMemcpyAsync(p_distributed.vec_d[i + 1], p_distributed.vec_h[i + 1], p_distributed.counts[neighbour_idx] * sizeof(double), cudaMemcpyHostToDevice, default_stream));
            }
        }
        MPI_Waitall(A_distributed.number_of_neighbours - 1, send_requests, MPI_STATUSES_IGNORE);

        cublasErrchk(cublasDdot(default_cublasHandle, rows_per_rank, p_distributed.vec_d[0], 1, Ap_local_d, 1, dot_h));
        // allreduce
        MPI_Allreduce(MPI_IN_PLACE, dot_h, 1, MPI_DOUBLE, MPI_SUM, comm);

        a = r_norm2_h[0] / dot_h[0];

        // xk+1 = xk + ak * pk
        cublasErrchk(cublasDaxpy(default_cublasHandle, rows_per_rank, &a, p_distributed.vec_d[0], 1, x_local_d, 1));

        // rk+1 = rk - ak * A * pk
        na = -a;
        cublasErrchk(cublasDaxpy(default_cublasHandle, rows_per_rank, &na, Ap_local_d, 1, r_local_d, 1));
        r0 = r_norm2_h[0];

        // r_norm2_h = r0*r0
        cublasErrchk(cublasDdot(default_cublasHandle, rows_per_rank, r_local_d, 1, r_local_d, 1, r_norm2_h));
        // allreduce
        MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);
        k++;
    }

    // end CG
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaStreamSynchronize(default_stream));
    time_taken[0] += omp_get_wtime();

    steps_taken[0] = k;
    if (rank == 0)
    {
        std::printf("iteration = %3d, residual = %e\n", k, sqrt(r_norm2_h[0]));
    }

    std::cout << "rank " << rank << " time_taken[0] " << time_taken[0] << std::endl;

    // copy solution to host
    cudaErrchk(cudaMemcpy(r_local_h, x_local_d, rows_per_rank * sizeof(double), cudaMemcpyDeviceToHost));

    double difference = 0;
    double sum_ref = 0;
    for (int i = 0; i < rows_per_rank; ++i)
    {
        difference += std::sqrt((r_local_h[i] - reference_solution[i + row_start_index]) * (r_local_h[i] - reference_solution[i + row_start_index]));
        sum_ref += std::sqrt((reference_solution[i + row_start_index]) * (reference_solution[i + row_start_index]));
    }
    MPI_Allreduce(MPI_IN_PLACE, &difference, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &sum_ref, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (rank == 0)
    {
        std::cout << "difference/sum_ref " << difference / sum_ref << std::endl;
    }

    cusparseErrchk(cusparseDestroy(default_cusparseHandle));
    cublasErrchk(cublasDestroy(default_cublasHandle));
    cudaErrchk(cudaStreamDestroy(default_stream));
    cusparseErrchk(cusparseDestroyDnVec(vecAp_local));
    cudaErrchk(cudaFree(row_indptr_local_d));
    cudaErrchk(cudaFree(col_indices_local_d));
    cudaErrchk(cudaFree(data_local_d));
    cudaErrchk(cudaFree(r_local_d));
    cudaErrchk(cudaFree(x_local_d));
    cudaErrchk(cudaFree(Ap_local_d));

    cudaErrchk(cudaFree(r_norm2_d));
    cudaErrchk(cudaFree(dot_d));

    delete[] row_indptr_local_h;
    delete[] col_indices_local_h;
    delete[] data_local_h;
    delete[] r_local_h;
    cudaErrchk(cudaFreeHost(r_norm2_h));
    cudaErrchk(cudaFreeHost(dot_h));
}

void solve_cg_nonblocking_point_to_point3(
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

    // cudaHostMalloc instead of new
    // cuda aware mpi
    // no manual copy ofdot_h and r_norm2_h

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

    int *row_indptr_local_h = new int[rows_per_rank + 1];
    double *r_local_h = new double[rows_per_rank];
    for (int i = 0; i < rows_per_rank + 1; ++i)
    {
        row_indptr_local_h[i] = row_indptr_h[i + row_start_index] - row_indptr_h[row_start_index];
    }
    for (int i = 0; i < rows_per_rank; ++i)
    {
        r_local_h[i] = r_h[i + row_start_index];
    }

    int nnz_local = row_indptr_local_h[rows_per_rank];

    int *col_indices_local_h = new int[nnz_local];
    double *data_local_h = new double[nnz_local];

    for (int i = 0; i < nnz_local; ++i)
    {
        col_indices_local_h[i] = col_indices_h[i + row_indptr_h[row_start_index]];
        data_local_h[i] = data_h[i + row_indptr_h[row_start_index]];
    }

    // initialize cuda
    cudaStream_t default_stream = NULL;
    cublasHandle_t default_cublasHandle = 0;
    cublasErrchk(cublasCreate(&default_cublasHandle));

    cusparseHandle_t default_cusparseHandle = 0;
    cusparseErrchk(cusparseCreate(&default_cusparseHandle));

    cudaErrchk(cudaStreamCreate(&default_stream));
    cusparseErrchk(cusparseSetStream(default_cusparseHandle, default_stream));
    cublasErrchk(cublasSetStream(default_cublasHandle, default_stream));

    double a, b, na;
    double alpha, alpham1, r0;
    double *r_norm2_h;
    double *dot_h;
    cudaErrchk(cudaMallocHost((void **)&r_norm2_h, sizeof(double)));
    cudaErrchk(cudaMallocHost((void **)&dot_h, sizeof(double)));

    alpha = 1.0;
    alpham1 = -1.0;
    r0 = 0.0;

    double *r_norm2_d = NULL;
    double *dot_d = NULL;
    cudaErrchk(cudaMalloc((void **)&r_norm2_d, sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&dot_d, sizeof(double)));

    // allocate memory on device
    int *row_indptr_local_d = NULL;
    int *col_indices_local_d = NULL;
    double *data_local_d = NULL;
    double *r_local_d = NULL;
    double *x_local_d = NULL;

    double *starting_guess_local_h = starting_guess_h + row_start_index;

    cudaErrchk(cudaMalloc((void **)&row_indptr_local_d, (rows_per_rank + 1) * sizeof(int)));
    cudaErrchk(cudaMalloc((void **)&col_indices_local_d, nnz_local * sizeof(int)));
    cudaErrchk(cudaMalloc((void **)&data_local_d, nnz_local * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&r_local_d, rows_per_rank * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&x_local_d, rows_per_rank * sizeof(double)));

    cudaErrchk(cudaMemcpy(row_indptr_local_d, row_indptr_local_h, (rows_per_rank + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(col_indices_local_d, col_indices_local_h, nnz_local * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(data_local_d, data_local_h, nnz_local * sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(r_local_d, r_local_h, rows_per_rank * sizeof(double), cudaMemcpyHostToDevice));
    // copy data to device
    //  starting guess for x
    cudaErrchk(cudaMemcpy(x_local_d, starting_guess_local_h, rows_per_rank * sizeof(double), cudaMemcpyHostToDevice));

    double *Ap_local_d = NULL;
    cudaErrchk(cudaMalloc((void **)&Ap_local_d, rows_per_rank * sizeof(double)));
    cudaErrchk(cudaMemset(Ap_local_d, 0, rows_per_rank * sizeof(double)));

    cusparseDnVecDescr_t vecAp_local = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecAp_local, rows_per_rank, Ap_local_d, CUDA_R_64F));

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
        default_cusparseHandle);

    Distributed_vector p_distributed(
        matrix_size,
        counts,
        displacements,
        A_distributed.number_of_neighbours,
        A_distributed.neighbours,
        MPI_COMM_WORLD,
        default_cusparseHandle);

    MPI_Request send_requests[A_distributed.number_of_neighbours - 1];
    MPI_Request recv_requests[A_distributed.number_of_neighbours - 1];

    // begin CG
    std::printf("CG starts\n");
    cudaErrchk(cudaStreamSynchronize(default_stream));
    cudaErrchk(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    time_taken[0] = -omp_get_wtime();

    // norm of rhs for convergence check
    double norm2_rhs = 0;
    cublasErrchk(cublasDdot(default_cublasHandle, rows_per_rank, r_local_d, 1, r_local_d, 1, &norm2_rhs));
    // allreduce
    MPI_Allreduce(MPI_IN_PLACE, &norm2_rhs, 1, MPI_DOUBLE, MPI_SUM, comm);

    cudaErrchk(cudaMemcpy(p_distributed.vec_d[0], starting_guess_local_h,
                            p_distributed.counts[rank] * sizeof(double), cudaMemcpyHostToDevice));
    std::memcpy(p_distributed.vec_h[0], starting_guess_local_h,
                p_distributed.counts[rank] * sizeof(double));

    // this stuff only works due to the symmetry of the matrix
    // i.e. a rank knows which other ranks needs its data
    // without symmetry, this would be more complicated

    // post all send requests
    for (int i = 1; i < A_distributed.number_of_neighbours; i++)
    {
        int send_idx = p_distributed.neighbours[A_distributed.number_of_neighbours - i];
        int send_tag = std::abs(send_idx - rank);
        MPI_Isend(p_distributed.vec_h[0], p_distributed.rows_this_rank,
                    MPI_DOUBLE, send_idx, send_tag, comm, &send_requests[i - 1]);
    }

    for (int i = 0; i < A_distributed.number_of_neighbours; i++)
    {
        // loop over neighbors
        if (i < A_distributed.number_of_neighbours - 1)
        {
            int recv_idx = p_distributed.neighbours[i + 1];
            int recv_tag = std::abs(recv_idx - rank);
            MPI_Irecv(p_distributed.vec_h[i + 1], p_distributed.counts[recv_idx],
                        MPI_DOUBLE, recv_idx, recv_tag, comm, &recv_requests[i]);
        }

        // calc A*x0
        cusparseErrchk(cusparseSpMV(
            default_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            A_distributed.descriptors[i], p_distributed.descriptors[i],
            &alpha, vecAp_local, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, A_distributed.buffer_d[i]));

        if (i < A_distributed.number_of_neighbours - 1)
        {
            MPI_Wait(&recv_requests[i], MPI_STATUS_IGNORE);
            int neighbour_idx = p_distributed.neighbours[i + 1];
            cudaErrchk(cudaMemcpyAsync(p_distributed.vec_d[i + 1], p_distributed.vec_h[i + 1], p_distributed.counts[neighbour_idx] * sizeof(double), cudaMemcpyHostToDevice, default_stream));
        }
    }
    MPI_Waitall(A_distributed.number_of_neighbours - 1, send_requests, MPI_STATUSES_IGNORE);

    // cal residual r0 = b - A*x0
    // r_norm2_h = r0*r0
    cublasErrchk(cublasDaxpy(default_cublasHandle, rows_per_rank, &alpham1, Ap_local_d, 1, r_local_d, 1));
    cublasErrchk(cublasDdot(default_cublasHandle, rows_per_rank, r_local_d, 1, r_local_d, 1, r_norm2_h));
    // allreduce
    MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);

    int k = 1;
    while (r_norm2_h[0] / norm2_rhs > relative_tolerance * relative_tolerance && k <= max_iterations)
    {
        if (k > 1)
        {
            // pk+1 = rk+1 + b*pk
            b = r_norm2_h[0] / r0;
            cublasErrchk(cublasDscal(default_cublasHandle, rows_per_rank, &b, p_distributed.vec_d[0], 1));
            cublasErrchk(cublasDaxpy(default_cublasHandle, rows_per_rank, &alpha, r_local_d, 1, p_distributed.vec_d[0], 1));
        }
        else
        {
            // p0 = r0
            cublasErrchk(cublasDcopy(default_cublasHandle, rows_per_rank, r_local_d, 1, p_distributed.vec_d[0], 1));
        }

        // ak = rk^T * rk / pk^T * A * pk
        // has to be done for k=0 if x0 != 0
        // result is accumulated in Ap_local_d
        cudaErrchk(cudaMemset(Ap_local_d, 0, rows_per_rank * sizeof(double)));

        // post all send requests
        for (int i = 1; i < A_distributed.number_of_neighbours; i++)
        {
            int send_idx = p_distributed.neighbours[A_distributed.number_of_neighbours - i];
            int send_tag = std::abs(send_idx - rank);
            MPI_Isend(p_distributed.vec_d[0], p_distributed.rows_this_rank,
                        MPI_DOUBLE, send_idx, send_tag, comm, &send_requests[i - 1]);
        }

        for (int i = 0; i < A_distributed.number_of_neighbours; i++)
        {
            // loop over neighbors
            if (i < A_distributed.number_of_neighbours - 1)
            {
                int recv_idx = p_distributed.neighbours[i + 1];
                int recv_tag = std::abs(recv_idx - rank);
                MPI_Irecv(p_distributed.vec_d[i + 1], p_distributed.counts[recv_idx],
                            MPI_DOUBLE, recv_idx, recv_tag, comm, &recv_requests[i]);
            }

            // calc A*x0
            cusparseErrchk(cusparseSpMV(
                default_cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                A_distributed.descriptors[i], p_distributed.descriptors[i],
                &alpha, vecAp_local, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, A_distributed.buffer_d[i]));

            if (i < A_distributed.number_of_neighbours - 1)
            {
                MPI_Wait(&recv_requests[i], MPI_STATUS_IGNORE);
            }
        }
        MPI_Waitall(A_distributed.number_of_neighbours - 1, send_requests, MPI_STATUSES_IGNORE);

        cublasErrchk(cublasDdot(default_cublasHandle, rows_per_rank, p_distributed.vec_d[0], 1, Ap_local_d, 1, dot_h));
        // allreduce
        MPI_Allreduce(MPI_IN_PLACE, dot_h, 1, MPI_DOUBLE, MPI_SUM, comm);

        a = r_norm2_h[0] / dot_h[0];

        // xk+1 = xk + ak * pk
        cublasErrchk(cublasDaxpy(default_cublasHandle, rows_per_rank, &a, p_distributed.vec_d[0], 1, x_local_d, 1));

        // rk+1 = rk - ak * A * pk
        na = -a;
        cublasErrchk(cublasDaxpy(default_cublasHandle, rows_per_rank, &na, Ap_local_d, 1, r_local_d, 1));
        r0 = r_norm2_h[0];

        // r_norm2_h = r0*r0
        cublasErrchk(cublasDdot(default_cublasHandle, rows_per_rank, r_local_d, 1, r_local_d, 1, r_norm2_h));
        // allreduce
        MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);
        k++;
    }

    // end CG
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaStreamSynchronize(default_stream));
    time_taken[0] += omp_get_wtime();

    steps_taken[0] = k;
    if (rank == 0)
    {
        std::printf("iteration = %3d, residual = %e\n", k, sqrt(r_norm2_h[0]));
    }

    std::cout << "rank " << rank << " time_taken[0] " << time_taken[0] << std::endl;

    // copy solution to host
    cudaErrchk(cudaMemcpy(r_local_h, x_local_d, rows_per_rank * sizeof(double), cudaMemcpyDeviceToHost));

    double difference = 0;
    double sum_ref = 0;
    for (int i = 0; i < rows_per_rank; ++i)
    {
        difference += std::sqrt((r_local_h[i] - reference_solution[i + row_start_index]) * (r_local_h[i] - reference_solution[i + row_start_index]));
        sum_ref += std::sqrt((reference_solution[i + row_start_index]) * (reference_solution[i + row_start_index]));
    }
    MPI_Allreduce(MPI_IN_PLACE, &difference, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &sum_ref, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (rank == 0)
    {
        std::cout << "difference/sum_ref " << difference / sum_ref << std::endl;
    }

    cusparseErrchk(cusparseDestroy(default_cusparseHandle));
    cublasErrchk(cublasDestroy(default_cublasHandle));
    cudaErrchk(cudaStreamDestroy(default_stream));
    cusparseErrchk(cusparseDestroyDnVec(vecAp_local));
    cudaErrchk(cudaFree(row_indptr_local_d));
    cudaErrchk(cudaFree(col_indices_local_d));
    cudaErrchk(cudaFree(data_local_d));
    cudaErrchk(cudaFree(r_local_d));
    cudaErrchk(cudaFree(x_local_d));
    cudaErrchk(cudaFree(Ap_local_d));

    cudaErrchk(cudaFree(r_norm2_d));
    cudaErrchk(cudaFree(dot_d));

    delete[] row_indptr_local_h;
    delete[] col_indices_local_h;
    delete[] data_local_h;
    delete[] r_local_h;
    cudaErrchk(cudaFreeHost(r_norm2_h));
    cudaErrchk(cudaFreeHost(dot_h));
}

void solve_cg_nonblocking_point_to_point4(
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

    // cudaHostMalloc instead of new
    // no manual copy ofdot_h and r_norm2_h
    // multiple streams

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

    int *row_indptr_local_h = new int[rows_per_rank + 1];
    double *r_local_h = new double[rows_per_rank];
    for (int i = 0; i < rows_per_rank + 1; ++i)
    {
        row_indptr_local_h[i] = row_indptr_h[i + row_start_index] - row_indptr_h[row_start_index];
    }
    for (int i = 0; i < rows_per_rank; ++i)
    {
        r_local_h[i] = r_h[i + row_start_index];
    }

    int nnz_local = row_indptr_local_h[rows_per_rank];

    int *col_indices_local_h = new int[nnz_local];
    double *data_local_h = new double[nnz_local];

    for (int i = 0; i < nnz_local; ++i)
    {
        col_indices_local_h[i] = col_indices_h[i + row_indptr_h[row_start_index]];
        data_local_h[i] = data_h[i + row_indptr_h[row_start_index]];
    }

    // initialize cuda
    cudaStream_t default_stream = NULL;
    cublasHandle_t default_cublasHandle = 0;
    cublasErrchk(cublasCreate(&default_cublasHandle));

    cusparseHandle_t default_cusparseHandle = 0;
    cusparseErrchk(cusparseCreate(&default_cusparseHandle));

    cudaErrchk(cudaStreamCreate(&default_stream));
    cusparseErrchk(cusparseSetStream(default_cusparseHandle, default_stream));
    cublasErrchk(cublasSetStream(default_cublasHandle, default_stream));

    double a, b, na;
    double alpha, alpham1, r0;
    double *r_norm2_h;
    double *dot_h;
    cudaErrchk(cudaMallocHost((void **)&r_norm2_h, sizeof(double)));
    cudaErrchk(cudaMallocHost((void **)&dot_h, sizeof(double)));

    alpha = 1.0;
    alpham1 = -1.0;
    r0 = 0.0;

    double *r_norm2_d = NULL;
    double *dot_d = NULL;
    cudaErrchk(cudaMalloc((void **)&r_norm2_d, sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&dot_d, sizeof(double)));

    // allocate memory on device
    int *row_indptr_local_d = NULL;
    int *col_indices_local_d = NULL;
    double *data_local_d = NULL;
    double *r_local_d = NULL;
    double *x_local_d = NULL;

    double *starting_guess_local_h = starting_guess_h + row_start_index;

    cudaErrchk(cudaMalloc((void **)&row_indptr_local_d, (rows_per_rank + 1) * sizeof(int)));
    cudaErrchk(cudaMalloc((void **)&col_indices_local_d, nnz_local * sizeof(int)));
    cudaErrchk(cudaMalloc((void **)&data_local_d, nnz_local * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&r_local_d, rows_per_rank * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&x_local_d, rows_per_rank * sizeof(double)));

    cudaErrchk(cudaMemcpy(row_indptr_local_d, row_indptr_local_h, (rows_per_rank + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(col_indices_local_d, col_indices_local_h, nnz_local * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(data_local_d, data_local_h, nnz_local * sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(r_local_d, r_local_h, rows_per_rank * sizeof(double), cudaMemcpyHostToDevice));
    // copy data to device
    //  starting guess for x
    cudaErrchk(cudaMemcpy(x_local_d, starting_guess_local_h, rows_per_rank * sizeof(double), cudaMemcpyHostToDevice));

    double *Ap_local_d = NULL;
    cudaErrchk(cudaMalloc((void **)&Ap_local_d, rows_per_rank * sizeof(double)));
    cudaErrchk(cudaMemset(Ap_local_d, 0, rows_per_rank * sizeof(double)));

    cusparseDnVecDescr_t vecAp_local = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecAp_local, rows_per_rank, Ap_local_d, CUDA_R_64F));

    // cudaStream_t default_stream = default_stream;
    // cusparseHandle_t default_cusparseHandle = default_cusparseHandle;

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
        default_cusparseHandle);

    Distributed_vector p_distributed(
        matrix_size,
        counts,
        displacements,
        A_distributed.number_of_neighbours,
        A_distributed.neighbours,
        MPI_COMM_WORLD,
        default_cusparseHandle);

    cudaStream_t streams[A_distributed.number_of_neighbours];
    cublasHandle_t cublasHandles[A_distributed.number_of_neighbours];
    cusparseHandle_t cusparseHandles[A_distributed.number_of_neighbours];

    for (int i = 0; i < A_distributed.number_of_neighbours; i++)
    {
        cudaErrchk(cudaStreamCreate(&streams[i]));
        cublasErrchk(cublasCreate(&cublasHandles[i]));
        cusparseErrchk(cusparseCreate(&cusparseHandles[i]));
        cusparseErrchk(cusparseSetStream(cusparseHandles[i], streams[i]));
        cublasErrchk(cublasSetStream(cublasHandles[i], streams[i]));
    }

    MPI_Request send_requests[A_distributed.number_of_neighbours - 1];
    MPI_Request recv_requests[A_distributed.number_of_neighbours - 1];

    // begin CG
    std::printf("CG starts\n");
    cudaErrchk(cudaStreamSynchronize(default_stream));
    cudaErrchk(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    time_taken[0] = -omp_get_wtime();

    // norm of rhs for convergence check
    double norm2_rhs = 0;
    cublasErrchk(cublasDdot(default_cublasHandle, rows_per_rank, r_local_d, 1, r_local_d, 1, &norm2_rhs));
    // allreduce
    MPI_Allreduce(MPI_IN_PLACE, &norm2_rhs, 1, MPI_DOUBLE, MPI_SUM, comm);

    cudaErrchk(cudaMemcpy(p_distributed.vec_d[0], starting_guess_local_h,
                            p_distributed.counts[rank] * sizeof(double), cudaMemcpyHostToDevice));
    std::memcpy(p_distributed.vec_h[0], starting_guess_local_h,
                p_distributed.counts[rank] * sizeof(double));

    // this stuff only works due to the symmetry of the matrix
    // i.e. a rank knows which other ranks needs its data
    // without symmetry, this would be more complicated

    // post all send requests
    for (int i = 1; i < A_distributed.number_of_neighbours; i++)
    {
        int send_idx = p_distributed.neighbours[A_distributed.number_of_neighbours - i];
        int send_tag = std::abs(send_idx - rank);
        MPI_Isend(p_distributed.vec_h[0], p_distributed.rows_this_rank,
                    MPI_DOUBLE, send_idx, send_tag, comm, &send_requests[i - 1]);
    }

    for (int i = 0; i < A_distributed.number_of_neighbours; i++)
    {
        // loop over neighbors
        if (i < A_distributed.number_of_neighbours - 1)
        {
            int recv_idx = p_distributed.neighbours[i + 1];
            int recv_tag = std::abs(recv_idx - rank);
            MPI_Irecv(p_distributed.vec_h[i + 1], p_distributed.counts[recv_idx],
                        MPI_DOUBLE, recv_idx, recv_tag, comm, &recv_requests[i]);
        }

        // calc A*x0
        cusparseErrchk(cusparseSpMV(
            cusparseHandles[i], CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            A_distributed.descriptors[i], p_distributed.descriptors[i],
            &alpha, vecAp_local, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, A_distributed.buffer_d[i]));

        if (i < A_distributed.number_of_neighbours - 1)
        {
            MPI_Wait(&recv_requests[i], MPI_STATUS_IGNORE);
            int neighbour_idx = p_distributed.neighbours[i + 1];
            cudaErrchk(cudaMemcpyAsync(p_distributed.vec_d[i + 1], p_distributed.vec_h[i + 1], p_distributed.counts[neighbour_idx] * sizeof(double), cudaMemcpyHostToDevice, streams[i + 1]));
        }
    }
    MPI_Waitall(A_distributed.number_of_neighbours - 1, send_requests, MPI_STATUSES_IGNORE);

    for (int i = 0; i < A_distributed.number_of_neighbours; i++)
    {
        cudaErrchk(cudaStreamSynchronize(streams[i]));
    }

    // cal residual r0 = b - A*x0
    // r_norm2_h = r0*r0
    cublasErrchk(cublasDaxpy(default_cublasHandle, rows_per_rank, &alpham1, Ap_local_d, 1, r_local_d, 1));
    cublasErrchk(cublasDdot(default_cublasHandle, rows_per_rank, r_local_d, 1, r_local_d, 1, r_norm2_h));
    // allreduce
    MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);

    int k = 1;
    while (r_norm2_h[0] / norm2_rhs > relative_tolerance * relative_tolerance && k <= max_iterations)
    {
        if (k > 1)
        {
            // pk+1 = rk+1 + b*pk
            b = r_norm2_h[0] / r0;
            cublasErrchk(cublasDscal(default_cublasHandle, rows_per_rank, &b, p_distributed.vec_d[0], 1));
            cublasErrchk(cublasDaxpy(default_cublasHandle, rows_per_rank, &alpha, r_local_d, 1, p_distributed.vec_d[0], 1));
        }
        else
        {
            // p0 = r0
            cublasErrchk(cublasDcopy(default_cublasHandle, rows_per_rank, r_local_d, 1, p_distributed.vec_d[0], 1));
        }

        // ak = rk^T * rk / pk^T * A * pk
        // has to be done for k=0 if x0 != 0
        // memcpy
        // allgather
        // result is accumulated in Ap_local_d
        cudaErrchk(cudaMemset(Ap_local_d, 0, rows_per_rank * sizeof(double)));
        // memcpy
        cudaErrchk(cudaMemcpy(p_distributed.vec_h[0], p_distributed.vec_d[0], rows_per_rank * sizeof(double), cudaMemcpyDeviceToHost));

        // post all send requests
        for (int i = 1; i < A_distributed.number_of_neighbours; i++)
        {
            int send_idx = p_distributed.neighbours[A_distributed.number_of_neighbours - i];
            int send_tag = std::abs(send_idx - rank);
            MPI_Isend(p_distributed.vec_h[0], p_distributed.rows_this_rank,
                        MPI_DOUBLE, send_idx, send_tag, comm, &send_requests[i - 1]);
        }

        for (int i = 0; i < A_distributed.number_of_neighbours; i++)
        {
            // loop over neighbors
            if (i < A_distributed.number_of_neighbours - 1)
            {
                int recv_idx = p_distributed.neighbours[i + 1];
                int recv_tag = std::abs(recv_idx - rank);
                MPI_Irecv(p_distributed.vec_h[i + 1], p_distributed.counts[recv_idx],
                            MPI_DOUBLE, recv_idx, recv_tag, comm, &recv_requests[i]);
            }

            // calc A*x0
            cusparseErrchk(cusparseSpMV(
                cusparseHandles[i], CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                A_distributed.descriptors[i], p_distributed.descriptors[i],
                &alpha, vecAp_local, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, A_distributed.buffer_d[i]));

            if (i < A_distributed.number_of_neighbours - 1)
            {
                MPI_Wait(&recv_requests[i], MPI_STATUS_IGNORE);
                int neighbour_idx = p_distributed.neighbours[i + 1];
                cudaErrchk(cudaMemcpyAsync(p_distributed.vec_d[i + 1], p_distributed.vec_h[i + 1], p_distributed.counts[neighbour_idx] * sizeof(double), cudaMemcpyHostToDevice, streams[i + 1]));
            }
        }
        MPI_Waitall(A_distributed.number_of_neighbours - 1, send_requests, MPI_STATUSES_IGNORE);

        for (int i = 0; i < A_distributed.number_of_neighbours; i++)
        {
            cudaErrchk(cudaStreamSynchronize(streams[i]));
        }

        cublasErrchk(cublasDdot(default_cublasHandle, rows_per_rank, p_distributed.vec_d[0], 1, Ap_local_d, 1, dot_h));
        // allreduce
        MPI_Allreduce(MPI_IN_PLACE, dot_h, 1, MPI_DOUBLE, MPI_SUM, comm);

        a = r_norm2_h[0] / dot_h[0];

        // xk+1 = xk + ak * pk
        cublasErrchk(cublasDaxpy(default_cublasHandle, rows_per_rank, &a, p_distributed.vec_d[0], 1, x_local_d, 1));

        // rk+1 = rk - ak * A * pk
        na = -a;
        cublasErrchk(cublasDaxpy(default_cublasHandle, rows_per_rank, &na, Ap_local_d, 1, r_local_d, 1));
        r0 = r_norm2_h[0];

        // r_norm2_h = r0*r0
        cublasErrchk(cublasDdot(default_cublasHandle, rows_per_rank, r_local_d, 1, r_local_d, 1, r_norm2_h));
        // allreduce
        MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);
        k++;
    }

    // end CG
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaStreamSynchronize(default_stream));
    time_taken[0] += omp_get_wtime();

    steps_taken[0] = k;
    if (rank == 0)
    {
        std::printf("iteration = %3d, residual = %e\n", k, sqrt(r_norm2_h[0]));
    }

    std::cout << "rank " << rank << " time_taken[0] " << time_taken[0] << std::endl;

    // copy solution to host
    cudaErrchk(cudaMemcpy(r_local_h, x_local_d, rows_per_rank * sizeof(double), cudaMemcpyDeviceToHost));

    double difference = 0;
    double sum_ref = 0;
    for (int i = 0; i < rows_per_rank; ++i)
    {
        difference += std::sqrt((r_local_h[i] - reference_solution[i + row_start_index]) * (r_local_h[i] - reference_solution[i + row_start_index]));
        sum_ref += std::sqrt((reference_solution[i + row_start_index]) * (reference_solution[i + row_start_index]));
    }
    MPI_Allreduce(MPI_IN_PLACE, &difference, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &sum_ref, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (rank == 0)
    {
        std::cout << "difference/sum_ref " << difference / sum_ref << std::endl;
    }

    cusparseErrchk(cusparseDestroy(default_cusparseHandle));
    cublasErrchk(cublasDestroy(default_cublasHandle));
    cudaErrchk(cudaStreamDestroy(default_stream));
    cusparseErrchk(cusparseDestroyDnVec(vecAp_local));
    cudaErrchk(cudaFree(row_indptr_local_d));
    cudaErrchk(cudaFree(col_indices_local_d));
    cudaErrchk(cudaFree(data_local_d));
    cudaErrchk(cudaFree(r_local_d));
    cudaErrchk(cudaFree(x_local_d));
    cudaErrchk(cudaFree(Ap_local_d));

    cudaErrchk(cudaFree(r_norm2_d));
    cudaErrchk(cudaFree(dot_d));

    delete[] row_indptr_local_h;
    delete[] col_indices_local_h;
    delete[] data_local_h;
    delete[] r_local_h;
    cudaErrchk(cudaFreeHost(r_norm2_h));
    cudaErrchk(cudaFreeHost(dot_h));

    for (int i = 0; i < A_distributed.number_of_neighbours; i++)
    {
        cudaErrchk(cudaStreamDestroy(streams[i]));
        cublasErrchk(cublasDestroy(cublasHandles[i]));
        cusparseErrchk(cusparseDestroy(cusparseHandles[i]));
    }
}

} // namespace own_test