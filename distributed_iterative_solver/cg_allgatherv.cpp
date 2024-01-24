#include "own_cg_to_compare.h"


namespace own_test{

void solve_cg_allgatherv2(
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
    // replaces new with cudaMallocHost
    // for memcpy



    MPI_Barrier(comm);
    std::printf("CG starts\n");

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);


    // prepare for allgatherv
    int recvcounts[size];
    int displs[size];
    int rows_per_rank = matrix_size / size;    
    for (int i = 0; i < size; ++i) {
        if(i < matrix_size % size){
            recvcounts[i] = rows_per_rank+1;
        }
        else{
            recvcounts[i] = rows_per_rank;
        }
    }
    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
        displs[i] = displs[i-1] + recvcounts[i-1];
    }
    int row_start_index = displs[rank];
    rows_per_rank = recvcounts[rank];



    double *p_local_h;
    cudaErrchk(cudaMallocHost((void**)&p_local_h, rows_per_rank * sizeof(double)));


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


    cusparseDnVecDescr_t vecAp_local = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecAp_local, rows_per_rank, Ax_local_d, CUDA_R_64F));




    //figure out extra amount of memory needed
    cusparseErrchk(cusparseSpMV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA_local, vecp,
        &beta, vecAp_local, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    cudaErrchk(cudaMalloc(&buffer, bufferSize));


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

    MPI_Allgatherv(p_local_h, rows_per_rank, MPI_DOUBLE, p_h, recvcounts, displs, MPI_DOUBLE, comm);
    //memcpy
    cudaErrchk(cudaMemcpy(p_d, p_h, matrix_size * sizeof(double), cudaMemcpyHostToDevice));
    // calc A*x0
    cusparseErrchk(cusparseSpMV(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA_local, vecp,
        &beta, vecAp_local, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));


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
        cudaErrchk(cudaMemcpy(p_local_h, p_local_d, rows_per_rank * sizeof(double), cudaMemcpyDeviceToHost));
        MPI_Allgatherv(p_local_h, rows_per_rank, MPI_DOUBLE, p_h, recvcounts, displs, MPI_DOUBLE, comm);
        cudaErrchk(cudaMemcpy(p_d, p_h, matrix_size * sizeof(double), cudaMemcpyHostToDevice));
        cusparseErrchk(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA_local, vecp,
            &beta, vecAp_local, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));

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

        k++;
    }

    //end CG
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaStreamSynchronize(stream));
    time_taken[0] += omp_get_wtime();

    steps_taken[0] = k;
    if(rank == 0){
        std::printf("iteration = %3d, residual = %e\n", k, sqrt(r_norm2));
    }


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
    delete[] p_h;
    cudaErrchk(cudaFreeHost(p_local_h));
}
    
} //namespace own_test  