#include "own_cg_to_compare.h"


namespace own_test{

// cusolver has CUSOLVER_STATUS_SUCCESS and not cudaSuccess, but they are the same
// this seems again kinda hacky
#define cudaErrchk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDAassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#define cublasErrchk(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUBLAS_STATUS_SUCCESS) 
   {
        //Did not find a counter part to cudaGetErrorString in cusolver
        fprintf(stderr,"CUBLASassert: %s %d\n", file, line);
        if (abort) exit(code);
   }
}

#define cusparseErrchk(ans) { cusparseAssert((ans), __FILE__, __LINE__); }
inline void cusparseAssert(cusparseStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUSPARSE_STATUS_SUCCESS) 
   {
        //Did not find a counter part to cudaGetErrorString in cusolver
        fprintf(stderr,"CUSPARSEassert: %s %d\n", file, line);
        if (abort) exit(code);
   }
}


void solve_cg_mpi(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution,
    double *starting_guess_h,
    int nnz,
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
    std::printf("rank %d size %d\n", rank, size);
    int rows_per_rank = matrix_size / size;

    if(matrix_size % size != 0){
        std::printf("Error: not dividable\n");
        // exit(1);
    }
    else{
        std::printf("Dividable\n");
    }
    int row_start_index = rank * rows_per_rank;

    cudaStream_t stream = NULL;
    
    cublasHandle_t cublasHandle = 0;
    cublasErrchk(cublasCreate(&cublasHandle));
    
    cusparseHandle_t cusparseHandle = 0;
    cusparseErrchk(cusparseCreate(&cusparseHandle));    


    cudaErrchk(cudaStreamCreate(&stream));
    cusparseErrchk(cusparseSetStream(cusparseHandle, stream));
    cublasErrchk(cublasSetStream(cublasHandle, stream));


    double *data_d = NULL;
    int *col_indices_d = NULL;
    int *row_indptr_d = NULL;
    double *rhs_d = NULL;
    double *x_d = NULL;
    double *p_d = NULL;
    double *Ax_d = NULL;
    double dot;

    cusparseSpMatDescr_t matA_local = NULL;


    double a, b, na;
    double alpha, beta, alpham1, r0, r1;
    size_t bufferSize = 0;
    void *buffer = NULL;

    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.0;


    //allocate memory on device
    cudaErrchk(cudaMalloc((void**)&data_d, nnz*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&col_indices_d, nnz*sizeof(int)));
    cudaErrchk(cudaMalloc((void**)&row_indptr_d, (matrix_size+1)*sizeof(int)));
    cudaErrchk(cudaMalloc((void**)&rhs_d, matrix_size*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&x_d, matrix_size*sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&p_d, matrix_size * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&Ax_d, matrix_size * sizeof(double)));


    int *row_indptr_local_d = NULL;
    int *col_indices_local_d = NULL;
    double *data_local_d = NULL;
    double *rhs_local_d = NULL;
    double *x_local_d = NULL;

    cudaErrchk(cudaMalloc((void**)&row_indptr_local_d, (rows_per_rank+1)*sizeof(int)));
    cudaErrchk(cudaMalloc((void**)&col_indices_local_d, nnz*sizeof(int)));
    cudaErrchk(cudaMalloc((void**)&data_local_d, nnz*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&rhs_local_d, rows_per_rank*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&x_local_d, rows_per_rank*sizeof(double)));

    int *row_indptr_local_h = new int[rows_per_rank+1];

    for (int i = 0; i < rows_per_rank+1; ++i) {
        row_indptr_local_h[i] = row_indptr_h[i+rank*rows_per_rank] - row_indptr_h[rank*rows_per_rank];
    }
    int nnz_local = row_indptr_local_h[rows_per_rank];
    int *col_indices_local_h = new int[nnz_local];
    double *data_local_h = new double[nnz_local];

    for (int i = 0; i < nnz_local; ++i) {
        col_indices_local_h[i] = col_indices_h[i+row_indptr_h[rank*rows_per_rank]];
        data_local_h[i] = data_h[i+row_indptr_h[rank*rows_per_rank]];
    }

    double *rhs_local_h = new double[rows_per_rank];
    for (int i = 0; i < rows_per_rank; ++i) {
        rhs_local_h[i] = rhs_h[i+rank*rows_per_rank];
    }

    double *p_local_h = new double[rows_per_rank];
    double *p_h = new double[matrix_size];

    cudaErrchk(cudaMemcpy(row_indptr_local_d, row_indptr_local_h, (rows_per_rank+1)*sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(col_indices_local_d, col_indices_local_h, nnz_local*sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(data_local_d, data_local_h, nnz_local*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(rhs_local_d, rhs_local_h, rows_per_rank*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemset(x_local_d, 0, rows_per_rank*sizeof(double)));


    // /* Wrap raw data into cuSPARSE generic API objects */
    // cusparseErrchk(cusparseCreateCsr(&matA, matrix_size, matrix_size,
    //                                     nnz, row_indptr_d, col_indices_d, data_d,
    //                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    //                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));


    /* Wrap raw data into cuSPARSE generic API objects */
    cusparseErrchk(cusparseCreateCsr(&matA_local, rows_per_rank, matrix_size,
                                        nnz_local, row_indptr_local_d, col_indices_local_d, data_local_d,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));



    cusparseDnVecDescr_t vecx = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecx, matrix_size, x_d, CUDA_R_64F));
    cusparseDnVecDescr_t vecp = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecp, matrix_size, p_d, CUDA_R_64F));
    cusparseDnVecDescr_t vecAx = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecAx, matrix_size, Ax_d, CUDA_R_64F));

    double *p_local_d = NULL;
    double *Ax_local_d = NULL;
    cudaErrchk(cudaMalloc((void **)&p_local_d, rows_per_rank * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&Ax_local_d, rows_per_rank * sizeof(double)));


    cusparseDnVecDescr_t vecp_local = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecp_local, rows_per_rank, p_local_d, CUDA_R_64F));
    cusparseDnVecDescr_t vecAx_local = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecAx_local, rows_per_rank, Ax_local_d, CUDA_R_64F));


    //copy data to device
    cudaErrchk(cudaMemcpy(rhs_d, rhs_h, matrix_size*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(col_indices_d, col_indices_h, nnz * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(row_indptr_d, row_indptr_h, (matrix_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(data_d, data_h, nnz * sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(x_d, starting_guess_h, matrix_size * sizeof(double), cudaMemcpyHostToDevice));    

    //figure out extra amount of memory needed
    cusparseErrchk(cusparseSpMV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA_local, vecx,
        &beta, vecAx_local, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    cudaErrchk(cudaMalloc(&buffer, bufferSize));



    // // calc A*x
    // cusparseErrchk(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                            &alpha, matA, vecx, &beta, vecAx, CUDA_R_64F,
    //                            CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    // // r = b - A*x
    // cublasErrchk(cublasDaxpy(cublasHandle, matrix_size, &alpham1, Ax_d, 1, rhs_d, 1));
    // cublasErrchk(cublasDdot(cublasHandle, matrix_size, rhs_d, 1, rhs_d, 1, &r1));
    // std::cout << r1 << std::endl;

    // int k = 1;
    // while (r1 > relative_tolerance * relative_tolerance && k <= 3) {
    //     if(k > 1){
    //         b = r1 / r0;
    //         cublasErrchk(cublasDscal(cublasHandle, matrix_size, &b, p_d, 1));
    //         cublasErrchk(cublasDaxpy(cublasHandle, matrix_size, &alpha, rhs_d, 1, p_d, 1));            
    //     }
    //     else {
    //         cublasErrchk(cublasDcopy(cublasHandle, matrix_size, rhs_d, 1, p_d, 1));
    //     }

    //     cusparseErrchk(cusparseSpMV(
    //         cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp,
    //         &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    
    //     cublasErrchk(cublasDdot(cublasHandle, matrix_size, p_d, 1, Ax_d, 1, &dot));
    //     std::cout << dot << std::endl;
    //     a = r1 / dot;

    //     cublasErrchk(cublasDaxpy(cublasHandle, matrix_size, &a, p_d, 1, x_d, 1));
    //     na = -a;
    //     cublasErrchk(cublasDaxpy(cublasHandle, matrix_size, &na, Ax_d, 1, rhs_d, 1));

    //     r0 = r1;
    //     cublasErrchk(cublasDdot(cublasHandle, matrix_size, rhs_d, 1, rhs_d, 1, &r1));
    //     cudaErrchk(cudaStreamSynchronize(stream));

    //     k++;
    // }


    //begin CG

    std::printf("CG starts\n");
    cudaErrchk(cudaStreamSynchronize(stream));
    cudaErrchk(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    time_taken[0] = -omp_get_wtime();

    // norm of rhs for convergence check
    double norm2_rhs = 0;
    cublasErrchk(cublasDdot(cublasHandle, rows_per_rank, rhs_local_d, 1, rhs_local_d, 1, &norm2_rhs));
    //allreduce
    MPI_Allreduce(MPI_IN_PLACE, &norm2_rhs, 1, MPI_DOUBLE, MPI_SUM, comm);

    // std::cout << norm2_rhs << std::endl;

    MPI_Allgather(rhs_h, rows_per_rank, MPI_DOUBLE, rhs_local_h, rows_per_rank, MPI_DOUBLE, comm);
    //memcpy
    cudaErrchk(cudaMemcpy(rhs_d, rhs_h, matrix_size * sizeof(double), cudaMemcpyHostToDevice));
    // calc A*x
    cusparseErrchk(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA_local, vecx, &beta, vecAx_local, CUDA_R_64F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    // r = b - A*x
    cublasErrchk(cublasDaxpy(cublasHandle, rows_per_rank, &alpham1, Ax_local_d, 1, rhs_local_d, 1));
    cublasErrchk(cublasDdot(cublasHandle, rows_per_rank, rhs_local_d, 1, rhs_local_d, 1, &r1));
    //allreduce
    MPI_Allreduce(MPI_IN_PLACE, &r1, 1, MPI_DOUBLE, MPI_SUM, comm);
    // std::cout << r1 << std::endl;

    int k = 1;
    while (r1/norm2_rhs > relative_tolerance * relative_tolerance && k <= max_iterations) {
        if(k > 1){
            b = r1 / r0;
            cublasErrchk(cublasDscal(cublasHandle, rows_per_rank, &b, p_local_d, 1));
            cublasErrchk(cublasDaxpy(cublasHandle, rows_per_rank, &alpha, rhs_local_d, 1, p_local_d, 1)); 
            // std::cout << b << std::endl;
        }
        else {
            cublasErrchk(cublasDcopy(cublasHandle, rows_per_rank, rhs_local_d, 1, p_local_d, 1));
        }
        // has to be done for k=0 if x0 != 0
        //memcpy
        //allgather
        //memcpy
        cudaErrchk(cudaMemcpy(p_local_h, p_local_d, rows_per_rank * sizeof(double), cudaMemcpyDeviceToHost));
        MPI_Allgather(p_local_h, rows_per_rank, MPI_DOUBLE, p_h, rows_per_rank, MPI_DOUBLE, comm);
        cudaErrchk(cudaMemcpy(p_d, p_h, matrix_size * sizeof(double), cudaMemcpyHostToDevice));
        cusparseErrchk(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA_local, vecp,
            &beta, vecAx_local, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));

        cublasErrchk(cublasDdot(cublasHandle, rows_per_rank, p_local_d, 1, Ax_local_d, 1, &dot));
        //allreduce        
        MPI_Allreduce(MPI_IN_PLACE, &dot, 1, MPI_DOUBLE, MPI_SUM, comm);
        // std::cout << dot << std::endl;
        a = r1 / dot;
        cublasErrchk(cublasDaxpy(cublasHandle, rows_per_rank, &a, p_local_d, 1, x_local_d, 1));
        na = -a;
        // std::cout << na << std::endl;
        cublasErrchk(cublasDaxpy(cublasHandle, rows_per_rank, &na, Ax_local_d, 1, rhs_local_d, 1));
        r0 = r1;

        cublasErrchk(cublasDdot(cublasHandle, rows_per_rank, rhs_local_d, 1, rhs_local_d, 1, &r1));
        //allreduce
        MPI_Allreduce(MPI_IN_PLACE, &r1, 1, MPI_DOUBLE, MPI_SUM, comm);
        // std::cout << r1 << std::endl;
        cudaErrchk(cudaStreamSynchronize(stream));

        k++;
    }

    steps_taken[0] = k;
    if(rank == 0){
        std::printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
    }


    //end CG
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaStreamSynchronize(stream));
    time_taken[0] += omp_get_wtime();

    std::cout << "rank " << rank << " time_taken[0] " << time_taken[0] << std::endl;

    //copy solution to host
    cudaErrchk(cudaMemcpy(rhs_local_h, x_local_d, rows_per_rank * sizeof(double), cudaMemcpyDeviceToHost));

    double difference = 0;
    double sum_ref = 0;
    for (int i = 0; i < rows_per_rank; ++i) {
        difference += std::sqrt( (rhs_local_h[i] - reference_solution[i+row_start_index]) * (rhs_local_h[i] - reference_solution[i+row_start_index]) );
        sum_ref += std::sqrt( (reference_solution[i+row_start_index]) * (reference_solution[i+row_start_index]) );
    }
    MPI_Allreduce(MPI_IN_PLACE, &difference, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &sum_ref, 1, MPI_DOUBLE, MPI_SUM, comm);
    if(rank == 0){
        std::cout << "difference/sum_ref " << difference/sum_ref << std::endl;
    }

    cudaErrchk(cudaDeviceSynchronize());

    MPI_Barrier(comm);
    std::cout << "cusparseHandle" << std::endl;
    cusparseErrchk(cusparseDestroy(cusparseHandle));
    MPI_Barrier(comm);
    std::cout << "cublasHandle" << std::endl;
    cublasErrchk(cublasDestroy(cublasHandle));
    MPI_Barrier(comm);
    std::cout << "stream" << std::endl;
    cudaErrchk(cudaStreamDestroy(stream));
    // MPI_Barrier(comm);
    // std::cout << "matA" << std::endl;
    // cusparseErrchk(cusparseDestroySpMat(matA));
    MPI_Barrier(comm);
    std::cout << "vecx" << std::endl;
    cusparseErrchk(cusparseDestroyDnVec(vecx));
    MPI_Barrier(comm);
    std::cout << "vecAx" << std::endl;
    cusparseErrchk(cusparseDestroyDnVec(vecAx));
    MPI_Barrier(comm);
    std::cout << "vecp" << std::endl;
    cusparseErrchk(cusparseDestroyDnVec(vecp));
    MPI_Barrier(comm);
    std::cout << "matA_local" << std::endl;
    cusparseErrchk(cusparseDestroySpMat(matA_local));
    MPI_Barrier(comm);
    std::cout << "vecAx_local" << std::endl;
    cusparseErrchk(cusparseDestroyDnVec(vecAx_local));
    MPI_Barrier(comm);
    std::cout << "vecp_local" << std::endl;
    cusparseErrchk(cusparseDestroyDnVec(vecp_local));
    MPI_Barrier(comm);
    std::cout << "buffer" << std::endl;
    cudaErrchk(cudaFree(buffer));
    MPI_Barrier(comm);
    std::cout << "data_d" << std::endl;
    cudaErrchk(cudaFree(data_d));
    MPI_Barrier(comm);
    std::cout << "col_indices_d" << std::endl;
    cudaErrchk(cudaFree(col_indices_d));
    MPI_Barrier(comm);
    std::cout << "row_indptr_d" << std::endl;
    cudaErrchk(cudaFree(row_indptr_d));
    MPI_Barrier(comm);
    std::cout << "rhs_d" << std::endl;
    cudaErrchk(cudaFree(rhs_d));
    MPI_Barrier(comm);
    std::cout << "x_d" << std::endl;
    cudaErrchk(cudaFree(x_d));
    MPI_Barrier(comm);
    std::cout << "p_d" << std::endl;
    cudaErrchk(cudaFree(p_d));
    MPI_Barrier(comm);
    std::cout << "Ax_d" << std::endl;
    cudaErrchk(cudaFree(Ax_d));
    MPI_Barrier(comm);
    std::cout << "row_indptr_local_d" << std::endl;
    cudaErrchk(cudaFree(row_indptr_local_d));
    MPI_Barrier(comm);
    std::cout << "col_indices_local_d" << std::endl;
    cudaErrchk(cudaFree(col_indices_local_d));
    MPI_Barrier(comm);
    std::cout << "data_local_d" << std::endl;
    cudaErrchk(cudaFree(data_local_d));
    MPI_Barrier(comm);
    std::cout << "rhs_local_d" << std::endl;
    cudaErrchk(cudaFree(rhs_local_d));
    MPI_Barrier(comm);
    std::cout << "x_local_d" << std::endl;
    cudaErrchk(cudaFree(x_local_d));
    MPI_Barrier(comm);
    std::cout << "p_local_d" << std::endl;
    cudaErrchk(cudaFree(p_local_d));
    MPI_Barrier(comm);
    std::cout << "Ax_local_d" << std::endl;
    cudaErrchk(cudaFree(Ax_local_d));
    MPI_Barrier(comm);
    std::cout << "row_indptr_local_h" << std::endl;
    delete[] row_indptr_local_h;
    MPI_Barrier(comm);
    std::cout << "col_indices_local_h" << std::endl;
    delete[] col_indices_local_h;
    MPI_Barrier(comm);
    std::cout << "data_local_h" << std::endl;
    // delete[] data_local_h;
    MPI_Barrier(comm);
    std::cout << "rhs_local_h" << std::endl;
    // delete[] rhs_local_h;
    MPI_Barrier(comm);
    std::cout << "p_local_h" << std::endl;
    // delete[] p_local_h;
    MPI_Barrier(comm);
    std::cout << "p_h" << std::endl;
    // delete[] p_h;

    MPI_Barrier(comm);
    std::cout << "end" << std::endl;
    MPI_Barrier(comm);
}

} // namespace own_test