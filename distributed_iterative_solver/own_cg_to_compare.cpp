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
    int rows_per_rank = matrix_size / size;
    
    // if(matrix_size % size != 0){
    //     std::printf("Error: not dividable\n");
    //     // exit(1);
    // }
    // else{
    //     std::printf("Dividable\n");
    // }
    int row_start_index = rank * rows_per_rank;
    if(rank == size-1){
        rows_per_rank += matrix_size % size;
    }

    int *row_indptr_local_h = new int[rows_per_rank+1];

    for (int i = 0; i < rows_per_rank+1; ++i) {
        row_indptr_local_h[i] = row_indptr_h[i+row_start_index] - row_indptr_h[row_start_index];
    }
    int nnz_local = row_indptr_local_h[rows_per_rank];
    int *col_indices_local_h = new int[nnz_local];
    double *data_local_h = new double[nnz_local];

    for (int i = 0; i < nnz_local; ++i) {
        col_indices_local_h[i] = col_indices_h[i+row_indptr_h[row_start_index]];
        data_local_h[i] = data_h[i+row_indptr_h[row_start_index]];
    }

    double *rhs_local_h = new double[rows_per_rank];
    for (int i = 0; i < rows_per_rank; ++i) {
        rhs_local_h[i] = rhs_h[i+row_start_index];
    }

    double *p_local_h = new double[rows_per_rank];
    double *p_h = new double[matrix_size];


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

    double *r1_d = NULL;
    double *dot_d = NULL;
    cudaErrchk(cudaMalloc((void**)&r1_d, sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&dot_d, sizeof(double)));

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
    cudaErrchk(cudaMalloc((void**)&col_indices_local_d, nnz_local*sizeof(int)));
    cudaErrchk(cudaMalloc((void**)&data_local_d, nnz_local*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&rhs_local_d, rows_per_rank*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&x_local_d, rows_per_rank*sizeof(double)));



    cudaErrchk(cudaMemcpy(row_indptr_local_d, row_indptr_local_h, (rows_per_rank+1)*sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(col_indices_local_d, col_indices_local_h, nnz_local*sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(data_local_d, data_local_h, nnz_local*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(rhs_local_d, rhs_local_h, rows_per_rank*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemset(x_local_d, 0, rows_per_rank*sizeof(double)));


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



    // prepare for allgatherv
    int recvcounts[size];
    int displs[size];
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = matrix_size / size;
        displs[i] = i * (matrix_size / size);
    }
    recvcounts[size-1] += matrix_size % size;

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

    //MPI_Allgather(rhs_local_h, rows_per_rank, MPI_DOUBLE, rhs_h, rows_per_rank, MPI_DOUBLE, comm);
    MPI_Allgatherv(rhs_local_h, rows_per_rank, MPI_DOUBLE, rhs_h, recvcounts, displs, MPI_DOUBLE, comm);
    //memcpy
    cudaErrchk(cudaMemcpy(rhs_d, rhs_h, matrix_size * sizeof(double), cudaMemcpyHostToDevice));
    // calc A*x
    cusparseErrchk(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA_local, vecx, &beta, vecAx_local, CUDA_R_64F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    // r = b - A*x
    cublasErrchk(cublasDaxpy(cublasHandle, rows_per_rank, &alpham1, Ax_local_d, 1, rhs_local_d, 1));
    cublasErrchk(cublasDdot(cublasHandle, rows_per_rank, rhs_local_d, 1, rhs_local_d, 1, r1_d));
    //memcpy
    cudaErrchk(cudaMemcpy(&r1, r1_d, sizeof(double), cudaMemcpyDeviceToHost));
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
        //MPI_Allgather(p_local_h, rows_per_rank, MPI_DOUBLE, p_h, rows_per_rank, MPI_DOUBLE, comm);
        MPI_Allgatherv(p_local_h, rows_per_rank, MPI_DOUBLE, p_h, recvcounts, displs, MPI_DOUBLE, comm);
        cudaErrchk(cudaMemcpy(p_d, p_h, matrix_size * sizeof(double), cudaMemcpyHostToDevice));
        cusparseErrchk(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA_local, vecp,
            &beta, vecAx_local, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));

        cublasErrchk(cublasDdot(cublasHandle, rows_per_rank, p_local_d, 1, Ax_local_d, 1, dot_d));
        //memcpy
        cudaErrchk(cudaMemcpy(&dot, dot_d, sizeof(double), cudaMemcpyDeviceToHost));
        //allreduce        
        MPI_Allreduce(MPI_IN_PLACE, &dot, 1, MPI_DOUBLE, MPI_SUM, comm);
        // std::cout << dot << std::endl;
        a = r1 / dot;
        cublasErrchk(cublasDaxpy(cublasHandle, rows_per_rank, &a, p_local_d, 1, x_local_d, 1));
        na = -a;
        // std::cout << na << std::endl;
        cublasErrchk(cublasDaxpy(cublasHandle, rows_per_rank, &na, Ax_local_d, 1, rhs_local_d, 1));
        r0 = r1;

        cublasErrchk(cublasDdot(cublasHandle, rows_per_rank, rhs_local_d, 1, rhs_local_d, 1, r1_d));
        //memcpy
        cudaErrchk(cudaMemcpy(&r1, r1_d, sizeof(double), cudaMemcpyDeviceToHost));
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

    cusparseErrchk(cusparseDestroy(cusparseHandle));
    cublasErrchk(cublasDestroy(cublasHandle));
    cudaErrchk(cudaStreamDestroy(stream));
    cusparseErrchk(cusparseDestroyDnVec(vecx));
    cusparseErrchk(cusparseDestroyDnVec(vecAx));
    cusparseErrchk(cusparseDestroyDnVec(vecp));
    cusparseErrchk(cusparseDestroySpMat(matA_local));
    cusparseErrchk(cusparseDestroyDnVec(vecAx_local));
    cusparseErrchk(cusparseDestroyDnVec(vecp_local));
    cudaErrchk(cudaFree(buffer));
    cudaErrchk(cudaFree(data_d));
    cudaErrchk(cudaFree(col_indices_d));
    cudaErrchk(cudaFree(row_indptr_d));
    cudaErrchk(cudaFree(rhs_d));
    cudaErrchk(cudaFree(x_d));
    cudaErrchk(cudaFree(p_d));
    cudaErrchk(cudaFree(Ax_d));
    cudaErrchk(cudaFree(row_indptr_local_d));
    cudaErrchk(cudaFree(col_indices_local_d));
    cudaErrchk(cudaFree(data_local_d));
    cudaErrchk(cudaFree(rhs_local_d));
    cudaErrchk(cudaFree(x_local_d));
    cudaErrchk(cudaFree(p_local_d));
    cudaErrchk(cudaFree(Ax_local_d));

    cudaErrchk(cudaFree(r1_d));
    cudaErrchk(cudaFree(dot_d));

    delete[] row_indptr_local_h;
    delete[] col_indices_local_h;
    delete[] data_local_h;
    delete[] rhs_local_h;
    delete[] p_local_h;
    delete[] p_h;


}

} // namespace own_test