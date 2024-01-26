#include "cg_own_implementations.h"


namespace own_test{

void solve_cg1(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution_h,
    double *starting_guess_h,
    int nnz,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    int *steps_taken,
    double *time_taken)
{

    cudaStream_t stream = NULL;
    
    
    cusparseHandle_t cusparseHandle = 0;
    cusparseErrchk(cusparseCreate(&cusparseHandle));    

    cublasHandle_t cublasHandle = 0;
    cublasErrchk(cublasCreate(&cublasHandle));

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
    double dot_h;

    cusparseSpMatDescr_t matA = NULL;

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

    /* Wrap raw data into cuSPARSE generic API objects */
    cusparseErrchk(cusparseCreateCsr(&matA, matrix_size, matrix_size,
                                        nnz, row_indptr_d, col_indices_d, data_d,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));



    cusparseDnVecDescr_t vecx = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecx, matrix_size, x_d, CUDA_R_64F));
    cusparseDnVecDescr_t vecp = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecp, matrix_size, p_d, CUDA_R_64F));
    cusparseDnVecDescr_t vecAx = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecAx, matrix_size, Ax_d, CUDA_R_64F));


    //copy data to device

    cudaErrchk(cudaMemcpy(rhs_d, rhs_h, matrix_size*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(col_indices_d, col_indices_h, nnz * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(row_indptr_d, row_indptr_h, (matrix_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(data_d, data_h, nnz * sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(x_d, starting_guess_h, matrix_size * sizeof(double), cudaMemcpyHostToDevice));    

    //figure out extra amount of memory needed

    cusparseErrchk(cusparseSpMV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
        &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    cudaErrchk(cudaMalloc(&buffer, bufferSize));


    //begin CG
    cudaErrchk(cudaStreamSynchronize(stream));
    cudaErrchk(cudaDeviceSynchronize());
    time_taken[0] = -omp_get_wtime();
    // dot_h of rhs for convergence check
    double norm2_rhs = 0;
    cublasErrchk(cublasDdot(cublasHandle, matrix_size, rhs_d, 1, rhs_d, 1, &norm2_rhs));

    // calc A*x
    cusparseErrchk(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecx, &beta, vecAx, CUDA_R_64F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    // r = b - A*x
    cublasErrchk(cublasDaxpy(cublasHandle, matrix_size, &alpham1, Ax_d, 1, rhs_d, 1));
    cublasErrchk(cublasDdot(cublasHandle, matrix_size, rhs_d, 1, rhs_d, 1, &r1));


    int k = 1;
    while (r1 / norm2_rhs > relative_tolerance * relative_tolerance && k <= max_iterations) {
        if(k > 1){
            b = r1 / r0;
            cublasErrchk(cublasDscal(cublasHandle, matrix_size, &b, p_d, 1));
            cublasErrchk(cublasDaxpy(cublasHandle, matrix_size, &alpha, rhs_d, 1, p_d, 1));            
        }
        else {
            cublasErrchk(cublasDcopy(cublasHandle, matrix_size, rhs_d, 1, p_d, 1));
        }

        cusparseErrchk(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp,
            &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    
        cublasErrchk(cublasDdot(cublasHandle, matrix_size, p_d, 1, Ax_d, 1, &dot_h));
        a = r1 / dot_h;

        cublasErrchk(cublasDaxpy(cublasHandle, matrix_size, &a, p_d, 1, x_d, 1));
        na = -a;
        cublasErrchk(cublasDaxpy(cublasHandle, matrix_size, &na, Ax_d, 1, rhs_d, 1));

        r0 = r1;
        cublasErrchk(cublasDdot(cublasHandle, matrix_size, rhs_d, 1, rhs_d, 1, &r1));
        k++;
    }

    //end CG
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaStreamSynchronize(stream));
    time_taken[0] += omp_get_wtime();
    std::cout << "time_taken[0] " << time_taken[0] << std::endl;

    std::printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
    steps_taken[0] = k;


    //copy solution to host
    double *x_h = new double[matrix_size];
    cudaErrchk(cudaMemcpy(x_h, x_d, matrix_size * sizeof(double), cudaMemcpyDeviceToHost));


    double difference = 0;
    double sum_ref = 0;
    for (int i = 0; i < matrix_size; ++i) {
        difference += std::sqrt( (x_h[i] - reference_solution_h[i]) * (x_h[i] - reference_solution_h[i]) );
        sum_ref += std::sqrt( (reference_solution_h[i]) * (reference_solution_h[i]) );
    }
    delete[] x_h;
    std::cout << "difference/sum_ref " << difference/sum_ref << std::endl;

    if(cusparseHandle) {
        cusparseErrchk(cusparseDestroy(cusparseHandle));
    }
    if(cublasHandle) {
        cublasErrchk(cublasDestroy(cublasHandle));
    }
    if(stream) {
        cudaErrchk(cudaStreamDestroy(stream));
    }
    if(matA) {
        cusparseErrchk(cusparseDestroySpMat(matA));
    }
    if(vecx) {
        cusparseErrchk(cusparseDestroyDnVec(vecx));
    }
    if(vecAx) {
        cusparseErrchk(cusparseDestroyDnVec(vecAx));
    }
    if(vecp) {
        cusparseErrchk(cusparseDestroyDnVec(vecp));
    }

    if (buffer) {
        cudaErrchk(cudaFree(buffer));
    }
    if(data_d){
        cudaErrchk(cudaFree(data_d));
    }
    if(col_indices_d){
        cudaErrchk(cudaFree(col_indices_d));
    }
    if(row_indptr_d){
        cudaErrchk(cudaFree(row_indptr_d));
    }
    if(rhs_d){
        cudaErrchk(cudaFree(rhs_d));
    }
    if(x_d){
        cudaErrchk(cudaFree(x_d));
    }
    if(p_d){
        cudaErrchk(cudaFree(p_d));
    }
    if(Ax_d){
        cudaErrchk(cudaFree(Ax_d));
    }

}



void solve_cg2(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution_h,
    double *starting_guess_h,
    int nnz,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    int *steps_taken,
    double *time_taken)
{

    //fuses some operations with custom kernels

    cudaStream_t stream = NULL;
    
    
    cusparseHandle_t cusparseHandle = 0;
    cusparseErrchk(cusparseCreate(&cusparseHandle));    

    cublasHandle_t cublasHandle = 0;
    cublasErrchk(cublasCreate(&cublasHandle));

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
    double dot_h;

    cusparseSpMatDescr_t matA = NULL;

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

    /* Wrap raw data into cuSPARSE generic API objects */
    cusparseErrchk(cusparseCreateCsr(&matA, matrix_size, matrix_size,
                                        nnz, row_indptr_d, col_indices_d, data_d,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));



    cusparseDnVecDescr_t vecx = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecx, matrix_size, x_d, CUDA_R_64F));
    cusparseDnVecDescr_t vecp = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecp, matrix_size, p_d, CUDA_R_64F));
    cusparseDnVecDescr_t vecAx = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecAx, matrix_size, Ax_d, CUDA_R_64F));


    //copy data to device

    cudaErrchk(cudaMemcpy(rhs_d, rhs_h, matrix_size*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(col_indices_d, col_indices_h, nnz * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(row_indptr_d, row_indptr_h, (matrix_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(data_d, data_h, nnz * sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(x_d, starting_guess_h, matrix_size * sizeof(double), cudaMemcpyHostToDevice));    

    //figure out extra amount of memory needed

    cusparseErrchk(cusparseSpMV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
        &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    cudaErrchk(cudaMalloc(&buffer, bufferSize));


    //begin CG
    cudaErrchk(cudaStreamSynchronize(stream));
    cudaErrchk(cudaDeviceSynchronize());
    time_taken[0] = -omp_get_wtime();
    // dot_h of rhs for convergence check
    double norm2_rhs = 0;
    cublasErrchk(cublasDdot(cublasHandle, matrix_size, rhs_d, 1, rhs_d, 1, &norm2_rhs));

    // calc A*x
    cusparseErrchk(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecx, &beta, vecAx, CUDA_R_64F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    // r = b - A*x
    cublasErrchk(cublasDaxpy(cublasHandle, matrix_size, &alpham1, Ax_d, 1, rhs_d, 1));
    cublasErrchk(cublasDdot(cublasHandle, matrix_size, rhs_d, 1, rhs_d, 1, &r1));


    int k = 1;
    while (r1 / norm2_rhs > relative_tolerance * relative_tolerance && k <= max_iterations) {
        if(k > 1){
            b = r1 / r0;
            // pk+1 = rk+1 + b*pk
            cg_addvec(rhs_d, b, p_d, matrix_size, stream);          
        }
        else {
            cublasErrchk(cublasDcopy(cublasHandle, matrix_size, rhs_d, 1, p_d, 1));
        }

        cusparseErrchk(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp,
            &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    
        cublasErrchk(cublasDdot(cublasHandle, matrix_size, p_d, 1, Ax_d, 1, &dot_h));
        a = r1 / dot_h;
        // xk+1 = xk + ak * pk
        cublasErrchk(cublasDaxpy(cublasHandle, matrix_size, &a, p_d, 1, x_d, 1));
        // rk+1 = rk - ak * A * pk
        na = -a;
        cublasErrchk(cublasDaxpy(cublasHandle, matrix_size, &na, Ax_d, 1, rhs_d, 1));

        r0 = r1;
        cublasErrchk(cublasDdot(cublasHandle, matrix_size, rhs_d, 1, rhs_d, 1, &r1));
        k++;
    }

    //end CG
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaStreamSynchronize(stream));
    time_taken[0] += omp_get_wtime();
    std::cout << "time_taken[0] " << time_taken[0] << std::endl;

    std::printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
    steps_taken[0] = k;


    //copy solution to host
    double *x_h = new double[matrix_size];
    cudaErrchk(cudaMemcpy(x_h, x_d, matrix_size * sizeof(double), cudaMemcpyDeviceToHost));


    double difference = 0;
    double sum_ref = 0;
    for (int i = 0; i < matrix_size; ++i) {
        difference += std::sqrt( (x_h[i] - reference_solution_h[i]) * (x_h[i] - reference_solution_h[i]) );
        sum_ref += std::sqrt( (reference_solution_h[i]) * (reference_solution_h[i]) );
    }
    delete[] x_h;
    std::cout << "difference/sum_ref " << difference/sum_ref << std::endl;

    if(cusparseHandle) {
        cusparseErrchk(cusparseDestroy(cusparseHandle));
    }
    if(cublasHandle) {
        cublasErrchk(cublasDestroy(cublasHandle));
    }
    if(stream) {
        cudaErrchk(cudaStreamDestroy(stream));
    }
    if(matA) {
        cusparseErrchk(cusparseDestroySpMat(matA));
    }
    if(vecx) {
        cusparseErrchk(cusparseDestroyDnVec(vecx));
    }
    if(vecAx) {
        cusparseErrchk(cusparseDestroyDnVec(vecAx));
    }
    if(vecp) {
        cusparseErrchk(cusparseDestroyDnVec(vecp));
    }

    if (buffer) {
        cudaErrchk(cudaFree(buffer));
    }
    if(data_d){
        cudaErrchk(cudaFree(data_d));
    }
    if(col_indices_d){
        cudaErrchk(cudaFree(col_indices_d));
    }
    if(row_indptr_d){
        cudaErrchk(cudaFree(row_indptr_d));
    }
    if(rhs_d){
        cudaErrchk(cudaFree(rhs_d));
    }
    if(x_d){
        cudaErrchk(cudaFree(x_d));
    }
    if(p_d){
        cudaErrchk(cudaFree(p_d));
    }
    if(Ax_d){
        cudaErrchk(cudaFree(Ax_d));
    }

}



void solve_cg3(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution_h,
    double *starting_guess_h,
    int nnz,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    int *steps_taken,
    double *time_taken)
{

    //fuses some operations with custom kernels

    cudaStream_t stream = NULL;
    
    
    cusparseHandle_t cusparseHandle = 0;
    cusparseErrchk(cusparseCreate(&cusparseHandle));    

    cublasHandle_t cublasHandle = 0;
    cublasErrchk(cublasCreate(&cublasHandle));

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
    double dot_h;

    cusparseSpMatDescr_t matA = NULL;

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

    /* Wrap raw data into cuSPARSE generic API objects */
    cusparseErrchk(cusparseCreateCsr(&matA, matrix_size, matrix_size,
                                        nnz, row_indptr_d, col_indices_d, data_d,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));



    cusparseDnVecDescr_t vecx = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecx, matrix_size, x_d, CUDA_R_64F));
    cusparseDnVecDescr_t vecp = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecp, matrix_size, p_d, CUDA_R_64F));
    cusparseDnVecDescr_t vecAx = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecAx, matrix_size, Ax_d, CUDA_R_64F));


    //copy data to device

    cudaErrchk(cudaMemcpy(rhs_d, rhs_h, matrix_size*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(col_indices_d, col_indices_h, nnz * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(row_indptr_d, row_indptr_h, (matrix_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(data_d, data_h, nnz * sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(x_d, starting_guess_h, matrix_size * sizeof(double), cudaMemcpyHostToDevice));    

    //figure out extra amount of memory needed

    cusparseErrchk(cusparseSpMV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
        &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    cudaErrchk(cudaMalloc(&buffer, bufferSize));


    //begin CG
    cudaErrchk(cudaStreamSynchronize(stream));
    cudaErrchk(cudaDeviceSynchronize());
    time_taken[0] = -omp_get_wtime();
    // dot_h of rhs for convergence check
    double norm2_rhs = 0;
    cublasErrchk(cublasDdot(cublasHandle, matrix_size, rhs_d, 1, rhs_d, 1, &norm2_rhs));

    // calc A*x
    cusparseErrchk(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecx, &beta, vecAx, CUDA_R_64F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    // r = b - A*x
    cublasErrchk(cublasDaxpy(cublasHandle, matrix_size, &alpham1, Ax_d, 1, rhs_d, 1));
    cublasErrchk(cublasDdot(cublasHandle, matrix_size, rhs_d, 1, rhs_d, 1, &r1));


    int k = 1;
    while (r1 / norm2_rhs > relative_tolerance * relative_tolerance && k <= max_iterations) {
        if(k > 1){
            b = r1 / r0;
            // pk+1 = rk+1 + b*pk
            cg_addvec(rhs_d, b, p_d, matrix_size, stream);          
        }
        else {
            cublasErrchk(cublasDcopy(cublasHandle, matrix_size, rhs_d, 1, p_d, 1));
        }

        cusparseErrchk(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp,
            &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    
        cublasErrchk(cublasDdot(cublasHandle, matrix_size, p_d, 1, Ax_d, 1, &dot_h));
        // xk+1 = xk + ak * pk
        // rk+1 = rk - ak * A * pk
        a = r1 / dot_h;
        na = -a;
        fused_daxpy(a, na, p_d, Ax_d, x_d, rhs_d, matrix_size, stream);
        r0 = r1;
        cublasErrchk(cublasDdot(cublasHandle, matrix_size, rhs_d, 1, rhs_d, 1, &r1));
        k++;
    }

    //end CG
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaStreamSynchronize(stream));
    time_taken[0] += omp_get_wtime();
    std::cout << "time_taken[0] " << time_taken[0] << std::endl;

    std::printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
    steps_taken[0] = k;


    //copy solution to host
    double *x_h = new double[matrix_size];
    cudaErrchk(cudaMemcpy(x_h, x_d, matrix_size * sizeof(double), cudaMemcpyDeviceToHost));


    double difference = 0;
    double sum_ref = 0;
    for (int i = 0; i < matrix_size; ++i) {
        difference += std::sqrt( (x_h[i] - reference_solution_h[i]) * (x_h[i] - reference_solution_h[i]) );
        sum_ref += std::sqrt( (reference_solution_h[i]) * (reference_solution_h[i]) );
    }
    delete[] x_h;
    std::cout << "difference/sum_ref " << difference/sum_ref << std::endl;

    if(cusparseHandle) {
        cusparseErrchk(cusparseDestroy(cusparseHandle));
    }
    if(cublasHandle) {
        cublasErrchk(cublasDestroy(cublasHandle));
    }
    if(stream) {
        cudaErrchk(cudaStreamDestroy(stream));
    }
    if(matA) {
        cusparseErrchk(cusparseDestroySpMat(matA));
    }
    if(vecx) {
        cusparseErrchk(cusparseDestroyDnVec(vecx));
    }
    if(vecAx) {
        cusparseErrchk(cusparseDestroyDnVec(vecAx));
    }
    if(vecp) {
        cusparseErrchk(cusparseDestroyDnVec(vecp));
    }

    if (buffer) {
        cudaErrchk(cudaFree(buffer));
    }
    if(data_d){
        cudaErrchk(cudaFree(data_d));
    }
    if(col_indices_d){
        cudaErrchk(cudaFree(col_indices_d));
    }
    if(row_indptr_d){
        cudaErrchk(cudaFree(row_indptr_d));
    }
    if(rhs_d){
        cudaErrchk(cudaFree(rhs_d));
    }
    if(x_d){
        cudaErrchk(cudaFree(x_d));
    }
    if(p_d){
        cudaErrchk(cudaFree(p_d));
    }
    if(Ax_d){
        cudaErrchk(cudaFree(Ax_d));
    }

}


void solve_cg4(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution_h,
    double *starting_guess_h,
    int nnz,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    int *steps_taken,
    double *time_taken)
{

    //fuses some operations with custom kernels

    cudaStream_t stream = NULL;
    
    
    cusparseHandle_t cusparseHandle = 0;
    cusparseErrchk(cusparseCreate(&cusparseHandle));    

    cublasHandle_t cublasHandle = 0;
    cublasErrchk(cublasCreate(&cublasHandle));

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
    double dot_h;

    cusparseSpMatDescr_t matA = NULL;

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

    /* Wrap raw data into cuSPARSE generic API objects */
    cusparseErrchk(cusparseCreateCsr(&matA, matrix_size, matrix_size,
                                        nnz, row_indptr_d, col_indices_d, data_d,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));



    cusparseDnVecDescr_t vecx = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecx, matrix_size, x_d, CUDA_R_64F));
    cusparseDnVecDescr_t vecp = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecp, matrix_size, p_d, CUDA_R_64F));
    cusparseDnVecDescr_t vecAx = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecAx, matrix_size, Ax_d, CUDA_R_64F));


    //copy data to device

    cudaErrchk(cudaMemcpy(rhs_d, rhs_h, matrix_size*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(col_indices_d, col_indices_h, nnz * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(row_indptr_d, row_indptr_h, (matrix_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(data_d, data_h, nnz * sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(x_d, starting_guess_h, matrix_size * sizeof(double), cudaMemcpyHostToDevice));    

    //figure out extra amount of memory needed

    cusparseErrchk(cusparseSpMV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
        &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    cudaErrchk(cudaMalloc(&buffer, bufferSize));


    //begin CG
    cudaErrchk(cudaStreamSynchronize(stream));
    cudaErrchk(cudaDeviceSynchronize());
    time_taken[0] = -omp_get_wtime();
    // dot_h of rhs for convergence check
    double norm2_rhs = 0;
    cublasErrchk(cublasDdot(cublasHandle, matrix_size, rhs_d, 1, rhs_d, 1, &norm2_rhs));

    // calc A*x
    cusparseErrchk(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecx, &beta, vecAx, CUDA_R_64F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    // r = b - A*x
    cublasErrchk(cublasDaxpy(cublasHandle, matrix_size, &alpham1, Ax_d, 1, rhs_d, 1));
    cublasErrchk(cublasDdot(cublasHandle, matrix_size, rhs_d, 1, rhs_d, 1, &r1));


    int k = 1;
    while (r1 / norm2_rhs > relative_tolerance * relative_tolerance && k <= max_iterations) {
        if(k > 1){
            b = r1 / r0;
            // pk+1 = rk+1 + b*pk
            cg_addvec(rhs_d, b, p_d, matrix_size, stream);          
        }
        else {
            cublasErrchk(cublasDcopy(cublasHandle, matrix_size, rhs_d, 1, p_d, 1));
        }

        cusparseErrchk(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp,
            &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    
        cublasErrchk(cublasDdot(cublasHandle, matrix_size, p_d, 1, Ax_d, 1, &dot_h));
        // xk+1 = xk + ak * pk
        // rk+1 = rk - ak * A * pk
        a = r1 / dot_h;
        na = -a;
        r0 = r1;
        // TODO: fuse daxpy and ddot
        fused_daxpy2(a, na, p_d, Ax_d, x_d, rhs_d, matrix_size, stream);
        cublasErrchk(cublasDdot(cublasHandle, matrix_size, rhs_d, 1, rhs_d, 1, &r1));
        k++;
    }

    //end CG
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaStreamSynchronize(stream));
    time_taken[0] += omp_get_wtime();
    std::cout << "time_taken[0] " << time_taken[0] << std::endl;

    std::printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
    steps_taken[0] = k;


    //copy solution to host
    double *x_h = new double[matrix_size];
    cudaErrchk(cudaMemcpy(x_h, x_d, matrix_size * sizeof(double), cudaMemcpyDeviceToHost));


    double difference = 0;
    double sum_ref = 0;
    for (int i = 0; i < matrix_size; ++i) {
        difference += std::sqrt( (x_h[i] - reference_solution_h[i]) * (x_h[i] - reference_solution_h[i]) );
        sum_ref += std::sqrt( (reference_solution_h[i]) * (reference_solution_h[i]) );
    }
    delete[] x_h;
    std::cout << "difference/sum_ref " << difference/sum_ref << std::endl;

    if(cusparseHandle) {
        cusparseErrchk(cusparseDestroy(cusparseHandle));
    }
    if(cublasHandle) {
        cublasErrchk(cublasDestroy(cublasHandle));
    }
    if(stream) {
        cudaErrchk(cudaStreamDestroy(stream));
    }
    if(matA) {
        cusparseErrchk(cusparseDestroySpMat(matA));
    }
    if(vecx) {
        cusparseErrchk(cusparseDestroyDnVec(vecx));
    }
    if(vecAx) {
        cusparseErrchk(cusparseDestroyDnVec(vecAx));
    }
    if(vecp) {
        cusparseErrchk(cusparseDestroyDnVec(vecp));
    }

    if (buffer) {
        cudaErrchk(cudaFree(buffer));
    }
    if(data_d){
        cudaErrchk(cudaFree(data_d));
    }
    if(col_indices_d){
        cudaErrchk(cudaFree(col_indices_d));
    }
    if(row_indptr_d){
        cudaErrchk(cudaFree(row_indptr_d));
    }
    if(rhs_d){
        cudaErrchk(cudaFree(rhs_d));
    }
    if(x_d){
        cudaErrchk(cudaFree(x_d));
    }
    if(p_d){
        cudaErrchk(cudaFree(p_d));
    }
    if(Ax_d){
        cudaErrchk(cudaFree(Ax_d));
    }

}

} // namespace own_test