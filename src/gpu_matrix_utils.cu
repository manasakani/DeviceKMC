#include "cuda_wrapper.h"

// check that sparse and dense versions are the same
void check_sparse_dense_match(int m, int nnz, double *dense_matrix, int* d_csrRowPtr, int* d_csrColInd, double* d_csrVal){
    
    double *h_D = (double *)calloc(m*m, sizeof(double));
    double *h_D_csr = (double *)calloc(nnz, sizeof(double));
    int *h_pointers = (int *)calloc((m + 1), sizeof(int));
    int *h_inds = (int *)calloc(nnz, sizeof(int));

    gpuErrchk( cudaMemcpy(h_D, dense_matrix, m*m * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_D_csr, d_csrVal, nnz * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_pointers, d_csrRowPtr, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_inds, d_csrColInd, nnz * sizeof(int), cudaMemcpyDeviceToHost) );

    int nnz_count = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < m; col++) {
            int i = row * m + col;  // Linear index in dense matrix
            // Check if the element in the dense matrix is non-zero
            if (h_D[i] != 0) {
                // Compare the row and column indices
                if (h_D[i] != h_D_csr[nnz_count] || col != h_inds[nnz_count]) {
                    std::cout << "Mismatch found at (row, col) = (" << row << ", " << col << ")\n";
                }
                nnz_count++;
            }
        }
    }
}

// dump sparse matrix into a file
void dump_csr_matrix_txt(int m, int nnz, int* d_csrRowPtr, int* d_csrColIndices, double* d_csrValues, int kmc_step_count){

    // Copy matrix back to host memory
    double *h_csrValues = (double *)calloc(nnz, sizeof(double));
    int *h_csrRowPtr = (int *)calloc((m + 1), sizeof(int));
    int *h_csrColIndices = (int *)calloc(nnz, sizeof(int));
    gpuErrchk( cudaMemcpy(h_csrValues, d_csrValues, nnz * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_csrRowPtr, d_csrRowPtr, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_csrColIndices, d_csrColIndices, nnz * sizeof(int), cudaMemcpyDeviceToHost) );

    // print to file, tagged with the kmc step number
    std::ofstream fout_val("csrValues_step#" + std::to_string(kmc_step_count) + ".txt");
    for(int i = 0; i < nnz; i++){
        fout_val << h_csrValues[i] << " "; 
    }
    std::ofstream fout_row("csrRowPtr_step#" + std::to_string(kmc_step_count) + ".txt");
    for(int i = 0; i < (m + 1); i++){
        fout_row << h_csrRowPtr[i] << " "; 
    }
    std::ofstream fout_col("csrColIndices_step#" + std::to_string(kmc_step_count) + ".txt");
    for(int i = 0; i < nnz; i++){
        fout_col << h_csrColIndices[i] << " "; 
    }

    free(h_csrValues);
    free(h_csrRowPtr);
    free(h_csrColIndices);
}

// Solution of A*x = y using cusolver in host pointer mode
void sparse_system_solve(cusolverSpHandle_t handle, int* d_csrRowPtr, int* d_csrColInd, double* d_csrVal,
                         int nnz, int m, double *d_x, double *d_y){

    // Ref: https://stackoverflow.com/questions/31840341/solving-general-sparse-linear-systems-in-cuda

    // cusolverSpDcsrlsvlu only supports the host path
    int *h_A_RowIndices = (int *)malloc((m + 1) * sizeof(int));
    int *h_A_ColIndices = (int *)malloc(nnz * sizeof(int));
    double *h_A_Val = (double *)malloc(nnz * sizeof(double));
    double *h_x = (double *)malloc(m * sizeof(double));
    double *h_y = (double *)malloc(m * sizeof(double));
    gpuErrchk( cudaMemcpy(h_A_RowIndices, d_csrRowPtr, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_A_ColIndices, d_csrColInd, nnz * sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_A_Val, d_csrVal, nnz * sizeof(double), cudaMemcpyDeviceToHost) );   
    gpuErrchk( cudaMemcpy(h_x, d_x, m * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_y, d_y, m * sizeof(double), cudaMemcpyDeviceToHost) );

    cusparseMatDescr_t matDescrA;
    cusparseCreateMatDescr(&matDescrA);
    cusparseSetMatType(matDescrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(matDescrA, CUSPARSE_INDEX_BASE_ZERO);

    int singularity;
    double tol = 0.00000001;

    // Solve with LU
    // CheckCusolverDnError( cusolverSpDcsrlsvluHost(handle, m, nnz, matDescrA, h_A_Val, h_A_RowIndices, 
    //                       h_A_ColIndices, h_y, tol, 0, h_x, &singularity) );
    
    // Solve with QR
    // CheckCusolverDnError( cusolverSpDcsrlsvqrHost(handle, m, nnz, matDescrA, h_A_Val, h_A_RowIndices, 
    //                       h_A_ColIndices, h_y, tol, 1, h_x, &singularity) );

    // Solve with Cholesky
    CheckCusolverDnError( cusolverSpDcsrlsvcholHost(handle, m, nnz, matDescrA, h_A_Val, h_A_RowIndices,
                          h_A_ColIndices, h_y, tol, 1, h_x, &singularity) );

    gpuErrchk( cudaDeviceSynchronize() );
    if (singularity != -1){
        std::cout << "In sparse_system_solve: Matrix has a singularity at : " << singularity << "\n";
    }

    // copy back the solution vector:
    gpuErrchk( cudaMemcpy(d_x, h_x, m * sizeof(double), cudaMemcpyHostToDevice) );

    cusolverSpDestroy(handle);
    cusparseDestroyMatDescr(matDescrA);
    free(h_A_RowIndices);
    free(h_A_ColIndices);
    free(h_A_Val);
    free(h_x);
    free(h_y);
}

// Iterative sparse linear solver using CG steps
void sparse_system_solve_iterative(cublasHandle_t handle_cublas, cusparseHandle_t handle, 
								   cusparseSpMatDescr_t matA, int m, double *d_x, double *d_y){

    // A is an m x m sparse matrix represented by CSR format
    // - d_x is right hand side vector in gpu memory,
    // - d_y is solution vector in gpu memory.
    // - d_z is intermediate result on gpu memory.

    // Sets the initial guess for the solution vector to zero
    bool zero_guess = 0;

    // Error tolerance for the norm of the residual in the CG steps
    double tol = 1e-12;

    double one = 1.0;
    double n_one = -1.0;
    double zero = 0.0;
    double *one_d, *n_one_d, *zero_d;
    gpuErrchk( cudaMalloc((void**)&one_d, sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&n_one_d, sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&zero_d, sizeof(double)) );
    gpuErrchk( cudaMemcpy(one_d, &one, sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(n_one_d, &n_one, sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(zero_d, &zero, sizeof(double), cudaMemcpyHostToDevice) );
    cusparseStatus_t status;

    // ************************************
    // ** Precondioner and Initial Guess **

    if (zero_guess)
    {
        // Set the initial guess for the solution vector to zero
        gpuErrchk( cudaMemset(d_y, 0, m * sizeof(double)) ); 
        gpuErrchk( cudaDeviceSynchronize() );
    }

    // *******************************
    // ** Iterative refinement loop **

    // initialize variables for the residual calculation
    double h_norm;
    double *d_r, *d_p, *d_temp;
    gpuErrchk( cudaMalloc((void**)&d_r, m * sizeof(double)) ); 
    gpuErrchk( cudaMalloc((void**)&d_p, m * sizeof(double)) ); 
    gpuErrchk( cudaMalloc((void**)&d_temp, m * sizeof(double)) ); 

    // for SpMV:
    // - d_x is right hand side vector
    // - d_y is solution vector
    cusparseDnVecDescr_t vecY, vecR, vecP, vectemp; 
    cusparseCreateDnVec(&vecY, m, d_y, CUDA_R_64F);
    cusparseCreateDnVec(&vecR, m, d_r, CUDA_R_64F);
    cusparseCreateDnVec(&vecP, m, d_p, CUDA_R_64F);
    cusparseCreateDnVec(&vectemp, m, d_temp, CUDA_R_64F);

    size_t MVBufferSize;
    void *MVBuffer = 0;
    status = cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA, 
                          vecY, zero_d, vectemp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &MVBufferSize);
    gpuErrchk( cudaMalloc((void**)&MVBuffer, sizeof(double) * MVBufferSize) );
    
    // Initialize the residual and conjugate vectors
    // r = A*y - x & p = -r
    status = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA, 
                          vecY, zero_d, vecR, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, MVBuffer);         // r = A*y
    //gpuErrchk( cudaDeviceSynchronize() );
    CheckCublasError( cublasDaxpy(handle_cublas, m, &n_one, d_x, 1, d_r, 1) );                          // r = -x + r
    //gpuErrchk( cudaDeviceSynchronize() );
    CheckCublasError(cublasDcopy(handle_cublas, m, d_r, 1, d_p, 1));                                    // p = r
    //gpuErrchk( cudaDeviceSynchronize() );
    CheckCublasError(cublasDscal(handle_cublas, m, &n_one, d_p, 1));                                    // p = -p
    //gpuErrchk( cudaDeviceSynchronize() );

    // calculate the error (norm of the residual)
    CheckCublasError( cublasDnrm2(handle_cublas, m, d_r, 1, &h_norm) );
    gpuErrchk( cudaDeviceSynchronize() );
    
    // Conjugate Gradient steps
    int counter = 0;
    double t, tnew, alpha, beta, alpha_temp;
    while (h_norm > tol){

        // alpha = rT * r / (pT * A * p)
        CheckCublasError( cublasDdot (handle_cublas, m, d_r, 1, d_r, 1, &t) );                         // t = rT * r
        //gpuErrchk( cudaDeviceSynchronize() );
        status = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA, 
                              vecP, zero_d, vectemp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, MVBuffer); // temp = A*p
        //gpuErrchk( cudaDeviceSynchronize() );
        CheckCublasError( cublasDdot (handle_cublas, m, d_p, 1, d_temp, 1, &alpha_temp) );             // alpha = pT*temp = pT*A*p
        //gpuErrchk( cudaDeviceSynchronize() );
        alpha = t / alpha_temp; 

        // y = y + alpha * p
        CheckCublasError(cublasDaxpy(handle_cublas, m, &alpha, d_p, 1, d_y, 1));                       // y = y + alpha * p
        //gpuErrchk( cudaDeviceSynchronize() );

        // r = r + alpha * A * p 
        CheckCublasError(cublasDaxpy(handle_cublas, m, &alpha, d_temp, 1, d_r, 1));                    // r = r + alpha * temp
        //gpuErrchk( cudaDeviceSynchronize() );

        // beta = (rT * r) / t
        CheckCublasError( cublasDdot (handle_cublas, m, d_r, 1, d_r, 1, &tnew) );                       // tnew = rT * r
        //gpuErrchk( cudaDeviceSynchronize() );
        beta = tnew / t;

        // p = -r + beta * p
        CheckCublasError(cublasDscal(handle_cublas, m, &beta, d_p, 1));                                  // p = p * beta
        //gpuErrchk( cudaDeviceSynchronize() );
        CheckCublasError(cublasDaxpy(handle_cublas, m, &n_one, d_r, 1, d_p, 1));                         // p = p - r
        //gpuErrchk( cudaDeviceSynchronize() );

        // calculate the error (norm of the residual)
        CheckCublasError( cublasDnrm2(handle_cublas, m, d_r, 1, &h_norm) );
        //gpuErrchk( cudaDeviceSynchronize() );
        //std::cout << h_norm << "\n";

        counter++;
        if (counter > 10000){
            std::cout << "WARNING: probably stuck in diverging CG iterations, check the residual!\n";
        }
    }
    std::cout << "# CG steps: " << counter << "\n";

    // // check solution vector
    // double *copy_back = (double *)calloc(m, sizeof(double));
    // gpuErrchk( cudaMemcpy(copy_back, d_y, m * sizeof(double), cudaMemcpyDeviceToHost) );
    // for (int i = 0; i < m; i++){
    //     std::cout << copy_back[i] << " ";
    // }
    
}