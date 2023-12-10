#include "cuda_wrapper.h"
#include <cub/cub.cuh>

void initialize_sparsity(GPUBuffers &gpubuf, int pbc, const double nn_dist, int num_atoms_contact)
{
    int N_left_tot = num_atoms_contact;
    int N_right_tot = num_atoms_contact;
    int N_interface = gpubuf.N_ - (N_left_tot + N_right_tot);

    Assemble_K_sparsity(gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
                        gpubuf.lattice, pbc, nn_dist,
                        N_interface, N_left_tot, N_right_tot,
                        &gpubuf.Device_row_ptr_d, &gpubuf.Device_col_indices_d, &gpubuf.Device_nnz,
                        &gpubuf.contact_left_col_indices, &gpubuf.contact_left_row_ptr, &gpubuf.contact_left_nnz,
                        &gpubuf.contact_right_col_indices, &gpubuf.contact_right_row_ptr, &gpubuf.contact_right_nnz);

}

__device__ double site_dist_gpu_2(double pos1x, double pos1y, double pos1z,
                                double pos2x, double pos2y, double pos2z,
                                double lattx, double latty, double lattz, bool pbc)
{

    double dist = 0;

    if (pbc == 1)
    {
        double dist_x = pos1x - pos2x;
        double distance_frac[3];

        distance_frac[1] = (pos1y - pos2y) / latty;
        distance_frac[1] -= round(distance_frac[1]);
        distance_frac[2] = (pos1z - pos2z) / lattz;
        distance_frac[2] -= round(distance_frac[2]);

        double dist_xyz[3];
        dist_xyz[0] = dist_x;

        dist_xyz[1] = distance_frac[1] * latty;
        dist_xyz[2] = distance_frac[2] * lattz;

        dist = sqrt(dist_xyz[0] * dist_xyz[0] + dist_xyz[1] * dist_xyz[1] + dist_xyz[2] * dist_xyz[2]);
        
    }
    else
    {
        dist = sqrt(pow(pos2x - pos1x, 2) + pow(pos2y - pos1y, 2) + pow(pos2z - pos1z, 2));
    }

    return dist;
}

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

// Extracts the inverse sqrt of the diagonal values into a vector to use for the preconditioning
__global__ void computeDiagonalInvSqrt(const double* A_data, const int* A_row_ptr,
                                       const int* A_col_indices, double* diagonal_values_inv_sqrt_d,
                                       const int matrix_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < matrix_size) {
        // Find the range of non-zero elements for the current row
        int row_start = A_row_ptr[tid];
        int row_end = A_row_ptr[tid + 1];

        // Initialize the sum for the diagonal element
        double diagonal_sum = 0.0;

        // Loop through the non-zero elements in the current row
        for (int i = row_start; i < row_end; ++i) {
            if (A_col_indices[i] == tid) {
                // Found the diagonal element
                diagonal_sum = A_data[i];
                break;
            }
        }

        double diagonal_inv_sqrt = 1.0 / sqrt(diagonal_sum);

        // Store the result in the output array
        diagonal_values_inv_sqrt_d[tid] = diagonal_inv_sqrt;
    }
}

// apply Jacobi preconditioner to an rhs vector
__global__ void jacobi_precondition_array(
    double *array,
    double *diagonal_values_inv_sqrt,
    int matrix_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        array[i] = array[i] * diagonal_values_inv_sqrt[i];
    }

}

// apply Jacobi preconditioner to matrix
__global__ void jacobi_precondition_matrix(
    double *data,
    const int *col_indices,
    const int *row_indptr,
    double *diagonal_values_inv_sqrt,
    int matrix_size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        // Iterate over the row elements
        for(int j = row_indptr[i]; j < row_indptr[i+1]; j++){
            // Use temporary variables to store the original values
            double original_value = data[j];

            // Update data with the preconditioned value
            data[j] = original_value * diagonal_values_inv_sqrt[i] * diagonal_values_inv_sqrt[col_indices[j]];
        }
    }
}

// apply Jacobi preconditioner to starting guess
__global__ void jacobi_unprecondition_array(
    double *array,
    double *diagonal_values_inv_sqrt,
    int matrix_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        array[i] = array[i] * 1/diagonal_values_inv_sqrt[i];
    }

}

// Iterative sparse linear solver using CG steps
void solve_sparse_CG_Jacobi(cublasHandle_t handle_cublas, cusparseHandle_t handle, 
							double* A_data, int* A_row_ptr,
                            int* A_col_indices, const int A_nnz, int m, double *d_x, double *d_y){

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
    // ** Initial Guess **

    if (zero_guess)
    {
        // Set the initial guess for the solution vector to zero
        gpuErrchk( cudaMemset(d_y, 0, m * sizeof(double)) ); 
        gpuErrchk( cudaDeviceSynchronize() );
    }

    // *******************************
    // ** Preconditioner **

    double* diagonal_values_inv_sqrt_d;
    cudaMalloc((void**)&diagonal_values_inv_sqrt_d, sizeof(double) * m);

    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;

    computeDiagonalInvSqrt<<<grid_size, block_size>>>(A_data, A_row_ptr, A_col_indices,
                                                      diagonal_values_inv_sqrt_d, m);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // scale rhs
    jacobi_precondition_array<<<grid_size, block_size>>>(d_x, diagonal_values_inv_sqrt_d, m);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    // scale matrix
    jacobi_precondition_matrix<<<grid_size, block_size>>>(A_data, A_col_indices, A_row_ptr, 
                                                          diagonal_values_inv_sqrt_d, m);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // scale starting guess
    jacobi_unprecondition_array<<<grid_size, block_size>>>(d_y, diagonal_values_inv_sqrt_d, m);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cusparseSpMatDescr_t matA;
    status = cusparseCreateCsr(&matA, m, m, A_nnz, A_row_ptr, A_col_indices, A_data, 
                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "ERROR: creation of sparse matrix descriptor in solve_sparse_CG_Jacobi() failed!\n";
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
                          vecY, zero_d, vecR, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, MVBuffer);           // r = A*y
    CheckCublasError( cublasDaxpy(handle_cublas, m, &n_one, d_x, 1, d_r, 1) );                            // r = -x + r
    CheckCublasError( cublasDcopy(handle_cublas, m, d_r, 1, d_p, 1) );                                    // p = r
    CheckCublasError( cublasDscal(handle_cublas, m, &n_one, d_p, 1) );                                    // p = -p

    // calculate the error (norm of the residual)
    CheckCublasError( cublasDnrm2(handle_cublas, m, d_r, 1, &h_norm) );
    gpuErrchk( cudaDeviceSynchronize() );
    
    // Conjugate Gradient steps
    int counter = 0;
    double t, tnew, alpha, beta, alpha_temp;
    while (h_norm > tol*tol){

        // alpha = rT * r / (pT * A * p)
        CheckCublasError( cublasDdot (handle_cublas, m, d_r, 1, d_r, 1, &t) );                           // t = rT * r
        status = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA, 
                              vecP, zero_d, vectemp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, MVBuffer);   // temp = A*p
        CheckCublasError( cublasDdot (handle_cublas, m, d_p, 1, d_temp, 1, &alpha_temp) );               // alpha = pT*temp = pT*A*p
        alpha = t / alpha_temp; 

        // y = y + alpha * p
        CheckCublasError( cublasDaxpy(handle_cublas, m, &alpha, d_p, 1, d_y, 1) );                       // y = y + alpha * p

        // r = r + alpha * A * p 
        CheckCublasError( cublasDaxpy(handle_cublas, m, &alpha, d_temp, 1, d_r, 1) );                    // r = r + alpha * temp

        // beta = (rT * r) / t
        CheckCublasError( cublasDdot (handle_cublas, m, d_r, 1, d_r, 1, &tnew) );                        // tnew = rT * r
        beta = tnew / t;

        // p = -r + beta * p
        CheckCublasError( cublasDscal(handle_cublas, m, &beta, d_p, 1) );                                 // p = p * beta
        CheckCublasError( cublasDaxpy(handle_cublas, m, &n_one, d_r, 1, d_p, 1) );                        // p = p - r

        // calculate the error (norm of the residual)
        CheckCublasError( cublasDdot(handle_cublas, m, d_r, 1, d_r, 1, &h_norm) );
        // std::cout << h_norm << "\n";

        counter++;
        if (counter > 50000){
            std::cout << "WARNING: might be stuck in diverging CG iterations, check the residual!\n";
        }
    }
    std::cout << "# CG steps: " << counter << "\n";

    // unprecondition the solution vector
    jacobi_precondition_array<<<grid_size, block_size>>>(d_y, diagonal_values_inv_sqrt_d, m);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // // check solution vector
    // double *copy_back = (double *)calloc(m, sizeof(double));
    // gpuErrchk( cudaMemcpy(copy_back, d_y, m * sizeof(double), cudaMemcpyDeviceToHost) );
    // for (int i = 0; i < m; i++){
    //     std::cout << copy_back[i] << " ";
    // }
    // std::cout << "\nPrinted solution vector, now exiting\n";
    // exit(1);

    cudaFree(diagonal_values_inv_sqrt_d);
    cudaFree(MVBuffer); 
    cudaFree(one_d);
    cudaFree(n_one_d);
    cudaFree(zero_d);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_temp);
}

// Iterative sparse linear solver using CG steps
void solve_sparse_CG(cublasHandle_t handle_cublas, cusparseHandle_t handle, 
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
    // ** Set Initial Guess **

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
        // std::cout << h_norm << "\n";

        counter++;
        if (counter > 50000){
            std::cout << "WARNING: might be stuck in diverging CG iterations, check the residual!\n";
        }
    }
    std::cout << "# CG steps: " << counter << "\n";

    cudaFree(MVBuffer); 
    cudaFree(one_d);
    cudaFree(n_one_d);
    cudaFree(zero_d);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_temp);

    // // check solution vector
    // double *copy_back = (double *)calloc(m, sizeof(double));
    // gpuErrchk( cudaMemcpy(copy_back, d_y, m * sizeof(double), cudaMemcpyDeviceToHost) );
    // for (int i = 0; i < m; i++){
    //     std::cout << copy_back[i] << " ";
    // }
    // exit(1);
    
}

__global__ void assemble_K_indices_gpu(
    const double *posx_d, const double *posy_d, const double *posz_d,
    const double *lattice_d, const bool pbc,
    const double cutoff_radius,
    int matrix_size,
    int *nnz_per_row_d,
    int *row_ptr_d,
    int *col_indices_d)
{
    // row ptr is already calculated
    // exclusive scam of nnz_per_row

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    //TODO can be optimized with a 2D grid instead of 1D
    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        int nnz_row = 0;
        for(int j = 0; j < matrix_size; j++){
        
            double dist = site_dist_gpu_2(posx_d[i], posy_d[i], posz_d[i],
                                        posx_d[j], posy_d[j], posz_d[j],
                                        lattice_d[0], lattice_d[1], lattice_d[2], pbc);
            if(dist < cutoff_radius){
                col_indices_d[row_ptr_d[i] + nnz_row] = j;
                nnz_row++;
            }
        }
    }
}

__global__ void calc_nnz_per_row_gpu(
    const double *posx_d, const double *posy_d, const double *posz_d,
    const double *lattice_d, const bool pbc,
    const double cutoff_radius,
    int matrix_size,
    int *nnz_per_row_d
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO optimize this with a 2D grid instead of 1D
    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        int nnz_row = 0;
        for(int j = 0; j < matrix_size; j++){
            double dist = site_dist_gpu_2(posx_d[i], posy_d[i], posz_d[i],
                                        posx_d[j], posy_d[j], posz_d[j],
                                        lattice_d[0], lattice_d[1], lattice_d[2], pbc);
            if(dist < cutoff_radius){
                nnz_row++;
            }
        }
        nnz_per_row_d[i] = nnz_row;
    }

}

void indices_creation_gpu(
    const double *posx_d, const double *posy_d, const double *posz_d,
    const double *lattice_d, const bool pbc,
    const double cutoff_radius,
    const int matrix_size,
    int **col_indices_d,
    int **row_ptr_d,
    int *nnz
)
{
    // parallelize over rows
    int threads = 512;
    int blocks = (matrix_size + threads - 1) / threads;

    int *nnz_per_row_d;
    gpuErrchk( cudaMalloc((void **)row_ptr_d, (matrix_size + 1) * sizeof(int)) );
    gpuErrchk( cudaMalloc((void **)&nnz_per_row_d, matrix_size * sizeof(int)) );
    gpuErrchk(cudaMemset((*row_ptr_d), 0, (matrix_size + 1) * sizeof(int)) );

    // calculate the nnz per row
    calc_nnz_per_row_gpu<<<blocks, threads>>>(posx_d, posy_d, posz_d, lattice_d, pbc, cutoff_radius, matrix_size, nnz_per_row_d);

    void     *temp_storage_d = NULL;
    size_t   temp_storage_bytes = 0;
    // determines temporary device storage requirements for inclusive prefix sum
    cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*row_ptr_d)+1, matrix_size);

    // Allocate temporary storage for inclusive prefix sum
    gpuErrchk(cudaMalloc(&temp_storage_d, temp_storage_bytes));
    // Run inclusive prefix sum
    // inclusive sum starting at second value to get the row ptr
    // which is the same as inclusive sum starting at first value and last value filled with nnz
    cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*row_ptr_d)+1, matrix_size);
    
    // nnz is the same as (*row_ptr_d)[matrix_size]
    gpuErrchk( cudaMemcpy(nnz, (*row_ptr_d) + matrix_size, sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMalloc((void **)col_indices_d, nnz[0] * sizeof(int)) );

    // assemble the indices of K
    assemble_K_indices_gpu<<<blocks, threads>>>(
        posx_d, posy_d, posz_d,
        lattice_d, pbc,
        cutoff_radius,
        matrix_size,
        nnz_per_row_d,
        (*row_ptr_d),
        (*col_indices_d)
    );

    cudaFree(temp_storage_d);
    cudaFree(nnz_per_row_d);
}

__global__ void calc_nnz_per_row_gpu_off_diagonal_block(
    const double *posx_d, const double *posy_d, const double *posz_d,
    const double *lattice_d, const bool pbc,
    const double cutoff_radius,
    int block_size_i,
    int block_size_j,
    int block_start_i,
    int block_start_j,
    int *nnz_per_row_d
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO optimize this with a 2D grid instead of 1D
    for(int row = idx; row < block_size_i; row += blockDim.x * gridDim.x){
        int nnz_row = 0;
        for(int col = 0; col < block_size_j; col++){
            int i = block_start_i + row;
            int j = block_start_j + col;
            double dist = site_dist_gpu_2(posx_d[i], posy_d[i], posz_d[i],
                                        posx_d[j], posy_d[j], posz_d[j],
                                        lattice_d[0], lattice_d[1], lattice_d[2], pbc);
            if(dist < cutoff_radius){
                nnz_row++;
            }
        }
        nnz_per_row_d[row] = nnz_row;
    }

}

__global__ void assemble_K_indices_gpu_off_diagonal_block(
    const double *posx_d, const double *posy_d, const double *posz_d,
    const double *lattice_d, const bool pbc,
    const double cutoff_radius,
    int block_size_i,
    int block_size_j,
    int block_start_i,
    int block_start_j,
    int *nnz_per_row_d,
    int *row_ptr_d,
    int *col_indices_d)
{
    // row ptr is already calculated
    // exclusive scam of nnz_per_row

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    //TODO can be optimized with a 2D grid instead of 1D
    for(int row = idx; row < block_size_i; row += blockDim.x * gridDim.x){
        int nnz_row = 0;
        for(int col = 0; col < block_size_j; col++){
            int i = block_start_i + row;
            int j = block_start_j + col;
            double dist = site_dist_gpu_2(posx_d[i], posy_d[i], posz_d[i],
                                        posx_d[j], posy_d[j], posz_d[j],
                                        lattice_d[0], lattice_d[1], lattice_d[2], pbc);
            if(dist < cutoff_radius){
                col_indices_d[row_ptr_d[row] + nnz_row] = col;
                nnz_row++;
            }
        }
    }
}

void indices_creation_gpu_off_diagonal_block(
    const double *posx_d, const double *posy_d, const double *posz_d,
    const double *lattice_d, const bool pbc,
    const double cutoff_radius,
    int block_size_i,
    int block_size_j,
    int block_start_i,
    int block_start_j,
    int **col_indices_d,
    int **row_ptr_d,
    int *nnz
)
{
    // parallelize over rows
    int threads = 512;
    int blocks = (block_size_i + threads - 1) / threads;

    int *nnz_per_row_d;
    gpuErrchk( cudaMalloc((void **)row_ptr_d, (block_size_i + 1) * sizeof(int)) );
    gpuErrchk( cudaMalloc((void **)&nnz_per_row_d, block_size_i * sizeof(int)) );
    gpuErrchk(cudaMemset((*row_ptr_d), 0, (block_size_i + 1) * sizeof(int)) );

    // calculate the nnz per row
    calc_nnz_per_row_gpu_off_diagonal_block<<<blocks, threads>>>(posx_d, posy_d, posz_d, lattice_d, pbc, cutoff_radius,
        block_size_i, block_size_j, block_start_i, block_start_j, nnz_per_row_d);

    void     *temp_storage_d = NULL;
    size_t   temp_storage_bytes = 0;

    // determines temporary device storage requirements for inclusive prefix sum
    cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*row_ptr_d)+1, block_size_i);

    // Allocate temporary storage for inclusive prefix sum
    gpuErrchk(cudaMalloc(&temp_storage_d, temp_storage_bytes));

    // Run inclusive prefix sum
    // inclusive sum starting at second value to get the row ptr
    // which is the same as inclusive sum starting at first value and last value filled with nnz
    cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*row_ptr_d)+1, block_size_i);
    
    // nnz is the same as (*row_ptr_d)[block_size_i]
    gpuErrchk( cudaMemcpy(nnz, (*row_ptr_d) + block_size_i, sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMalloc((void **)col_indices_d, nnz[0] * sizeof(int)) );

    // assemble the indices of K
    assemble_K_indices_gpu_off_diagonal_block<<<blocks, threads>>>(
        posx_d, posy_d, posz_d,
        lattice_d, pbc,
        cutoff_radius,
        block_size_i,
        block_size_j,
        block_start_i,
        block_start_j,
        nnz_per_row_d,
        (*row_ptr_d),
        (*col_indices_d)
    );

    cudaFree(temp_storage_d);
    cudaFree(nnz_per_row_d);
}

void Assemble_K_sparsity(const double *posx, const double *posy, const double *posz,
                         const double *lattice, const bool pbc,
                         const double cutoff_radius,
                         int system_size, int contact_left_size, int contact_right_size,
                         int **A_row_ptr, int **A_col_indices, int *A_nnz, 
                         int **contact_left_col_indices, int **contact_left_row_ptr, int *contact_left_nnz, 
                         int **contact_right_col_indices, int **contact_right_row_ptr, int *contact_right_nnz){

    // indices of A (the device submatrix)
    indices_creation_gpu(
        posx + contact_left_size,
        posy + contact_left_size,
        posz + contact_left_size,
        lattice, pbc,
        cutoff_radius,
        system_size,
        A_col_indices,
        A_row_ptr,
        A_nnz
    );

    // indices of the off-diagonal leftcontact-A matrix
    indices_creation_gpu_off_diagonal_block(
        posx, posy, posz,
        lattice, pbc,
        cutoff_radius,
        system_size,
        contact_left_size,
        contact_left_size,
        0,
        contact_left_col_indices,
        contact_left_row_ptr,
        contact_left_nnz
    );
    // std::cout << "contact_left_nnz " << *contact_left_nnz << std::endl;

    // indices of the off-diagonal A-rightcontact matrix
    indices_creation_gpu_off_diagonal_block(
        posx, posy, posz,
        lattice, pbc,
        cutoff_radius,
        system_size,
        contact_right_size,
        contact_left_size,
        contact_left_size + system_size,
        contact_right_col_indices,
        contact_right_row_ptr,
        contact_right_nnz
    );
    // std::cout << "contact_right_nnz " << *contact_right_nnz << std::endl;
}