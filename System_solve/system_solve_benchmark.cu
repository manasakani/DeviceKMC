#include <string> 
#include <omp.h>

#include "utils.h"

#include "mkl.h"
#include "cusolverDn.h"
#include "cusolverSp.h"
#include <cusparse.h>



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

#define cusolverErrchk(ans) { cusolverAssert((ans), __FILE__, __LINE__); }
inline void cusolverAssert(cusolverStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUSOLVER_STATUS_SUCCESS) 
   {
        //Did not find a counter part to cudaGetErrorString in cusolver
        fprintf(stderr,"CUSOLVERassert: %s %d\n", file, line);
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

cusolverDnHandle_t CreateCusolverDnHandle(int device) {
    if (cudaSetDevice(device) != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device.");
    }
    cusolverDnHandle_t handle;
    cusolverErrchk(cusolverDnCreate(&handle));
    return handle;
}


double solve_mkl_dgesv(
    double *matrix_dense,
    double *rhs,
    double *reference_solution,
    int matrix_size,
    double abstol,
    double reltol,
    bool flag_verbose)
{

    double time = -1.0;


    int ipiv[matrix_size];
    int nrhs = 1;
    int info;
    time = -omp_get_wtime();
    info = LAPACKE_dgesv(LAPACK_COL_MAJOR, matrix_size, nrhs,
                        matrix_dense, matrix_size, ipiv, rhs, matrix_size);
    time += omp_get_wtime();

    if(info != 0){
        std::printf("Error in MKL dgesv\n");
        std::printf("info: %d\n", info);
        if(info > 0){
            std::printf("Singular");
        }
    }

    if(flag_verbose){
        std::printf("MKL dgesv done\n");
    }
    double relative_error[1];
    if(!assert_array_magnitude<double>(rhs,
            reference_solution, 
            abstol,
            reltol,
            matrix_size,
            relative_error)){
        
        
        std::printf("Error: MKL dgesv solution is not the same as the reference solution\n");
    }
    else{
        std::printf("MKL dgesv solution is the same as the reference solution\n");
    }
    return time;
}

double solve_mkl_dposv(
    double *matrix_dense,
    double *rhs,
    double *reference_solution,
    int matrix_size,
    double abstol,
    double reltol,
    bool flag_verbose)
{

    double time = -1.0;


    int nrhs = 1;
    int info;
    time = -omp_get_wtime();
    char uplo = 'U';
    info = LAPACKE_dposv(LAPACK_COL_MAJOR,
                        uplo,
                        matrix_size,
                        nrhs,
                        matrix_dense,
                        matrix_size,
                        rhs,
                        matrix_size);
    time += omp_get_wtime();

    if(info != 0){
        std::printf("Error in MKL dposv\n");
        std::printf("info: %d\n", info);
        if(info > 0){
            std::printf("Singular");
        }
    }

    if(flag_verbose){
        std::printf("MKL dposv done\n");
    }
    double relative_error[1];
    if(!assert_array_magnitude<double>(rhs,
            reference_solution, 
            abstol,
            reltol,
            matrix_size,
            relative_error)){
        std::printf("Error: MKL dposv solution is not the same as the reference solution\n");
    }
    else{
        std::printf("MKL dposv solution is the same as the reference solution\n");
    }
    return time;
}

double solve_mkl_dgbsv(
    double *matrix_band,
    double *rhs,
    double *reference_solution,
    int matrix_size,
    int kl,
    int ku,
    double abstol,
    double reltol,
    bool flag_verbose)
{

    double time = -1.0;


    int ipiv[matrix_size];
    int nrhs = 1;
    int info;
    int ldab = 2*kl + ku + 1;
    time = -omp_get_wtime();
    info = LAPACKE_dgbsv(LAPACK_COL_MAJOR, matrix_size, kl, ku, nrhs,
            matrix_band, ldab,
            ipiv, rhs, matrix_size);
    time += omp_get_wtime();

    if(info != 0){
        std::printf("Error in MKL dgbsv\n");
        std::printf("info: %d\n", info);
        if(info > 0){
            std::printf("Singular");
        }
    }

    if(flag_verbose){
        std::printf("MKL dgesv done\n");
    }
    double relative_error[1];
    if(!assert_array_magnitude<double>(rhs,
            reference_solution,
            abstol,
            reltol,
            matrix_size,
            relative_error)){
        std::printf("Error: MKL dgbsv solution is not the same as the reference solution\n");
    }
    else{
        std::printf("MKL dgbsv solution is the same as the reference solution\n");
    }
    return time;
}


double solve_mkl_dpbsv(
    double *matrix_band,
    double *rhs,
    double *reference_solution,
    int matrix_size,
    int kd,
    double abstol,
    double reltol,
    bool flag_verbose)
{

    double time = -1.0;


    int nrhs = 1;
    int info;
    int ldab = kd + 1;
    char order = 'U';
    time = -omp_get_wtime();
    info = LAPACKE_dpbsv(LAPACK_COL_MAJOR,
                order,
                matrix_size,
                kd,
                nrhs,
                matrix_band,
                ldab,
                rhs,
                matrix_size);

    time += omp_get_wtime();

    if(info != 0){
        std::printf("Error in MKL LAPACKE_dpbsv\n");
        std::printf("info: %d\n", info);
        if(info > 0){
            std::printf("Singular");
        }
    }

    if(flag_verbose){
        std::printf("MKL pbsv done\n");
    }
    double relative_error[1];
    if(!assert_array_magnitude<double>(rhs,
            reference_solution,
            abstol,
            reltol,
            matrix_size,
            relative_error)){
        std::printf("Error: MKL dpbsv solution is not the same as the reference solution\n");
    }
    else{
        std::printf("MKL dpbsv solution is the same as the reference solution\n");
    }
    return time;
}


double solve_cusolver_dense_LU(
    double *matrix_dense_h,
    double *rhs_h,
    double *reference_solution_h,
    int matrix_size,
    double abstol,
    double reltol,
    bool flag_verbose)
{

    double time = -1.0;
    cudaStream_t stream = NULL;
    cusolverDnHandle_t handle = CreateCusolverDnHandle(0);
    cudaErrchk(cudaStreamCreate(&stream));
    cusolverErrchk(cusolverDnSetStream(handle, stream));



    int info_h = 0;
    int bufferSize = 0;

    double *matrix_dense_d = NULL;
    double *rhs_d = NULL;
    int *ipiv_d = NULL;
    int *info_d = NULL;
    double *buffer = NULL;

    //allocate memory on device
    cudaErrchk(cudaMalloc((void**)&info_d, sizeof(int)))
    cudaErrchk(cudaMalloc((void**)&matrix_dense_d, matrix_size*matrix_size*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&rhs_d, matrix_size*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&ipiv_d, matrix_size*sizeof(int)));


    //copy data to device
    if(flag_verbose){
        std::printf("Copy data to device\n");
    }
    cudaErrchk(cudaMemcpy(matrix_dense_d, matrix_dense_h, matrix_size*matrix_size*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemset(info_d, 0, sizeof(int)));
    cudaErrchk(cudaMemcpy(rhs_d, rhs_h, matrix_size*sizeof(double), cudaMemcpyHostToDevice));


    //figure out extra amount of memory needed
    cusolverErrchk(cusolverDnDgetrf_bufferSize(handle, matrix_size, matrix_size,
                                            (double *)matrix_dense_d,
                                              matrix_size, &bufferSize));
    cudaErrchk(cudaMalloc(&buffer, sizeof(double) * bufferSize));

    //LU factorization
    if(flag_verbose){
        std::printf("LU factorization\n");
    }
    time = -omp_get_wtime();
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaStreamSynchronize(stream));
    cusolverErrchk(cusolverDnDgetrf(handle, matrix_size, matrix_size,
                                matrix_dense_d, matrix_size, buffer, ipiv_d, info_d));
    
    //copy info to host
    cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_h != 0) {
        fprintf(stderr, "Error: LU factorization failed\n");
    }
    else{
        std::printf("LU factorization done\n");
    }

    if(flag_verbose){
        std::printf("Back substitution\n");
    }
    //back substitution
    cusolverErrchk(cusolverDnDgetrs(handle, CUBLAS_OP_N, matrix_size,
                                    1, matrix_dense_d, matrix_size, ipiv_d,
                                    rhs_d, matrix_size, info_d));
    cudaErrchk(cudaStreamSynchronize(stream));
    cudaErrchk(cudaDeviceSynchronize());
    time += omp_get_wtime();


    cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_h != 0) {
        fprintf(stderr, "Error: Back substitution failed\n");
    }
    else{
        std::printf("Back substitution done\n");
    }

    //copy solution to host
    if(flag_verbose){
        std::printf("Copy solution to host\n");
    }
    cudaErrchk(cudaMemcpy(rhs_h, rhs_d, matrix_size*sizeof(double), cudaMemcpyDeviceToHost));
    double relative_error[1];
    if(!assert_array_magnitude<double>(rhs_h,
            reference_solution_h,
            abstol,
            reltol,
            matrix_size,
            relative_error)){
        std::printf("Error: CuSolver LU solution is not the same as the reference solution\n");
    }
    else{
        std::printf("CuSolver LU solution is the same as the reference solution\n");
    }


    if (info_d) {
        cudaErrchk(cudaFree(info_d));
    }
    if (buffer) {
        cudaErrchk(cudaFree(buffer));
    }
    if (matrix_dense_d) {
        cudaErrchk(cudaFree(matrix_dense_d));
    }
    if(rhs_d) {
        cudaErrchk(cudaFree(rhs_d));
    }
    if (ipiv_d) {
        cudaErrchk(cudaFree(ipiv_d));
    }


    if (handle) {
        cusolverErrchk(cusolverDnDestroy(handle));
    }
    if (stream) {
        cudaErrchk(cudaStreamDestroy(stream));
    }

    return time;
}


double solve_cusolver_dense_CHOL(
    double *matrix_dense_h,
    double *rhs_h,
    double *reference_solution_h,
    int matrix_size,
    double abstol,
    double reltol,
    bool flag_verbose)
{

    double time = -1.0;
    cudaStream_t stream = NULL;
    cusolverDnHandle_t handle = CreateCusolverDnHandle(0);
    cudaErrchk(cudaStreamCreate(&stream));
    cusolverErrchk(cusolverDnSetStream(handle, stream));



    int info_h = 0;
    int bufferSize = 0;

    double *matrix_dense_d = NULL;
    double *rhs_d = NULL;
    int *info_d = NULL;
    double *buffer = NULL;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    //allocate memory on device
    cudaErrchk(cudaMalloc((void**)&info_d, sizeof(int)))
    cudaErrchk(cudaMalloc((void**)&matrix_dense_d, matrix_size*matrix_size*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&rhs_d, matrix_size*sizeof(double)));


    //copy data to device
    if(flag_verbose){
        std::printf("Copy data to device\n");
    }
    cudaErrchk(cudaMemcpy(matrix_dense_d, matrix_dense_h, matrix_size*matrix_size*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemset(info_d, 0, sizeof(int)));
    cudaErrchk(cudaMemcpy(rhs_d, rhs_h, matrix_size*sizeof(double), cudaMemcpyHostToDevice));


    //figure out extra amount of memory needed
    cusolverErrchk(cusolverDnDpotrf_bufferSize(handle, uplo, matrix_size,
                                            (double *)matrix_dense_d,
                                              matrix_size, &bufferSize));
    cudaErrchk(cudaMalloc(&buffer, sizeof(double) * bufferSize));

    //LU factorization
    if(flag_verbose){
        std::printf("CHOL factorization\n");
    }
    time = -omp_get_wtime();
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaStreamSynchronize(stream));
    cusolverErrchk(cusolverDnDpotrf(handle, uplo, matrix_size,
                                matrix_dense_d, matrix_size, buffer, bufferSize, info_d));
    
    //copy info to host
    cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));

    if (info_h != 0) {
        fprintf(stderr, "Error: CHOL factorization failed\n");
    }
    else{
        std::printf("CHOL factorization done\n");
    }

    if(flag_verbose){
        std::printf("Back substitution\n");
    }
    //back substitution
    cusolverErrchk(cusolverDnDpotrs(handle, uplo, matrix_size,
                                    1, matrix_dense_d, matrix_size,
                                    rhs_d, matrix_size, info_d));
    cudaErrchk(cudaStreamSynchronize(stream));
    cudaErrchk(cudaDeviceSynchronize());
    time += omp_get_wtime();


    cudaErrchk(cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_h != 0) {
        fprintf(stderr, "Error: Back substitution failed\n");
    }
    else{
        std::printf("Back substitution done\n");
    }

    //copy solution to host
    if(flag_verbose){
        std::printf("Copy solution to host\n");
    }
    cudaErrchk(cudaMemcpy(rhs_h, rhs_d, matrix_size*sizeof(double), cudaMemcpyDeviceToHost));
    double relative_error[1];
    if(!assert_array_magnitude<double>(rhs_h,
            reference_solution_h,
            abstol,
            reltol,
            matrix_size,
            relative_error)){
        std::printf("Error: CuSolver CHOL solution is not the same as the reference solution\n");
    }
    else{
        std::printf("CuSolver CHOL solution is the same as the reference solution\n");
    }


    if (info_d) {
        cudaErrchk(cudaFree(info_d));
    }
    if (buffer) {
        cudaErrchk(cudaFree(buffer));
    }
    if (matrix_dense_d) {
        cudaErrchk(cudaFree(matrix_dense_d));
    }
    if(rhs_d) {
        cudaErrchk(cudaFree(rhs_d));
    }

    if (handle) {
        cusolverErrchk(cusolverDnDestroy(handle));
    }
    if (stream) {
        cudaErrchk(cudaStreamDestroy(stream));
    }

    return time;
}


double solve_cusparse_CG(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution_h,
    double *starting_guess_h,
    int nnz,
    int matrix_size,
    double abstol,
    double reltol,
    double restol,
    bool flag_verbose,
    int *steps_taken,
    double *relative_error)
{

    double time = -1.0;
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
    double dot;

    cusparseSpMatDescr_t matA = NULL;

    const int max_iter = 100000;
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
    if(flag_verbose){
        std::printf("Copy data to device\n");
    }
    cudaErrchk(cudaMemcpy(rhs_d, rhs_h, matrix_size*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(col_indices_d, col_indices_h, nnz * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(row_indptr_d, row_indptr_h, (matrix_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(data_d, data_h, nnz * sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(x_d, starting_guess_h, matrix_size * sizeof(double), cudaMemcpyHostToDevice));    

    //figure out extra amount of memory needed
    if(flag_verbose){
        std::printf("Figure out extra amount of memory needed\n");
    }
    cusparseErrchk(cusparseSpMV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
        &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    cudaErrchk(cudaMalloc(&buffer, bufferSize));


    //begin CG
    time = -omp_get_wtime();
    cudaErrchk(cudaStreamSynchronize(stream));
    cudaErrchk(cudaDeviceSynchronize());
    if(flag_verbose){
        std::printf("CG starts\n");
    }

    // calc A*x
    cusparseErrchk(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecx, &beta, vecAx, CUDA_R_64F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    // r = b - A*x
    cublasErrchk(cublasDaxpy(cublasHandle, matrix_size, &alpham1, Ax_d, 1, rhs_d, 1));
    cublasErrchk(cublasDdot(cublasHandle, matrix_size, rhs_d, 1, rhs_d, 1, &r1));


    int k = 1;
    while (r1 > restol * restol && k <= max_iter) {
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
    
        cublasErrchk(cublasDdot(cublasHandle, matrix_size, p_d, 1, Ax_d, 1, &dot));
        a = r1 / dot;

        cublasErrchk(cublasDaxpy(cublasHandle, matrix_size, &a, p_d, 1, x_d, 1));
        na = -a;
        cublasErrchk(cublasDaxpy(cublasHandle, matrix_size, &na, Ax_d, 1, rhs_d, 1));

        r0 = r1;
        cublasErrchk(cublasDdot(cublasHandle, matrix_size, rhs_d, 1, rhs_d, 1, &r1));
        cudaErrchk(cudaStreamSynchronize(stream));

        k++;
    }

    steps_taken[0] = k;
    std::printf("iteration = %3d, residual = %e\n", k, sqrt(r1));


    //end CG
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaStreamSynchronize(stream));
    time += omp_get_wtime();

    //copy solution to host
    if(flag_verbose){
        std::printf("Copy solution to host\n");
    }
    cudaErrchk(cudaMemcpy(rhs_h, x_d, matrix_size * sizeof(double), cudaMemcpyDeviceToHost));

    if(!assert_array_magnitude<double>(
            rhs_h,
            reference_solution_h,
            abstol,
            reltol,
            matrix_size,
            relative_error)){
        std::printf("Error: CG solution is not the same as the reference solution\n");
    }
    else{
        std::printf("CG solution is the same as the reference solution\n");
    }



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

    return time;
}

double solve_cusparse_ILU_CG(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution_h,
    int nnz,
    int matrix_size,
    double abstol,
    double reltol,
    double restol,
    bool flag_verbose,
    int *steps_taken,
    double *relative_error)
{

    double time = -1.0;

    
    
    cusparseHandle_t cusparseHandle = 0;
    cusparseErrchk(cusparseCreate(&cusparseHandle));    

    cublasHandle_t cublasHandle = 0;
    cublasErrchk(cublasCreate(&cublasHandle));

    cudaStream_t stream = NULL;
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
    double *valsILU0_d = NULL;
    double *zm1_d = NULL;
    double *zm2_d = NULL;
    double *rm2_d = NULL;
    double *omega_d = NULL;
    double *y_d = NULL;

    const int max_iter = 100000;
    double alpha, beta, r1;
    double numerator, denominator, nalpha;
    const double doubleone = 1.0;
    const double doublezero = 0.0;

    alpha = 1.0;
    beta = 0.0;


    cusparseSpMatDescr_t matA = NULL;
    cusparseSpMatDescr_t matM_lower = NULL;
    cusparseSpMatDescr_t matM_upper = NULL;
    cusparseFillMode_t   fill_lower    = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t   diag_unit     = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseFillMode_t   fill_upper    = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t   diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;


    int                 bufferSizeLU = 0;
    size_t              bufferSizeMV, bufferSizeL, bufferSizeU;
    void*               bufferLU_d, *bufferMV_d,  *bufferL_d, *bufferU_d;
    cusparseSpSVDescr_t spsvDescrL, spsvDescrU;
    cusparseMatDescr_t   matLU;
    csrilu02Info_t      infoILU = NULL;


    /* Description of the A matrix */
    cusparseMatDescr_t descr = 0;
    cusparseErrchk(cusparseCreateMatDescr(&descr));
    cusparseErrchk(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    cusparseErrchk(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

    //allocate memory on device
    cudaErrchk(cudaMalloc((void**)&data_d, nnz*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&col_indices_d, nnz*sizeof(int)));
    cudaErrchk(cudaMalloc((void**)&row_indptr_d, (matrix_size+1)*sizeof(int)));
    cudaErrchk(cudaMalloc((void**)&rhs_d, matrix_size*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&x_d, matrix_size*sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&y_d, matrix_size * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&p_d, matrix_size * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&Ax_d, matrix_size * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&omega_d, matrix_size * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&valsILU0_d, nnz * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&zm1_d, (matrix_size) * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&zm2_d, (matrix_size) * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&rm2_d, (matrix_size) * sizeof(double)));


    /* Wrap raw data into cuSPARSE generic API objects */
    cusparseDnVecDescr_t vecp = NULL, vecX=NULL, vecY = NULL, vecR = NULL, vecZM1=NULL;
    cusparseDnVecDescr_t vecomega = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecp, matrix_size, p_d, CUDA_R_64F));
    cusparseErrchk(cusparseCreateDnVec(&vecX, matrix_size, x_d, CUDA_R_64F));
    cusparseErrchk(cusparseCreateDnVec(&vecY, matrix_size, y_d, CUDA_R_64F));
    cusparseErrchk(cusparseCreateDnVec(&vecR, matrix_size, rhs_d, CUDA_R_64F));
    cusparseErrchk(cusparseCreateDnVec(&vecZM1, matrix_size, zm1_d, CUDA_R_64F));
    cusparseErrchk(cusparseCreateDnVec(&vecomega, matrix_size, omega_d, CUDA_R_64F));


    //copy data to device
    if(flag_verbose){
        std::printf("Copy data to device\n");
    }
    cudaErrchk(cudaMemcpy(rhs_d, rhs_h, matrix_size*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(col_indices_d, col_indices_h, nnz * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(row_indptr_d, row_indptr_h, (matrix_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(data_d, data_h, nnz * sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(valsILU0_d, data_d, nnz*sizeof(double), cudaMemcpyDeviceToDevice));
    // setting starting guess to zero
    cudaErrchk(cudaMemset(x_d, 0.0, matrix_size*sizeof(double)))


    cusparseErrchk(cusparseCreateCsr(
        &matA, matrix_size, matrix_size, nnz, row_indptr_d, col_indices_d, data_d, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    
    //Lower Part 
     cusparseErrchk(cusparseCreateCsr(&matM_lower, matrix_size, matrix_size, nnz, row_indptr_d, col_indices_d, valsILU0_d,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    cusparseErrchk(cusparseSpMatSetAttribute(matM_lower,
                                              CUSPARSE_SPMAT_FILL_MODE,
                                              &fill_lower, sizeof(fill_lower)));
    cusparseErrchk(cusparseSpMatSetAttribute(matM_lower,
                                              CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diag_unit, sizeof(diag_unit)));
    // M_upper
    cusparseErrchk(cusparseCreateCsr(&matM_upper, matrix_size, matrix_size, nnz, row_indptr_d, col_indices_d, valsILU0_d,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    cusparseErrchk(cusparseSpMatSetAttribute(matM_upper,
                                              CUSPARSE_SPMAT_FILL_MODE,
                                              &fill_upper, sizeof(fill_upper)));
    cusparseErrchk(cusparseSpMatSetAttribute(matM_upper,
                                              CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diag_non_unit,
                                              sizeof(diag_non_unit)));


    /* Create ILU(0) info object */
    cusparseErrchk(cusparseCreateCsrilu02Info(&infoILU));
    cusparseErrchk(cusparseCreateMatDescr(&matLU) );
    cusparseErrchk(cusparseSetMatType(matLU, CUSPARSE_MATRIX_TYPE_GENERAL) );
    cusparseErrchk(cusparseSetMatIndexBase(matLU, CUSPARSE_INDEX_BASE_ZERO) );

    /* Allocate workspace for cuSPARSE */
    if(flag_verbose){
        std::printf("Figure out extra amount of memory needed\n");
    }
    cusparseErrchk(cusparseSpMV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matA,
        vecp, &doublezero, vecomega, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
        &bufferSizeMV));
    cudaErrchk( cudaMalloc(&bufferMV_d, bufferSizeMV) );

    cusparseErrchk(cusparseDcsrilu02_bufferSize(
        cusparseHandle, matrix_size, nnz, matLU, data_d, row_indptr_d, col_indices_d, infoILU, &bufferSizeLU));
    cudaErrchk( cudaMalloc(&bufferLU_d, bufferSizeLU) );

    cusparseErrchk(cusparseSpSV_createDescr(&spsvDescrL) );
    cusparseErrchk(cusparseSpSV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matM_lower, vecR, vecX, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL));
    cudaErrchk(cudaMalloc(&bufferL_d, bufferSizeL) );

    cusparseErrchk(cusparseSpSV_createDescr(&spsvDescrU) );
    cusparseErrchk(cusparseSpSV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matM_upper, vecR, vecX, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, &bufferSizeU));
    cudaErrchk(cudaMalloc(&bufferU_d, bufferSizeU) );



    //begin CG
    time = -omp_get_wtime();
    cudaErrchk(cudaStreamSynchronize(stream));
    cudaErrchk(cudaDeviceSynchronize());
    if(flag_verbose){
        std::printf("CG starts\n");
    }

    /* Preconditioned Conjugate Gradient using ILU.
       --------------------------------------------
       Follows the description by Golub & Van Loan,
       "Matrix Computations 3rd ed.", Algorithm 10.3.1  */

    printf("Convergence of CG using ILU(0) preconditioning: \n");



    /* Perform analysis for ILU(0) */
    cusparseErrchk(cusparseDcsrilu02_analysis(
        cusparseHandle, matrix_size, nnz, descr, valsILU0_d, row_indptr_d, col_indices_d, infoILU,
        CUSPARSE_SOLVE_POLICY_USE_LEVEL, bufferLU_d));

    /* generate the ILU(0) factors */
    cusparseErrchk(cusparseDcsrilu02(
        cusparseHandle, matrix_size, nnz, matLU, valsILU0_d, row_indptr_d, col_indices_d, infoILU,
        CUSPARSE_SOLVE_POLICY_USE_LEVEL, bufferLU_d));

    /* perform triangular solve analysis */
    cusparseErrchk(cusparseSpSV_analysis(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone,
        matM_lower, vecR, vecX, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, bufferL_d));

    cusparseErrchk(cusparseSpSV_analysis(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone,
        matM_upper, vecR, vecX, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, bufferU_d));

    // /* reset the initial guess of the solution to zero */
    // for (int i = 0; i < matrix_size; i++)
    // {
    //     x[i] = 0.0;
    // }
    // cudaErrchk(cudaMemcpy(
    //     rhs_d, rhs, matrix_size * sizeof(double), cudaMemcpyHostToDevice));
    // cudaErrchk(cudaMemcpy(
    //     x_d, x, matrix_size * sizeof(double), cudaMemcpyHostToDevice));

    int k = 0;
    cublasErrchk(cublasDdot(cublasHandle, matrix_size, rhs_d, 1, rhs_d, 1, &r1));

    while (r1 > restol * restol && k <= max_iter)
    {
        // preconditioner application: zm1_d = U^-1 L^-1 rhs_d
        cusparseErrchk(cusparseSpSV_solve(cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone,
            matM_lower, vecR, vecY, CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT,
            spsvDescrL) );
            
        cusparseErrchk(cusparseSpSV_solve(cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matM_upper,
            vecY, vecZM1,
            CUDA_R_64F,
            CUSPARSE_SPSV_ALG_DEFAULT,
            spsvDescrU));
        k++;

        if (k == 1)
        {
            cublasErrchk(cublasDcopy(cublasHandle, matrix_size, zm1_d, 1, p_d, 1));
        }
        else
        {
            cublasErrchk(cublasDdot(
                cublasHandle, matrix_size, rhs_d, 1, zm1_d, 1, &numerator));
            cublasErrchk(cublasDdot(
                cublasHandle, matrix_size, rm2_d, 1, zm2_d, 1, &denominator));
            beta = numerator / denominator;
            cublasErrchk(cublasDscal(cublasHandle, matrix_size, &beta, p_d, 1));
            cublasErrchk(cublasDaxpy(
                cublasHandle, matrix_size, &doubleone, zm1_d, 1, p_d, 1));
        }

        cusparseErrchk(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &doubleone, matA,
            vecp, &doublezero, vecomega, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
            bufferMV_d));
    
        cublasErrchk(cublasDdot(
            cublasHandle, matrix_size, rhs_d, 1, zm1_d, 1, &numerator));
        cublasErrchk(cublasDdot(
            cublasHandle, matrix_size, p_d, 1, omega_d, 1, &denominator));

        alpha = numerator / denominator;
        cublasErrchk(cublasDaxpy(cublasHandle, matrix_size, &alpha, p_d, 1, x_d, 1));
        cublasErrchk(cublasDcopy(cublasHandle, matrix_size, rhs_d, 1, rm2_d, 1));
        cublasErrchk(cublasDcopy(cublasHandle, matrix_size, zm1_d, 1, zm2_d, 1));
        nalpha = -alpha;
        cublasErrchk(cublasDaxpy(
            cublasHandle, matrix_size, &nalpha, omega_d, 1, rhs_d, 1));
        cublasErrchk(cublasDdot(cublasHandle, matrix_size, rhs_d, 1, rhs_d, 1, &r1));
    }

    steps_taken[0] = k;
    std::printf("iteration = %3d, residual = %e \n", k, sqrt(r1));


    //end CG
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaStreamSynchronize(stream));
    time += omp_get_wtime();

    //copy solution to host
    if(flag_verbose){
        std::printf("Copy solution to host\n");
    }
    cudaErrchk(cudaMemcpy(rhs_h, x_d, matrix_size * sizeof(double), cudaMemcpyDeviceToHost));

    if(!assert_array_magnitude<double>(rhs_h,
            reference_solution_h,
            abstol,
            reltol,
            matrix_size,
            relative_error)){
        std::printf("Error: ILU CG solution is not the same as the reference solution\n");
    }
    else{
        std::printf("ILU CG solution is the same as the reference solution\n");
    }


    /* Destroy descriptors */
    if(descr) {
        cusparseErrchk(cusparseDestroyMatDescr(descr));
    }
    if(matA) {
        cusparseErrchk(cusparseDestroySpMat(matA));
    }
    if(vecp) {
        cusparseErrchk(cusparseDestroyDnVec(vecp));
    }
    if(vecX) {
        cusparseErrchk(cusparseDestroyDnVec(vecX));
    }
    if(vecY) {
        cusparseErrchk(cusparseDestroyDnVec(vecY));
    }
    if(vecR) {
        cusparseErrchk(cusparseDestroyDnVec(vecR));
    }
    if(vecZM1) {
        cusparseErrchk(cusparseDestroyDnVec(vecZM1));
    }
    if(vecomega) {
        cusparseErrchk(cusparseDestroyDnVec(vecomega));
    }
    if(matM_lower) {
        cusparseErrchk(cusparseDestroySpMat(matM_lower));
    }
    if(matM_upper) {
        cusparseErrchk(cusparseDestroySpMat(matM_upper));
    }
    if(matLU) {
        cusparseErrchk(cusparseDestroyMatDescr(matLU));
    }
    if(spsvDescrL) {
        cusparseErrchk(cusparseSpSV_destroyDescr(spsvDescrL));
    }
    if(spsvDescrU) {
        cusparseErrchk(cusparseSpSV_destroyDescr(spsvDescrU));
    }
    if(infoILU) {
        cusparseErrchk(cusparseDestroyCsrilu02Info(infoILU));
    }


    //Destroy handles
    if(cusparseHandle) {
        cusparseErrchk(cusparseDestroy(cusparseHandle));
    }
    if(cublasHandle) {
        cublasErrchk(cublasDestroy(cublasHandle));
    }
    if(stream) {
        cudaErrchk(cudaStreamDestroy(stream));
    }


    // Destroy buffer
    //bufferLU_d, *bufferMV_d,  *bufferL_d, *bufferU_d;
    if (bufferLU_d) {
        cudaErrchk(cudaFree(bufferLU_d));
    }
    if (bufferMV_d) {
        cudaErrchk(cudaFree(bufferMV_d));
    }
    if (bufferL_d) {
        cudaErrchk(cudaFree(bufferL_d));
    }
    if (bufferU_d) {
        cudaErrchk(cudaFree(bufferU_d));
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
    if(y_d){
        cudaErrchk(cudaFree(y_d));
    }
    if(p_d){
        cudaErrchk(cudaFree(p_d));
    }
    if(omega_d){
        cudaErrchk(cudaFree(omega_d));
    }
    if(Ax_d){
        cudaErrchk(cudaFree(Ax_d));
    }
    if(valsILU0_d){
        cudaErrchk(cudaFree(valsILU0_d));
    }
    if(zm1_d){
        cudaErrchk(cudaFree(zm1_d));
    }
    if(zm2_d){
        cudaErrchk(cudaFree(zm2_d));
    }
    if(rm2_d){
        cudaErrchk(cudaFree(rm2_d));
    }

    return time;
}

void extract_diagonal_values(
    double *data,
    int *col_indices,
    int *row_indptr,
    double *diagonal_values_inv_sqrt,
    int matrix_size
)
{
    #pragma omp parallel for
    for(int i = 0; i < matrix_size; i++){
        for(int j = row_indptr[i]; j < row_indptr[i+1]; j++){
            if(col_indices[j] == i){
                diagonal_values_inv_sqrt[i] = 1/std::sqrt(data[j]);
                break;
            }
        }
    }

}

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




__global__ void jacobi_precondition_matrix(
    double *data,
    int *col_indices,
    int *row_indptr,
    double *diagonal_values_inv_sqrt,
    int matrix_size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        for(int j = row_indptr[i]; j < row_indptr[i+1]; j++){
            data[j] = data[j] *
            diagonal_values_inv_sqrt[i] * diagonal_values_inv_sqrt[col_indices[j]];
        }
    }

}


double solve_cusparse_CG_jacobi(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution_h,
    double *starting_guess_h,
    int nnz,
    int matrix_size,
    double abstol,
    double reltol,
    double restol,
    bool flag_verbose,
    int *steps_taken,
    double *relative_error)
{

    double time = -1.0;
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
    double dot;

    cusparseSpMatDescr_t matA = NULL;

    const int max_iter = 100000;
    double a, b, na;
    double alpha, beta, alpham1, r0, r1;
    size_t bufferSize = 0;
    void *buffer = NULL;

    alpha = 1.0;
    alpham1 = -1.0;
    beta = 0.0;
    r0 = 0.0;

    double *diagonal_values_inv_sqrt_h = (double *)malloc(matrix_size * sizeof(double));
    extract_diagonal_values(
        data_h,
        col_indices_h,
        row_indptr_h,
        diagonal_values_inv_sqrt_h,
        matrix_size
    );
    double *diagonal_values_inv_sqrt_d = NULL;
    


    //allocate memory on device
    cudaErrchk(cudaMalloc((void**)&data_d, nnz*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&col_indices_d, nnz*sizeof(int)));
    cudaErrchk(cudaMalloc((void**)&row_indptr_d, (matrix_size+1)*sizeof(int)));
    cudaErrchk(cudaMalloc((void**)&rhs_d, matrix_size*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&x_d, matrix_size*sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&p_d, matrix_size * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&Ax_d, matrix_size * sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&diagonal_values_inv_sqrt_d, matrix_size*sizeof(double)));

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
    if(flag_verbose){
        std::printf("Copy data to device\n");
    }
    cudaErrchk(cudaMemcpy(rhs_d, rhs_h, matrix_size*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(col_indices_d, col_indices_h, nnz * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(row_indptr_d, row_indptr_h, (matrix_size + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(data_d, data_h, nnz * sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(x_d, starting_guess_h, matrix_size * sizeof(double), cudaMemcpyHostToDevice));    
    cudaErrchk(cudaMemcpy(diagonal_values_inv_sqrt_d, diagonal_values_inv_sqrt_h, matrix_size * sizeof(double), cudaMemcpyHostToDevice));


    // precondition the matrix and right hand side
    // do it directly and not as solving another system
    int num_threads = 256;
    int num_blocks = (matrix_size + num_threads - 1) / num_threads;


    //figure out extra amount of memory needed
    if(flag_verbose){
        std::printf("Figure out extra amount of memory needed\n");
    }
    cusparseErrchk(cusparseSpMV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx,
        &beta, vecAx, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    cudaErrchk(cudaMalloc(&buffer, bufferSize));


    //begin CG
    time = -omp_get_wtime();

    // scale rhs
    jacobi_precondition_array<<<num_blocks, num_threads>>>(
        rhs_d,
        diagonal_values_inv_sqrt_d,
        matrix_size    
    );
    cudaErrchk( cudaDeviceSynchronize() );
    // scale matrix
    jacobi_precondition_matrix<<<num_blocks, num_threads>>>(
        data_d,
        col_indices_d,
        row_indptr_d,
        diagonal_values_inv_sqrt_d,
        matrix_size
    );
    cudaErrchk( cudaDeviceSynchronize() );
    // scale starting guess
    jacobi_unprecondition_array<<<num_blocks, num_threads>>>(
        x_d,
        diagonal_values_inv_sqrt_d,
        matrix_size    
    );

    cudaErrchk(cudaStreamSynchronize(stream));
    cudaErrchk(cudaDeviceSynchronize());
    if(flag_verbose){
        std::printf("CG starts\n");
    }

    // calc A*x
    cusparseErrchk(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecx, &beta, vecAx, CUDA_R_64F,
                               CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    // r = b - A*x
    cublasErrchk(cublasDaxpy(cublasHandle, matrix_size, &alpham1, Ax_d, 1, rhs_d, 1));
    cublasErrchk(cublasDdot(cublasHandle, matrix_size, rhs_d, 1, rhs_d, 1, &r1));


    int k = 1;
    while (r1 > restol * restol && k <= max_iter) {
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
    
        cublasErrchk(cublasDdot(cublasHandle, matrix_size, p_d, 1, Ax_d, 1, &dot));
        a = r1 / dot;

        cublasErrchk(cublasDaxpy(cublasHandle, matrix_size, &a, p_d, 1, x_d, 1));
        na = -a;
        cublasErrchk(cublasDaxpy(cublasHandle, matrix_size, &na, Ax_d, 1, rhs_d, 1));

        r0 = r1;
        cublasErrchk(cublasDdot(cublasHandle, matrix_size, rhs_d, 1, rhs_d, 1, &r1));
        cudaErrchk(cudaStreamSynchronize(stream));

        k++;
    }

    steps_taken[0] = k;
    std::printf("iteration = %3d, residual = %e\n", k, sqrt(r1));


    //end CG
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaStreamSynchronize(stream));
    

    // unprecondition solution
    jacobi_precondition_array<<<num_blocks, num_threads>>>(
        x_d,
        diagonal_values_inv_sqrt_d,
        matrix_size    
    );
    time += omp_get_wtime();
    cudaErrchk( cudaDeviceSynchronize() );
    //copy solution to host
    if(flag_verbose){
        std::printf("Copy solution to host\n");
    }
    cudaErrchk(cudaMemcpy(rhs_h, x_d, matrix_size * sizeof(double), cudaMemcpyDeviceToHost));


    if(!assert_array_magnitude<double>(
            rhs_h,
            reference_solution_h,
            abstol,
            reltol,
            matrix_size,
            relative_error)){
        std::printf("Error: Jacobi CG solution is not the same as the reference solution\n");
    }
    else{
        std::printf("Jacobi CG solution is the same as the reference solution\n");
    }




    cusparseErrchk(cusparseDestroy(cusparseHandle));

    cublasErrchk(cublasDestroy(cublasHandle));

    cudaErrchk(cudaStreamDestroy(stream));

    cusparseErrchk(cusparseDestroySpMat(matA));

    cusparseErrchk(cusparseDestroyDnVec(vecx));

    cusparseErrchk(cusparseDestroyDnVec(vecAx));

    cusparseErrchk(cusparseDestroyDnVec(vecp));


    cudaErrchk(cudaFree(buffer));


    cudaErrchk(cudaFree(data_d));

    cudaErrchk(cudaFree(col_indices_d));


    cudaErrchk(cudaFree(row_indptr_d));


    cudaErrchk(cudaFree(rhs_d));


    cudaErrchk(cudaFree(x_d));


    cudaErrchk(cudaFree(p_d));


    cudaErrchk(cudaFree(Ax_d));


    cudaErrchk(cudaFree(diagonal_values_inv_sqrt_d));

    free(diagonal_values_inv_sqrt_h);

    return time;
}

double solve_cusolver_sparse_CHOL(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution_h,
    int nnz,
    int matrix_size,
    double abstol,
    double reltol,
    bool flag_verbose)
{


    cusolverSpHandle_t handle = NULL;
    cusparseHandle_t cusparseHandle = NULL; /* used in residual evaluation */
    cudaStream_t stream = NULL;
    cusparseMatDescr_t descrA = NULL;

    cudaErrchk(cudaStreamCreate(&stream));
    cusolverErrchk(cusolverSpCreate(&handle));
    cusparseErrchk(cusparseCreate(&cusparseHandle));

    cusolverErrchk(cusolverSpSetStream(handle, stream));
    cusparseErrchk(cusparseSetStream(cusparseHandle, stream));


    double *data_d = NULL;
    int *col_indices_d = NULL;
    int *row_indptr_d = NULL;
    double *rhs_d = NULL;
    double *x_d = NULL;

    const int reorder = 0;
    int singularity = 0;
    double singular_tol = 1.e-12;
    double time = -1.0;

    cudaErrchk(cudaMalloc((void **)&row_indptr_d, sizeof(int) * (matrix_size + 1)));
    cudaErrchk(cudaMalloc((void **)&col_indices_d, sizeof(int) * nnz));
    cudaErrchk(cudaMalloc((void **)&data_d, sizeof(double) * nnz));
    cudaErrchk(cudaMalloc((void **)&rhs_d, sizeof(double) * matrix_size));
    cudaErrchk(cudaMalloc((void **)&x_d, sizeof(double) * matrix_size));

    // load data to device
    if(flag_verbose){
        std::printf("Copy data to device\n");
    }
    cudaErrchk(cudaMemcpy(row_indptr_d, row_indptr_h, sizeof(int) * (matrix_size + 1), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(col_indices_d, col_indices_h, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(data_d, data_h, sizeof(double) * nnz, cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(rhs_d, rhs_h, sizeof(double) * matrix_size, cudaMemcpyHostToDevice));

    cudaErrchk(cudaMemset(x_d, 0.0, matrix_size*sizeof(double)))


    cusparseErrchk(cusparseCreateMatDescr(&descrA));
    cusparseErrchk(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    cusparseErrchk(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    time = -omp_get_wtime();
    if(flag_verbose){
        std::printf("Cholesky factorization\n");
    }
    cudaErrchk(cudaStreamSynchronize(stream));

    cusolverErrchk(cusolverSpDcsrlsvchol(
        handle, matrix_size, nnz, descrA, data_d, row_indptr_d, col_indices_d,
        rhs_d, singular_tol, reorder, x_d, &singularity));

    cudaErrchk(cudaStreamSynchronize(stream));
    if(flag_verbose){
        std::printf("Cholesky factorization done\n");
    }
    time += omp_get_wtime();

    if (0 <= singularity) {
        printf("WARNING: the matrix is singular at row %d under tolerance (%E)\n",
            singularity, singular_tol);
    }


    cudaErrchk(cudaMemcpy(rhs_h, x_d, matrix_size * sizeof(double), cudaMemcpyDeviceToHost));

    double relative_error[1];

    if(!assert_array_magnitude<double>(rhs_h,
            reference_solution_h,
            abstol,
            reltol,
            matrix_size,
            relative_error)){
        std::printf("Error: CHOL solution is not the same as the reference solution\n");
    }
    else{
        std::printf("CHOL solution is the same as the reference solution\n");
    }

    //Destroy handles
    if(handle) {
        cusolverErrchk(cusolverSpDestroy(handle));
    }
    if(cusparseHandle) {
        cusparseErrchk(cusparseDestroy(cusparseHandle));
    }
    if(stream) {
        cudaErrchk(cudaStreamDestroy(stream));
    }
    if(descrA) {
        cusparseErrchk(cusparseDestroyMatDescr(descrA));
    }

    //Destroy buffers
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


    return time;
}
