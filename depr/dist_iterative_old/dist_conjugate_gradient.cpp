#include "dist_conjugate_gradient.h"
#include "dist_spmv.h"
namespace iterative_solver{

template <void (*distributed_spmv)(Distributed_matrix&, Distributed_vector&, hipsparseDnVecDescr_t&, hipStream_t&, hipsparseHandle_t&)>
void conjugate_gradient(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm)
{

    std::cout << "inititalize_cuda" << std::endl; MPI_Barrier(MPI_COMM_WORLD); fflush(stdout);

    // initialize cuda
    hipStream_t default_stream = NULL;
    hipblasHandle_t default_cublasHandle = 0;
    cublasErrchk(hipblasCreate(&default_cublasHandle));
    
    hipsparseHandle_t default_cusparseHandle = 0;
    cusparseErrchk(hipsparseCreate(&default_cusparseHandle));    

    cudaErrchk(hipStreamCreate(&default_stream));
    cusparseErrchk(hipsparseSetStream(default_cusparseHandle, default_stream));
    cublasErrchk(hipblasSetStream(default_cublasHandle, default_stream));

    MPI_Barrier(MPI_COMM_WORLD); std::cout << "done here" << std::endl; MPI_Barrier(MPI_COMM_WORLD); fflush(stdout);

    double a, b, na;
    double alpha, alpham1, r0;
    double *r_norm2_h;
    double *dot_h;    
    cudaErrchk(hipHostMalloc((void**)&r_norm2_h, sizeof(double)));
    cudaErrchk(hipHostMalloc((void**)&dot_h, sizeof(double)));

    alpha = 1.0;
    alpham1 = -1.0;
    r0 = 0.0;

    //copy data to device
    // starting guess for p
    cudaErrchk(hipMemcpy(p_distributed.vec_d[0], x_local_d,
        p_distributed.counts[A_distributed.rank] * sizeof(double), hipMemcpyDeviceToDevice));

    double *Ap_local_d = NULL;
    hipsparseDnVecDescr_t vecAp_local = NULL;
    cudaErrchk(hipMalloc((void **)&Ap_local_d, A_distributed.rows_this_rank * sizeof(double)));
    cudaErrchk(hipMemset(Ap_local_d, 0, A_distributed.rows_this_rank * sizeof(double)));
    cusparseErrchk(hipsparseCreateDnVec(&vecAp_local, A_distributed.rows_this_rank, Ap_local_d, HIP_R_64F));

    //begin CG

    // norm of rhs for convergence check
    double norm2_rhs = 0;
    cublasErrchk(hipblasDdot(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, r_local_d, 1, &norm2_rhs));
    MPI_Allreduce(MPI_IN_PLACE, &norm2_rhs, 1, MPI_DOUBLE, MPI_SUM, comm);

    MPI_Barrier(MPI_COMM_WORLD); std::cout << "distributed_spmv" << std::endl; MPI_Barrier(MPI_COMM_WORLD); fflush(stdout);

    // A*x0
    distributed_spmv(
        A_distributed,
        p_distributed,
        vecAp_local,
        default_stream,
        default_cusparseHandle
    );

    MPI_Barrier(MPI_COMM_WORLD); std::cout << "distributed_spmv end" << std::endl; MPI_Barrier(MPI_COMM_WORLD); fflush(stdout);

    // cal residual r0 = b - A*x0
    // r_norm2_h = r0*r0
    cublasErrchk(hipblasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, &alpham1, Ap_local_d, 1, r_local_d, 1));
    cublasErrchk(hipblasDdot(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, r_local_d, 1, r_norm2_h));
    MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);


    int k = 1;
    while (r_norm2_h[0]/norm2_rhs > relative_tolerance * relative_tolerance && k <= max_iterations) {
        if(k > 1){
            // pk+1 = rk+1 + b*pk
            b = r_norm2_h[0] / r0;
            cublasErrchk(hipblasDscal(default_cublasHandle, A_distributed.rows_this_rank, &b, p_distributed.vec_d[0], 1));
            cublasErrchk(hipblasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, &alpha, r_local_d, 1, p_distributed.vec_d[0], 1)); 
        }
        else {
            // p0 = r0
            cublasErrchk(hipblasDcopy(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, p_distributed.vec_d[0], 1));
        }


        // ak = rk^T * rk / pk^T * A * pk
        // has to be done for k=0 if x0 != 0
        distributed_spmv(
            A_distributed,
            p_distributed,
            vecAp_local,
            default_stream,
            default_cusparseHandle
        );

        cublasErrchk(hipblasDdot(default_cublasHandle, A_distributed.rows_this_rank, p_distributed.vec_d[0], 1, Ap_local_d, 1, dot_h));
        MPI_Allreduce(MPI_IN_PLACE, dot_h, 1, MPI_DOUBLE, MPI_SUM, comm);

        a = r_norm2_h[0] / dot_h[0];

        // xk+1 = xk + ak * pk
        cublasErrchk(hipblasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, &a, p_distributed.vec_d[0], 1, x_local_d, 1));

        // rk+1 = rk - ak * A * pk
        na = -a;
        cublasErrchk(hipblasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, &na, Ap_local_d, 1, r_local_d, 1));
        r0 = r_norm2_h[0];

        // r_norm2_h = r0*r0
        cublasErrchk(hipblasDdot(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, r_local_d, 1, r_norm2_h));
        MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);
        k++;
    }

    //end CG
    cudaErrchk(hipDeviceSynchronize());
    if(A_distributed.rank == 0){
        std::cout << "iteration = " << k << ", relative residual = " << sqrt(r_norm2_h[0]/norm2_rhs) << std::endl;
    }

    cusparseErrchk(hipsparseDestroy(default_cusparseHandle));
    cublasErrchk(hipblasDestroy(default_cublasHandle));
    cudaErrchk(hipStreamDestroy(default_stream));
    cusparseErrchk(hipsparseDestroyDnVec(vecAp_local));
    cudaErrchk(hipFree(Ap_local_d));
    

    cudaErrchk(hipHostFree(r_norm2_h));
    cudaErrchk(hipHostFree(dot_h));

}
template 
void conjugate_gradient<dspmv::gpu_packing>(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);
template 
void conjugate_gradient<dspmv::gpu_packing_cam>(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);



template <void (*distributed_spmv)(Distributed_matrix&, Distributed_vector&, hipsparseDnVecDescr_t&, hipStream_t&, hipsparseHandle_t&)>
void conjugate_gradient_jacobi(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm)
{

    // initialize cuda
    hipStream_t default_stream = NULL;
    hipblasHandle_t default_cublasHandle = 0;
    cublasErrchk(hipblasCreate(&default_cublasHandle));
    
    hipsparseHandle_t default_cusparseHandle = 0;
    cusparseErrchk(hipsparseCreate(&default_cusparseHandle));    

    cudaErrchk(hipStreamCreate(&default_stream));
    cusparseErrchk(hipsparseSetStream(default_cusparseHandle, default_stream));
    cublasErrchk(hipblasSetStream(default_cublasHandle, default_stream));

    double a, b, na;
    double alpha, alpham1, r0;
    // double *r_norm2_h;
    // double *dot_h;    
    // cudaErrchk(hipHostMalloc((void**)&r_norm2_h, sizeof(double)));  // change this
    // cudaErrchk(hipHostMalloc((void**)&dot_h, sizeof(double)));
    double    *r_norm2_h = new    double[1];
    double    *dot_h     = new    double[1];

    alpha = 1.0;
    alpham1 = -1.0;
    r0 = 0.0;

    //copy data to device
    // starting guess for p
    cudaErrchk(hipMemcpy(p_distributed.vec_d[0], x_local_d,
        p_distributed.counts[A_distributed.rank] * sizeof(double), hipMemcpyDeviceToDevice));

    double *Ap_local_d = NULL;
    hipsparseDnVecDescr_t vecAp_local = NULL;
    cudaErrchk(hipMalloc((void **)&Ap_local_d, A_distributed.rows_this_rank * sizeof(double)));
    cudaErrchk(hipMemset(Ap_local_d, 0, A_distributed.rows_this_rank * sizeof(double)));
    cusparseErrchk(hipsparseCreateDnVec(&vecAp_local, A_distributed.rows_this_rank, Ap_local_d, HIP_R_64F));
    
    double *z_local_d = NULL;
    cudaErrchk(hipMalloc((void **)&z_local_d, A_distributed.rows_this_rank * sizeof(double)));

    //begin CG

    // norm of rhs for convergence check
    double norm2_rhs = 0;
    cublasErrchk(hipblasDdot(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, r_local_d, 1, &norm2_rhs));
    MPI_Allreduce(MPI_IN_PLACE, &norm2_rhs, 1, MPI_DOUBLE, MPI_SUM, comm);

    // A*x0
    distributed_spmv(
        A_distributed,
        p_distributed,
        vecAp_local,
        default_stream,
        default_cusparseHandle
    );

    // cal residual r0 = b - A*x0
    // r_norm2_h = r0*r0
    cublasErrchk(hipblasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, &alpham1, Ap_local_d, 1, r_local_d, 1));
    
    // Mz = r
    elementwise_vector_vector(
        r_local_d,
        diag_inv_local_d,
        z_local_d,
        A_distributed.rows_this_rank,
        default_stream
    ); 

    cublasErrchk(hipblasDdot(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, z_local_d, 1, r_norm2_h));
    MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);

    int k = 1;

    while (r_norm2_h[0]/norm2_rhs > relative_tolerance * relative_tolerance && k <= max_iterations) {
        
        if(k > 1){
            // pk+1 = rk+1 + b*pk
            b = r_norm2_h[0] / r0;
            cublasErrchk(hipblasDscal(default_cublasHandle, A_distributed.rows_this_rank, &b, p_distributed.vec_d[0], 1));
            cublasErrchk(hipblasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, &alpha, z_local_d, 1, p_distributed.vec_d[0], 1)); 
        }
        else {
            // p0 = r0
            cublasErrchk(hipblasDcopy(default_cublasHandle, A_distributed.rows_this_rank, z_local_d, 1, p_distributed.vec_d[0], 1));
        }


        // ak = rk^T * rk / pk^T * A * pk
        // has to be done for k=0 if x0 != 0
        distributed_spmv(
            A_distributed,
            p_distributed,
            vecAp_local,
            default_stream,
            default_cusparseHandle
        );

        cublasErrchk(hipblasDdot(default_cublasHandle, A_distributed.rows_this_rank, p_distributed.vec_d[0], 1, Ap_local_d, 1, dot_h));
        MPI_Allreduce(MPI_IN_PLACE, dot_h, 1, MPI_DOUBLE, MPI_SUM, comm);

        a = r_norm2_h[0] / dot_h[0];

        // xk+1 = xk + ak * pk
        cublasErrchk(hipblasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, &a, p_distributed.vec_d[0], 1, x_local_d, 1));

        // rk+1 = rk - ak * A * pk
        na = -a;
        cublasErrchk(hipblasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, &na, Ap_local_d, 1, r_local_d, 1));
        r0 = r_norm2_h[0];

        // Mz = r
        elementwise_vector_vector(
            r_local_d,
            diag_inv_local_d,
            z_local_d,
            A_distributed.rows_this_rank,
            default_stream
        ); 
        

        // r_norm2_h = r0*r0
        cublasErrchk(hipblasDdot(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, z_local_d, 1, r_norm2_h));
        MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);
        k++;
    }

    //end CG
    cudaErrchk(hipDeviceSynchronize());
    if(A_distributed.rank == 0){
        std::cout << "iteration = " << k << ", relative residual = " << sqrt(r_norm2_h[0]/norm2_rhs) << std::endl;
    }

    cusparseErrchk(hipsparseDestroy(default_cusparseHandle));
    cublasErrchk(hipblasDestroy(default_cublasHandle));
    cudaErrchk(hipStreamDestroy(default_stream));
    cusparseErrchk(hipsparseDestroyDnVec(vecAp_local));
    cudaErrchk(hipFree(Ap_local_d));
    cudaErrchk(hipFree(z_local_d));

    // cudaErrchk(hipHostFree(r_norm2_h));
    // cudaErrchk(hipHostFree(dot_h));
    free(r_norm2_h);
    free(dot_h);

}
template 
void conjugate_gradient_jacobi<dspmv::gpu_packing>(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);
template 
void conjugate_gradient_jacobi<dspmv::gpu_packing_cam>(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);

} // namespace iterative_solver
