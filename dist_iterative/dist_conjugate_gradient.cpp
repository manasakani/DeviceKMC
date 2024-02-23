#include "dist_conjugate_gradient.h"
#include "dist_spmv.h"
namespace iterative_solver{

template <void (*distributed_spmv)(Distributed_matrix&, Distributed_vector&, cusparseDnVecDescr_t&, cudaStream_t&, cusparseHandle_t&)>
void conjugate_gradient(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm)
{
    // initialize cuda
    cudaStream_t default_stream = NULL;
    cublasHandle_t cublasHandle = 0;
    cublasErrchk(cublasCreate(&cublasHandle));
    
    cusparseHandle_t default_cusparseHandle = 0;
    cusparseErrchk(cusparseCreate(&default_cusparseHandle));    

    cudaErrchk(cudaStreamCreate(&default_stream));
    cusparseErrchk(cusparseSetStream(default_cusparseHandle, default_stream));
    cublasErrchk(cublasSetStream(cublasHandle, default_stream));

    double a, b, na;
    double alpha, alpham1, r0;
    double *r_norm2_h;
    double *dot_h;    
    cudaErrchk(cudaMallocHost((void**)&r_norm2_h, sizeof(double)));
    cudaErrchk(cudaMallocHost((void**)&dot_h, sizeof(double)));

    alpha = 1.0;
    alpham1 = -1.0;
    r0 = 0.0;

    //copy data to device
    // starting guess for p
    cudaErrchk(cudaMemcpy(p_distributed.vec_d[0], x_local_d,
        p_distributed.counts[A_distributed.rank] * sizeof(double), cudaMemcpyDeviceToDevice));

    double *Ap_local_d = NULL;
    cusparseDnVecDescr_t vecAp_local = NULL;
    cudaErrchk(cudaMalloc((void **)&Ap_local_d, A_distributed.rows_this_rank * sizeof(double)));
    cudaErrchk(cudaMemset(Ap_local_d, 0, A_distributed.rows_this_rank * sizeof(double)));
    cusparseErrchk(cusparseCreateDnVec(&vecAp_local, A_distributed.rows_this_rank, Ap_local_d, CUDA_R_64F));

    //begin CG

    // norm of rhs for convergence check
    double norm2_rhs = 0;
    cublasErrchk(cublasDdot(cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, r_local_d, 1, &norm2_rhs));
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
    cublasErrchk(cublasDaxpy(cublasHandle, A_distributed.rows_this_rank, &alpham1, Ap_local_d, 1, r_local_d, 1));
    cublasErrchk(cublasDdot(cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, r_local_d, 1, r_norm2_h));
    MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);


    int k = 1;
    while (r_norm2_h[0]/norm2_rhs > relative_tolerance * relative_tolerance && k <= max_iterations) {
        if(k > 1){
            // pk+1 = rk+1 + b*pk
            b = r_norm2_h[0] / r0;
            cublasErrchk(cublasDscal(cublasHandle, A_distributed.rows_this_rank, &b, p_distributed.vec_d[0], 1));
            cublasErrchk(cublasDaxpy(cublasHandle, A_distributed.rows_this_rank, &alpha, r_local_d, 1, p_distributed.vec_d[0], 1)); 
        }
        else {
            // p0 = r0
            cublasErrchk(cublasDcopy(cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, p_distributed.vec_d[0], 1));
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

        cublasErrchk(cublasDdot(cublasHandle, A_distributed.rows_this_rank, p_distributed.vec_d[0], 1, Ap_local_d, 1, dot_h));
        MPI_Allreduce(MPI_IN_PLACE, dot_h, 1, MPI_DOUBLE, MPI_SUM, comm);

        a = r_norm2_h[0] / dot_h[0];

        // xk+1 = xk + ak * pk
        cublasErrchk(cublasDaxpy(cublasHandle, A_distributed.rows_this_rank, &a, p_distributed.vec_d[0], 1, x_local_d, 1));

        // rk+1 = rk - ak * A * pk
        na = -a;
        cublasErrchk(cublasDaxpy(cublasHandle, A_distributed.rows_this_rank, &na, Ap_local_d, 1, r_local_d, 1));
        r0 = r_norm2_h[0];

        // r_norm2_h = r0*r0
        cublasErrchk(cublasDdot(cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, r_local_d, 1, r_norm2_h));
        MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);
        k++;
    }

    //end CG
    cudaErrchk(cudaDeviceSynchronize());
    if(A_distributed.rank == 0){
        std::cout << "iteration = " << k << ", relative residual = " << sqrt(r_norm2_h[0]/norm2_rhs) << std::endl;
    }

    cusparseErrchk(cusparseDestroy(default_cusparseHandle));
    cublasErrchk(cublasDestroy(cublasHandle));
    cudaErrchk(cudaStreamDestroy(default_stream));
    cusparseErrchk(cusparseDestroyDnVec(vecAp_local));
    cudaErrchk(cudaFree(Ap_local_d));
    

    cudaErrchk(cudaFreeHost(r_norm2_h));
    cudaErrchk(cudaFreeHost(dot_h));

    MPI_Barrier(comm);
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



template <void (*distributed_spmv)(Distributed_matrix&, Distributed_vector&, cusparseDnVecDescr_t&, cudaStream_t&, cusparseHandle_t&)>
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
    cudaStream_t default_stream = NULL;
    cublasHandle_t cublasHandle = 0;
    cublasErrchk(cublasCreate(&cublasHandle));
    
    cusparseHandle_t default_cusparseHandle = 0;
    cusparseErrchk(cusparseCreate(&default_cusparseHandle));    

    cudaErrchk(cudaStreamCreate(&default_stream));
    cusparseErrchk(cusparseSetStream(default_cusparseHandle, default_stream));
    cublasErrchk(cublasSetStream(cublasHandle, default_stream));

    double a, b, na;
    double alpha, alpham1, r0;
    double *r_norm2_h;
    double *dot_h;    
    cudaErrchk(cudaMallocHost((void**)&r_norm2_h, sizeof(double)));
    cudaErrchk(cudaMallocHost((void**)&dot_h, sizeof(double)));

    alpha = 1.0;
    alpham1 = -1.0;
    r0 = 0.0;

    //copy data to device
    // starting guess for p
    cudaErrchk(cudaMemcpy(p_distributed.vec_d[0], x_local_d,
        p_distributed.counts[A_distributed.rank] * sizeof(double), cudaMemcpyDeviceToDevice));

    double *Ap_local_d = NULL;
    cusparseDnVecDescr_t vecAp_local = NULL;
    cudaErrchk(cudaMalloc((void **)&Ap_local_d, A_distributed.rows_this_rank * sizeof(double)));
    cudaErrchk(cudaMemset(Ap_local_d, 0, A_distributed.rows_this_rank * sizeof(double)));
    cusparseErrchk(cusparseCreateDnVec(&vecAp_local, A_distributed.rows_this_rank, Ap_local_d, CUDA_R_64F));
    
    double *z_local_d = NULL;
    cudaErrchk(cudaMalloc((void **)&z_local_d, A_distributed.rows_this_rank * sizeof(double)));

    //begin CG

    // norm of rhs for convergence check
    double norm2_rhs = 0;
    cublasErrchk(cublasDdot(cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, r_local_d, 1, &norm2_rhs));
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
    cublasErrchk(cublasDaxpy(cublasHandle, A_distributed.rows_this_rank, &alpham1, Ap_local_d, 1, r_local_d, 1));
    
    // Mz = r
    elementwise_vector_vector(
        r_local_d,
        diag_inv_local_d,
        z_local_d,
        A_distributed.rows_this_rank,
        default_stream
    ); 
    
    cublasErrchk(cublasDdot(cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, z_local_d, 1, r_norm2_h));
    MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);


    int k = 1;
    while (r_norm2_h[0]/norm2_rhs > relative_tolerance * relative_tolerance && k <= max_iterations) {
        if(k > 1){
            // pk+1 = rk+1 + b*pk
            b = r_norm2_h[0] / r0;
            cublasErrchk(cublasDscal(cublasHandle, A_distributed.rows_this_rank, &b, p_distributed.vec_d[0], 1));
            cublasErrchk(cublasDaxpy(cublasHandle, A_distributed.rows_this_rank, &alpha, z_local_d, 1, p_distributed.vec_d[0], 1)); 
        }
        else {
            // p0 = r0
            cublasErrchk(cublasDcopy(cublasHandle, A_distributed.rows_this_rank, z_local_d, 1, p_distributed.vec_d[0], 1));
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

        cublasErrchk(cublasDdot(cublasHandle, A_distributed.rows_this_rank, p_distributed.vec_d[0], 1, Ap_local_d, 1, dot_h));
        MPI_Allreduce(MPI_IN_PLACE, dot_h, 1, MPI_DOUBLE, MPI_SUM, comm);

        a = r_norm2_h[0] / dot_h[0];

        // xk+1 = xk + ak * pk
        cublasErrchk(cublasDaxpy(cublasHandle, A_distributed.rows_this_rank, &a, p_distributed.vec_d[0], 1, x_local_d, 1));

        // rk+1 = rk - ak * A * pk
        na = -a;
        cublasErrchk(cublasDaxpy(cublasHandle, A_distributed.rows_this_rank, &na, Ap_local_d, 1, r_local_d, 1));
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
        cublasErrchk(cublasDdot(cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, z_local_d, 1, r_norm2_h));
        MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);
        k++;
    }

    //end CG
    cudaErrchk(cudaDeviceSynchronize());
    if(A_distributed.rank == 0){
        std::cout << "iteration = " << k << ", relative residual = " << sqrt(r_norm2_h[0]/norm2_rhs) << std::endl;
    }

    cusparseErrchk(cusparseDestroy(default_cusparseHandle));
    cublasErrchk(cublasDestroy(cublasHandle));
    cudaErrchk(cudaStreamDestroy(default_stream));
    cusparseErrchk(cusparseDestroyDnVec(vecAp_local));
    cudaErrchk(cudaFree(Ap_local_d));
    cudaErrchk(cudaFree(z_local_d));

    cudaErrchk(cudaFreeHost(r_norm2_h));
    cudaErrchk(cudaFreeHost(dot_h));

    MPI_Barrier(comm);
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