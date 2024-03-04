#include "dist_conjugate_gradient.h"
#include "dist_spmv.h"
namespace iterative_solver{

template <void (*distributed_spmv_split)
    (Distributed_subblock &,
    Distributed_matrix &,    
    double *,
    double *,
    Distributed_vector &,
    double *,
    cusparseDnVecDescr_t &,
    double *,
    cudaStream_t &,
    cusparseHandle_t &,
    cublasHandle_t &)>
void conjugate_gradient_split(
    Distributed_subblock &A_subblock,
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

    // dense subblock product y=Ax
    double *p_subblock_d = NULL;
    double *Ap_subblock_d = NULL;
    cudaErrchk(cudaMalloc((void **)&p_subblock_d,
        A_subblock.subblock_size * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&Ap_subblock_d,
        A_subblock.count_subblock_h[A_distributed.rank] * sizeof(double)));
    double *p_subblock_h = new double[A_subblock.subblock_size];


    //begin CG

    // norm of rhs for convergence check
    double norm2_rhs = 0;
    cublasErrchk(cublasDdot(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, r_local_d, 1, &norm2_rhs));
    MPI_Allreduce(MPI_IN_PLACE, &norm2_rhs, 1, MPI_DOUBLE, MPI_SUM, comm);


    distributed_spmv_split(
        A_subblock,
        A_distributed,
        p_subblock_d,
        p_subblock_h,
        p_distributed,
        Ap_subblock_d,
        vecAp_local,
        Ap_local_d,
        default_stream,
        default_cusparseHandle,
        default_cublasHandle
    );

    // cal residual r0 = b - A*x0
    // r_norm2_h = r0*r0
    cublasErrchk(cublasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, &alpham1, Ap_local_d, 1, r_local_d, 1));
    cublasErrchk(cublasDdot(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, r_local_d, 1, r_norm2_h));
    MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);


    int k = 1;
    while (r_norm2_h[0]/norm2_rhs > relative_tolerance * relative_tolerance && k <= max_iterations) {
        if(k > 1){
            // pk+1 = rk+1 + b*pk
            b = r_norm2_h[0] / r0;
            cublasErrchk(cublasDscal(default_cublasHandle, A_distributed.rows_this_rank, &b, p_distributed.vec_d[0], 1));
            cublasErrchk(cublasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, &alpha, r_local_d, 1, p_distributed.vec_d[0], 1)); 
        }
        else {
            // p0 = r0
            cublasErrchk(cublasDcopy(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, p_distributed.vec_d[0], 1));
        }


        distributed_spmv_split(
            A_subblock,
            A_distributed,
            p_subblock_d,
            p_subblock_h,
            p_distributed,
            Ap_subblock_d,
            vecAp_local,
            Ap_local_d,
            default_stream,
            default_cusparseHandle,
            default_cublasHandle
        );

        cublasErrchk(cublasDdot(default_cublasHandle, A_distributed.rows_this_rank, p_distributed.vec_d[0], 1, Ap_local_d, 1, dot_h));
        MPI_Allreduce(MPI_IN_PLACE, dot_h, 1, MPI_DOUBLE, MPI_SUM, comm);

        a = r_norm2_h[0] / dot_h[0];

        // xk+1 = xk + ak * pk
        cublasErrchk(cublasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, &a, p_distributed.vec_d[0], 1, x_local_d, 1));

        // rk+1 = rk - ak * A * pk
        na = -a;
        cublasErrchk(cublasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, &na, Ap_local_d, 1, r_local_d, 1));
        r0 = r_norm2_h[0];

        // r_norm2_h = r0*r0
        cublasErrchk(cublasDdot(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, r_local_d, 1, r_norm2_h));
        MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);
        k++;
    }

    //end CG
    cudaErrchk(cudaDeviceSynchronize());
    if(A_distributed.rank == 0){
        std::cout << "iteration = " << k << ", relative residual = " << sqrt(r_norm2_h[0]/norm2_rhs) << std::endl;
    }

    cusparseErrchk(cusparseDestroy(default_cusparseHandle));
    cublasErrchk(cublasDestroy(default_cublasHandle));
    cudaErrchk(cudaStreamDestroy(default_stream));
    cusparseErrchk(cusparseDestroyDnVec(vecAp_local));
    cudaErrchk(cudaFree(Ap_local_d));
    cudaErrchk(cudaFreeHost(r_norm2_h));
    cudaErrchk(cudaFreeHost(dot_h));
    cudaErrchk(cudaFree(p_subblock_d));
    cudaErrchk(cudaFree(Ap_subblock_d));

    delete[] p_subblock_h;

    MPI_Barrier(comm);
}
template 
void conjugate_gradient_split<dspmv_split::spmm_split1>(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);
template 
void conjugate_gradient_split<dspmv_split::spmm_split2>(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);
template 
void conjugate_gradient_split<dspmv_split::spmm_split3>(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);
template 
void conjugate_gradient_split<dspmv_split::spmm_split4>(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);
template 
void conjugate_gradient_split<dspmv_split::spmm_split5>(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);
template 
void conjugate_gradient_split<dspmv_split::spmm_split6>(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);



template <void (*distributed_spmv_split)
    (Distributed_subblock &,
    Distributed_matrix &,    
    double *,
    double *,
    Distributed_vector &,
    double *,
    cusparseDnVecDescr_t &,
    double *,
    cudaStream_t &,
    cusparseHandle_t &,
    cublasHandle_t &)>
void conjugate_gradient_jacobi_split(
    Distributed_subblock &A_subblock,
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

    // dense subblock product y=Ax
    double *p_subblock_d = NULL;
    double *Ap_subblock_d = NULL;
    cudaErrchk(cudaMalloc((void **)&p_subblock_d,
        A_subblock.subblock_size * sizeof(double)));
    cudaErrchk(cudaMalloc((void **)&Ap_subblock_d,
        A_subblock.count_subblock_h[A_distributed.rank] * sizeof(double)));
    double *p_subblock_h = new double[A_subblock.subblock_size];

    //begin CG

    // norm of rhs for convergence check
    double norm2_rhs = 0;
    cublasErrchk(cublasDdot(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, r_local_d, 1, &norm2_rhs));
    MPI_Allreduce(MPI_IN_PLACE, &norm2_rhs, 1, MPI_DOUBLE, MPI_SUM, comm);

    // A*x0
    distributed_spmv_split(
        A_subblock,
        A_distributed,
        p_subblock_d,
        p_subblock_h,
        p_distributed,
        Ap_subblock_d,
        vecAp_local,
        Ap_local_d,
        default_stream,
        default_cusparseHandle,
        default_cublasHandle
    );


    // cal residual r0 = b - A*x0
    // r_norm2_h = r0*r0
    cublasErrchk(cublasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, &alpham1, Ap_local_d, 1, r_local_d, 1));
    
    // Mz = r
    elementwise_vector_vector(
        r_local_d,
        diag_inv_local_d,
        z_local_d,
        A_distributed.rows_this_rank,
        default_stream
    ); 
    
    cublasErrchk(cublasDdot(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, z_local_d, 1, r_norm2_h));
    MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);


    int k = 1;
    while (r_norm2_h[0]/norm2_rhs > relative_tolerance * relative_tolerance && k <= max_iterations) {
        if(k > 1){
            // pk+1 = rk+1 + b*pk
            b = r_norm2_h[0] / r0;
            cublasErrchk(cublasDscal(default_cublasHandle, A_distributed.rows_this_rank, &b, p_distributed.vec_d[0], 1));
            cublasErrchk(cublasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, &alpha, z_local_d, 1, p_distributed.vec_d[0], 1)); 
        }
        else {
            // p0 = r0
            cublasErrchk(cublasDcopy(default_cublasHandle, A_distributed.rows_this_rank, z_local_d, 1, p_distributed.vec_d[0], 1));
        }


        // ak = rk^T * rk / pk^T * A * pk
        // has to be done for k=0 if x0 != 0
        distributed_spmv_split(
            A_subblock,
            A_distributed,
            p_subblock_d,
            p_subblock_h,
            p_distributed,
            Ap_subblock_d,
            vecAp_local,
            Ap_local_d,
            default_stream,
            default_cusparseHandle,
            default_cublasHandle
        );

        cublasErrchk(cublasDdot(default_cublasHandle, A_distributed.rows_this_rank, p_distributed.vec_d[0], 1, Ap_local_d, 1, dot_h));
        MPI_Allreduce(MPI_IN_PLACE, dot_h, 1, MPI_DOUBLE, MPI_SUM, comm);

        a = r_norm2_h[0] / dot_h[0];

        // xk+1 = xk + ak * pk
        cublasErrchk(cublasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, &a, p_distributed.vec_d[0], 1, x_local_d, 1));

        // rk+1 = rk - ak * A * pk
        na = -a;
        cublasErrchk(cublasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, &na, Ap_local_d, 1, r_local_d, 1));
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
        cublasErrchk(cublasDdot(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, z_local_d, 1, r_norm2_h));
        MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);
        k++;
    }

    //end CG
    cudaErrchk(cudaDeviceSynchronize());
    if(A_distributed.rank == 0){
        std::cout << "iteration = " << k << ", relative residual = " << sqrt(r_norm2_h[0]/norm2_rhs) << std::endl;
    }

    cusparseErrchk(cusparseDestroy(default_cusparseHandle));
    cublasErrchk(cublasDestroy(default_cublasHandle));
    cudaErrchk(cudaStreamDestroy(default_stream));
    cusparseErrchk(cusparseDestroyDnVec(vecAp_local));
    cudaErrchk(cudaFree(Ap_local_d));
    cudaErrchk(cudaFree(z_local_d));

    cudaErrchk(cudaFreeHost(r_norm2_h));
    cudaErrchk(cudaFreeHost(dot_h));
    cudaErrchk(cudaFree(p_subblock_d));
    cudaErrchk(cudaFree(Ap_subblock_d));
    delete[] p_subblock_h;

    MPI_Barrier(comm);
}
template 
void conjugate_gradient_jacobi_split<dspmv_split::spmm_split1>(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);
template 
void conjugate_gradient_jacobi_split<dspmv_split::spmm_split2>(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);
template 
void conjugate_gradient_jacobi_split<dspmv_split::spmm_split3>(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);
template 
void conjugate_gradient_jacobi_split<dspmv_split::spmm_split4>(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);
template 
void conjugate_gradient_jacobi_split<dspmv_split::spmm_split5>(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);
template 
void conjugate_gradient_jacobi_split<dspmv_split::spmm_split6>(
    Distributed_subblock &A_subblock,
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);


} // namespace iterative_solver