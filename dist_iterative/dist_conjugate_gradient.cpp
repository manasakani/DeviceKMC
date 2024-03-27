#include "dist_conjugate_gradient.h"
#include "dist_spmv.h"

#include <cfloat>
#include <cmath>
#include <iostream>
#include <limits>

namespace iterative_solver{

template <void (*distributed_spmv)(
    Distributed_matrix&,
    Distributed_vector&,
    rocsparse_dnvec_descr&,
    hipStream_t&,
    rocsparse_handle&)>
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
    hipStream_t default_stream = NULL;
    hipblasHandle_t default_cublasHandle = 0;
    cublasErrchk(hipblasCreate(&default_cublasHandle));
    
    hipsparseHandle_t default_cusparseHandle = 0;
    cusparseErrchk(hipsparseCreate(&default_cusparseHandle));    

    rocsparse_handle default_rocsparseHandle;
    rocsparse_create_handle(&default_rocsparseHandle);

    cudaErrchk(hipStreamCreate(&default_stream));
    cusparseErrchk(hipsparseSetStream(default_cusparseHandle, default_stream));
    cublasErrchk(hipblasSetStream(default_cublasHandle, default_stream));
    rocsparse_set_stream(default_rocsparseHandle, default_stream);

    double a, b, na;
    double alpha, alpham1, r0;
    // double *r_norm2_h;
    // double *dot_h;    
    // cudaErrchk(hipHostMalloc((void**)&r_norm2_h, sizeof(double)));
    // cudaErrchk(hipHostMalloc((void**)&dot_h, sizeof(double)));
    double    r_norm2_h[1];
    double    dot_h[1];

    alpha = 1.0;
    alpham1 = -1.0;
    r0 = 0.0;

    // set pointer mode to host
    cublasErrchk(hipblasSetPointerMode(default_cublasHandle, HIPBLAS_POINTER_MODE_HOST)); // TEST @Manasa

    //copy data to device
    // starting guess for p
    cudaErrchk(hipMemcpy(p_distributed.vec_d[0], x_local_d,
        p_distributed.counts[A_distributed.rank] * sizeof(double), hipMemcpyDeviceToDevice));

    double *Ap_local_d = NULL;
    cudaErrchk(hipMalloc((void **)&Ap_local_d, A_distributed.rows_this_rank * sizeof(double)));
    cudaErrchk(hipMemset(Ap_local_d, 0, A_distributed.rows_this_rank * sizeof(double)));
    rocsparse_dnvec_descr vecAp_local = NULL;
    rocsparse_create_dnvec_descr(&vecAp_local,
                                A_distributed.rows_this_rank,
                                Ap_local_d,
                                rocsparse_datatype_f64_r);

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
        default_rocsparseHandle
    );

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
            default_rocsparseHandle
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

        // if(A_distributed.rank == 0){
        //     std::cout << "iteration = " << k << ", relative residual = " << sqrt(r_norm2_h[0]/norm2_rhs) << std::endl;
        // }

        // r_norm2_h = r0*r0
        cublasErrchk(hipblasDdot(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, r_local_d, 1, r_norm2_h));
        MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);
        k++;
    }

    //end CG
    cudaErrchk(hipDeviceSynchronize());
    if(A_distributed.rank == 0){
        std::cout << "iteration K = " << k << ", relative residual = " << sqrt(r_norm2_h[0]/norm2_rhs) << std::endl;
    }

    rocsparse_destroy_handle(default_rocsparseHandle);
    cusparseErrchk(hipsparseDestroy(default_cusparseHandle));
    cublasErrchk(hipblasDestroy(default_cublasHandle));
    cudaErrchk(hipStreamDestroy(default_stream));
    rocsparse_destroy_dnvec_descr(vecAp_local);
    cudaErrchk(hipFree(Ap_local_d));
    
    // cudaErrchk(hipHostFree(r_norm2_h));
    // cudaErrchk(hipHostFree(dot_h));

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



template <void (*distributed_spmv)(
    Distributed_matrix&,
    Distributed_vector&,
    rocsparse_dnvec_descr&,
    hipStream_t&,
    rocsparse_handle&)>
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

    rocsparse_handle default_rocsparseHandle;
    rocsparse_create_handle(&default_rocsparseHandle);

    cudaErrchk(hipStreamCreate(&default_stream));
    cusparseErrchk(hipsparseSetStream(default_cusparseHandle, default_stream));
    cublasErrchk(hipblasSetStream(default_cublasHandle, default_stream));
    rocsparse_set_stream(default_rocsparseHandle, default_stream);

    // set pointer mode to host
    cublasErrchk(hipblasSetPointerMode(default_cublasHandle, HIPBLAS_POINTER_MODE_HOST)); // TEST @Manasa

    double a, b, na;
    double alpha, alpham1, r0;
    // double *r_norm2_h;
    // double *dot_h;    
    // cudaErrchk(hipHostMalloc((void**)&r_norm2_h, sizeof(double)));
    // cudaErrchk(hipHostMalloc((void**)&dot_h, sizeof(double)));
    double    r_norm2_h[1];
    double    dot_h[1];

    alpha = 1.0;
    alpham1 = -1.0;
    r0 = 0.0;

    //copy data to device
    // starting guess for p
    cudaErrchk(hipMemcpy(p_distributed.vec_d[0], x_local_d,
        p_distributed.counts[A_distributed.rank] * sizeof(double), hipMemcpyDeviceToDevice));

    double *Ap_local_d = NULL;
    cudaErrchk(hipMalloc((void **)&Ap_local_d, A_distributed.rows_this_rank * sizeof(double)));
    cudaErrchk(hipMemset(Ap_local_d, 0, A_distributed.rows_this_rank * sizeof(double)));
    rocsparse_dnvec_descr vecAp_local = NULL;
    rocsparse_create_dnvec_descr(&vecAp_local,
                                A_distributed.rows_this_rank,
                                Ap_local_d,
                                rocsparse_datatype_f64_r);
    
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
        default_rocsparseHandle
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
            default_rocsparseHandle
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

        // if(A_distributed.rank == 0){
        //     std::cout << "iteration = " << k << ", relative residual = " << sqrt(r_norm2_h[0]/norm2_rhs) << std::endl;
        // }

    }

    //end CG
    cudaErrchk(hipDeviceSynchronize());
    if(A_distributed.rank == 0){
        std::cout << "iteration K = " << k << ", relative residual = " << sqrt(r_norm2_h[0]/norm2_rhs) << std::endl;
    }

    rocsparse_destroy_handle(default_rocsparseHandle);
    cusparseErrchk(hipsparseDestroy(default_cusparseHandle));
    cublasErrchk(hipblasDestroy(default_cublasHandle));
    cudaErrchk(hipStreamDestroy(default_stream));
    rocsparse_destroy_dnvec_descr(vecAp_local);
    cudaErrchk(hipFree(Ap_local_d));
    cudaErrchk(hipFree(z_local_d));

    // cudaErrchk(hipHostFree(r_norm2_h));
    // cudaErrchk(hipHostFree(dot_h));

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


__global__ void r_divided_dot(
    double *r_norm2_d,
    double *dot_d,
    double *a_d
){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx == 0){
        *a_d = *r_norm2_d / *dot_d;
    }
}

__global__ void r_divided_dot2(
    double *r_norm2_d,
    double *dot_global_d,
    double *a_d,
    int size
){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx == 0){
        double tmp = 0;
        for(int i = 0; i < size; i++){
            tmp += dot_global_d[i];
        }
        *a_d = *r_norm2_d / tmp;
    }
}



__global__ void negate_a(
    double *a_d,
    double *na_d
){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx == 0){
        *na_d = -(*a_d);
    }
}

__global__ void equal_r0(
    double *r_norm2_d,
    double *r0_d
){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx == 0){
        *r0_d = *r_norm2_d;
    }
}


template <void (*distributed_spmv)(
    Distributed_matrix&,
    Distributed_vector&,
    rocsparse_dnvec_descr&,
    hipStream_t&,
    rocsparse_handle&)>
void conjugate_gradient_jacobi2(
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

    rocsparse_handle default_rocsparseHandle;
    rocsparse_create_handle(&default_rocsparseHandle);

    cudaErrchk(hipStreamCreate(&default_stream));
    cusparseErrchk(hipsparseSetStream(default_cusparseHandle, default_stream));
    cublasErrchk(hipblasSetStream(default_cublasHandle, default_stream));
    rocsparse_set_stream(default_rocsparseHandle, default_stream);

    //set pointer mode for cublas
    cublasErrchk(hipblasSetPointerMode(default_cublasHandle, HIPBLAS_POINTER_MODE_DEVICE));

    double a, b, na;
    double alpha, alpham1, r0;

    double *a_d, *b_d, *na_d;
    cudaErrchk(hipMalloc((void**)&a_d, sizeof(double)));
    cudaErrchk(hipMalloc((void**)&b_d, sizeof(double)));
    cudaErrchk(hipMalloc((void**)&na_d, sizeof(double)));

    int blocks = 1;
    int threads = 64;

    alpha = 1.0;
    alpham1 = -1.0;
    r0 = 0.0;
    double *alpha_d;
    cudaErrchk(hipMalloc((void**)&alpha_d, sizeof(double)));
    cudaErrchk(hipMemcpy(alpha_d, &alpha, sizeof(double), hipMemcpyHostToDevice));
    double *alpham1_d;
    cudaErrchk(hipMalloc((void**)&alpham1_d, sizeof(double)));
    cudaErrchk(hipMemcpy(alpham1_d, &alpham1, sizeof(double), hipMemcpyHostToDevice));
    double *r0_d;
    cudaErrchk(hipMalloc((void**)&r0_d, sizeof(double)));

    double *dot_h;
    cudaErrchk(hipHostMalloc((void**)&dot_h, sizeof(double)));
    double *dot_d;
    cudaErrchk(hipMalloc((void**)&dot_d, sizeof(double)));


    //copy data to device
    // starting guess for p
    cudaErrchk(hipMemcpy(p_distributed.vec_d[0], x_local_d,
        p_distributed.counts[A_distributed.rank] * sizeof(double), hipMemcpyDeviceToDevice));

    double *Ap_local_d = NULL;
    cudaErrchk(hipMalloc((void **)&Ap_local_d, A_distributed.rows_this_rank * sizeof(double)));
    cudaErrchk(hipMemset(Ap_local_d, 0, A_distributed.rows_this_rank * sizeof(double)));
    rocsparse_dnvec_descr vecAp_local = NULL;
    rocsparse_create_dnvec_descr(&vecAp_local,
                                A_distributed.rows_this_rank,
                                Ap_local_d,
                                rocsparse_datatype_f64_r);

    double *z_local_d = NULL;
    cudaErrchk(hipMalloc((void **)&z_local_d, A_distributed.rows_this_rank * sizeof(double)));

    //begin CG

    // norm of rhs for convergence check
    double *norm2_rhs_h;
    hipHostMalloc((void**)&norm2_rhs_h, sizeof(double));
    double *norm2_rhs_d;
    hipMalloc((void**)&norm2_rhs_d, sizeof(double));
    double *r_norm2_h;
    double *r_norm2_d;
    hipHostMalloc((void**)&r_norm2_h, sizeof(double));
    hipMalloc((void**)&r_norm2_d, sizeof(double));


    cublasErrchk(hipblasDdot(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, r_local_d, 1, norm2_rhs_d));
    hipStreamSynchronize(default_stream);
    MPI_Allreduce(MPI_IN_PLACE, norm2_rhs_d, 1, MPI_DOUBLE, MPI_SUM, comm);

    // A*x0
    distributed_spmv(
        A_distributed,
        p_distributed,
        vecAp_local,
        default_stream,
        default_rocsparseHandle
    );

    // cal residual r0 = b - A*x0
    // r_norm2_h = r0*r0
    cublasErrchk(hipblasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, alpham1_d, Ap_local_d, 1, r_local_d, 1));
    
    // Mz = r
    elementwise_vector_vector(
        r_local_d,
        diag_inv_local_d,
        z_local_d,
        A_distributed.rows_this_rank,
        default_stream
    ); 

    cublasErrchk(hipblasDdot(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, z_local_d, 1, r_norm2_d));
    hipStreamSynchronize(default_stream);
    hipMemcpyAsync(norm2_rhs_h, norm2_rhs_d, sizeof(double), hipMemcpyDeviceToHost, default_stream);
    MPI_Allreduce(MPI_IN_PLACE, r_norm2_d, 1, MPI_DOUBLE, MPI_SUM, comm);
    hipMemcpyAsync(r_norm2_h, r_norm2_d, sizeof(double), hipMemcpyDeviceToHost, default_stream);

    int k = 0;
    hipStreamSynchronize(default_stream);
    while(r_norm2_h[0]/norm2_rhs_h[0] > relative_tolerance * relative_tolerance && k < max_iterations) {
        if(k > 0){
            // pk+1 = rk+1 + b*pk
            // b = r_norm2_h[0] / r0;
            r_divided_dot<<<blocks, threads, 0, default_stream>>>(r_norm2_d, r0_d, b_d);

            cublasErrchk(hipblasDscal(default_cublasHandle, A_distributed.rows_this_rank, b_d, p_distributed.vec_d[0], 1));
            cublasErrchk(hipblasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, alpha_d, z_local_d, 1, p_distributed.vec_d[0], 1)); 
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
            default_rocsparseHandle
        );

        cublasErrchk(hipblasDdot(default_cublasHandle, A_distributed.rows_this_rank, p_distributed.vec_d[0], 1, Ap_local_d, 1, dot_d));
        hipStreamSynchronize(default_stream);
        MPI_Allreduce(MPI_IN_PLACE, dot_d, 1, MPI_DOUBLE, MPI_SUM, comm);

        // a = r_norm2_h[0] / dot_h[0];
        r_divided_dot<<<blocks, threads, 0, default_stream>>>(r_norm2_d, dot_d, a_d);

        // xk+1 = xk + ak * pk
        cublasErrchk(hipblasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, a_d, p_distributed.vec_d[0], 1, x_local_d, 1));

        // rk+1 = rk - ak * A * pk
        // na = -a;
        negate_a<<<blocks, threads, 0, default_stream>>>(a_d, na_d);
        cublasErrchk(hipblasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, na_d, Ap_local_d, 1, r_local_d, 1));
        // r0 = r_norm2_h[0];
        equal_r0<<<blocks, threads, 0, default_stream>>>(r_norm2_d, r0_d);

        // Mz = r
        elementwise_vector_vector(
            r_local_d,
            diag_inv_local_d,
            z_local_d,
            A_distributed.rows_this_rank,
            default_stream
        ); 
        

        // r_norm2_h = r0*r0
        cublasErrchk(hipblasDdot(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, z_local_d, 1, r_norm2_d));
        hipStreamSynchronize(default_stream);
        MPI_Allreduce(MPI_IN_PLACE, r_norm2_d, 1, MPI_DOUBLE, MPI_SUM, comm);
        hipMemcpy(r_norm2_h, r_norm2_d, sizeof(double), hipMemcpyDeviceToHost);
        k++;

    }

    //end CG
    cudaErrchk(hipDeviceSynchronize());
    if(A_distributed.rank == 0){
        std::cout << "iteration K = " << k << ", relative residual = " << sqrt(r_norm2_h[0]/norm2_rhs_h[0]) << std::endl;
    }

    rocsparse_destroy_handle(default_rocsparseHandle);
    cusparseErrchk(hipsparseDestroy(default_cusparseHandle));
    cublasErrchk(hipblasDestroy(default_cublasHandle));
    cudaErrchk(hipStreamDestroy(default_stream));
    rocsparse_destroy_dnvec_descr(vecAp_local);
    cudaErrchk(hipFree(Ap_local_d));
    cudaErrchk(hipFree(z_local_d));

    hipFree(a_d);
    hipFree(b_d);
    hipFree(na_d);

    hipFree(alpha_d);
    hipFree(alpham1_d);
    hipFree(r0_d);

    hipHostFree(norm2_rhs_h);
    hipFree(norm2_rhs_d);

    hipHostFree(r_norm2_h);
    hipFree(r_norm2_d);

    hipHostFree(dot_h);
    hipFree(dot_d);

}
template 
void conjugate_gradient_jacobi2<dspmv::gpu_packing>(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);
template 
void conjugate_gradient_jacobi2<dspmv::gpu_packing_cam>(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);



template <void (*distributed_spmv)(
    Distributed_matrix&,
    Distributed_vector&,
    rocsparse_dnvec_descr&,
    hipStream_t&,
    rocsparse_handle&)>
void conjugate_gradient_jacobi3(
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

    rocsparse_handle default_rocsparseHandle;
    rocsparse_create_handle(&default_rocsparseHandle);

    cudaErrchk(hipStreamCreate(&default_stream));
    cusparseErrchk(hipsparseSetStream(default_cusparseHandle, default_stream));
    cublasErrchk(hipblasSetStream(default_cublasHandle, default_stream));
    rocsparse_set_stream(default_rocsparseHandle, default_stream);

    //set pointer mode for cublas
    cublasErrchk(hipblasSetPointerMode(default_cublasHandle, HIPBLAS_POINTER_MODE_DEVICE));

    double a, b, na;
    double alpha, alpham1, r0;

    double *a_d, *b_d, *na_d;
    cudaErrchk(hipMalloc((void**)&a_d, sizeof(double)));
    cudaErrchk(hipMalloc((void**)&b_d, sizeof(double)));
    cudaErrchk(hipMalloc((void**)&na_d, sizeof(double)));

    int blocks = 1;
    int threads = 64;

    alpha = 1.0;
    alpham1 = -1.0;
    r0 = 0.0;
    double *alpha_d;
    cudaErrchk(hipMalloc((void**)&alpha_d, sizeof(double)));
    cudaErrchk(hipMemcpy(alpha_d, &alpha, sizeof(double), hipMemcpyHostToDevice));
    double *alpham1_d;
    cudaErrchk(hipMalloc((void**)&alpham1_d, sizeof(double)));
    cudaErrchk(hipMemcpy(alpham1_d, &alpham1, sizeof(double), hipMemcpyHostToDevice));
    double *r0_d;
    cudaErrchk(hipMalloc((void**)&r0_d, sizeof(double)));

    double *dot_h;
    cudaErrchk(hipHostMalloc((void**)&dot_h, sizeof(double)));
    double *dot_d;
    cudaErrchk(hipMalloc((void**)&dot_d, sizeof(double)));
    double *dot_global_d;
    cudaErrchk(hipMalloc((void**)&dot_global_d, A_distributed.size*sizeof(double)));

    //copy data to device
    // starting guess for p
    cudaErrchk(hipMemcpy(p_distributed.vec_d[0], x_local_d,
        p_distributed.counts[A_distributed.rank] * sizeof(double), hipMemcpyDeviceToDevice));

    double *Ap_local_d = NULL;
    cudaErrchk(hipMalloc((void **)&Ap_local_d, A_distributed.rows_this_rank * sizeof(double)));
    cudaErrchk(hipMemset(Ap_local_d, 0, A_distributed.rows_this_rank * sizeof(double)));
    rocsparse_dnvec_descr vecAp_local = NULL;
    rocsparse_create_dnvec_descr(&vecAp_local,
                                A_distributed.rows_this_rank,
                                Ap_local_d,
                                rocsparse_datatype_f64_r);

    double *z_local_d = NULL;
    cudaErrchk(hipMalloc((void **)&z_local_d, A_distributed.rows_this_rank * sizeof(double)));

    //begin CG

    // norm of rhs for convergence check
    double *norm2_rhs_h;
    hipHostMalloc((void**)&norm2_rhs_h, sizeof(double));
    double *norm2_rhs_d;
    hipMalloc((void**)&norm2_rhs_d, sizeof(double));
    double *r_norm2_h;
    double *r_norm2_d;
    hipHostMalloc((void**)&r_norm2_h, sizeof(double));
    hipMalloc((void**)&r_norm2_d, sizeof(double));


    cublasErrchk(hipblasDdot(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, r_local_d, 1, norm2_rhs_d));
    hipStreamSynchronize(default_stream);
    MPI_Allreduce(MPI_IN_PLACE, norm2_rhs_d, 1, MPI_DOUBLE, MPI_SUM, comm);

    // A*x0
    distributed_spmv(
        A_distributed,
        p_distributed,
        vecAp_local,
        default_stream,
        default_rocsparseHandle
    );

    // cal residual r0 = b - A*x0
    // r_norm2_h = r0*r0
    cublasErrchk(hipblasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, alpham1_d, Ap_local_d, 1, r_local_d, 1));
    
    // Mz = r
    elementwise_vector_vector(
        r_local_d,
        diag_inv_local_d,
        z_local_d,
        A_distributed.rows_this_rank,
        default_stream
    ); 

    cublasErrchk(hipblasDdot(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, z_local_d, 1, r_norm2_d));
    hipStreamSynchronize(default_stream);
    hipMemcpyAsync(norm2_rhs_h, norm2_rhs_d, sizeof(double), hipMemcpyDeviceToHost, default_stream);
    MPI_Allreduce(MPI_IN_PLACE, r_norm2_d, 1, MPI_DOUBLE, MPI_SUM, comm);
    hipMemcpyAsync(r_norm2_h, r_norm2_d, sizeof(double), hipMemcpyDeviceToHost, default_stream);

    int k = 0;
    hipStreamSynchronize(default_stream);
    while(r_norm2_h[0]/norm2_rhs_h[0] > relative_tolerance * relative_tolerance && k < max_iterations) {
        if(k > 0){
            // pk+1 = rk+1 + b*pk
            // b = r_norm2_h[0] / r0;
            r_divided_dot<<<blocks, threads, 0, default_stream>>>(r_norm2_d, r0_d, b_d);

            cublasErrchk(hipblasDscal(default_cublasHandle, A_distributed.rows_this_rank, b_d, p_distributed.vec_d[0], 1));
            cublasErrchk(hipblasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, alpha_d, z_local_d, 1, p_distributed.vec_d[0], 1)); 
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
            default_rocsparseHandle
        );

        cublasErrchk(hipblasDdot(default_cublasHandle, A_distributed.rows_this_rank, p_distributed.vec_d[0], 1, Ap_local_d, 1, dot_d));
        hipStreamSynchronize(default_stream);
        // MPI_Allreduce(MPI_IN_PLACE, dot_d, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allgather(dot_d, 1, MPI_DOUBLE, dot_global_d, 1, MPI_DOUBLE, comm);

        // a = r_norm2_h[0] / dot_h[0];
        r_divided_dot2<<<blocks, threads, 0, default_stream>>>(r_norm2_d, dot_global_d, a_d, A_distributed.size);


        // xk+1 = xk + ak * pk
        cublasErrchk(hipblasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, a_d, p_distributed.vec_d[0], 1, x_local_d, 1));

        // rk+1 = rk - ak * A * pk
        // na = -a;
        negate_a<<<blocks, threads, 0, default_stream>>>(a_d, na_d);
        equal_r0<<<blocks, threads, 0, default_stream>>>(r_norm2_d, r0_d);

        cublasErrchk(hipblasDaxpy(default_cublasHandle, A_distributed.rows_this_rank, na_d, Ap_local_d, 1, r_local_d, 1));
        // r0 = r_norm2_h[0];
        

        // Mz = r
        elementwise_vector_vector(
            r_local_d,
            diag_inv_local_d,
            z_local_d,
            A_distributed.rows_this_rank,
            default_stream
        ); 
        

        // r_norm2_h = r0*r0
        cublasErrchk(hipblasDdot(default_cublasHandle, A_distributed.rows_this_rank, r_local_d, 1, z_local_d, 1, r_norm2_d));
        hipStreamSynchronize(default_stream);
        MPI_Allreduce(MPI_IN_PLACE, r_norm2_d, 1, MPI_DOUBLE, MPI_SUM, comm);
        hipMemcpy(r_norm2_h, r_norm2_d, sizeof(double), hipMemcpyDeviceToHost);
        k++;

    }

    //end CG
    cudaErrchk(hipDeviceSynchronize());
    if(A_distributed.rank == 0){
        std::cout << "iteration K = " << k << ", relative residual = " << sqrt(r_norm2_h[0]/norm2_rhs_h[0]) << std::endl;
    }

    rocsparse_destroy_handle(default_rocsparseHandle);
    cusparseErrchk(hipsparseDestroy(default_cusparseHandle));
    cublasErrchk(hipblasDestroy(default_cublasHandle));
    cudaErrchk(hipStreamDestroy(default_stream));
    rocsparse_destroy_dnvec_descr(vecAp_local);
    cudaErrchk(hipFree(Ap_local_d));
    cudaErrchk(hipFree(z_local_d));

    hipFree(a_d);
    hipFree(b_d);
    hipFree(na_d);

    hipFree(alpha_d);
    hipFree(alpham1_d);
    hipFree(r0_d);

    hipHostFree(norm2_rhs_h);
    hipFree(norm2_rhs_d);

    hipHostFree(r_norm2_h);
    hipFree(r_norm2_d);

    hipHostFree(dot_h);
    hipFree(dot_d);
    hipFree(dot_global_d);

}
template 
void conjugate_gradient_jacobi3<dspmv::gpu_packing>(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);
template 
void conjugate_gradient_jacobi3<dspmv::gpu_packing_cam>(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    double *r_local_d,
    double *x_local_d,
    double *diag_inv_local_d,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);

} // namespace iterative_solver