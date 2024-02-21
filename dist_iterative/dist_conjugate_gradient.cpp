#include "dist_conjugate_gradient.h"
#include "dist_spmv.h"
namespace iterative_solver{

void split_matrix_uniform(
    int matrix_size,
    int size,
    int *counts,
    int *displacements)
{
    int rows_per_rank = matrix_size / size;    
    for (int i = 0; i < size; ++i) {
        if(i < matrix_size % size){
            counts[i] = rows_per_rank+1;
        }
        else{
            counts[i] = rows_per_rank;
        }
    }
    displacements[0] = 0;
    for (int i = 1; i < size; ++i) {
        displacements[i] = displacements[i-1] + counts[i-1];
    }

}

//template <void (*distributed_spmv)()>
template <void (*distributed_spmv)(Distributed_matrix&, Distributed_vector&, cusparseDnVecDescr_t&, cudaStream_t&, cusparseHandle_t&)>
void conjugate_gradient(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
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


    // prepare for allgatherv
    int counts[size];
    int displacements[size];
    int rows_per_rank = matrix_size / size;    
    split_matrix_uniform(matrix_size, size, counts, displacements);

    int row_start_index = displacements[rank];
    rows_per_rank = counts[rank];



    int *row_indptr_local_h = new int[rows_per_rank+1];
    double *r_local_h = new double[rows_per_rank];
    for (int i = 0; i < rows_per_rank+1; ++i) {
        row_indptr_local_h[i] = row_indptr_h[i+row_start_index] - row_indptr_h[row_start_index];
    }
    for (int i = 0; i < rows_per_rank; ++i) {
        r_local_h[i] = r_h[i+row_start_index];
    }

    int nnz_local = row_indptr_local_h[rows_per_rank];

    int *col_indices_local_h = new int[nnz_local];
    double *data_local_h = new double[nnz_local];

    for (int i = 0; i < nnz_local; ++i) {
        col_indices_local_h[i] = col_indices_h[i+row_indptr_h[row_start_index]];
        data_local_h[i] = data_h[i+row_indptr_h[row_start_index]];
    }

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

    double *r_norm2_d = NULL;
    double *dot_d = NULL;
    cudaErrchk(cudaMalloc((void**)&r_norm2_d, sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&dot_d, sizeof(double)));

    //allocate memory on device
    int *row_indptr_local_d = NULL;
    int *col_indices_local_d = NULL;
    double *data_local_d = NULL;
    double *r_local_d = NULL;
    double *x_local_d = NULL;

    double *starting_guess_local_h = starting_guess_h + row_start_index;


    cudaErrchk(cudaMalloc((void**)&row_indptr_local_d, (rows_per_rank+1)*sizeof(int)));
    cudaErrchk(cudaMalloc((void**)&col_indices_local_d, nnz_local*sizeof(int)));
    cudaErrchk(cudaMalloc((void**)&data_local_d, nnz_local*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&r_local_d, rows_per_rank*sizeof(double)));
    cudaErrchk(cudaMalloc((void**)&x_local_d, rows_per_rank*sizeof(double)));


    cudaErrchk(cudaMemcpy(row_indptr_local_d, row_indptr_local_h, (rows_per_rank+1)*sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(col_indices_local_d, col_indices_local_h, nnz_local*sizeof(int), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(data_local_d, data_local_h, nnz_local*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrchk(cudaMemcpy(r_local_d, r_local_h, rows_per_rank*sizeof(double), cudaMemcpyHostToDevice));
    //copy data to device
    // starting guess for x
    cudaErrchk(cudaMemcpy(x_local_d, starting_guess_local_h, rows_per_rank * sizeof(double), cudaMemcpyHostToDevice));

    double *Ap_local_d = NULL;
    cudaErrchk(cudaMalloc((void **)&Ap_local_d, rows_per_rank * sizeof(double)));
    cudaErrchk(cudaMemset(Ap_local_d, 0, rows_per_rank * sizeof(double)));

    cusparseDnVecDescr_t vecAp_local = NULL;
    cusparseErrchk(cusparseCreateDnVec(&vecAp_local, rows_per_rank, Ap_local_d, CUDA_R_64F));


    // create distributed matrix
    std::printf("Creating distributed matrix\n");
    Distributed_matrix A_distributed(
        matrix_size,
        nnz_local,
        counts,
        displacements,
        col_indices_local_h,
        row_indptr_local_h,
        data_local_h,
        MPI_COMM_WORLD,
        default_cusparseHandle
    );
    std::printf("Creating distributed vector\n");
    Distributed_vector p_distributed(
        matrix_size,
        counts,
        displacements,
        A_distributed.number_of_neighbours,
        A_distributed.neighbours,
        MPI_COMM_WORLD,
        default_cusparseHandle
    );
    //begin CG
    std::printf("CG starts\n");
    cudaErrchk(cudaStreamSynchronize(default_stream));
    cudaErrchk(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    time_taken[0] = -omp_get_wtime();
    // norm of rhs for convergence check
    double norm2_rhs = 0;
    cublasErrchk(cublasDdot(cublasHandle, rows_per_rank, r_local_d, 1, r_local_d, 1, &norm2_rhs));
    //allreduce
    MPI_Allreduce(MPI_IN_PLACE, &norm2_rhs, 1, MPI_DOUBLE, MPI_SUM, comm);

    cudaErrchk(cudaMemcpy(p_distributed.vec_d[0], starting_guess_local_h,
        p_distributed.counts[rank] * sizeof(double), cudaMemcpyHostToDevice));
    std::memcpy(p_distributed.vec_h[0], starting_guess_local_h,
        p_distributed.counts[rank] * sizeof(double));

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
    cublasErrchk(cublasDaxpy(cublasHandle, rows_per_rank, &alpham1, Ap_local_d, 1, r_local_d, 1));
    cublasErrchk(cublasDdot(cublasHandle, rows_per_rank, r_local_d, 1, r_local_d, 1, r_norm2_h));
    //allreduce
    MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);

    int k = 1;
    while (r_norm2_h[0]/norm2_rhs > relative_tolerance * relative_tolerance && k <= max_iterations) {
        if(k > 1){
            // pk+1 = rk+1 + b*pk
            b = r_norm2_h[0] / r0;
            cublasErrchk(cublasDscal(cublasHandle, rows_per_rank, &b, p_distributed.vec_d[0], 1));
            cublasErrchk(cublasDaxpy(cublasHandle, rows_per_rank, &alpha, r_local_d, 1, p_distributed.vec_d[0], 1)); 
        }
        else {
            // p0 = r0
            cublasErrchk(cublasDcopy(cublasHandle, rows_per_rank, r_local_d, 1, p_distributed.vec_d[0], 1));
        }


        // ak = rk^T * rk / pk^T * A * pk
        // has to be done for k=0 if x0 != 0
        //memcpy
        //allgather
        //memcpy
        distributed_spmv(
            A_distributed,
            p_distributed,
            vecAp_local,
            default_stream,
            default_cusparseHandle
        );

        cublasErrchk(cublasDdot(cublasHandle, rows_per_rank, p_distributed.vec_d[0], 1, Ap_local_d, 1, dot_h));
        //allreduce        
        MPI_Allreduce(MPI_IN_PLACE, dot_h, 1, MPI_DOUBLE, MPI_SUM, comm);

        a = r_norm2_h[0] / dot_h[0];

        // xk+1 = xk + ak * pk
        cublasErrchk(cublasDaxpy(cublasHandle, rows_per_rank, &a, p_distributed.vec_d[0], 1, x_local_d, 1));

        // rk+1 = rk - ak * A * pk
        na = -a;
        cublasErrchk(cublasDaxpy(cublasHandle, rows_per_rank, &na, Ap_local_d, 1, r_local_d, 1));
        r0 = r_norm2_h[0];

        // r_norm2_h = r0*r0
        cublasErrchk(cublasDdot(cublasHandle, rows_per_rank, r_local_d, 1, r_local_d, 1, r_norm2_h));
        //allreduce
        MPI_Allreduce(MPI_IN_PLACE, r_norm2_h, 1, MPI_DOUBLE, MPI_SUM, comm);
        k++;
    }

    //end CG
    cudaErrchk(cudaDeviceSynchronize());
    cudaErrchk(cudaStreamSynchronize(default_stream));
    time_taken[0] += omp_get_wtime();

    steps_taken[0] = k;
    if(rank == 0){
        std::printf("iteration = %3d, relative residual = %e\n", k, sqrt(r_norm2_h[0]/norm2_rhs));
    }


    std::cout << "rank " << rank << " time_taken[0] " << time_taken[0] << std::endl;

    //copy solution to host
    cudaErrchk(cudaMemcpy(r_local_h, x_local_d, rows_per_rank * sizeof(double), cudaMemcpyDeviceToHost));


    double difference = 0;
    double sum_ref = 0;
    for (int i = 0; i < rows_per_rank; ++i) {
        difference += std::sqrt( (r_local_h[i] - reference_solution[i+row_start_index]) * (r_local_h[i] - reference_solution[i+row_start_index]) );
        sum_ref += std::sqrt( (reference_solution[i+row_start_index]) * (reference_solution[i+row_start_index]) );
    }
    MPI_Allreduce(MPI_IN_PLACE, &difference, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &sum_ref, 1, MPI_DOUBLE, MPI_SUM, comm);
    if(rank == 0){
        std::cout << "difference/sum_ref " << difference/sum_ref << std::endl;
    }

    cusparseErrchk(cusparseDestroy(default_cusparseHandle));
    cublasErrchk(cublasDestroy(cublasHandle));
    cudaErrchk(cudaStreamDestroy(default_stream));
    cusparseErrchk(cusparseDestroyDnVec(vecAp_local));
    cudaErrchk(cudaFree(row_indptr_local_d));
    cudaErrchk(cudaFree(col_indices_local_d));
    cudaErrchk(cudaFree(data_local_d));
    cudaErrchk(cudaFree(r_local_d));
    cudaErrchk(cudaFree(x_local_d));
    cudaErrchk(cudaFree(Ap_local_d));

    cudaErrchk(cudaFree(r_norm2_d));
    cudaErrchk(cudaFree(dot_d));

    delete[] row_indptr_local_h;
    delete[] col_indices_local_h;
    delete[] data_local_h;
    delete[] r_local_h;

    cudaErrchk(cudaFreeHost(r_norm2_h));
    cudaErrchk(cudaFreeHost(dot_h));

    MPI_Barrier(comm);
}

template 
void conjugate_gradient<dspmv::gpu_packing>(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    int *steps_taken,
    double *time_taken);
template 
void conjugate_gradient<dspmv::gpu_packing_cam>(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *rhs_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    int *steps_taken,
    double *time_taken);


} // namespace iterative_solver

