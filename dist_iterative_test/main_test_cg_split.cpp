#include <iostream>
#include <string>
#include "utils.h"
#include <mpi.h>
#include <cuda_runtime.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "utils_gpu.h"
#include <cublas_v2.h>
#include "../dist_iterative/dist_conjugate_gradient.h"
#include "../dist_iterative/dist_spmv.h"
#include <pthread.h>

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
void test_preconditioned_split(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    int *subblock_indices_h,
    double *A_subblock_h,
    int subblock_size,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm)
{
    MPI_Barrier(comm);


    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    // prepare for allgatherv
    int counts[size];
    int displacements[size];
    int rows_per_rank = matrix_size / size;    
    split_matrix(matrix_size, size, counts, displacements);

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
        comm
    );
    std::printf("Creating distributed vector\n");
    Distributed_vector p_distributed(
        matrix_size,
        counts,
        displacements,
        A_distributed.number_of_neighbours,
        A_distributed.neighbours,
        comm
    );
    double *r_local_d;
    double *x_local_d;
    cudaMalloc(&r_local_d, rows_per_rank * sizeof(double));
    cudaMalloc(&x_local_d, rows_per_rank * sizeof(double));
    cudaMemcpy(r_local_d, r_local_h, rows_per_rank * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_local_d, starting_guess_h + row_start_index,
        rows_per_rank * sizeof(double), cudaMemcpyHostToDevice);

    int *dense_subblock_indices_d;
    double *A_subblock_d;
    cudaMalloc(&dense_subblock_indices_d, subblock_size * sizeof(int));
    cudaMalloc(&A_subblock_d, subblock_size * subblock_size * sizeof(double));
    cudaMemcpy(dense_subblock_indices_d, subblock_indices_h, subblock_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(A_subblock_d, A_subblock_h, subblock_size * subblock_size * sizeof(double), cudaMemcpyHostToDevice);
    MPI_Barrier(comm);
    cudaDeviceSynchronize();

    int *count_subblock = new int[size];
    int *displ_subblock = new int[size];
    double *diag_local_h = new double[rows_per_rank];


    for(int i = 0; i < size; i++){
        count_subblock[i] = 0;
    }
    for(int i = 0; i < subblock_size; i++){
        for(int j = 0; j < size; j++){
            if( subblock_indices_h[i] >= displacements[j] && subblock_indices_h[i] < displacements[j] + counts[j]){
                count_subblock[j]++;
            }
        }
    }
    displ_subblock[0] = 0;
    for(int i = 1; i < size; i++){
        displ_subblock[i] = displ_subblock[i-1] + count_subblock[i-1];
    }

    
    int *subblock_indices_local_h = new int[count_subblock[rank]];
    for(int i = 0; i < count_subblock[rank]; i++){
        subblock_indices_local_h[i] = subblock_indices_h[displ_subblock[rank] + i] - displacements[rank];
    }

    int *subblock_indices_local_d;
    cudaMalloc(&subblock_indices_local_d, count_subblock[rank] * sizeof(int));
    cudaMemcpy(subblock_indices_local_d, subblock_indices_local_h, count_subblock[rank] * sizeof(int), cudaMemcpyHostToDevice);


    double *diag_inv_local_d;
    cudaMalloc(&diag_inv_local_d, rows_per_rank * sizeof(double));
    double *A_subblock_local_h = new double[count_subblock[rank] * subblock_size];
    double *A_subblock_local_d;
    cudaMalloc(&A_subblock_local_d, count_subblock[rank] * subblock_size * sizeof(double));

    
    for(int i = 0; i < count_subblock[rank]; i++){
        for(int j = 0; j < subblock_size; j++){
            A_subblock_local_h[i + j * count_subblock[rank]] = A_subblock_h[i + displ_subblock[rank] + j * subblock_size];
        }
    }
    cudaMemcpy(A_subblock_local_d, A_subblock_local_h, count_subblock[rank] * subblock_size * sizeof(double), cudaMemcpyHostToDevice);



    for (int i = 0; i < rows_per_rank; ++i) {
        diag_local_h[i] = 0.0;
    }
    for (int i = 0; i < rows_per_rank; ++i) {
        for (int j = row_indptr_local_h[i]; j < row_indptr_local_h[i+1]; ++j) {
            if (col_indices_local_h[j] == i + row_start_index) {
                diag_local_h[i] = data_local_h[j];
            }
        }
    }
    // only diagonal block matters for the preconditioner
    for(int i = 0; i < count_subblock[rank]; i++){
        for(int j = 0; j < count_subblock[rank]; j++){
            if(subblock_indices_local_h[i] == subblock_indices_local_h[j]){
                diag_local_h[subblock_indices_local_h[i]] += A_subblock_local_h[i + j * count_subblock[rank]];
            }
        }
    }

    for (int i = 0; i < rows_per_rank; ++i) {
        diag_local_h[i] = 1.0 / diag_local_h[i];
    }
    cudaMemcpy(diag_inv_local_d, diag_local_h, rows_per_rank * sizeof(double), cudaMemcpyHostToDevice);    





    Distributed_subblock A_subblock;
    A_subblock.subblock_indices_local_d = subblock_indices_local_d;
    A_subblock.A_subblock_local_d = A_subblock_local_d;
    A_subblock.subblock_size = subblock_size;
    A_subblock.count_subblock_h = count_subblock;
    A_subblock.displ_subblock_h = displ_subblock;
    A_subblock.send_subblock_requests = new MPI_Request[size-1];
    A_subblock.recv_subblock_requests = new MPI_Request[size-1];
    A_subblock.streams_recv_subblock = new cudaStream_t[size-1];
    for(int i = 0; i < size-1; i++){
        cudaStreamCreate(&A_subblock.streams_recv_subblock[i]);
    }
    A_subblock.events_recv_subblock = new cudaEvent_t[size];
    for(int i = 0; i < size; i++){
        cudaEventCreateWithFlags(&A_subblock.events_recv_subblock[i], cudaEventDisableTiming);
    }

    iterative_solver::conjugate_gradient_split<distributed_spmv_split>(
        A_subblock,
        A_distributed,
        p_distributed,
        r_local_d,
        x_local_d,
        relative_tolerance,
        max_iterations,
        comm);


    for(int i = 0; i < size-1; i++){
        cudaStreamDestroy(A_subblock.streams_recv_subblock[i]);
    }
    delete[] A_subblock.streams_recv_subblock;


    delete[] A_subblock.send_subblock_requests;
    delete[] A_subblock.recv_subblock_requests;

    // //copy solution to host
    double *solution = new double[rows_per_rank];
    cudaErrchk(cudaMemcpy(solution,
        x_local_d, rows_per_rank * sizeof(double), cudaMemcpyDeviceToHost));
    MPI_Allgatherv(solution, rows_per_rank, MPI_DOUBLE, reference_solution, counts, displacements, MPI_DOUBLE, comm);

    for(int i = 0; i < size; i++){
        cudaEventDestroy(A_subblock.events_recv_subblock[i]);
    }
    delete[] A_subblock.events_recv_subblock;

    delete[] solution;
    delete[] count_subblock;
    delete[] displ_subblock;
    delete[] diag_local_h;
    delete[] subblock_indices_local_h;
    delete[] A_subblock_local_h;
    cudaFree(subblock_indices_local_d);
    cudaFree(A_subblock_local_d);

    delete[] row_indptr_local_h;
    delete[] r_local_h;
    delete[] col_indices_local_h;
    delete[] data_local_h;
    cudaFree(r_local_d);
    cudaFree(x_local_d);

    MPI_Barrier(comm);
}
template 
void test_preconditioned_split<dspmv_split::spmm_split1>(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    int *subblock_indices_h,
    double *A_subblock_h,
    int subblock_size,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);
template 
void test_preconditioned_split<dspmv_split::spmm_split2>(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    int *subblock_indices_h,
    double *A_subblock_h,
    int subblock_size,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);
template 
void test_preconditioned_split<dspmv_split::spmm_split3>(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    int *subblock_indices_h,
    double *A_subblock_h,
    int subblock_size,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);
template 
void test_preconditioned_split<dspmv_split::spmm_split4>(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    int *subblock_indices_h,
    double *A_subblock_h,
    int subblock_size,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);
template 
void test_preconditioned_split<dspmv_split::spmm_split5>(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    int *subblock_indices_h,
    double *A_subblock_h,
    int subblock_size,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm);


int main(int argc, char **argv) {

    // MPI_Init(&argc, &argv);

    // Init thread multiple
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        std::cout << "MPI_THREAD_MULTIPLE not supported by MPI, aborting" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cudaError_t set_device_error = cudaSetDevice(0);
    std::cout << "rank " << rank << " set_device_error " << set_device_error << std::endl;

    std::string data_path = "/usr/scratch/mont-fort23/almaeder/kmc_split/";

    int matrix_size = 2048;
    int nnz_sparse = 118122;
    int nnz_tot = 133930;

    int subblock_size = 128;    


    int max_iterations = 10000;
    double relative_tolerance = 1e-8;

    int rows_per_rank = matrix_size / size;
    int remainder = matrix_size % size;
    int row_start_index = rank * rows_per_rank;
    int col_start_index = row_start_index + rows_per_rank;
    if (rank == size-1) {
        col_start_index += remainder;
        rows_per_rank += remainder;
    }

    std::cout << "rank " << rank << " row_start_index " << row_start_index << " row_end_index " << col_start_index << std::endl;
    std::cout << "rank " << rank << " rows_per_rank " << rows_per_rank << std::endl;

    double *data_sparse = new double[nnz_sparse];
    int *row_ptr_sparse = new int[matrix_size+1];
    int *col_indices_sparse = new int[nnz_sparse];

    double *data_tot = new double[nnz_tot];
    int *row_ptr_tot = new int[matrix_size+1];
    int *col_indices_tot = new int[nnz_tot];

    double *reference_solution = new double[matrix_size];

    double *rhs = new double[matrix_size];
    for(int i = 0; i < matrix_size; i++){
        rhs[i] = 1.0;
    }

    double *starting_guess = new double[matrix_size];
    for (int i = 0; i < matrix_size; ++i) {
        starting_guess[i] = 0.0;
    }

    std::string data_sparse_filename = data_path + "sparse_data.bin";
    std::string row_ptr_sparse_filename = data_path + "sparse_indptr.bin";
    std::string col_indices_sparse_filename = data_path + "sparse_indices.bin";

    std::string data_tot_filename = data_path + "data.bin";
    std::string row_ptr_tot_filename = data_path + "indptr.bin";
    std::string col_indices_tot_filename = data_path + "indices.bin";

    std::string solution_filename = data_path + "solution.bin";

    load_binary_array<double>(data_sparse_filename, data_sparse, nnz_sparse);
    load_binary_array<int>(row_ptr_sparse_filename, row_ptr_sparse, matrix_size+1);
    load_binary_array<int>(col_indices_sparse_filename, col_indices_sparse, nnz_sparse);

    load_binary_array<double>(data_tot_filename, data_tot, nnz_tot);
    load_binary_array<int>(row_ptr_tot_filename, row_ptr_tot, matrix_size+1);
    load_binary_array<int>(col_indices_tot_filename, col_indices_tot, nnz_tot);

    int *dense_subblock_indices = new int[subblock_size];
    double *dense_subblock_data = new double[subblock_size * subblock_size];

    std::string dense_subblock_indices_filename = data_path + "dense_subblock_indices.bin";
    std::string dense_subblock_data_filename = data_path + "dense.bin";

    load_binary_array<int>(dense_subblock_indices_filename, dense_subblock_indices, subblock_size);
    load_binary_array<double>(dense_subblock_data_filename, dense_subblock_data, subblock_size * subblock_size);
    load_binary_array<double>(solution_filename, reference_solution, matrix_size);

    std::cout << "rank " << rank << " data loaded" << std::endl;

    double *dense_tot = new double[matrix_size * matrix_size];
    double *dense_split = new double[matrix_size * matrix_size];
    for (int i = 0; i < matrix_size; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            dense_tot[i * matrix_size + j] = 0.0;
            dense_split[i * matrix_size + j] = 0.0;
        }
    }
    for (int i = 0; i < matrix_size; ++i) {
        for (int j = row_ptr_tot[i]; j < row_ptr_tot[i+1]; ++j) {
            dense_tot[i * matrix_size + col_indices_tot[j]] = data_tot[j];
        }
    }
    for (int i = 0; i < matrix_size; ++i) {
        for (int j = row_ptr_sparse[i]; j < row_ptr_sparse[i+1]; ++j) {
            dense_split[i * matrix_size + col_indices_sparse[j]] = data_sparse[j];
        }
    }
    for (int i = 0; i < subblock_size; ++i) {
        for (int j = 0; j < subblock_size; ++j) {
            dense_split[dense_subblock_indices[i] * matrix_size + dense_subblock_indices[j]] +=
                dense_subblock_data[i * subblock_size + j];
        }
    }

    double sum_matrix = 0.0;
    double diff_matrix = 0.0;
    for (int i = 0; i < matrix_size; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            sum_matrix += std::abs(dense_tot[i * matrix_size + j]) * std::abs(dense_tot[i * matrix_size + j]);
            diff_matrix += std::abs(dense_tot[i * matrix_size + j] - dense_split[i * matrix_size + j]) *
                std::abs(dense_tot[i * matrix_size + j] - dense_split[i * matrix_size + j]);
        }
    }
    std::cout << "rank " << rank << " relative between matrices " << std::sqrt(diff_matrix / sum_matrix) << std::endl;

    int number_of_measurements = 5;

    for(int measurement = 0; measurement < number_of_measurements; measurement++){
    double *test_solution_split = new double[matrix_size];
    test_preconditioned_split<dspmv_split::spmm_split5>(
            data_sparse,
            col_indices_sparse,
            row_ptr_sparse,
            dense_subblock_indices,
            dense_subblock_data,
            subblock_size,
            rhs,
            test_solution_split,
            starting_guess,
            matrix_size,
            relative_tolerance,
            max_iterations,
            MPI_COMM_WORLD
    );


    double sum = 0.0;
    double diff_split = 0.0;
    for (int i = 0; i < matrix_size; ++i) {
        sum += std::abs(reference_solution[i]) * std::abs(reference_solution[i]);
        diff_split += std::abs(reference_solution[i] - test_solution_split[i]) * std::abs(reference_solution[i] - test_solution_split[i]);
    }
    if(rank == 0){
        std::cout << " relative error split " << std::sqrt(diff_split / sum) << std::endl; 
    }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
