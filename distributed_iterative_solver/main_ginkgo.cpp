#include <iostream>
#include <string>
#include "utils.h"
#include <mpi.h>


#include "utils_gpu.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "ginkgo_implementations_to_compare.h"

int main(int argc, char* argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    std::cout << "provided " << provided << std::endl;

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);



    // std::string data_path = "/usr/scratch/mont-fort17/almaeder/kmc_80k/system_K";
    int matsize = 80;
    std::string data_path = "/scratch/snx3000/amaeder/"+std::to_string(matsize)+"k_piz_daint_data";
    std::string save_path ="/scratch/snx3000/amaeder/measurements/single_node_libraries/";
    int step = 0;
    std::string data_filename = data_path + "/A_data"+std::to_string(step)+".bin";
    std::string row_ptr_filename = data_path + "/A_row_ptr"+std::to_string(step)+".bin";
    std::string col_indices_filename = data_path + "/A_col_indices"+std::to_string(step)+".bin";
    std::string rhs_filename = data_path + "/A_rhs"+std::to_string(step)+".bin";
    std::string solution_filename = data_path + "/solution"+std::to_string(step)+".bin";

    int matrix_size = 70630;
    int nnz = 1719652;    


    int rows_per_rank = matrix_size / size;
    int remainder = matrix_size % size;
    int row_start_index = rank * rows_per_rank;
    int row_end_index = row_start_index + rows_per_rank;
    if (rank == size-1) {
        row_end_index += remainder;
        rows_per_rank += remainder;
    }
    std::cout << "rank " << rank << " row_start_index " << row_start_index << std::endl;
    std::cout << "rank " << rank << " row_end_index " << row_end_index << std::endl;

    // load custom matrix
    double *data = new double[nnz];
    int *row_ptr = new int[matrix_size+1];
    int *col_indices = new int[nnz];
    double *rhs = new double[matrix_size];
    double *reference_solution = new double[matrix_size];
    int *row_ptr_local = new int[rows_per_rank+1];

    load_binary_array<double>(data_filename, data, nnz);
    load_binary_array<int>(row_ptr_filename, row_ptr, matrix_size+1);
    load_binary_array<int>(col_indices_filename, col_indices, nnz);
    load_binary_array<double>(rhs_filename, rhs, matrix_size);
    load_binary_array<double>(solution_filename, reference_solution, matrix_size);


    for (int i = 0; i < rows_per_rank+1; ++i) {
        row_ptr_local[i] = row_ptr[i+row_start_index] - row_ptr[row_start_index];
    }
    int nnz_local = row_ptr_local[rows_per_rank];
    int nnz_start_index = row_ptr[row_start_index];
    int nnz_end_index = nnz_start_index + nnz_local;
    // std::cout << "rank " << rank << " nnz_local " << nnz_local << std::endl;
    // std::cout << "rank " << rank << " nnz_start_index " << nnz_start_index << std::endl;
    // std::cout << "rank " << rank << " nnz_end_index " << nnz_end_index << std::endl;
    int *col_indices_local = col_indices + nnz_start_index;
    double *data_local = data + nnz_start_index;
    double relative_tolerance = 1e-16;
    int max_iterations = 1000;

    double *data_d;
    int *row_ptr_d;
    int *col_indices_d;
    double *rhs_d;
    double *reference_solution_d;
    double *diagonal_d;

    cudaMalloc(&data_d, nnz * sizeof(double));
    cudaMalloc(&row_ptr_d, (matrix_size+1) * sizeof(int));
    cudaMalloc(&col_indices_d, nnz * sizeof(int));
    cudaMalloc(&rhs_d, matrix_size * sizeof(double));
    cudaMalloc(&reference_solution_d, matrix_size * sizeof(double));
    cudaMalloc(&diagonal_d, matrix_size * sizeof(double));

    cudaMemcpy(data_d, data, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(row_ptr_d, row_ptr, (matrix_size+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(col_indices_d, col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(rhs_d, rhs, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(reference_solution_d, reference_solution, matrix_size * sizeof(double), cudaMemcpyHostToDevice);

    extract_diagonal_gpu(
        data_d,
        col_indices_d,
        row_ptr_d,
        diagonal_d,
        matrix_size
    );
    symmetric_precondition_matrix_gpu(
        data_d,
        col_indices_d,
        row_ptr_d,
        diagonal_d,
        matrix_size
    );
    precondition_vector_gpu(
        rhs_d,
        diagonal_d,
        matrix_size
    );
    unpreecondition_vector_gpu(
        reference_solution_d,
        diagonal_d,
        matrix_size
    );

    cudaMemcpy(rhs, rhs_d, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(reference_solution, reference_solution_d, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(data, data_d, nnz * sizeof(double), cudaMemcpyDeviceToHost);


    int number_of_measurements = 30;

    int iterations;
    double time_taken[number_of_measurements];
    bool correct_solution;


    for(int i = 0; i < number_of_measurements; i++) {
        gko_test::gpu_solve(
            data_local,
            row_ptr_local,
            col_indices_local,
            rhs,
            reference_solution,
            matrix_size,
            max_iterations,
            relative_tolerance,
            &iterations,
            &time_taken[i],
            &correct_solution
        );
    }

    // std::string method_name = "ginkgo";
    // std::ofstream outputFile_times;
    // std::string path_times = save_path + method_name + "_times"+ std::to_string(matsize) + ".txt";
    // outputFile_times.open(path_times);
    // if(outputFile_times.is_open()){
    //     for(int i = 0; i < number_of_measurements; i++){
    //         outputFile_times << time_taken[i] << " ";
    //     }
    //     outputFile_times << '\n';
    // }
    // else{
    //     std::printf("Error opening file\n");
    // }
    // outputFile_times.close();

    delete[] data;
    delete[] row_ptr;
    delete[] col_indices;
    delete[] rhs;
    delete[] reference_solution;
    delete[] row_ptr_local;

    cudaFree(data_d);
    cudaFree(row_ptr_d);
    cudaFree(col_indices_d);
    cudaFree(rhs_d);
    cudaFree(reference_solution_d);
    cudaFree(diagonal_d);


}