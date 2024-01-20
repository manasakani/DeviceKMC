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



    int matsize = 260;
    std::string data_path = "/scratch/snx3000/amaeder/"+std::to_string(matsize)+"k_piz_daint_data";
    //std::string save_path ="/scratch/snx3000/amaeder/measurements/self_preconditioned_scaling_measurement/";
    std::string save_path ="/scratch/snx3000/amaeder/measurements/own_260/";
    data_path = "/usr/scratch/mont-fort17/almaeder/kmc_"+std::to_string(matsize)+"k/system_K";

    int matrix_size;
    int nnz;     
    if(matsize == 7){
        save_path = "/scratch/snx3000/amaeder/measurements/dump/";
        matrix_size = 7302;
        nnz = 186684;        
    }
    else if(matsize == 80){
        matrix_size = 70630;
        nnz = 1719652;        
    }
    else{
        data_path = "/scratch/snx3000/amaeder/kmc_random";
        matrix_size = 262144;
        nnz = 16481266;
    } 
    int step = 0;
    std::string data_filename = data_path + "/A_data"+std::to_string(step)+".bin";
    std::string row_ptr_filename = data_path + "/A_row_ptr"+std::to_string(step)+".bin";
    std::string col_indices_filename = data_path + "/A_col_indices"+std::to_string(step)+".bin";
    std::string rhs_filename = data_path + "/A_rhs"+std::to_string(step)+".bin";
    std::string solution_filename = data_path + "/solution"+std::to_string(step)+".bin";


    int counts[size];
    int displacements[size];
    int rows_per_rank = matrix_size / size;    
    split_matrix(matrix_size, size, counts, displacements);

    int row_start_index = displacements[rank];
    rows_per_rank = counts[rank];

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
    int *col_indices_local = col_indices + nnz_start_index;
    double *data_local = data + nnz_start_index;

    double relative_tolerance = 1e-12;
    int max_iterations = 5000;

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


    int number_of_measurements = 22;

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
    int number_of_kmc_steps = 1;
    std::ofstream outputFile_times;
    std::string path_times = save_path + "solve_ginkgo" +
                std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps) 
                +"_" + std::to_string(size) +"_" + std::to_string(rank) +"_.txt";

    outputFile_times.open(path_times);
    if(outputFile_times.is_open()){
        for(int i = 0; i < number_of_measurements; i++){
            outputFile_times << time_taken[i] << " ";
        }
        outputFile_times << '\n';
    }
    else{
        std::printf("Error opening file\n");
    }
    outputFile_times.close();

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