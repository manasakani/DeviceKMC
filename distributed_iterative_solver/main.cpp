#include <iostream>
#include <string>
#include "utils.h"
#include <mpi.h>
#include <cuda_runtime.h>
#include "petsc_implementations_to_compare.h"

#include <petscksp.h>
#include <petscvec.h>
#include <petscdevice.h> 
int main(int argc, char **argv) {
    // older version of petsc on daint
    // replace by PetscCall()
    CHKERRQ(PetscInitialize(&argc, &argv, NULL, NULL));
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    std::cout << "Hello World from rank " << rank << std::endl;
    int matrix_size = 7302;
    int nnz = 186684;

    int rows_per_rank = matrix_size / size;
    int remainder = matrix_size % size;
    int row_start_index = rank * rows_per_rank;
    int col_start_index = row_start_index + rows_per_rank;
    if (rank == size-1) {
        col_start_index += remainder;
        rows_per_rank += remainder;
    }

    std::cout << "rank " << rank << " row_start_index " << row_start_index << " col_start_index " << col_start_index << std::endl;
    std::cout << "rank " << rank << " rows_per_rank " << rows_per_rank << std::endl;

    double *data = new double[nnz];
    int *row_ptr = new int[matrix_size+1];
    int *col_indices = new int[nnz];
    double *rhs = new double[matrix_size];
    double *reference_solution = new double[matrix_size];

    std::string data_path = "/scratch/snx3000/amaeder/7k_piz_daint_data";
    std::string save_path ="/scratch/snx3000/amaeder/measurements/7k/";

    int number_of_measurements = 20;
    int number_of_kmc_steps = 500;

    int max_iterations = 1000;
    double relative_tolerance = 1e-15;
    double absolute_tolerance = 1e-20;
    double divergence_tolerance = 1e+50;

    int number_of_methods = 6;
    std::string method_names[number_of_methods] = {
        "cg_jacobi",
        "bicg_jacobi",
        "gmres_jacobi",
        "cg_sor",
        "bicg_sor",
        "gmres_sor"
    };
    KSPType solver_types[number_of_methods] = {
        KSPCG,
        KSPBCGS,
        KSPGMRES,
        KSPCG,
        KSPBCGS,
        KSPGMRES
    };
    PCType preconditioners[number_of_methods] = {
        PCJACOBI,
        PCJACOBI,
        PCJACOBI,
        PCSOR,
        PCSOR,
        PCSOR
    };
    int iterations[number_of_methods][number_of_kmc_steps];




    bool correct_solution_global = true;
    for(int step = 0; step < number_of_kmc_steps; step++){
        std::cout << "rank " << rank << " step " << step << std::endl;
        bool correct_solution_iteration;

        double times[number_of_methods][number_of_measurements];

        std::string data_filename = data_path + "/A_data"+std::to_string(step)+".bin";
        std::string row_ptr_filename = data_path + "/A_row_ptr"+std::to_string(step)+".bin";
        std::string col_indices_filename = data_path + "/A_col_indices"+std::to_string(step)+".bin";
        std::string rhs_filename = data_path + "/A_rhs"+std::to_string(step)+".bin";
        std::string solution_filename = data_path + "/solution"+std::to_string(step)+".bin";

        load_binary_array<double>(data_filename, data, nnz);
        load_binary_array<int>(row_ptr_filename, row_ptr, matrix_size+1);
        load_binary_array<int>(col_indices_filename, col_indices, nnz);
        load_binary_array<double>(rhs_filename, rhs, matrix_size);
        load_binary_array<double>(solution_filename, reference_solution, matrix_size);


        int *row_ptr_local = new int[rows_per_rank+1];
        for (int i = 0; i < rows_per_rank+1; ++i) {
            row_ptr_local[i] = row_ptr[i+row_start_index] - row_ptr[row_start_index];
        }
        int nnz_local = row_ptr_local[rows_per_rank];
        int nnz_start_index = row_ptr[row_start_index];
        int nnz_end_index = nnz_start_index + nnz_local;
        std::cout << "rank " << rank << " nnz_local " << nnz_local << std::endl;
        std::cout << "rank " << rank << " nnz_start_index " << nnz_start_index << std::endl;
        std::cout << "rank " << rank << " nnz_end_index " << nnz_end_index << std::endl;

        int *col_indices_local = col_indices + nnz_start_index;
        double *data_local = data + nnz_start_index;

        int *row_indices_local = new int[nnz_local];


        for (int i = 0; i < rows_per_rank; ++i) {
            for (int j = row_ptr_local[i]; j < row_ptr_local[i+1]; ++j) {
                row_indices_local[j] = i;
            }
        }
        for(int method = 0; method < number_of_methods; method++){
            std::cout << "rank " << rank << " method " << method_names[method] << std::endl;
            for(int measurement = 0; measurement < number_of_measurements; measurement++){
                gpu_solve(
                    rank,
                    data_local,
                    row_ptr_local,
                    col_indices_local,
                    rhs,
                    reference_solution,
                    row_start_index,
                    rows_per_rank, 
                    matrix_size,
                    max_iterations,
                    solver_types[method],
                    preconditioners[method],
                    relative_tolerance,
                    absolute_tolerance,
                    divergence_tolerance,
                    &iterations[method][step],
                    &times[method][measurement],
                    &correct_solution_iteration
                );
                correct_solution_global = correct_solution_global && correct_solution_iteration;
            }
        }

        delete[] row_ptr_local;
        delete[] row_indices_local;

        for(int method = 0; method < number_of_methods; method++){
            std::ofstream outputFile_times;
            std::string path_times = save_path + method_names[method] + "_times"+std::to_string(step)+ "_" + std::to_string(size) + "_" + std::to_string(rank) + ".txt";
            outputFile_times.open(path_times);
            if(outputFile_times.is_open()){
                for(int i = 0; i < number_of_measurements; i++){
                    outputFile_times << times[method][i] << " ";
                }
                outputFile_times << '\n';
            }
            else{
                std::printf("Error opening file\n");
            }
            outputFile_times.close();
        }

    }

    if(rank == 0){
        std::cout << "Correct measurements: " << correct_solution_global << std::endl;
    }
    
    if(rank == 0){
        for(int method = 0; method < number_of_methods; method++){
            std::ofstream outputFile_iterations;
            std::string path_iterations = save_path + method_names[method] + "_iterations.txt";
            outputFile_iterations.open(path_iterations);
            if(outputFile_iterations.is_open()){
                for(int i = 0; i < number_of_kmc_steps; i++){
                    outputFile_iterations << iterations[method][i] << " ";
                }
                outputFile_iterations << '\n';
            }
            else{
                std::printf("Error opening file\n");
            }
            outputFile_iterations.close();
        }

    }

    delete[] data;
    delete[] row_ptr;
    delete[] col_indices;
    delete[] rhs;
    delete[] reference_solution;


    CHKERRQ(PetscFinalize());
    return 0;
}
