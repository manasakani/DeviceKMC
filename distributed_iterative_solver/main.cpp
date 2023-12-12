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

    int number_of_measurements = 2;
    int number_of_kmc_steps = 2;

    int max_iterations = 10000;
    double relative_tolerance = 1e-15;
    double absolute_tolerance = 1e-13;
    double divergence_tolerance = 1e+50;


    std::string method_names[9] = {
        "cg",
        "bicg",
        "gmres",
        "cg_jacobi",
        "bicg_jacobi",
        "gmres_jacobi",
        "cg_sor",
        "bicg_sor",
        "gmres_sor"
    };
    KSPType solver_types[9] = {
        KSPCG,
        KSPBCGS,
        KSPGMRES,
        KSPCG,
        KSPBCGS,
        KSPGMRES,
        KSPCG,
        KSPBCGS,
        KSPGMRES
    };
    PCType preconditioners[9] = {
        PCNONE,
        PCNONE,
        PCNONE,
        PCJACOBI,
        PCJACOBI,
        PCJACOBI,
        PCSOR,
        PCSOR,
        PCSOR
    };
    int iterations[9][number_of_kmc_steps];


    // int cg_ssoreisenstat_iterations[number_of_kmc_steps];
    // int gmres_ssoreisenstat_iterations[number_of_kmc_steps];
    // int bicg_ssoreisenstat_iterations[number_of_kmc_steps];
    // int cg_eisenstat_iterations[number_of_kmc_steps];


    bool correct_solution_global = true;
    for(int step = 0; step < number_of_measurements; step++){
        bool correct_solution_iteration;

        double times[9][number_of_measurements];

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
                KSPCG,
                PCNONE,
                relative_tolerance,
                absolute_tolerance,
                divergence_tolerance,
                &cg_iterations[step],
                &cg_times[measurement],
                &correct_solution_iteration
            );
            correct_solution_global = correct_solution_global && correct_solution_iteration;
        }
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
                KSPGMRES,
                PCNONE,
                relative_tolerance,
                absolute_tolerance,
                divergence_tolerance,
                &gmres_iterations[step],
                &gmres_times[measurement],
                &correct_solution_iteration
            );
            correct_solution_global = correct_solution_global && correct_solution_iteration;
        }
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
                KSPBCGS,
                PCNONE,
                relative_tolerance,
                absolute_tolerance,
                divergence_tolerance,
                &bicg_iterations[step],
                &bicg_times[measurement],
                &correct_solution_iteration
            );
            correct_solution_global = correct_solution_global && correct_solution_iteration;
        }
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
                KSPCG,
                PCJACOBI,
                relative_tolerance,
                absolute_tolerance,
                divergence_tolerance,
                &cg_jacobi_iterations[step],
                &cg_jacobi_times[measurement],
                &correct_solution_iteration
            );
            correct_solution_global = correct_solution_global && correct_solution_iteration;
        }
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
                KSPGMRES,
                PCJACOBI,
                relative_tolerance,
                absolute_tolerance,
                divergence_tolerance,
                &gmres_jacobi_iterations[step],
                &gmres_jacobi_times[measurement],
                &correct_solution_iteration
            );
            correct_solution_global = correct_solution_global && correct_solution_iteration;
        }
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
                KSPBCGS,
                PCJACOBI,
                relative_tolerance,
                absolute_tolerance,
                divergence_tolerance,
                &bicg_jacobi_iterations[step],
                &bicg_jacobi_times[measurement],
                &correct_solution_iteration
            );
            correct_solution_global = correct_solution_global && correct_solution_iteration;
        }
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
                KSPCG,
                PCSOR,
                relative_tolerance,
                absolute_tolerance,
                divergence_tolerance,
                &cg_sor_iterations[step],
                &cg_sor_times[measurement],
                &correct_solution_iteration
            );
            correct_solution_global = correct_solution_global && correct_solution_iteration;
        }
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
                KSPGMRES,
                PCSOR,
                relative_tolerance,
                absolute_tolerance,
                divergence_tolerance,
                &gmres_sor_iterations[step],
                &gmres_sor_times[measurement],
                &correct_solution_iteration
            );
            correct_solution_global = correct_solution_global && correct_solution_iteration;
        }
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
                KSPBCGS,
                PCSOR,
                relative_tolerance,
                absolute_tolerance,
                divergence_tolerance,
                &bicg_sor_iterations[step],
                &bicg_sor_times[measurement],
                &correct_solution_iteration
            );
            correct_solution_global = correct_solution_global && correct_solution_iteration;
        }

        delete[] row_ptr_local;
        delete[] row_indices_local;

        std::ofstream outputFile_cg_times;
        std::string path_cg_times = save_path + "cg_times"+std::to_string(step)+ "_" + std::to_string(size) + "_" + std::to_string(rank) + ".txt";
        outputFile_cg_times.open(path_cg_times);
        if(outputFile_cg_times.is_open()){
            for(int i = 0; i < number_of_measurements; i++){
                outputFile_cg_times << cg_times[i] << " ";
            }
            outputFile_cg_times << '\n';
        }
        else{
            std::printf("Error opening file\n");
        }
        outputFile_cg_times.close();

        std::ofstream outputFile_gmres_times;
        std::string path_gmres_times = save_path + "gmres_times"+std::to_string(step)+ "_" + std::to_string(size) + "_" + std::to_string(rank) + ".txt";
        outputFile_gmres_times.open(path_gmres_times);
        if(outputFile_gmres_times.is_open()){
            for(int i = 0; i < number_of_measurements; i++){
                outputFile_gmres_times << gmres_times[i] << " ";
            }
            outputFile_gmres_times << '\n';
        }
        else{
            std::printf("Error opening file\n");
        }
        outputFile_gmres_times.close();

        std::ofstream outputFile_bicg_times;
        std::string path_bicg_times = save_path + "bicg_times"+std::to_string(step)+ "_" + std::to_string(size) + "_" + std::to_string(rank) + ".txt";
        outputFile_bicg_times.open(path_bicg_times);
        if(outputFile_bicg_times.is_open()){
            for(int i = 0; i < number_of_measurements; i++){
                outputFile_bicg_times << bicg_times[i] << " ";
            }
            outputFile_bicg_times << '\n';
        }
        else{
            std::printf("Error opening file\n");
        }
        outputFile_bicg_times.close();

        std::ofstream outputFile_cg_jacobi_times;
        std::string path_cg_jacobi_times = save_path + "cg_jacobi_times"+std::to_string(step)+ "_" + std::to_string(size) + "_" + std::to_string(rank) + ".txt";
        outputFile_cg_jacobi_times.open(path_cg_jacobi_times);
        if(outputFile_cg_jacobi_times.is_open()){
            for(int i = 0; i < number_of_measurements; i++){
                outputFile_cg_jacobi_times << cg_jacobi_times[i] << " ";
            }
            outputFile_cg_jacobi_times << '\n';
        }
        else{
            std::printf("Error opening file\n");
        }
        outputFile_cg_jacobi_times.close();

        std::ofstream outputFile_gmres_jacobi_times;
        std::string path_gmres_jacobi_times = save_path + "gmres_jacobi_times"+std::to_string(step)+ "_" + std::to_string(size) + "_" + std::to_string(rank) + ".txt";
        outputFile_gmres_jacobi_times.open(path_gmres_jacobi_times);

    }

    if(rank == 0){
        std::cout << "Correct measurements: " << correct_solution_global << std::endl;
    }
    
    if(rank == 0){
        std::ofstream outputFile_cg_iterations;
        std::string path_cg_iterations = save_path + "cg_iterations.txt";
        outputFile_cg_iterations.open(path_cg_iterations);
        if(outputFile_cg_iterations.is_open()){
            for(int i = 0; i < number_of_kmc_steps; i++){
                outputFile_cg_iterations << cg_iterations[i] << " ";
            }
            outputFile_cg_iterations << '\n';
        }
        else{
            std::printf("Error opening file\n");
        }
        outputFile_cg_iterations.close();

        std::ofstream outputFile_gmres_iterations;
        std::string path_gmres_iterations = save_path + "gmres_iterations.txt";
        outputFile_gmres_iterations.open(path_gmres_iterations);
        if(outputFile_gmres_iterations.is_open()){
            for(int i = 0; i < number_of_kmc_steps; i++){
                outputFile_gmres_iterations << gmres_iterations[i] << " ";
            }
            outputFile_gmres_iterations << '\n';
        }
        else{
            std::printf("Error opening file\n");
        }
        outputFile_gmres_iterations.close();

        std::ofstream outputFile_bicg_iterations;
        std::string path_bicg_iterations = save_path + "bicg_iterations.txt";
        outputFile_bicg_iterations.open(path_bicg_iterations);
        if(outputFile_bicg_iterations.is_open()){
            for(int i = 0; i < number_of_kmc_steps; i++){
                outputFile_bicg_iterations << bicg_iterations[i] << " ";
            }
            outputFile_bicg_iterations << '\n';
        }
        else{
            std::printf("Error opening file\n");
        }
        outputFile_bicg_iterations.close();


    }

    delete[] data;
    delete[] row_ptr;
    delete[] col_indices;
    delete[] rhs;
    delete[] reference_solution;


    CHKERRQ(PetscFinalize());
    return 0;
}
