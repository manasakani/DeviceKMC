#include <iostream>
#include <string>
#include "utils.h"
#include <mpi.h>
#include <cuda_runtime.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "utils_gpu.h"
#include <cublas_v2.h>
#include "own_implementations/cg_own_implementations.h"

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cudaError_t set_device_error = cudaSetDevice(0);
    std::cout << "rank " << rank << " set_device_error " << set_device_error << std::endl;

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

    int number_of_measurements = 10;
    int number_of_kmc_steps = 1;

    int max_iterations = 5000;
    double relative_tolerance = 1e-12;

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

    double *data = new double[nnz];
    int *row_ptr = new int[matrix_size+1];
    int *col_indices = new int[nnz];
    double *rhs = new double[matrix_size];
    double *reference_solution = new double[matrix_size];



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




    // int iterations[number_of_kmc_steps];

    double *starting_guess = new double[matrix_size];
    for (int i = 0; i < matrix_size; ++i) {
        starting_guess[i] = 0.0;
    }


    for(int step = 0; step < number_of_kmc_steps; step++){
        std::cout << "rank " << rank << " step " << step << std::endl;

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



        double *data_copy = new double[nnz];
        int *row_ptr_copy = new int[matrix_size+1];
        int *col_indices_copy = new int[nnz];
        double *rhs_copy = new double[matrix_size];
        double *reference_solution_copy = new double[matrix_size];
        double *starting_guess_copy = new double[matrix_size];

        #pragma omp parallel for
        for(int i = 0; i < nnz; i++){
            data_copy[i] = data[i];
        }
        #pragma omp parallel for
        for(int i = 0; i < matrix_size+1; i++){
            row_ptr_copy[i] = row_ptr[i];
        }
        #pragma omp parallel for
        for(int i = 0; i < nnz; i++){
            col_indices_copy[i] = col_indices[i];
        }
        #pragma omp parallel for
        for(int i = 0; i < matrix_size; i++){
            rhs_copy[i] = rhs[i];
            reference_solution_copy[i] = reference_solution[i];
            starting_guess_copy[i] = starting_guess[i];
        }
        int iteration;
        std::cout << "rank " << rank << " starting measurement" << std::endl;
        double times1[number_of_measurements];
        double times2[number_of_measurements];
        double times3[number_of_measurements];
        double times4[number_of_measurements];
        double times_allgatherv1[number_of_measurements];
        double times_allgatherv2[number_of_measurements];
        double times_allgatherv3[number_of_measurements];
        double times_point_to_point1[number_of_measurements];
        double times_point_to_point2[number_of_measurements];
        double times_point_to_point3[number_of_measurements];
        double times_point_to_point4[number_of_measurements];
        double times_custom1[number_of_measurements];
        double times_custom2[number_of_measurements];
        double times_gpu_packing1[number_of_measurements];
        double times_gpu_packing2[number_of_measurements];
        double times_gpu_packing3[number_of_measurements];
        double times_gpu_packing4[number_of_measurements];
        double times_gpu_packing5[number_of_measurements];
        // if(size < 2){
        //     for(int measurement = 0; measurement < number_of_measurements; measurement++){
        //         std::cout << "rank " << rank << " solve_cg1 " << measurement << std::endl;
        //         own_test::solve_cg1(
        //             data,
        //             col_indices,
        //             row_ptr,
        //             rhs,
        //             reference_solution,
        //             starting_guess,
        //             nnz,
        //             matrix_size,
        //             relative_tolerance,
        //             max_iterations,
        //             &iteration,
        //             &times1[measurement]);
        //     }
        //     for(int measurement = 0; measurement < number_of_measurements; measurement++){
        //         std::cout << "rank " << rank << " solve_cg2 " << measurement << std::endl;
        //         own_test::solve_cg2(
        //             data,
        //             col_indices,
        //             row_ptr,
        //             rhs,
        //             reference_solution,
        //             starting_guess,
        //             nnz,
        //             matrix_size,
        //             relative_tolerance,
        //             max_iterations,
        //             &iteration,
        //             &times2[measurement]);
        //     }

        //     for(int measurement = 0; measurement < number_of_measurements; measurement++){
        //         std::cout << "rank " << rank << " solve_cg3 " << measurement << std::endl;
        //         own_test::solve_cg3(
        //             data,
        //             col_indices,
        //             row_ptr,
        //             rhs,
        //             reference_solution,
        //             starting_guess,
        //             nnz,
        //             matrix_size,
        //             relative_tolerance,
        //             max_iterations,
        //             &iteration,
        //             &times3[measurement]);
        //     }
        //     for(int measurement = 0; measurement < number_of_measurements; measurement++){
        //         std::cout << "rank " << rank << " solve_cg4 " << measurement << std::endl;
        //         own_test::solve_cg4(
        //             data,
        //             col_indices,
        //             row_ptr,
        //             rhs,
        //             reference_solution,
        //             starting_guess,
        //             nnz,
        //             matrix_size,
        //             relative_tolerance,
        //             max_iterations,
        //             &iteration,
        //             &times4[measurement]);
        //     }
        // }

        // for(int measurement = 0; measurement < number_of_measurements; measurement++){
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     std::cout << "rank " << rank << " solve_allgatherv1 " << measurement << std::endl;
        //     own_test::solve_cg_allgatherv1(
        //         data_copy,
        //         col_indices_copy,
        //         row_ptr_copy,
        //         rhs_copy,
        //         reference_solution_copy,
        //         starting_guess_copy,
        //         matrix_size,
        //         relative_tolerance,
        //         max_iterations,
        //         MPI_COMM_WORLD,
        //         &iteration,
        //         &times_allgatherv1[measurement]
        //     );
        // }
        // for(int measurement = 0; measurement < number_of_measurements; measurement++){
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     std::cout << "rank " << rank << " solve_allgatherv2 " << measurement << std::endl;
        //     own_test::solve_cg_allgatherv2(
        //         data_copy,
        //         col_indices_copy,
        //         row_ptr_copy,
        //         rhs_copy,
        //         reference_solution_copy,
        //         starting_guess_copy,
        //         matrix_size,
        //         relative_tolerance,
        //         max_iterations,
        //         MPI_COMM_WORLD,
        //         &iteration,
        //         &times_allgatherv2[measurement]
        //     );
        // }
        // for(int measurement = 0; measurement < number_of_measurements; measurement++){
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     std::cout << "rank " << rank << " solve_allgatherv3 " << measurement << std::endl;
        //     own_test::solve_cg_allgatherv3(
        //         data_copy,
        //         col_indices_copy,
        //         row_ptr_copy,
        //         rhs_copy,
        //         reference_solution_copy,
        //         starting_guess_copy,
        //         matrix_size,
        //         relative_tolerance,
        //         max_iterations,
        //         MPI_COMM_WORLD,
        //         &iteration,
        //         &times_allgatherv3[measurement]
        //     );
        // }

        // for(int measurement = 0; measurement < number_of_measurements; measurement++){
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     std::cout << "rank " << rank << " solve_cg_nonblocking_point_to_point1 " << measurement << std::endl;
        //     own_test::solve_cg_nonblocking_point_to_point1(
        //         data_copy,
        //         col_indices_copy,
        //         row_ptr_copy,
        //         rhs_copy,
        //         reference_solution_copy,
        //         starting_guess_copy,
        //         matrix_size,
        //         relative_tolerance,
        //         max_iterations,
        //         MPI_COMM_WORLD,
        //         &iteration,
        //         &times_point_to_point1[measurement]
        //     );
        // }

        // for(int measurement = 0; measurement < number_of_measurements; measurement++){
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     std::cout << "rank " << rank << " solve_cg_nonblocking_point_to_point2 " << measurement << std::endl;
        //     own_test::solve_cg_nonblocking_point_to_point2(
        //         data_copy,
        //         col_indices_copy,
        //         row_ptr_copy,
        //         rhs_copy,
        //         reference_solution_copy,
        //         starting_guess_copy,
        //         matrix_size,
        //         relative_tolerance,
        //         max_iterations,
        //         MPI_COMM_WORLD,
        //         &iteration,
        //         &times_point_to_point2[measurement]
        //     );
        // }
    
        // for(int measurement = 0; measurement < number_of_measurements; measurement++){
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     std::cout << "rank " << rank << " solve_cg_nonblocking_point_to_point3 " << measurement << std::endl;
        //     own_test::solve_cg_nonblocking_point_to_point3(
        //         data_copy,
        //         col_indices_copy,
        //         row_ptr_copy,
        //         rhs_copy,
        //         reference_solution_copy,
        //         starting_guess_copy,
        //         matrix_size,
        //         relative_tolerance,
        //         max_iterations,
        //         MPI_COMM_WORLD,
        //         &iteration,
        //         &times_point_to_point3[measurement]
        //     );
        // }
    
        // for(int measurement = 0; measurement < number_of_measurements; measurement++){
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     std::cout << "rank " << rank << " solve_cg_nonblocking_point_to_point4 " << measurement << std::endl;
        //     own_test::solve_cg_nonblocking_point_to_point4(
        //         data_copy,
        //         col_indices_copy,
        //         row_ptr_copy,
        //         rhs_copy,
        //         reference_solution_copy,
        //         starting_guess_copy,
        //         matrix_size,
        //         relative_tolerance,
        //         max_iterations,
        //         MPI_COMM_WORLD,
        //         &iteration,
        //         &times_point_to_point4[measurement]
        //     );
        // }
        // for(int measurement = 0; measurement < number_of_measurements; measurement++){
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     std::cout << "rank " << rank << " solve_cg_nonblocking_point_to_point_custom_datatype1 " << measurement << std::endl;
        //     own_test::solve_cg_nonblocking_point_to_point_custom_datatype1(
        //         data_copy,
        //         col_indices_copy,
        //         row_ptr_copy,
        //         rhs_copy,
        //         reference_solution_copy,
        //         starting_guess_copy,
        //         matrix_size,
        //         relative_tolerance,
        //         max_iterations,
        //         MPI_COMM_WORLD,
        //         &iteration,
        //         &times_custom1[measurement]
        //     );
        // }
        
        // for(int measurement = 0; measurement < number_of_measurements; measurement++){
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     std::cout << "rank " << rank << " solve_cg_nonblocking_point_to_point_custom_datatype2 " << measurement << std::endl;
        //     own_test::solve_cg_nonblocking_point_to_point_custom_datatype2(
        //         data_copy,
        //         col_indices_copy,
        //         row_ptr_copy,
        //         rhs_copy,
        //         reference_solution_copy,
        //         starting_guess_copy,
        //         matrix_size,
        //         relative_tolerance,
        //         max_iterations,
        //         MPI_COMM_WORLD,
        //         &iteration,
        //         &times_custom2[measurement]
        //     );
        // }


        for(int measurement = 0; measurement < number_of_measurements; measurement++){
            MPI_Barrier(MPI_COMM_WORLD);
            std::cout << "rank " << rank << " solve_cg_nonblocking_point_to_point_gpu_packing1 " << measurement << std::endl;
            own_test::solve_cg_nonblocking_point_to_point_gpu_packing1(
                data_copy,
                col_indices_copy,
                row_ptr_copy,
                rhs_copy,
                reference_solution_copy,
                starting_guess_copy,
                matrix_size,
                relative_tolerance,
                max_iterations,
                MPI_COMM_WORLD,
                &iteration,
                &times_gpu_packing1[measurement]
            );
        }

        for(int measurement = 0; measurement < number_of_measurements; measurement++){
            MPI_Barrier(MPI_COMM_WORLD);
            std::cout << "rank " << rank << " solve_cg_nonblocking_point_to_point_gpu_packing2 " << measurement << std::endl;
            own_test::solve_cg_nonblocking_point_to_point_gpu_packing2(
                data_copy,
                col_indices_copy,
                row_ptr_copy,
                rhs_copy,
                reference_solution_copy,
                starting_guess_copy,
                matrix_size,
                relative_tolerance,
                max_iterations,
                MPI_COMM_WORLD,
                &iteration,
                &times_gpu_packing2[measurement]
            );
        }

        for(int measurement = 0; measurement < number_of_measurements; measurement++){
            MPI_Barrier(MPI_COMM_WORLD);
            std::cout << "rank " << rank << " solve_cg_nonblocking_point_to_point_gpu_packing3 " << measurement << std::endl;
            own_test::solve_cg_nonblocking_point_to_point_gpu_packing3(
                data_copy,
                col_indices_copy,
                row_ptr_copy,
                rhs_copy,
                reference_solution_copy,
                starting_guess_copy,
                matrix_size,
                relative_tolerance,
                max_iterations,
                MPI_COMM_WORLD,
                &iteration,
                &times_gpu_packing3[measurement]
            );
        }

        for(int measurement = 0; measurement < number_of_measurements; measurement++){
            MPI_Barrier(MPI_COMM_WORLD);
            std::cout << "rank " << rank << " solve_cg_nonblocking_point_to_point_gpu_packing4 " << measurement << std::endl;
            own_test::solve_cg_nonblocking_point_to_point_gpu_packing4(
                data_copy,
                col_indices_copy,
                row_ptr_copy,
                rhs_copy,
                reference_solution_copy,
                starting_guess_copy,
                matrix_size,
                relative_tolerance,
                max_iterations,
                MPI_COMM_WORLD,
                &iteration,
                &times_gpu_packing4[measurement]
            );
        }
        for(int measurement = 0; measurement < number_of_measurements; measurement++){
            MPI_Barrier(MPI_COMM_WORLD);
            std::cout << "rank " << rank << " solve_cg_nonblocking_point_to_point_gpu_packing5 " << measurement << std::endl;
            own_test::solve_cg_nonblocking_point_to_point_gpu_packing5(
                data_copy,
                col_indices_copy,
                row_ptr_copy,
                rhs_copy,
                reference_solution_copy,
                starting_guess_copy,
                matrix_size,
                relative_tolerance,
                max_iterations,
                MPI_COMM_WORLD,
                &iteration,
                &times_gpu_packing5[measurement]
            );
        }

        delete[] data_copy;
        delete[] row_ptr_copy;
        delete[] col_indices_copy;
        delete[] rhs_copy;
        delete[] reference_solution_copy;
        delete[] starting_guess_copy;
        // if(size < 2){
        //     std::ofstream outputFile_solve1;
        //     std::string path_solve_cg1 = save_path + "solve1" +
        //         std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps) 
        //         +"_" + std::to_string(size) +"_" + std::to_string(rank) +"_.txt";
        //     outputFile_solve1.open(path_solve_cg1);
        //     if(outputFile_solve1.is_open()){
        //         for(int i = 0; i < number_of_measurements; i++){
        //             outputFile_solve1 << times1[i] << " ";
        //         }
        //         outputFile_solve1 << '\n';
        //     }
        //     else{
        //         std::printf("Error opening file\n");
        //     }
        //     outputFile_solve1.close();

        //     std::ofstream outputFile_solve2;
        //     std::string path_solve_cg2 = save_path + "solve2" +
        //         std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps) 
        //         +"_" + std::to_string(size) +"_" + std::to_string(rank) +"_.txt";
        //     outputFile_solve2.open(path_solve_cg2);
        //     if(outputFile_solve2.is_open()){
        //         for(int i = 0; i < number_of_measurements; i++){
        //             outputFile_solve2 << times2[i] << " ";
        //         }
        //         outputFile_solve2 << '\n';
        //     }
        //     else{
        //         std::printf("Error opening file\n");
        //     }
        //     outputFile_solve2.close();

        //     std::ofstream outputFile_solve3;
        //     std::string path_solve_cg3 = save_path + "solve3" +
        //         std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps) 
        //         +"_" + std::to_string(size) +"_" + std::to_string(rank) +"_.txt";
        //     outputFile_solve3.open(path_solve_cg3);
        //     if(outputFile_solve3.is_open()){
        //         for(int i = 0; i < number_of_measurements; i++){
        //             outputFile_solve3 << times3[i] << " ";
        //         }
        //         outputFile_solve3 << '\n';
        //     }
        //     else{
        //         std::printf("Error opening file\n");
        //     }
        //     outputFile_solve3.close();

        //     std::ofstream outputFile_solve4;
        //     std::string path_solve_cg4 = save_path + "solve4" +
        //         std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps) 
        //         +"_" + std::to_string(size) +"_" + std::to_string(rank) +"_.txt";
        //     outputFile_solve4.open(path_solve_cg4);
        //     if(outputFile_solve4.is_open()){
        //         for(int i = 0; i < number_of_measurements; i++){
        //             outputFile_solve4 << times4[i] << " ";
        //         }
        //         outputFile_solve4 << '\n';
        //     }
        //     else{
        //         std::printf("Error opening file\n");
        //     }
        //     outputFile_solve4.close();

        // }
        // std::ofstream outputFile_solve_allgatherv1;
        // std::string path_solve_allgatherv1 = save_path + "solve_allgatherv1" +
        //     std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps) 
        //     +"_" + std::to_string(size) +"_" + std::to_string(rank) +"_.txt";
        // outputFile_solve_allgatherv1.open(path_solve_allgatherv1);
        // if(outputFile_solve_allgatherv1.is_open()){
        //     for(int i = 0; i < number_of_measurements; i++){
        //         outputFile_solve_allgatherv1 << times_allgatherv1[i] << " ";
        //     }
        //     outputFile_solve_allgatherv1 << '\n';
        // }
        // else{
        //     std::printf("Error opening file\n");
        // }
        // outputFile_solve_allgatherv1.close();

        // std::ofstream outputFile_solve_allgatherv2;
        // std::string path_solve_allgatherv2 = save_path + "solve_allgatherv2" +
        //     std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps) 
        //     +"_" + std::to_string(size) +"_" + std::to_string(rank) +"_.txt";
        // outputFile_solve_allgatherv2.open(path_solve_allgatherv2);
        // if(outputFile_solve_allgatherv2.is_open()){
        //     for(int i = 0; i < number_of_measurements; i++){
        //         outputFile_solve_allgatherv2 << times_allgatherv2[i] << " ";
        //     }
        //     outputFile_solve_allgatherv2 << '\n';
        // }
        // else{
        //     std::printf("Error opening file\n");
        // }
        // outputFile_solve_allgatherv2.close();

        // std::ofstream outputFile_solve_allgatherv3;
        // std::string path_solve_allgatherv3 = save_path + "solve_allgatherv3" +
        //     std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps) 
        //     +"_" + std::to_string(size) +"_" + std::to_string(rank) +"_.txt";
        // outputFile_solve_allgatherv3.open(path_solve_allgatherv3);
        // if(outputFile_solve_allgatherv3.is_open()){
        //     for(int i = 0; i < number_of_measurements; i++){
        //         outputFile_solve_allgatherv3 << times_allgatherv3[i] << " ";
        //     }
        //     outputFile_solve_allgatherv3 << '\n';
        // }
        // else{
        //     std::printf("Error opening file\n");
        // }
        // outputFile_solve_allgatherv3.close();

        // std::ofstream outputFile_solve_point_to_point1;
        // std::string path_solve_point_to_point1 = save_path + "solve_point_to_point1" +
        //     std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps) 
        //     +"_" + std::to_string(size) +"_" + std::to_string(rank) +"_.txt";
        // outputFile_solve_point_to_point1.open(path_solve_point_to_point1);
        // if(outputFile_solve_point_to_point1.is_open()){
        //     for(int i = 0; i < number_of_measurements; i++){
        //         outputFile_solve_point_to_point1 << times_point_to_point1[i] << " ";
        //     }
        //     outputFile_solve_point_to_point1 << '\n';
        // }
        // else{
        //     std::printf("Error opening file\n");
        // }
        // outputFile_solve_point_to_point1.close();

        // std::ofstream outputFile_solve_point_to_point2;
        // std::string path_solve_point_to_point2 = save_path + "solve_point_to_point2" +
        //     std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps) 
        //     +"_" + std::to_string(size) +"_" + std::to_string(rank) +"_.txt";
        // outputFile_solve_point_to_point2.open(path_solve_point_to_point2);
        // if(outputFile_solve_point_to_point2.is_open()){
        //     for(int i = 0; i < number_of_measurements; i++){
        //         outputFile_solve_point_to_point2 << times_point_to_point2[i] << " ";
        //     }
        //     outputFile_solve_point_to_point2 << '\n';
        // }
        // else{
        //     std::printf("Error opening file\n");
        // }
        // outputFile_solve_point_to_point2.close();

        // std::ofstream outputFile_solve_point_to_point3;
        // std::string path_solve_point_to_point3 = save_path + "solve_point_to_point3" +
        //     std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps) 
        //     +"_" + std::to_string(size) +"_" + std::to_string(rank) +"_.txt";
        // outputFile_solve_point_to_point3.open(path_solve_point_to_point3);
        // if(outputFile_solve_point_to_point3.is_open()){
        //     for(int i = 0; i < number_of_measurements; i++){
        //         outputFile_solve_point_to_point3 << times_point_to_point3[i] << " ";
        //     }
        //     outputFile_solve_point_to_point3 << '\n';
        // }
        // else{
        //     std::printf("Error opening file\n");
        // }
        // outputFile_solve_point_to_point3.close();

        // std::ofstream outputFile_solve_point_to_point4;
        // std::string path_solve_point_to_point4 = save_path + "solve_point_to_point4" +
        //     std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps) 
        //     +"_" + std::to_string(size) +"_" + std::to_string(rank) +"_.txt";
        // outputFile_solve_point_to_point4.open(path_solve_point_to_point4);
        // if(outputFile_solve_point_to_point4.is_open()){
        //     for(int i = 0; i < number_of_measurements; i++){
        //         outputFile_solve_point_to_point4 << times_point_to_point4[i] << " ";
        //     }
        //     outputFile_solve_point_to_point4 << '\n';
        // }
        // else{
        //     std::printf("Error opening file\n");
        // }
        // outputFile_solve_point_to_point4.close();

        // std::ofstream outputFile_solve_datatype1;
        // std::string path_solve_custom_datatype1 = save_path + "solve_custom_datatype1" +
        //     std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps) 
        //     +"_" + std::to_string(size) +"_" + std::to_string(rank) +"_.txt";
        // outputFile_solve_datatype1.open(path_solve_custom_datatype1);
        // if(outputFile_solve_datatype1.is_open()){
        //     for(int i = 0; i < number_of_measurements; i++){
        //         outputFile_solve_datatype1 << times_custom1[i] << " ";
        //     }
        //     outputFile_solve_datatype1 << '\n';
        // }
        // else{
        //     std::printf("Error opening file\n");
        // }
        // outputFile_solve_datatype1.close();

        // std::ofstream outputFile_solve_datatype2;
        // std::string path_solve_custom_datatype2 = save_path + "solve_custom_datatype2" +
        //     std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps) 
        //     +"_" + std::to_string(size) +"_" + std::to_string(rank) +"_.txt";
        // outputFile_solve_datatype2.open(path_solve_custom_datatype2);
        // if(outputFile_solve_datatype2.is_open()){
        //     for(int i = 0; i < number_of_measurements; i++){
        //         outputFile_solve_datatype2 << times_custom2[i] << " ";
        //     }
        //     outputFile_solve_datatype2 << '\n';
        // }
        // else{
        //     std::printf("Error opening file\n");
        // }
        // outputFile_solve_datatype2.close();

        // std::ofstream outputFile_gpu_packing1;
        // std::string path_solve_gpu_packing1 = save_path + "solve_gpu_packing1" +
        //     std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps) 
        //     +"_" + std::to_string(size) +"_" + std::to_string(rank) +"_.txt";
        // outputFile_gpu_packing1.open(path_solve_gpu_packing1);
        // if(outputFile_gpu_packing1.is_open()){
        //     for(int i = 0; i < number_of_measurements; i++){
        //         outputFile_gpu_packing1 << times_gpu_packing1[i] << " ";
        //     }
        //     outputFile_gpu_packing1 << '\n';
        // }
        // else{
        //     std::printf("Error opening file\n");
        // }
        // outputFile_gpu_packing1.close();

        // std::ofstream outputFile_gpu_packing2;
        // std::string path_solve_gpu_packing2 = save_path + "solve_gpu_packing2" +
        //     std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps)
        //     +"_" + std::to_string(size) +"_" + std::to_string(rank) +"_.txt";
        // outputFile_gpu_packing2.open(path_solve_gpu_packing2);
        // if(outputFile_gpu_packing2.is_open()){
        //     for(int i = 0; i < number_of_measurements; i++){
        //         outputFile_gpu_packing2 << times_gpu_packing2[i] << " ";
        //     }
        //     outputFile_gpu_packing2 << '\n';
        // }
        // else{
        //     std::printf("Error opening file\n");
        // }
        // outputFile_gpu_packing2.close();

        // std::ofstream outputFile_gpu_packing3;
        // std::string path_solve_gpu_packing3 = save_path + "solve_gpu_packing3" +
        //     std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps)
        //     +"_" + std::to_string(size) +"_" + std::to_string(rank) +"_.txt";
        // outputFile_gpu_packing3.open(path_solve_gpu_packing3);
        // if(outputFile_gpu_packing3.is_open()){
        //     for(int i = 0; i < number_of_measurements; i++){
        //         outputFile_gpu_packing3 << times_gpu_packing3[i] << " ";
        //     }
        //     outputFile_gpu_packing3 << '\n';
        // }
        // else{
        //     std::printf("Error opening file\n");
        // }
        // outputFile_gpu_packing3.close();

        // std::ofstream outputFile_gpu_packing4;
        // std::string path_solve_gpu_packing4 = save_path + "solve_gpu_packing4" +
        //     std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps)
        //     +"_" + std::to_string(size) +"_" + std::to_string(rank) +"_.txt";
        // outputFile_gpu_packing4.open(path_solve_gpu_packing4);
        // if(outputFile_gpu_packing4.is_open()){
        //     for(int i = 0; i < number_of_measurements; i++){
        //         outputFile_gpu_packing4 << times_gpu_packing4[i] << " ";
        //     }
        //     outputFile_gpu_packing4 << '\n';
        // }
        // else{
        //     std::printf("Error opening file\n");
        // }
        // outputFile_gpu_packing4.close();

        // std::ofstream outputFile_gpu_packing5;
        // std::string path_solve_gpu_packing5 = save_path + "solve_gpu_packing5" +
        //     std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps)
        //     +"_" + std::to_string(size) +"_" + std::to_string(rank) +"_.txt";
        // outputFile_gpu_packing5.open(path_solve_gpu_packing5);
        // if(outputFile_gpu_packing5.is_open()){
        //     for(int i = 0; i < number_of_measurements; i++){
        //         outputFile_gpu_packing5 << times_gpu_packing5[i] << " ";
        //     }
        //     outputFile_gpu_packing5 << '\n';
        // }
        // else{
        //     std::printf("Error opening file\n");
        // }
        // outputFile_gpu_packing5.close();

    }




    delete[] data;
    delete[] row_ptr;
    delete[] col_indices;
    delete[] rhs;
    delete[] reference_solution;
    delete[] starting_guess;

    cudaFree(data_d);
    cudaFree(row_ptr_d);
    cudaFree(col_indices_d);
    cudaFree(rhs_d);
    cudaFree(reference_solution_d);
    cudaFree(diagonal_d);

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
