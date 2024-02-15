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
#include "own_implementations/mv_own_implementations.h"
int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cudaError_t set_device_error = cudaSetDevice(0);
    std::cout << "rank " << rank << " set_device_error " << set_device_error << std::endl;

    int matsize = 7;
    std::string data_path = "/scratch/snx3000/amaeder/"+std::to_string(matsize)+"k_piz_daint_data";
    //std::string save_path ="/scratch/snx3000/amaeder/measurements/self_preconditioned_scaling_measurement/";
    std::string save_path ="/scratch/snx3000/amaeder/measurements/own_260/";
    //data_path = "/usr/scratch/mont-fort17/almaeder/kmc_"+std::to_string(matsize)+"k/system_K";


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
    else if(matsize == 400){
        data_path = "/scratch/snx3000/amaeder/cross_bars";
        save_path ="/scratch/snx3000/amaeder/measurements/400/";
        matrix_size = 403605;
        nnz = 10007089;        
    }
    else{
        data_path = "/scratch/snx3000/amaeder/kmc_random";
        matrix_size = 262144;
        nnz = 16481266;
    }

    int start_up_measurements = 0;
    int true_number_of_measurements = 1;
    int number_of_measurements = start_up_measurements + true_number_of_measurements;
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
        std::string data_filename;
        std::string row_ptr_filename;
        std::string col_indices_filename;
        std::string rhs_filename;
        std::string solution_filename;

        std::cout << "rank " << rank << " data_path " << data_path << std::endl;
        if(matsize == 400){
            data_filename = data_path + "/A_data"+std::to_string(matrix_size)+"_"+std::to_string(step)+".bin";
            row_ptr_filename = data_path + "/A_row_ptr"+std::to_string(matrix_size)+"_"+std::to_string(step)+".bin";
            col_indices_filename = data_path + "/A_col_indices"+std::to_string(matrix_size)+"_"+std::to_string(step)+".bin";
            rhs_filename = data_path + "/A_rhs"+std::to_string(matrix_size)+"_"+std::to_string(step)+".bin";
        }
        else{
            data_filename = data_path + "/A_data"+std::to_string(step)+".bin";
            row_ptr_filename = data_path + "/A_row_ptr"+std::to_string(step)+".bin";
            col_indices_filename = data_path + "/A_col_indices"+std::to_string(step)+".bin";
            rhs_filename = data_path + "/A_rhs"+std::to_string(step)+".bin";
            solution_filename = data_path + "/solution"+std::to_string(step)+".bin";
        }
        std::cout << "rank " << rank << " Loading data" << std::endl;
        std::cout << "rank " << rank << " data_filename " << data_filename << std::endl;
        std::cout << "rank " << rank << " row_ptr_filename " << row_ptr_filename << std::endl;
        std::cout << "rank " << rank << " col_indices_filename " << col_indices_filename << std::endl;
        std::cout << "rank " << rank << " rhs_filename " << rhs_filename << std::endl;
        std::cout << "rank " << rank << " solution_filename " << solution_filename << std::endl;
        if(matsize == 400){
            load_binary_array<double>(data_filename, data, nnz);
            load_binary_array<int>(row_ptr_filename, row_ptr, matrix_size+1);
            load_binary_array<int>(col_indices_filename, col_indices, nnz);
            load_binary_array<double>(rhs_filename, rhs, matrix_size);
        }
        else{
            load_binary_array<double>(data_filename, data, nnz);
            load_binary_array<int>(row_ptr_filename, row_ptr, matrix_size+1);
            load_binary_array<int>(col_indices_filename, col_indices, nnz);
            load_binary_array<double>(rhs_filename, rhs, matrix_size);
            load_binary_array<double>(solution_filename, reference_solution, matrix_size);
        }

        std::cout << "rank " << rank << " data loaded" << std::endl;

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
        double times_custom1[number_of_measurements];
        double times_custom2[number_of_measurements];
        double times_gpu_packing1[number_of_measurements];
        double times_gpu_packing2[number_of_measurements];
        double times_gpu_packing3[number_of_measurements];
        double times_gpu_packing4[number_of_measurements];
        double times_gpu_packing5[number_of_measurements];
        if(size < 2){
            // for(int measurement = 0; measurement < number_of_measurements; measurement++){
            //     std::cout << "rank " << rank << " solve_cg1 " << measurement << std::endl;
            //     own_test::solve_cg1(
            //         data,
            //         col_indices,
            //         row_ptr,
            //         rhs,
            //         reference_solution,
            //         starting_guess,
            //         nnz,
            //         matrix_size,
            //         relative_tolerance,
            //         max_iterations,
            //         &iteration,
            //         &times1[measurement]);
            // }
            // for(int measurement = 0; measurement < number_of_measurements; measurement++){
            //     std::cout << "rank " << rank << " solve_cg2 " << measurement << std::endl;
            //     own_test::solve_cg2(
            //         data,
            //         col_indices,
            //         row_ptr,
            //         rhs,
            //         reference_solution,
            //         starting_guess,
            //         nnz,
            //         matrix_size,
            //         relative_tolerance,
            //         max_iterations,
            //         &iteration,
            //         &times2[measurement]);
            // }

            // for(int measurement = 0; measurement < number_of_measurements; measurement++){
            //     std::cout << "rank " << rank << " solve_cg3 " << measurement << std::endl;
            //     own_test::solve_cg3(
            //         data,
            //         col_indices,
            //         row_ptr,
            //         rhs,
            //         reference_solution,
            //         starting_guess,
            //         nnz,
            //         matrix_size,
            //         relative_tolerance,
            //         max_iterations,
            //         &iteration,
            //         &times3[measurement]);
            // }
            for(int measurement = 0; measurement < number_of_measurements; measurement++){
                std::cout << "rank " << rank << " solve_cg4 " << measurement << std::endl;
                own_test::solve_cg4(
                    data,
                    col_indices,
                    row_ptr,
                    rhs,
                    reference_solution,
                    starting_guess,
                    nnz,
                    matrix_size,
                    relative_tolerance,
                    max_iterations,
                    &iteration,
                    &times4[measurement]);
            }
        }

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
        //     own_test::solve_own_generic_mv<own_mv::distributed_mv_point_to_point1>(
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
        //     own_test::solve_own_generic_mv<own_mv::distributed_mv_point_to_point2>(
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
    
        for(int measurement = 0; measurement < number_of_measurements; measurement++){
            MPI_Barrier(MPI_COMM_WORLD);
            std::cout << "rank " << rank << " solve_cg_nonblocking_point_to_point3 " << measurement << std::endl;
            own_test::solve_own_generic_mv<own_mv::distributed_mv_point_to_point3>(
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
                &times_point_to_point3[measurement]
            );
        }

        // for(int measurement = 0; measurement < number_of_measurements; measurement++){
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     std::cout << "rank " << rank << " solve_cg_nonblocking_point_to_point_custom_datatype1 " << measurement << std::endl;
        //     own_test::solve_own_generic_mv<own_mv::distributed_mv_custom_datatype1>(
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
        //     own_test::solve_own_generic_mv<own_mv::distributed_mv_custom_datatype2>(
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




        // for(int measurement = 0; measurement < number_of_measurements; measurement++){
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     std::cout << "rank " << rank << " solve_cg_nonblocking_point_to_point_gpu_packing1 " << measurement << std::endl;
        //     own_test::solve_own_generic_mv<own_mv::distributed_mv_gpu_packing1>(
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
        //         &times_gpu_packing1[measurement]
        //     );
        // }

        // for(int measurement = 0; measurement < number_of_measurements; measurement++){
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     std::cout << "rank " << rank << " solve_cg_nonblocking_point_to_point_gpu_packing2 " << measurement << std::endl;
        //     own_test::solve_own_generic_mv<own_mv::distributed_mv_gpu_packing2>(
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
        //         &times_gpu_packing2[measurement]
        //     );
        // }





        // for(int measurement = 0; measurement < number_of_measurements; measurement++){
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     std::cout << "rank " << rank << " solve_cg_nonblocking_point_to_point_gpu_packing3 " << measurement << std::endl;
        //     own_test::solve_own_generic_mv<own_mv::distributed_mv_gpu_packing3>(
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
        //         &times_gpu_packing3[measurement]
        //     );
        // }




        // for(int measurement = 0; measurement < number_of_measurements; measurement++){
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     std::cout << "rank " << rank << " solve_cg_nonblocking_point_to_point_gpu_packing4 " << measurement << std::endl;
        //     own_test::solve_own_generic_mv<own_mv::distributed_mv_gpu_packing4>(
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
        //         &times_gpu_packing4[measurement]
        //     );
        // }
        // for(int measurement = 0; measurement < number_of_measurements; measurement++){
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     std::cout << "rank " << rank << " solve_cg_nonblocking_point_to_point_gpu_packing5 " << measurement << std::endl;
        //     own_test::solve_own_generic_mv<own_mv::distributed_mv_gpu_packing5>(
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
        //         &times_gpu_packing5[measurement]
        //     );
        // }

        delete[] data_copy;
        delete[] row_ptr_copy;
        delete[] col_indices_copy;
        delete[] rhs_copy;
        delete[] reference_solution_copy;
        delete[] starting_guess_copy;
        // if(size < 2){
        //     std::string path_solve_cg1 = get_filename(save_path, "solve1", number_of_kmc_steps, size, rank);
        //     std::string path_solve_cg2 = get_filename(save_path, "solve2", number_of_kmc_steps, size, rank);
        //     std::string path_solve_cg3 = get_filename(save_path, "solve3", number_of_kmc_steps, size, rank);
        //     std::string path_solve_cg4 = get_filename(save_path, "solve4", number_of_kmc_steps, size, rank);

        //     save_measurements(path_solve_cg1,
        //         times1 + start_up_measurements,
        //         true_number_of_measurements, true);
        //     save_measurements(path_solve_cg2,
        //         times2 + start_up_measurements,
        //         true_number_of_measurements, true);
        //     save_measurements(path_solve_cg3,
        //         times3 + start_up_measurements,
        //         true_number_of_measurements, true);
        //     save_measurements(path_solve_cg4,
        //         times4 + start_up_measurements,
        //         true_number_of_measurements, true);
        // }
        // std::string path_solve_allgatherv1 = get_filename(save_path, "solve_allgatherv1", number_of_kmc_steps, size, rank);
        // std::string path_solve_allgatherv2 = get_filename(save_path, "solve_allgatherv2", number_of_kmc_steps, size, rank);
        // std::string path_solve_allgatherv3 = get_filename(save_path, "solve_allgatherv3", number_of_kmc_steps, size, rank);
        // std::string path_solve_point_to_point1 = get_filename(save_path, "solve_point_to_point1", number_of_kmc_steps, size, rank);
        // std::string path_solve_point_to_point2 = get_filename(save_path, "solve_point_to_point2", number_of_kmc_steps, size, rank);
        // std::string path_solve_point_to_point3 = get_filename(save_path, "solve_point_to_point3", number_of_kmc_steps, size, rank);
        // std::string path_solve_custom_datatype1 = get_filename(save_path, "solve_custom_datatype1", number_of_kmc_steps, size, rank);
        // std::string path_solve_custom_datatype2 = get_filename(save_path, "solve_custom_datatype2", number_of_kmc_steps, size, rank);
        // std::string path_solve_gpu_packing1 = get_filename(save_path, "solve_gpu_packing1", number_of_kmc_steps, size, rank);
        // std::string path_solve_gpu_packing2 = get_filename(save_path, "solve_gpu_packing2", number_of_kmc_steps, size, rank);
        // std::string path_solve_gpu_packing3 = get_filename(save_path, "solve_gpu_packing3", number_of_kmc_steps, size, rank);
        // std::string path_solve_gpu_packing4 = get_filename(save_path, "solve_gpu_packing4", number_of_kmc_steps, size, rank);
        // std::string path_solve_gpu_packing5 = get_filename(save_path, "solve_gpu_packing5", number_of_kmc_steps, size, rank);

        // save_measurements(path_solve_allgatherv1,
        //     times_allgatherv1 + start_up_measurements,
        //     true_number_of_measurements, true);
        // save_measurements(path_solve_allgatherv2,
        //     times_allgatherv2 + start_up_measurements,
        //     true_number_of_measurements, true);
        // save_measurements(path_solve_allgatherv3,
        //     times_allgatherv3 + start_up_measurements,
        //     true_number_of_measurements, true);
        // save_measurements(path_solve_point_to_point1,
        //     times_point_to_point1 + start_up_measurements,
        //     true_number_of_measurements, true);
        // save_measurements(path_solve_point_to_point2,
        //     times_point_to_point2 + start_up_measurements,
        //     true_number_of_measurements, true);
        // save_measurements(path_solve_point_to_point3,
        //     times_point_to_point3 + start_up_measurements,
        //     true_number_of_measurements, true);
        // save_measurements(path_solve_custom_datatype1,
        //     times_custom1 + start_up_measurements,
        //     true_number_of_measurements, true);
        // save_measurements(path_solve_custom_datatype2,
        //     times_custom2 + start_up_measurements,
        //     true_number_of_measurements, true);
        // save_measurements(path_solve_gpu_packing1,
        //     times_gpu_packing1 + start_up_measurements,
        //     true_number_of_measurements, true);
        // save_measurements(path_solve_gpu_packing2,
        //     times_gpu_packing2 + start_up_measurements,
        //     true_number_of_measurements, true);
        // save_measurements(path_solve_gpu_packing3,
        //     times_gpu_packing3 + start_up_measurements,
        //     true_number_of_measurements, true);
        // save_measurements(path_solve_gpu_packing4,
        //     times_gpu_packing4 + start_up_measurements,
        //     true_number_of_measurements, true);
        // save_measurements(path_solve_gpu_packing5,
        //     times_gpu_packing5 + start_up_measurements,
        //     true_number_of_measurements, true);

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
