#include <iostream>
#include <string>
#include "utils.h"
#include <mpi.h>
#include <cuda_runtime.h>
#include "petsc_implementations_to_compare.h"

#include <petscksp.h>
#include <petscvec.h>
#include <petscdevice.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "utils_gpu.h"
#include <cublas_v2.h>

// export MPICH_MAX_THREAD_SAFETY=multiple
// export MPICH_ASYNC_PROGRESS=1

int main(int argc, char **argv) {
    // older version of petsc on daint
    // replace by PetscCall()
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    std::cout << "provided " << provided << std::endl;

    CHKERRQ(PetscInitialize(&argc, &argv, NULL, NULL));
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "Hello World from rank " << rank << std::endl;

    int matsize = 260;
    std::string data_path = "/scratch/snx3000/amaeder/"+std::to_string(matsize)+"k_piz_daint_data";
    //std::string save_path ="/scratch/snx3000/amaeder/measurements/self_preconditioned_scaling_measurement/";
    std::string save_path ="/scratch/snx3000/amaeder/measurements/single_node_libraries/";


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
        save_path = "/scratch/snx3000/amaeder/measurements/random_scaling/";
        matrix_size = 262144;
        nnz = 16481266;
    }



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
    // precondition the system myself
    double *diagonal = new double[matrix_size];


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

    // int number_of_measurements = 20;
    // int number_of_kmc_steps = 50;
    int number_of_measurements = 3;
    int number_of_kmc_steps = 1;

    int max_iterations = 5000;
    double relative_tolerance = 1e-16;
    double absolute_tolerance = 1e-30;
    double divergence_tolerance = 1e+50;

    // int number_of_methods = 6;
    // std::string method_names[number_of_methods] = {
    //     "cg_jacobi",
    //     "cg_gropp_jacobi",
    //     "bicg_jacobi",
    //     "cg_sor",
    //     "cg_gropp_sor",
    //     "bicg_sor"
    // };
    // KSPType solver_types[number_of_methods] = {
    //     KSPCG,
    //     KSPGROPPCG,
    //     KSPBCGS,
    //     KSPCG,
    //     KSPGROPPCG,
    //     KSPBCGS
    // };
    // PCType preconditioners[number_of_methods] = {
    //     PCJACOBI,
    //     PCJACOBI,
    //     PCJACOBI,
    //     PCSOR,
    //     PCSOR,
    //     PCSOR
    // };
    // int number_of_methods = 4;
    // std::string method_names[number_of_methods] = {
    //     "cg_jacobi",
    //     "bicg_jacobi",
    //     "cg_sor",
    //     "bicg_sor"
    // };
    // KSPType solver_types[number_of_methods] = {
    //     KSPCG,
    //     KSPBCGS,
    //     KSPCG,
    //     KSPBCGS
    // };
    // PCType preconditioners[number_of_methods] = {
    //     PCJACOBI,
    //     PCJACOBI,
    //     PCSOR,
    //     PCSOR
    // };
    // int number_of_methods = 3;
    // std::string method_names[number_of_methods] = {
    //     "cg_jacobi",
    //     "cg_gropp_jacobi",
    //     "bicg_jacobi"
    // };
    // KSPType solver_types[number_of_methods] = {
    //     KSPCG,
    //     KSPGROPPCG,
    //     KSPBICG
    // };
    // PCType preconditioners[number_of_methods] = {
    //     PCNONE,
    //     PCNONE,
    //     PCNONE
    // };
    int number_of_methods = 1;
    std::string method_names[number_of_methods] = {
        "cg_jacobi"
    };
    KSPType solver_types[number_of_methods] = {
        KSPCG
    };
    PCType preconditioners[number_of_methods] = {
        PCNONE
    };
    int iterations[number_of_methods][number_of_kmc_steps];




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

        // // precondition the system myself
        // extract_diagonal(data, row_ptr, col_indices, diagonal, matrix_size);
        // symmetric_precondition_matrix(data, row_ptr, col_indices, diagonal, matrix_size);
        // precondition_vector(rhs, diagonal, matrix_size);
        // unpreecondition_vector(reference_solution, diagonal, matrix_size);


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
            if(rank == 0){
                std::cout  << "method " << method_names[method] << std::endl;                
            }
            
            double *data_local_copy = new double[nnz_local];
            int *row_ptr_local_copy = new int[rows_per_rank+1];
            int *col_indices_local_copy = new int[nnz_local];
            double *rhs_copy = new double[matrix_size];
            double *reference_solution_copy = new double[matrix_size];


            #pragma omp parallel for
            for(int i = 0; i < nnz_local; i++){
                data_local_copy[i] = data_local[i];
            }
            #pragma omp parallel for
            for(int i = 0; i < rows_per_rank+1; i++){
                row_ptr_local_copy[i] = row_ptr_local[i];
            }
            #pragma omp parallel for
            for(int i = 0; i < nnz_local; i++){
                col_indices_local_copy[i] = col_indices_local[i];
            }
            #pragma omp parallel for
            for(int i = 0; i < matrix_size; i++){
                rhs_copy[i] = rhs[i];
                reference_solution_copy[i] = reference_solution[i];
            }


            for(int measurement = 0; measurement < number_of_measurements; measurement++){
                //MPI_Barrier(MPI_COMM_WORLD);
                petsc_test::gpu_solve(
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

                bool not_overwritten = true;
                double eps = 1e-20;
                for(int i = 0; i < nnz_local; i++){
                    if(std::abs(data_local[i] - data_local_copy[i]) > eps){
                        not_overwritten = false;
                    }
                }
                for(int i = 0; i < rows_per_rank+1; i++){
                    if(std::abs(row_ptr_local[i] - row_ptr_local_copy[i]) > eps){
                        not_overwritten = false;
                    }
                }
                for(int i = 0; i < nnz_local; i++){
                    if(std::abs(col_indices_local[i] - col_indices_local_copy[i]) > eps){
                        not_overwritten = false;
                    }
                }
                for(int i = 0; i < matrix_size; i++){
                    if(std::abs(rhs[i] - rhs_copy[i]) > eps){
                        not_overwritten = false;
                    }
                    if(std::abs(reference_solution[i] - reference_solution_copy[i]) > eps){
                        not_overwritten = false;
                    }
                }
                std::cout << "rank " << rank << " not_overwritten " << not_overwritten << std::endl;

            }

            delete[] data_local_copy;
            delete[] row_ptr_local_copy;
            delete[] col_indices_local_copy;
            delete[] rhs_copy;
            delete[] reference_solution_copy;

        }

        delete[] row_ptr_local;
        delete[] row_indices_local;

        for(int method = 0; method < number_of_methods; method++){
            std::ofstream outputFile_times;
            std::string path_times = save_path + method_names[method] + "_gpu_times"+ std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps) + "_" + std::to_string(step) 
                + "_" + std::to_string(size) + "_" + std::to_string(rank) + ".txt";
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

    
    // if(rank == 0){
    //     for(int method = 0; method < number_of_methods; method++){
    //         std::ofstream outputFile_iterations;
    //         std::string path_iterations = save_path + method_names[method] + "_iterations" + std::to_string(matsize) +"_" + std::to_string(number_of_kmc_steps) +"_.txt";
    //         outputFile_iterations.open(path_iterations);
    //         if(outputFile_iterations.is_open()){
    //             for(int i = 0; i < number_of_kmc_steps; i++){
    //                 outputFile_iterations << iterations[method][i] << " ";
    //             }
    //             outputFile_iterations << '\n';
    //         }
    //         else{
    //             std::printf("Error opening file\n");
    //         }
    //         outputFile_iterations.close();
    //     }

    // }

    delete[] data;
    delete[] row_ptr;
    delete[] col_indices;
    delete[] rhs;
    delete[] reference_solution;
    delete[] diagonal;

    cudaFree(data_d);
    cudaFree(row_ptr_d);
    cudaFree(col_indices_d);
    cudaFree(rhs_d);
    cudaFree(reference_solution_d);
    cudaFree(diagonal_d);


    CHKERRQ(PetscFinalize());
    return 0;
}
