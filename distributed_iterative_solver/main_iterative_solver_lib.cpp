#include <iostream>
#include <string>
#include "utils.h"
#include <mpi.h>
#include <cuda_runtime.h>
#include "iterative_solver_lib.h"

#include <petsc.h>
#include <HYPRE.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "utils_gpu.h"


int main(int argc, char **argv) {
    // older version of petsc on daint
    // replace by PetscCall()
    MPI_Init(&argc, &argv);

    CHKERRQ(PetscInitialize(&argc, &argv, NULL, NULL));
    HYPRE_Initialize();
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "Hello World from rank " << rank << std::endl;

    int matsize = 1600;
    std::string data_path = "/scratch/snx3000/amaeder/"+std::to_string(matsize)+"k_piz_daint_data";
    //std::string save_path ="/scratch/snx3000/amaeder/measurements/self_preconditioned_scaling_measurement/";
    std::string save_path ="/scratch/snx3000/amaeder/measurements/own_260/";
    //data_path = "/usr/scratch/mont-fort17/almaeder/kmc_"+std::to_string(matsize)+"k/system_K";
    data_path = "/usr/scratch/mont-fort23/almaeder/kmc_matrices";

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
    else if(matsize == 1600){
        data_path = "/scratch/snx3000/amaeder/cross_bars";
        save_path = "/scratch/snx3000/amaeder/measurements/1600/";
        matrix_size = 1632355;
        nnz = 41208963;        
    }
    else{
        data_path = "/scratch/snx3000/amaeder/kmc_random";
        matrix_size = 262144;
        nnz = 16481266;
    }



    int counts[size];
    int displacements[size];
    int rows_per_rank = matrix_size / size;    
    split_matrix(matrix_size, size, counts, displacements);

    int row_start_index = displacements[rank];
    int row_end_index = row_start_index + counts[rank] - 1;
    rows_per_rank = counts[rank];

    std::cout << "rank " << rank << " row_start_index " << row_start_index << std::endl;
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

    int start_up_measurements = 0;
    int true_number_of_measurements = 1;
    int number_of_measurements = start_up_measurements + true_number_of_measurements;
    int number_of_kmc_steps = 1;

    int max_iterations = 5000;
    double relative_tolerance = 1e-8;
    double absolute_tolerance = 1e-30;
    double divergence_tolerance = 1e+50;

    int iterations_petsc[number_of_kmc_steps];
    int iterations_hypre[number_of_kmc_steps];
    int iterations_ginkgo[number_of_kmc_steps];

    for(int step = 0; step < number_of_kmc_steps; step++){
        std::cout << "rank " << rank << " step " << step << std::endl;

        double times_petsc[number_of_measurements];
        double times_hypre[number_of_measurements];
        double times_ginkgo[number_of_measurements];
        std::string data_filename;
        std::string row_ptr_filename;
        std::string col_indices_filename;
        std::string rhs_filename;
        std::string solution_filename;
        std::cout << "rank " << rank << " data_path " << data_path << std::endl;
        if(matsize == 400 || matsize == 1600){
            data_filename = data_path + "/A_data_rcm"+std::to_string(matrix_size)+"_"+std::to_string(step)+".bin";
            row_ptr_filename = data_path + "/A_row_ptr_rcm"+std::to_string(matrix_size)+"_"+std::to_string(step)+".bin";
            col_indices_filename = data_path + "/A_col_indices_rcm"+std::to_string(matrix_size)+"_"+std::to_string(step)+".bin";
            rhs_filename = data_path + "/A_rhs_rcm"+std::to_string(matrix_size)+"_"+std::to_string(step)+".bin";
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
        if(matsize == 400 || matsize == 1600){
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

        // // precondition the system myself

        int *row_ptr_local = new int[rows_per_rank+1];
        for (int i = 0; i < rows_per_rank+1; ++i) {
            row_ptr_local[i] = row_ptr[i+row_start_index] - row_ptr[row_start_index];
        }
        int nnz_local = row_ptr_local[rows_per_rank];
        int nnz_start_index = row_ptr[row_start_index];

        int *col_indices_local = col_indices + nnz_start_index;
        double *data_local = data + nnz_start_index;
        double *rhs_local = rhs + row_start_index;

        for(int measurement = 0; measurement < number_of_measurements; measurement++){
            std::cout << "rank " << rank << " solve_petsc " << measurement << std::endl;
            lib_to_compare::solve_petsc(
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
                &iterations_petsc[step],
                &times_petsc[measurement]
            );
        }

        for(int i = 0; i < number_of_measurements; i++){
            std::cout << "rank " << rank << " solve_hypre " << i << std::endl;
            lib_to_compare::solve_hypre(
                data_local,
                row_ptr_local,
                col_indices_local,
                rhs_local,
                reference_solution,
                row_start_index,
                row_end_index,
                rows_per_rank,
                max_iterations,
                relative_tolerance,
                absolute_tolerance,
                &iterations_hypre[step],
                &times_hypre[i]
            );
        }
        for(int i = 0; i < number_of_measurements; i++) {
            std::cout << "rank " << rank << " solve_ginkgo " << i << std::endl;
            lib_to_compare::solve_ginkgo(
                data,
                row_ptr,
                col_indices,
                rhs,
                reference_solution,
                matrix_size,
                max_iterations,
                relative_tolerance,
                &iterations_ginkgo[step],
                &times_ginkgo[i]);
        }

        delete[] row_ptr_local;

        // std::string path_solve_petsc = get_filename(save_path, "solve_petsc", number_of_kmc_steps, size, rank);
        // std::string path_solve_hypre = get_filename(save_path, "solve_hypre", number_of_kmc_steps, size, rank);
        // std::string path_solve_ginkgo = get_filename(save_path, "solve_ginkgo", number_of_kmc_steps, size, rank);
        // save_measurements(path_solve_petsc,
        //     times_petsc + start_up_measurements,
        //     true_number_of_measurements, true);
        // save_measurements(path_solve_hypre,
        //     times_hypre + start_up_measurements,
        //     true_number_of_measurements, true);
        // save_measurements(path_solve_ginkgo,
        //     times_ginkgo + start_up_measurements,
        //     true_number_of_measurements, true);
    }

    

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

    HYPRE_Finalize();
    CHKERRQ(PetscFinalize());
    return 0;
}
