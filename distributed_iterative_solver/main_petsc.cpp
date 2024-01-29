#include <iostream>
#include <string>
#include "utils.h"
#include <mpi.h>
#include "iterative_solver_lib.h"

#include <petsc.h>

int main(int argc, char **argv) {
    // older version of petsc on daint
    // replace by PetscCall()
    MPI_Init(&argc, &argv);

    CHKERRQ(PetscInitialize(&argc, &argv, NULL, NULL));
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "Hello World from rank " << rank << std::endl;

    int matsize = 260;
    std::string data_path = "/scratch/snx3000/amaeder/"+std::to_string(matsize)+"k_piz_daint_data";
    //std::string save_path ="/scratch/snx3000/amaeder/measurements/self_preconditioned_scaling_measurement/";
    std::string save_path ="/scratch/snx3000/amaeder/measurements/single_node_libraries/";
    save_path ="/scratch/snx3000/amaeder/measurements/own_260/";

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



    int counts[size];
    int displacements[size];
    int rows_per_rank = matrix_size / size;    
    split_matrix(matrix_size, size, counts, displacements);

    int row_start_index = displacements[rank];
    rows_per_rank = counts[rank];

    std::cout << "rank " << rank << " row_start_index " << row_start_index << std::endl;
    std::cout << "rank " << rank << " rows_per_rank " << rows_per_rank << std::endl;

    double *data = new double[nnz];
    int *row_ptr = new int[matrix_size+1];
    int *col_indices = new int[nnz];
    double *rhs = new double[matrix_size];
    double *reference_solution = new double[matrix_size];

    int start_up_measurements = 2;
    int true_number_of_measurements = 5;
    int number_of_measurements = start_up_measurements + true_number_of_measurements;
    int number_of_kmc_steps = 1;

    int max_iterations = 5000;
    double relative_tolerance = 1e-12;
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
        "solve_petsc"
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
        int nnz_start_index = row_ptr[row_start_index];

        int *col_indices_local = col_indices + nnz_start_index;
        double *data_local = data + nnz_start_index;

        for(int method = 0; method < number_of_methods; method++){
            
            for(int measurement = 0; measurement < number_of_measurements; measurement++){
                std::cout << "rank " << rank << " " << method_names[method] << " " << measurement << std::endl;
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
                    solver_types[method],
                    preconditioners[method],
                    relative_tolerance,
                    absolute_tolerance,
                    divergence_tolerance,
                    &iterations[method][step],
                    &times[method][measurement]
                );


            }
        }

        delete[] row_ptr_local;

        for(int method = 0; method < number_of_methods; method++){
            std::string path_solve_petsc = get_filename(save_path, method_names[method], number_of_kmc_steps, size, rank);
            save_measurements(path_solve_petsc,
                times[method] + start_up_measurements,
                true_number_of_measurements, true);

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

    CHKERRQ(PetscFinalize());
    return 0;
}
