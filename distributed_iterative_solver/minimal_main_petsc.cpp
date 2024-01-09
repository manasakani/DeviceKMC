#include <iostream>
#include <string>
#include <mpi.h>
#include <cuda_runtime.h>
#include "petsc_implementations_to_compare.h"

#include <petscksp.h>
#include <petscvec.h>
#include <petscdevice.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "utils.h"


int main(int argc, char **argv) {
    // older version of petsc on daint
    // replace by PetscCall()
    int provided;
    // test without
    // test without thread multiple
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    std::cout << "provided " << provided << std::endl;

    CHKERRQ(PetscInitialize(&argc, &argv, NULL, NULL));
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "Hello World from rank " << rank << std::endl;
 


    std::string data_path = "/scratch/snx3000/amaeder/kmc_random";
    int matrix_size;
    int nnz;
    bool load_from_file = true;
    
    if(load_from_file){
        matrix_size = 262144;
        nnz = 16481266;
    } else {
        matrix_size = 600000;
        nnz = matrix_size;  
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

    if(load_from_file){
        int step = 0;
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
    }
    else{
        for(int i = 0; i < matrix_size; ++i){
            rhs[i] = i;
            data[i] = i+1;
            reference_solution[i] = rhs[i] / data[i];
            
            col_indices[i] = i;
        }
        for(int i = 0; i < matrix_size+1; ++i){
            row_ptr[i] = i;
        }
    }





    int number_of_measurements = 3;

    int max_iterations = 5000;
    double relative_tolerance = 1e-15;
    double absolute_tolerance = 1e-30;
    double divergence_tolerance = 1e+50;


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
    int iterations[number_of_methods];


    bool correct_solution_iteration;

    double times[number_of_methods][number_of_measurements];


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
                &iterations[method],
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
        delete[] row_ptr_local;
        delete[] row_indices_local;

    }

    delete[] data;
    delete[] row_ptr;
    delete[] col_indices;
    delete[] rhs;
    delete[] reference_solution;

    CHKERRQ(PetscFinalize());
    return 0;
}
