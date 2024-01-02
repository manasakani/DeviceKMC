#include <iostream>
#include <string>
#include "utils.h"
#include <mpi.h>
#include <cuda_runtime.h>
#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_utilities.h>
#include <HYPRE_krylov.h>

int main(int argc, char **argv) {
    // older version of petsc on daint
    // replace by PetscCall()
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    
    std::cout << "provided " << provided << std::endl;

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "Hello World from rank " << rank << std::endl;

    int matsize = 80;
    std::string data_path = "/scratch/snx3000/amaeder/"+std::to_string(matsize)+"k_piz_daint_data";
    std::string save_path ="/scratch/snx3000/amaeder/measurements/self_preconditioned_scaling_measurement/";

    int matrix_size;
    int nnz;     
    if(matsize == 7){
        matrix_size = 7302;
        nnz = 186684;        
    }
    else{
        matrix_size = 70630;
        nnz = 1719652;        
    }


    int rows_per_rank = matrix_size / size;
    int remainder = matrix_size % size;
    int row_start_index = rank * rows_per_rank;
    int row_end_index = row_start_index + rows_per_rank - 1;
    if (rank == size-1) {
        row_end_index += remainder;
        rows_per_rank += remainder;
    }

    std::cout << "rank " << rank << " row_start_index " << row_start_index << " row_end_index " << row_end_index << std::endl;
    std::cout << "rank " << rank << " rows_per_rank " << rows_per_rank << std::endl;

    double *data = new double[nnz];
    int *row_ptr = new int[matrix_size+1];
    int *col_indices = new int[nnz];
    double *rhs = new double[matrix_size];
    double *reference_solution = new double[matrix_size];
    // precondition the system myself
    double *diagonal = new double[matrix_size];

    std::string row_ptr_filename = data_path + "/A_row_ptr"+std::to_string(0)+".bin";
    std::string col_indices_filename = data_path + "/A_col_indices"+std::to_string(0)+".bin";
    // row ptr and col indices are the same in every step
    load_binary_array<int>(row_ptr_filename, row_ptr, matrix_size+1);
    load_binary_array<int>(col_indices_filename, col_indices, nnz);


    // int number_of_measurements = 20;
    // int number_of_kmc_steps = 50;
    int number_of_measurements = 1;
    int number_of_kmc_steps = 1;

    int max_iterations = 5000;
    double relative_tolerance = 1e-15;
    double absolute_tolerance = 1e-30;
    double divergence_tolerance = 1e+50;

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
    int *nnz_per_row_local = new int[rows_per_rank];
    for (int i = 0; i < rows_per_rank; ++i) {
        nnz_per_row_local[i] = row_ptr_local[i+1] - row_ptr_local[i];
    }
    int *linear_indices = new int[rows_per_rank];
    double *solution = new double[rows_per_rank];
    for (int i = 0; i < rows_per_rank; ++i) {
        linear_indices[i] = row_start_index + i;
        solution[i] = 0;
    }


    HYPRE_IJMatrix      ij_matrix;
    HYPRE_ParCSRMatrix  parcsr_matrix;
    HYPRE_IJVector   ij_rhs;
    HYPRE_ParVector  par_rhs;
    HYPRE_IJVector   ij_x;
    HYPRE_ParVector  par_x;

    int ilower = row_start_index;
    int iupper = row_end_index;
    int jlower = row_start_index;
    int jupper = row_end_index;

    std::cout << "Creating matrix" << std::endl;
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, jlower, jupper, &ij_matrix);
    HYPRE_IJMatrixSetObjectType(ij_matrix, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(ij_matrix);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, jlower, jupper, &ij_rhs);
    HYPRE_IJVectorSetObjectType(ij_rhs, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ij_rhs);
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, jlower, jupper, &ij_x);
    HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(ij_x);

    HYPRE_Solver pcg_solver;
    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg_solver);
    HYPRE_PCGSetMaxIter(pcg_solver, max_iterations);
    HYPRE_PCGSetTol(pcg_solver, relative_tolerance);
    HYPRE_PCGSetAbsoluteTol(pcg_solver, absolute_tolerance);
    HYPRE_PCGSetTwoNorm(pcg_solver, 1);
    // HYPRE_PCGSetPrintLevel(pcg_solver, 2);
    // HYPRE_PCGSetLogging(pcg_solver, 1);

    int num_iterations;
    double final_res_norm;

    std::cout << "Loop Steps" << std::endl;
    for(int step = 0; step < number_of_kmc_steps; step++){
        std::cout << "rank " << rank << " step " << step << std::endl;
        bool correct_solution_iteration;

        std::cout << "Loading data" << std::endl;
        std::string data_filename = data_path + "/A_data"+std::to_string(step)+".bin";
        // std::string row_ptr_filename = data_path + "/A_row_ptr"+std::to_string(step)+".bin";
        // std::string col_indices_filename = data_path + "/A_col_indices"+std::to_string(step)+".bin";
        std::string rhs_filename = data_path + "/A_rhs"+std::to_string(step)+".bin";
        std::string solution_filename = data_path + "/solution"+std::to_string(step)+".bin";

        load_binary_array<double>(data_filename, data, nnz);
        load_binary_array<double>(rhs_filename, rhs, matrix_size);
        load_binary_array<double>(solution_filename, reference_solution, matrix_size);
        // load_binary_array<int>(row_ptr_filename, row_ptr, matrix_size+1);
        // load_binary_array<int>(col_indices_filename, col_indices, nnz);

        std::cout << "Preconditioning" << std::endl;
        // precondition the system myself
        extract_diagonal(data, row_ptr, col_indices, diagonal, matrix_size);
        symmetric_precondition_matrix(data, row_ptr, col_indices, diagonal, matrix_size);
        precondition_vector(rhs, diagonal, matrix_size);
        unpreecondition_vector(reference_solution, diagonal, matrix_size);

        // set matrix
        HYPRE_IJMatrixSetValues(ij_matrix, rows_per_rank, nnz_per_row_local, linear_indices,
            col_indices_local, data_local);
        HYPRE_IJMatrixAssemble(ij_matrix);
        HYPRE_IJMatrixGetObject(ij_matrix, (void **) &parcsr_matrix);

        //set rhs
        HYPRE_IJVectorSetValues(ij_rhs, rows_per_rank, linear_indices, rhs+row_start_index);
        HYPRE_IJVectorAssemble(ij_rhs);
        HYPRE_IJVectorGetObject(ij_rhs, (void **) &par_rhs);

        //set x
        HYPRE_IJVectorSetValues(ij_x, rows_per_rank, linear_indices, solution);
        HYPRE_IJVectorAssemble(ij_x);
        HYPRE_IJVectorGetObject(ij_x, (void **) &par_x);

        // HYPRE_IJMatrixPrint(ij_matrix, "IJ.out.A");
        // HYPRE_IJVectorPrint(ij_rhs, "IJ.out.b");

        // solve
        HYPRE_ParCSRPCGSetup(pcg_solver, parcsr_matrix, par_rhs, par_x);
        MPI_Barrier(MPI_COMM_WORLD);
        double time_taken = -omp_get_wtime();
        HYPRE_ParCSRPCGSolve(pcg_solver, parcsr_matrix, par_rhs, par_x);
        time_taken += omp_get_wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "rank " << rank << " time_taken " << time_taken << std::endl;

        HYPRE_PCGGetNumIterations(pcg_solver, &num_iterations);
        HYPRE_PCGGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);
        std::cout << "Iterations = " << num_iterations << std::endl;
        std::cout << "Final Relative Residual Norm = " << final_res_norm << std::endl;

        HYPRE_IJVectorGetValues(ij_x, rows_per_rank, linear_indices, solution);

        double difference = 0;
        double sum_ref = 0;
        for (int i = 0; i < rows_per_rank; ++i) {
            difference += std::sqrt( (solution[i] - reference_solution[i+row_start_index]) * (solution[i] - reference_solution[i+row_start_index]) );
            sum_ref += std::sqrt( (reference_solution[i+row_start_index]) * (reference_solution[i+row_start_index]) );
        }
        std::cout << "difference/sum_ref " << difference/sum_ref << std::endl;

    }

    HYPRE_IJMatrixDestroy(ij_matrix);
    HYPRE_IJVectorDestroy(ij_rhs);
    HYPRE_IJVectorDestroy(ij_x);
    HYPRE_ParCSRPCGDestroy(pcg_solver);
    delete[] row_ptr_local;
    delete[] row_indices_local;
    delete[] data;
    delete[] row_ptr;
    delete[] col_indices;
    delete[] rhs;
    delete[] reference_solution;
    delete[] diagonal;
    delete[] nnz_per_row_local;
    delete[] linear_indices;
    delete[] solution;

    // HYPRE_Finalize();
    MPI_Finalize();

    return 0;
}
