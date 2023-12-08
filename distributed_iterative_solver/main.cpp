#include <iostream>
#include <petscksp.h>
#include <petscsys.h>
#include <string>
#include "utils.h"
#include <mpi.h>


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
    double *solution_ref = new double[matrix_size];
    int *linear_range = new int[matrix_size];
    double *solution;

    for (int i = 0; i < matrix_size; ++i) {
        linear_range[i] = i;
    }

    std::string base_path = "test_data";
    std::string data_filename = base_path + "/A_data0.bin";
    std::string row_ptr_filename = base_path + "/A_row_ptr0.bin";
    std::string col_indices_filename = base_path + "/A_col_indices0.bin";
    std::string rhs_filename = base_path + "/rhs0.bin";
    std::string solution_filename = base_path + "/solution0.bin";

    load_binary_array<double>(data_filename, data, nnz);
    load_binary_array<int>(row_ptr_filename, row_ptr, matrix_size+1);
    load_binary_array<int>(col_indices_filename, col_indices, nnz);
    load_binary_array<double>(rhs_filename, rhs, matrix_size);
    load_binary_array<double>(solution_filename, solution_ref, matrix_size);


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




    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Loaded data" << std::endl;
    }

    Vec x;
    CHKERRQ(VecCreate(MPI_COMM_WORLD,&x));
    CHKERRQ(VecSetSizes(x, PETSC_DECIDE, matrix_size));
    CHKERRQ(VecSetType(x, VECMPI));
    CHKERRQ(VecAssemblyBegin(x));
    CHKERRQ(VecAssemblyEnd(x));

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Created x" << std::endl;
    }


    Vec x_ref;
    CHKERRQ(VecCreate(MPI_COMM_WORLD,&x_ref));
    CHKERRQ(VecSetSizes(x_ref, PETSC_DECIDE, matrix_size));
    CHKERRQ(VecSetType(x_ref, VECMPI));
    CHKERRQ(VecSetValues(x_ref, rows_per_rank, linear_range+row_start_index, solution_ref+row_start_index, INSERT_VALUES));
    CHKERRQ(VecAssemblyBegin(x_ref));
    CHKERRQ(VecAssemblyEnd(x_ref));

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Created x_ref" << std::endl;
    }

    Vec b;
    CHKERRQ(VecCreate(MPI_COMM_WORLD,&b));
    CHKERRQ(VecSetSizes(b, PETSC_DECIDE, matrix_size));
    CHKERRQ(VecSetType(b, VECMPI));    
    CHKERRQ(VecSetValues(b, rows_per_rank, linear_range+row_start_index, rhs+row_start_index, INSERT_VALUES));
    CHKERRQ(VecAssemblyBegin(b));
    CHKERRQ(VecAssemblyEnd(b));

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Created b" << std::endl;
    }




    Mat A_local;
    MatCreateMPIAIJWithArrays(PETSC_COMM_SELF, rows_per_rank, PETSC_DECIDE, matrix_size, matrix_size,
        row_ptr_local, col_indices_local, data_local, &A_local);
    MatAssemblyBegin(A_local, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A_local, MAT_FINAL_ASSEMBLY);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Created A_local" << std::endl;
    }

    Mat A;
    MatCreateMPIAIJWithArrays(MPI_COMM_WORLD, rows_per_rank, PETSC_DECIDE, matrix_size, matrix_size,
        row_ptr_local, col_indices_local, data_local, &A);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Created A" << std::endl;
    }

    int A_row_rstart, A_row_rend;
    int A_col_rstart, A_col_rend;
    int b_rstart, b_rend;
    int x_ref_rstart, x_ref_rend;
    CHKERRQ(MatGetOwnershipRange(A, &A_row_rstart, &A_row_rend));
    CHKERRQ(MatGetOwnershipRangeColumn(A, &A_col_rstart, &A_col_rend));
    CHKERRQ(VecGetOwnershipRange(b, &b_rstart, &b_rend));
    CHKERRQ(VecGetOwnershipRange(x_ref, &x_ref_rstart, &x_ref_rend));
    std::cout << "rank " << rank << " A_row_rstart " << A_row_rstart << " A_row_rend " << A_row_rend << std::endl;
    std::cout << "rank " << rank << " A_col_rstart " << A_col_rstart << " A_col_rend " << A_col_rend << std::endl;
    std::cout << "rank " << rank << " b_rstart " << b_rstart << " b_rend " << b_rend << std::endl;
    std::cout << "rank " << rank << " x_ref_rstart " << x_ref_rstart << " x_ref_rend " << x_ref_rend << std::endl;

    KSP ksp;
    CHKERRQ(KSPCreate(MPI_COMM_WORLD, &ksp));
    CHKERRQ(KSPSetOperators(ksp, A, A));
    PC pc;
    CHKERRQ(KSPGetPC(ksp, &pc));
    CHKERRQ(PCSetType(pc, PCJACOBI));

    int maxits = 10000;
    double rtol = 1e-15;
    double atol = 1e-15;
    double dtol = 1e+10;
    CHKERRQ(KSPSetTolerances(ksp,rtol,atol,dtol,maxits));


    CHKERRQ(KSPSolve(ksp,b,x));

    int iterations;
    CHKERRQ(KSPGetIterationNumber(ksp, &iterations));
    // CHKERRQ(KSPGetSolution(ksp,x));
    if (rank == 0) {
        std::cout << "iterations " << iterations << std::endl;
    }
    CHKERRQ(VecGetArray(x, &solution));


    double difference = 0;
    double sum_ref = 0;
    for (int i = 0; i < rows_per_rank; ++i) {
        difference += std::sqrt( (solution[i] - solution_ref[i+row_start_index]) * (solution[i] - solution_ref[i+row_start_index]) );
        sum_ref += std::sqrt( (solution_ref[i+row_start_index]) * (solution_ref[i+row_start_index]) );
    }
    std::cout << "rank " << rank << " difference " << difference << std::endl;
    std::cout << "rank " << rank << " sum_ref " << sum_ref << std::endl;
    std::cout << "rank " << rank << " difference/sum_ref " << difference/sum_ref << std::endl;





    CHKERRQ(VecRestoreArray(x, &solution));
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&x_ref));
    CHKERRQ(VecDestroy(&b));
    CHKERRQ(MatDestroy(&A));
    CHKERRQ(MatDestroy(&A_local));
    CHKERRQ(KSPDestroy(&ksp));

    delete[] data;
    delete[] row_ptr;
    delete[] col_indices;
    delete[] rhs;
    delete[] solution_ref;
    delete[] linear_range;
    delete[] row_ptr_local;
    delete[] row_indices_local;

    CHKERRQ(PetscFinalize());
    return 0;
}
