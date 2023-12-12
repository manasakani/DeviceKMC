#include "petsc_implementations_to_compare.h"
#include <petscerror.h>
int cpu_solve(
    int rank,
    double *data_local,
    int *row_ptr_local,
    int *col_indices_local,
    double *rhs,
    double *reference_solution,
    int row_start_index,
    int rows_per_rank, 
    int matrix_size,
    int max_iterations,
    KSPType solver_type,
    PCType preconditioner,
    double relative_tolerance,
    double absolute_tolerance,
    double divergence_tolerance,
    int *iterations,
    double *time_taken,
    bool *correct_solution
){
    int *linear_range = new int[matrix_size];

    for (int i = 0; i < matrix_size; ++i) {
        linear_range[i] = i;
    }

    VecType vec_type = VECMPI;
    MatType mat_type = MATMPIAIJ;


    Vec x;
    PetscCall(VecCreate(MPI_COMM_WORLD,&x));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, matrix_size));
    PetscCall(VecSetType(x, vec_type));
    PetscCall(VecAssemblyBegin(x));
    PetscCall(VecAssemblyEnd(x));

    Vec b;
    PetscCall(VecCreate(MPI_COMM_WORLD,&b));
    PetscCall(VecSetSizes(b, PETSC_DECIDE, matrix_size));
    PetscCall(VecSetType(b, vec_type));    
    PetscCall(VecSetValues(b, rows_per_rank, linear_range+row_start_index, rhs+row_start_index, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));

    Mat A;
    PetscCall(MatCreateMPIAIJWithArrays(MPI_COMM_WORLD, rows_per_rank, PETSC_DECIDE, matrix_size, matrix_size,
        row_ptr_local, col_indices_local, data_local, &A));
    PetscCall(MatSetType(A, mat_type));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    int A_row_rstart, A_row_rend;
    int A_col_rstart, A_col_rend;
    int b_rstart, b_rend;
    int x_rstart, x_rend;
    PetscCall(MatGetOwnershipRange(A, &A_row_rstart, &A_row_rend));
    PetscCall(MatGetOwnershipRangeColumn(A, &A_col_rstart, &A_col_rend));
    PetscCall(VecGetOwnershipRange(b, &b_rstart, &b_rend));
    PetscCall(VecGetOwnershipRange(x, &x_rstart, &x_rend));
    // std::cout << "rank " << rank << " A_row_rstart " << A_row_rstart << " A_row_rend " << A_row_rend << std::endl;
    // std::cout << "rank " << rank << " A_col_rstart " << A_col_rstart << " A_col_rend " << A_col_rend << std::endl;
    // std::cout << "rank " << rank << " b_rstart " << b_rstart << " b_rend " << b_rend << std::endl;
    // std::cout << "rank " << rank << " x_rstart " << x_rstart << " x_rend " << x_rend << std::endl;

    KSP ksp;
    PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
    PetscCall(KSPSetType(ksp, solver_type));
    PetscCall(KSPSetOperators(ksp, A, A));
    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, preconditioner));
    PetscCall(KSPSetTolerances(ksp, relative_tolerance, absolute_tolerance, divergence_tolerance, max_iterations));

    *time_taken = -omp_get_wtime();
    PetscCall(KSPSolve(ksp,b,x));
    *time_taken += omp_get_wtime();

    PetscCall(KSPGetIterationNumber(ksp, iterations));
    // PetscCall(KSPGetSolution(ksp,x));
    if (rank == 0) {
        std::cout << "iterations " << *iterations << std::endl;
    }
    std::cout << "rank " << rank << " time_taken " << *time_taken << std::endl;

    double *solution;
    PetscCall(VecGetArray(x, &solution));
    double difference = 0;
    double sum_ref = 0;
    for (int i = 0; i < rows_per_rank; ++i) {
        difference += std::sqrt( (solution[i] - reference_solution[i+row_start_index]) * (solution[i] - reference_solution[i+row_start_index]) );
        sum_ref += std::sqrt( (reference_solution[i+row_start_index]) * (reference_solution[i+row_start_index]) );
    }

    MPI_Reduce(&difference, &difference, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_ref, &sum_ref, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0){
        std::cout << "difference " << difference << std::endl;
        std::cout << "sum_ref " << sum_ref << std::endl;
        std::cout << difference/sum_ref << std::endl;
        if(difference < relative_tolerance * sum_ref + absolute_tolerance){
            *correct_solution = true;
        } else {
            *correct_solution = false;
            
        }        
    }


    PetscCall(VecRestoreArray(x, &solution));


    delete[] linear_range;
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
    PetscCall(MatDestroy(&A));
    PetscCall(KSPDestroy(&ksp));
    return 0;
}

int gpu_solve(
    int rank,
    double *data_local,
    int *row_ptr_local,
    int *col_indices_local,
    double *rhs,
    double *reference_solution,
    int row_start_index,
    int rows_per_rank, 
    int matrix_size,
    int max_iterations,
    KSPType solver_type,
    PCType preconditioner,
    double relative_tolerance,
    double absolute_tolerance,
    double divergence_tolerance,
    int *iterations,
    double *time_taken,
    bool *correct_solution
){
    int *linear_range = new int[matrix_size];

    for (int i = 0; i < matrix_size; ++i) {
        linear_range[i] = i;
    }


    // Device context
    PetscDevice device;
    PetscCall(PetscDeviceCreate(PETSC_DEVICE_CUDA, PETSC_DECIDE, &device));
    PetscDeviceContext dctx;
    PetscCall(PetscDeviceContextCreate(&dctx));
    PetscCall(PetscDeviceContextSetDevice(dctx, device));
    PetscCall(PetscDeviceContextSynchronize(dctx));


    VecType vec_type = VECMPICUDA;
    MatType mat_type = MATMPIAIJCUSPARSE;

    Vec x;
    PetscCall(VecCreate(MPI_COMM_WORLD,&x));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, matrix_size));
    PetscCall(VecSetType(x, vec_type));
    PetscCall(VecAssemblyBegin(x));
    PetscCall(VecAssemblyEnd(x));

    Vec b;
    PetscCall(VecCreate(MPI_COMM_WORLD,&b));
    PetscCall(VecSetSizes(b, PETSC_DECIDE, matrix_size));
    PetscCall(VecSetType(b, vec_type));    
    PetscCall(VecSetValues(b, rows_per_rank, linear_range+row_start_index, rhs+row_start_index, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));

    Mat A;
    PetscCall(MatCreateMPIAIJWithArrays(MPI_COMM_WORLD, rows_per_rank, PETSC_DECIDE, matrix_size, matrix_size,
        row_ptr_local, col_indices_local, data_local, &A));
    PetscCall(MatSetType(A, mat_type));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    int A_row_rstart, A_row_rend;
    int A_col_rstart, A_col_rend;
    int b_rstart, b_rend;
    int x_rstart, x_rend;
    PetscCall(MatGetOwnershipRange(A, &A_row_rstart, &A_row_rend));
    PetscCall(MatGetOwnershipRangeColumn(A, &A_col_rstart, &A_col_rend));
    PetscCall(VecGetOwnershipRange(b, &b_rstart, &b_rend));
    PetscCall(VecGetOwnershipRange(x, &x_rstart, &x_rend));
    // std::cout << "rank " << rank << " A_row_rstart " << A_row_rstart << " A_row_rend " << A_row_rend << std::endl;
    // std::cout << "rank " << rank << " A_col_rstart " << A_col_rstart << " A_col_rend " << A_col_rend << std::endl;
    // std::cout << "rank " << rank << " b_rstart " << b_rstart << " b_rend " << b_rend << std::endl;
    // std::cout << "rank " << rank << " x_rstart " << x_rstart << " x_rend " << x_rend << std::endl;

    KSP ksp;
    PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
    PetscCall(KSPSetType(ksp, solver_type));
    PetscCall(KSPSetOperators(ksp, A, A));
    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, preconditioner));
    PetscCall(KSPSetTolerances(ksp, relative_tolerance, absolute_tolerance, divergence_tolerance, max_iterations));

    *time_taken = -omp_get_wtime();
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCall(KSPSolve(ksp,b,x));
    PetscCall(PetscDeviceContextSynchronize(dctx));
    *time_taken += omp_get_wtime();

    PetscCall(KSPGetIterationNumber(ksp, iterations));
    // PetscCall(KSPGetSolution(ksp,x));
    if (rank == 0) {
        std::cout << "iterations " << *iterations << std::endl;
    }
    std::cout << "rank " << rank << " time_taken " << *time_taken << std::endl;

    double *solution;
    PetscCall(VecGetArray(x, &solution));
    double difference = 0;
    double sum_ref = 0;
    double sum_b = 0;
    for (int i = 0; i < rows_per_rank; ++i) {
        difference += std::sqrt( (solution[i] - reference_solution[i+row_start_index]) * (solution[i] - reference_solution[i+row_start_index]) );
        sum_ref += std::sqrt( (reference_solution[i+row_start_index]) * (reference_solution[i+row_start_index]) );
        sum_b += std::sqrt( (rhs[i+row_start_index]) * (rhs[i+row_start_index]) );
    }


    MPI_Reduce(&difference, &difference, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_ref, &sum_ref, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_b, &sum_b, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0){
        std::cout << "difference " << difference << std::endl;
        std::cout << "sum_ref " << sum_ref << std::endl;
        std::cout << "difference/sum_ref " << difference/sum_ref << std::endl;
        std::cout << "sum_b " << sum_b << std::endl;
        if(difference < relative_tolerance * sum_ref + absolute_tolerance){
            *correct_solution = true;
        } else {
            *correct_solution = false;
            
        }        
    }


    PetscCall(VecRestoreArray(x, &solution));


    delete[] linear_range;
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
    PetscCall(MatDestroy(&A));
    PetscCall(KSPDestroy(&ksp));
    return 0;
}