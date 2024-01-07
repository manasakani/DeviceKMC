#include <iostream>
#include <string>
#include <mpi.h>
#include <cuda_runtime.h>
#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_utilities.h>
#include <HYPRE_krylov.h>
#include <cmath>


int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    HYPRE_Initialize();
    
    std::cout << "provided " << provided << std::endl;

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "Hello World from rank " << rank << std::endl;

    int matrix_size = 1025;
    int nnz = matrix_size;  


    int max_iterations = 5000;
    double relative_tolerance = 1e-15;
    double absolute_tolerance = 1e-30;
    HYPRE_MemoryLocation MEMORY_LOCATION = HYPRE_MEMORY_DEVICE;

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
    double *solution = new double[matrix_size];

    for(int i = 0; i < matrix_size; ++i){
        rhs[i] = i;
        data[i] = i+1;
        reference_solution[i] = rhs[i] / data[i];
        
        col_indices[i] = i;
        solution[i] = 0;
    }
    for(int i = 0; i < matrix_size+1; ++i){
        row_ptr[i] = i;
    }


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
    double *solution_local = solution + row_start_index;
    double *rhs_local = rhs + row_start_index;

    int *nnz_per_row_local = new int[rows_per_rank];
    int *linear_indices_local = new int[rows_per_rank];    
    for (int i = 0; i < rows_per_rank; ++i) {
        nnz_per_row_local[i] = row_ptr_local[i+1] - row_ptr_local[i];
        linear_indices_local[i] = row_start_index + i;
    }


    // gpu memory
    double *data_d;
    int *row_ptr_d;
    int *col_indices_d;
    double *rhs_d;
    double *solution_d;
    int *nnz_per_row_local_d;
    int *linear_indices_local_d;
    int *col_indices_local_d;
    cudaMalloc(&data_d, nnz * sizeof(double));
    cudaMalloc(&row_ptr_d, (matrix_size+1) * sizeof(int));
    cudaMalloc(&col_indices_d, nnz * sizeof(int));
    cudaMalloc(&rhs_d, matrix_size * sizeof(double));
    cudaMalloc(&solution_d, matrix_size * sizeof(double));
    cudaMalloc(&nnz_per_row_local_d, rows_per_rank * sizeof(int));
    cudaMalloc(&linear_indices_local_d, rows_per_rank * sizeof(int));
    cudaMalloc(&col_indices_local_d, nnz_local * sizeof(int));

    cudaMemcpy(data_d, data, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(row_ptr_d, row_ptr, (matrix_size+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(col_indices_d, col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(rhs_d, rhs, matrix_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(nnz_per_row_local_d, nnz_per_row_local, rows_per_rank * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(linear_indices_local_d, linear_indices_local, rows_per_rank * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(col_indices_local_d, col_indices_local, nnz_local * sizeof(int), cudaMemcpyHostToDevice);

    double *data_local_d = data_d + nnz_start_index;    
    double *solution_local_d = solution_d + row_start_index;
    double *rhs_local_d = rhs_d + row_start_index;

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
    HYPRE_IJMatrixInitialize_v2(ij_matrix, MEMORY_LOCATION);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, jlower, jupper, &ij_rhs);
    HYPRE_IJVectorSetObjectType(ij_rhs, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize_v2(ij_rhs, MEMORY_LOCATION);
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, jlower, jupper, &ij_x);
    HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize_v2(ij_x, MEMORY_LOCATION);

    HYPRE_Solver pcg_solver;
    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &pcg_solver);
    HYPRE_PCGSetMaxIter(pcg_solver, max_iterations);
    HYPRE_PCGSetTol(pcg_solver, relative_tolerance);
    HYPRE_PCGSetAbsoluteTol(pcg_solver, absolute_tolerance);
    HYPRE_PCGSetTwoNorm(pcg_solver, 1);



    if(MEMORY_LOCATION == HYPRE_MEMORY_DEVICE){
        std::cout << "set matrix" << std::endl;
        cudaDeviceSynchronize();

        // set matrix
        HYPRE_IJMatrixSetValues(ij_matrix, rows_per_rank, nnz_per_row_local_d, linear_indices_local_d,
            col_indices_local_d, data_local_d);
        HYPRE_IJMatrixAssemble(ij_matrix);
        HYPRE_IJMatrixGetObject(ij_matrix, (void **) &parcsr_matrix);


        std::cout << "set x" << std::endl;
        cudaDeviceSynchronize();

        //set x
        HYPRE_IJVectorSetValues(ij_x, rows_per_rank, linear_indices_local_d, solution_local_d);
        HYPRE_IJVectorAssemble(ij_x);
        HYPRE_IJVectorGetObject(ij_x, (void **) &par_x);

        std::cout << "set rhs" << std::endl;
        cudaDeviceSynchronize();

        //set rhs
        HYPRE_IJVectorSetValues(ij_rhs, rows_per_rank, linear_indices_local_d, rhs_local_d);
        HYPRE_IJVectorAssemble(ij_rhs);
        HYPRE_IJVectorGetObject(ij_rhs, (void **) &par_rhs);
    }
    else{
        // set matrix
        HYPRE_IJMatrixSetValues(ij_matrix, rows_per_rank, nnz_per_row_local, linear_indices_local,
            col_indices_local, data_local);
        HYPRE_IJMatrixAssemble(ij_matrix);
        HYPRE_IJMatrixGetObject(ij_matrix, (void **) &parcsr_matrix);

        //set x
        HYPRE_IJVectorSetValues(ij_x, rows_per_rank, linear_indices_local, solution_local);
        HYPRE_IJVectorAssemble(ij_x);
        HYPRE_IJVectorGetObject(ij_x, (void **) &par_x);

        //set rhs
        HYPRE_IJVectorSetValues(ij_rhs, rows_per_rank, linear_indices_local, rhs_local);
        HYPRE_IJVectorAssemble(ij_rhs);
        HYPRE_IJVectorGetObject(ij_rhs, (void **) &par_rhs);
    }


    std::cout << "setup" << std::endl;
    cudaDeviceSynchronize();
    // solve
    HYPRE_ParCSRPCGSetup(pcg_solver, parcsr_matrix, par_rhs, par_x);

    std::cout << "solve" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    cudaDeviceSynchronize();
    double time_taken = -omp_get_wtime();
    HYPRE_ParCSRPCGSolve(pcg_solver, parcsr_matrix, par_rhs, par_x);
    time_taken += omp_get_wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "rank " << rank << " time_taken " << time_taken << std::endl;

    int num_iterations;
    double final_res_norm;
    HYPRE_PCGGetNumIterations(pcg_solver, &num_iterations);
    HYPRE_PCGGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);
    std::cout << "Iterations = " << num_iterations << std::endl;
    std::cout << "Final Relative Residual Norm = " << final_res_norm << std::endl;

    // get solution
    if(MEMORY_LOCATION == HYPRE_MEMORY_DEVICE){
        HYPRE_IJVectorGetValues(ij_x, rows_per_rank, linear_indices_local_d, solution_d);
        cudaMemcpy(solution_local, solution_local_d, rows_per_rank * sizeof(double), cudaMemcpyDeviceToHost);
    }
    else{
        HYPRE_IJVectorGetValues(ij_x, rows_per_rank, linear_indices_local, solution_local);
    }

        
    double difference = 0;
    double sum_ref = 0;
    for (int i = 0; i < rows_per_rank; ++i) {
        difference += std::sqrt( (solution_local[i] - reference_solution[i+row_start_index]) * (solution_local[i] - reference_solution[i+row_start_index]) );
        sum_ref += std::sqrt( (reference_solution[i+row_start_index]) * (reference_solution[i+row_start_index]) );
    }
    MPI_Allreduce(MPI_IN_PLACE, &difference, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &sum_ref, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(rank == 0){
        std::cout << "difference/sum_ref " << difference/sum_ref << std::endl;
    }

    HYPRE_IJMatrixDestroy(ij_matrix);
    HYPRE_IJVectorDestroy(ij_rhs);
    HYPRE_IJVectorDestroy(ij_x);
    HYPRE_ParCSRPCGDestroy(pcg_solver);
    delete[] row_ptr_local;
    delete[] data;
    delete[] row_ptr;
    delete[] col_indices;
    delete[] rhs;
    delete[] reference_solution;
    delete[] nnz_per_row_local;
    delete[] linear_indices_local;
    delete[] solution;

    cudaFree(data_d);
    cudaFree(row_ptr_d);
    cudaFree(col_indices_d);
    cudaFree(rhs_d);
    cudaFree(solution_d);
    cudaFree(nnz_per_row_local_d);
    cudaFree(linear_indices_local_d);
    cudaFree(col_indices_local_d);

    HYPRE_Finalize();
    MPI_Finalize();


    return 0;
}
