#include "hypre_implementations_to_compare.h"

namespace hypre_test {

void gpu_solve(
    double *data_local,
    int *row_ptr_local,
    int *col_indices_local,
    double *rhs_local,
    double *reference_solution,
    int row_start_index,
    int row_end_index,
    int rows_per_rank,
    int max_iterations,
    double relative_tolerance,
    double absolute_tolerance,
    HYPRE_MemoryLocation MEMORY_LOCATION,
    int *iterations,
    double *time_taken)
{

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *linear_indices = new int[rows_per_rank];
    double *solution_local = new double[rows_per_rank];
    for (int i = 0; i < rows_per_rank; ++i) {
        linear_indices[i] = row_start_index + i;
        solution_local[i] = 0;
    }
    int *nnz_per_row_local = new int[rows_per_rank];
    for (int i = 0; i < rows_per_rank; ++i) {
        nnz_per_row_local[i] = row_ptr_local[i+1] - row_ptr_local[i];
    }
    int nnz_local = row_ptr_local[rows_per_rank];

    int *nnz_per_row_local_d;
    int *linear_indices_d;
    int *col_indices_local_d;
    double *solution_local_d;
    double *rhs_local_d;
    double *data_local_d;

    cudaMalloc(&nnz_per_row_local_d, rows_per_rank * sizeof(int));
    cudaMalloc(&linear_indices_d, rows_per_rank * sizeof(int));
    cudaMalloc(&col_indices_local_d, nnz_local * sizeof(int));
    cudaMalloc(&solution_local_d, rows_per_rank * sizeof(double));
    cudaMalloc(&rhs_local_d, rows_per_rank * sizeof(double));
    cudaMalloc(&data_local_d, nnz_local * sizeof(double));

    cudaMemcpy(nnz_per_row_local_d, nnz_per_row_local, rows_per_rank * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(linear_indices_d, linear_indices, rows_per_rank * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(col_indices_local_d, col_indices_local, nnz_local * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(solution_local_d, solution_local, rows_per_rank * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(rhs_local_d, rhs_local, rows_per_rank * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(data_local_d, data_local, nnz_local * sizeof(double), cudaMemcpyHostToDevice);


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

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        std::cout << "Creating matrix" << std::endl;
    }
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
        cudaDeviceSynchronize();

        // set matrix
        HYPRE_IJMatrixSetValues(ij_matrix, rows_per_rank, nnz_per_row_local_d, linear_indices_d,
            col_indices_local_d, data_local_d);
        HYPRE_IJMatrixAssemble(ij_matrix);
        HYPRE_IJMatrixGetObject(ij_matrix, (void **) &parcsr_matrix);


        cudaDeviceSynchronize();

        //set x
        HYPRE_IJVectorSetValues(ij_x, rows_per_rank, linear_indices_d, solution_local_d);
        HYPRE_IJVectorAssemble(ij_x);
        HYPRE_IJVectorGetObject(ij_x, (void **) &par_x);

        cudaDeviceSynchronize();

        //set rhs
        HYPRE_IJVectorSetValues(ij_rhs, rows_per_rank, linear_indices_d, rhs_local_d);
        HYPRE_IJVectorAssemble(ij_rhs);
        HYPRE_IJVectorGetObject(ij_rhs, (void **) &par_rhs);
    }
    else{
        // set matrix
        HYPRE_IJMatrixSetValues(ij_matrix, rows_per_rank, nnz_per_row_local, linear_indices,
            col_indices_local, data_local);
        HYPRE_IJMatrixAssemble(ij_matrix);
        HYPRE_IJMatrixGetObject(ij_matrix, (void **) &parcsr_matrix);

        //set x
        HYPRE_IJVectorSetValues(ij_x, rows_per_rank, linear_indices, solution_local);
        HYPRE_IJVectorAssemble(ij_x);
        HYPRE_IJVectorGetObject(ij_x, (void **) &par_x);

        //set rhs
        HYPRE_IJVectorSetValues(ij_rhs, rows_per_rank, linear_indices, rhs_local);
        HYPRE_IJVectorAssemble(ij_rhs);
        HYPRE_IJVectorGetObject(ij_rhs, (void **) &par_rhs);
    }

    if(rank == 0){
        std::cout << "setup" << std::endl;
    }
    cudaDeviceSynchronize();
    // solve
    HYPRE_ParCSRPCGSetup(pcg_solver, parcsr_matrix, par_rhs, par_x);

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        std::cout << "solve" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    cudaDeviceSynchronize();
    double time = -omp_get_wtime();
    HYPRE_ParCSRPCGSolve(pcg_solver, parcsr_matrix, par_rhs, par_x);
    time += omp_get_wtime();
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "rank " << rank << " time " << time << std::endl;
    time_taken[0] = time;

    double final_res_norm;
    HYPRE_PCGGetNumIterations(pcg_solver, iterations);
    HYPRE_PCGGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        std::cout << "Iterations = " << iterations[0] << std::endl;
        std::cout << "Final Relative Residual Norm = " << final_res_norm << std::endl;
    }

    if(MEMORY_LOCATION == HYPRE_MEMORY_DEVICE){
        // get solution_local
        HYPRE_IJVectorGetValues(ij_x, rows_per_rank, linear_indices_d, solution_local_d);
        cudaMemcpy(solution_local, solution_local_d, rows_per_rank * sizeof(double), cudaMemcpyDeviceToHost);
    }
    else{
        // get solution_local
        HYPRE_IJVectorGetValues(ij_x, rows_per_rank, linear_indices, solution_local);
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

    delete[] linear_indices;
    delete[] solution_local;
    delete[] nnz_per_row_local;

    cudaFree(nnz_per_row_local_d);
    cudaFree(linear_indices_d);
    cudaFree(col_indices_local_d);
    cudaFree(solution_local_d);
    cudaFree(rhs_local_d);
    cudaFree(data_local_d);

    HYPRE_IJMatrixDestroy(ij_matrix);
    HYPRE_IJVectorDestroy(ij_rhs);
    HYPRE_IJVectorDestroy(ij_x);
    HYPRE_ParCSRPCGDestroy(pcg_solver);


}


} // namespace hypre_test