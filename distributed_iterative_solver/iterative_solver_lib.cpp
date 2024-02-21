#include "iterative_solver_lib.h"

namespace lib_to_compare
{


int solve_petsc(
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
    double *time_taken
){
    int *linear_range = new int[matrix_size];

    for (int i = 0; i < matrix_size; ++i) {
        linear_range[i] = i;
    }


    VecType vec_type = VECMPICUDA;
    MatType mat_type = MATMPIAIJCUSPARSE;

    Vec x;
    PetscCall(VecCreate(MPI_COMM_WORLD,&x));
    PetscCall(VecSetSizes(x, rows_per_rank, matrix_size));
    PetscCall(VecSetType(x, vec_type));
    PetscCall(VecAssemblyBegin(x));
    PetscCall(VecAssemblyEnd(x));

    Vec b;
    PetscCall(VecCreate(MPI_COMM_WORLD,&b));
    PetscCall(VecSetSizes(b, rows_per_rank, matrix_size));
    PetscCall(VecSetType(b, vec_type));    
    PetscCall(VecSetValues(b, rows_per_rank, linear_range+row_start_index, rhs+row_start_index, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));

    Mat A;
    PetscCall(MatCreateMPIAIJWithArrays(MPI_COMM_WORLD, rows_per_rank, rows_per_rank, matrix_size, matrix_size,
        row_ptr_local, col_indices_local, data_local, &A));
    PetscCall(MatSetType(A, mat_type));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));


    // int A_row_rstart, A_row_rend;
    // int A_col_rstart, A_col_rend;
    // int b_rstart, b_rend;
    // int x_rstart, x_rend;
    // PetscCall(MatGetOwnershipRange(A, &A_row_rstart, &A_row_rend));
    // PetscCall(MatGetOwnershipRangeColumn(A, &A_col_rstart, &A_col_rend));
    // PetscCall(VecGetOwnershipRange(b, &b_rstart, &b_rend));
    // PetscCall(VecGetOwnershipRange(x, &x_rstart, &x_rend));

    KSP ksp;
    PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
    PetscCall(KSPSetType(ksp, solver_type));
    PetscCall(KSPSetOperators(ksp, A, A));


    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, preconditioner));
    PetscCall(KSPSetTolerances(ksp, relative_tolerance, absolute_tolerance, divergence_tolerance, max_iterations));

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    *time_taken = -omp_get_wtime();
    PetscCall(KSPSolve(ksp,b,x));
    *time_taken += omp_get_wtime();

    PetscCall(KSPGetIterationNumber(ksp, iterations));
    std::cout << "rank " << rank << " iterations " << *iterations << std::endl;
    sleep(1);
    std::cout << "rank " << rank << " time_taken " << *time_taken << std::endl;

    double residual_norm;
    PetscCall(KSPGetResidualNorm(ksp, &residual_norm));
    if (rank == 0) {
        std::cout << "residual_norm " << residual_norm << std::endl;
    }

    double *solution;
    PetscCall(VecGetArray(x, &solution));

    double difference = 0;
    double sum_ref = 0;
    for (int i = 0; i < rows_per_rank; ++i) {
        difference += std::sqrt( (solution[i] - reference_solution[i+row_start_index]) * (solution[i] - reference_solution[i+row_start_index]) );
        sum_ref += std::sqrt( (reference_solution[i+row_start_index]) * (reference_solution[i+row_start_index]) );
    }
    PetscCall(VecRestoreArray(x, &solution));

    MPI_Allreduce(MPI_IN_PLACE, &difference, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &sum_ref, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(rank == 0){
        std::cout << "difference/sum_ref " << difference/sum_ref << std::endl;
    }

    delete[] linear_range;
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
    PetscCall(MatDestroy(&A));
    PetscCall(KSPDestroy(&ksp));
    return 0;
}


int solve_petsc_precon(
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
    int blocks,
    int *block_sizes
){
    int *linear_range = new int[matrix_size];

    for (int i = 0; i < matrix_size; ++i) {
        linear_range[i] = i;
    }


    VecType vec_type = VECMPICUDA;
    MatType mat_type = MATMPIAIJCUSPARSE;

    Vec x;
    PetscCall(VecCreate(MPI_COMM_WORLD,&x));
    PetscCall(VecSetSizes(x, rows_per_rank, matrix_size));
    PetscCall(VecSetType(x, vec_type));
    PetscCall(VecAssemblyBegin(x));
    PetscCall(VecAssemblyEnd(x));

    Vec b;
    PetscCall(VecCreate(MPI_COMM_WORLD,&b));
    PetscCall(VecSetSizes(b, rows_per_rank, matrix_size));
    PetscCall(VecSetType(b, vec_type));    
    PetscCall(VecSetValues(b, rows_per_rank, linear_range+row_start_index, rhs+row_start_index, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));

    Mat A;
    PetscCall(MatCreateMPIAIJWithArrays(MPI_COMM_WORLD, rows_per_rank, rows_per_rank, matrix_size, matrix_size,
        row_ptr_local, col_indices_local, data_local, &A));
    PetscCall(MatSetType(A, mat_type));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));


    KSP ksp;
    PetscCall(KSPCreate(MPI_COMM_WORLD, &ksp));
    PetscCall(KSPSetType(ksp, solver_type));
    PetscCall(KSPSetOperators(ksp, A, A));


    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, preconditioner));

    // if(preconditioner == PCBJACOBI){
    //     PetscCall(PCBJacobiSetTotalBlocks(pc, blocks, block_sizes));
    // }


    PetscCall(KSPSetTolerances(ksp, relative_tolerance, absolute_tolerance, divergence_tolerance, max_iterations));

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    *time_taken = -omp_get_wtime();
    PetscCall(KSPSolve(ksp,b,x));
    *time_taken += omp_get_wtime();

    PetscCall(KSPGetIterationNumber(ksp, iterations));
    std::cout << "rank " << rank << " iterations " << *iterations << std::endl;
    sleep(1);
    std::cout << "rank " << rank << " time_taken " << *time_taken << std::endl;

    double residual_norm;
    PetscCall(KSPGetResidualNorm(ksp, &residual_norm));
    if (rank == 0) {
        std::cout << "residual_norm " << residual_norm << std::endl;
    }

    double *solution;
    PetscCall(VecGetArray(x, &solution));

    double difference = 0;
    double sum_ref = 0;
    for (int i = 0; i < rows_per_rank; ++i) {
        difference += std::sqrt( (solution[i] - reference_solution[i+row_start_index]) * (solution[i] - reference_solution[i+row_start_index]) );
        sum_ref += std::sqrt( (reference_solution[i+row_start_index]) * (reference_solution[i+row_start_index]) );
    }
    PetscCall(VecRestoreArray(x, &solution));

    MPI_Allreduce(MPI_IN_PLACE, &difference, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &sum_ref, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(rank == 0){
        std::cout << "difference/sum_ref " << difference/sum_ref << std::endl;
    }

    delete[] linear_range;
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
    PetscCall(MatDestroy(&A));
    PetscCall(KSPDestroy(&ksp));
    return 0;
}


void solve_hypre(
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
    int *iterations,
    double *time_taken)
{
    HYPRE_MemoryLocation MEMORY_LOCATION = HYPRE_MEMORY_DEVICE;

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


void solve_ginkgo(
    double *data,
    int *row_ptr,
    int *col_indices,
    double *rhs,
    double *reference_solution,
    int matrix_size,
    int max_iterations,
    double relative_tolerance,
    int *iterations,
    double *time_taken)
{
    
    // @sect3{Type Definitions}
    // Define the needed types. In a parallel program we need to differentiate
    // between global and local indices, thus we have two index types.
    using GlobalIndexType = gko::int64;
    using LocalIndexType = gko::int32;
    // The underlying value type.
    using ValueType = double;
    // As vector type we use the following, which implements a subset of @ref
    // gko::matrix::Dense.
    using dist_vec = gko::experimental::distributed::Vector<ValueType>;
    // As matrix type we simply use the following type, which can read
    // distributed data and be applied to a distributed vector.
    using dist_mtx =
        gko::experimental::distributed::Matrix<ValueType, LocalIndexType,
                                               GlobalIndexType>;
    // We still need a localized vector type to be used as scalars in the
    // advanced apply operations.
    using vec = gko::matrix::Dense<ValueType>;
    // The partition type describes how the rows of the matrices are
    // distributed.
    using part_type =
        gko::experimental::distributed::Partition<LocalIndexType,
                                                  GlobalIndexType>;
    // We can use here the same solver type as you would use in a
    // non-distributed program. Please note that not all solvers support
    // distributed systems at the moment.
    using solver = gko::solver::Cg<ValueType>;
    using schwarz = gko::experimental::distributed::preconditioner::Schwarz<
        ValueType, LocalIndexType, GlobalIndexType>;
    using bj = gko::preconditioner::Jacobi<ValueType, LocalIndexType>;

    ValueType t_init = gko::experimental::mpi::get_walltime();
    // Create an MPI communicator get the rank of the calling process.
    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();

    const auto grid_dim = static_cast<gko::size_type>(matrix_size);
    const auto num_iters = static_cast<gko::size_type>(max_iterations);


    int device_id = gko::experimental::mpi::map_rank_to_device_id(MPI_COMM_WORLD, gko::CudaExecutor::get_num_devices());

    auto exec = gko::CudaExecutor::create(device_id, gko::ReferenceExecutor::create());

    // @sect3{Creating the Distributed Matrix and Vectors}
    // As a first step, we create a partition of the rows. The partition
    // consists of ranges of consecutive rows which are assigned a part-id.
    // These part-ids will be used for the distributed data structures to
    // determine which rows will be stored locally. In this example each rank
    // has (nearly) the same number of rows, so we can use the following
    // specialized constructor. See @ref gko::distributed::Partition for other
    // modes of creating a partition.
    const auto num_rows = grid_dim;
    auto partition = gko::share(part_type::build_from_global_size_uniform(
        exec->get_master(), comm.size(),
        static_cast<GlobalIndexType>(num_rows)));


    // Assemble the matrix using a 3-pt stencil and fill the right-hand-side
    // with a sine value. The distributed matrix supports only constructing an
    // empty matrix of zero size and filling in the values with
    // gko::experimental::distributed::Matrix::read_distributed. Only the data
    // that belongs to the rows by this rank will be assembled.
    gko::matrix_data<ValueType, GlobalIndexType> A_data;
    gko::matrix_data<ValueType, GlobalIndexType> b_data;
    gko::matrix_data<ValueType, GlobalIndexType> x_data;
    A_data.size = {num_rows, num_rows};
    b_data.size = {num_rows, 1};
    x_data.size = {num_rows, 1};
    const auto range_start = partition->get_range_bounds()[rank];
    const auto range_end = partition->get_range_bounds()[rank + 1];


    std::cout << "rank: " << rank << " range_start: " << range_start << " range_end: " << range_end << std::endl;
    for (int i = range_start; i < range_end; i++) {
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            A_data.nonzeros.emplace_back(i, col_indices[j], data[j]);
        }
        b_data.nonzeros.emplace_back(i, 0, rhs[i]);
        x_data.nonzeros.emplace_back(i, 0, gko::zero<ValueType>());
    }

    // Take timings.
    comm.synchronize();
    
    ValueType t_init_end = gko::experimental::mpi::get_walltime();
    std::cout << "rank " << rank << " t_init_end " << t_init_end - t_init << std::endl;
    // Read the matrix data, currently this is only supported on CPU executors.
    // This will also set up the communication pattern needed for the
    // distributed matrix-vector multiplication.
    auto A_host = gko::share(dist_mtx::create(exec->get_master(), comm));
    auto x_host = dist_vec::create(exec->get_master(), comm);
    auto b_host = dist_vec::create(exec->get_master(), comm);
    A_host->read_distributed(A_data, partition);
    b_host->read_distributed(b_data, partition);
    x_host->read_distributed(x_data, partition);
    // After reading, the matrix and vector can be moved to the chosen executor,
    // since the distributed matrix supports SpMV also on devices.
    auto A = gko::share(dist_mtx::create(exec, comm));
    auto x = dist_vec::create(exec, comm);
    auto b = dist_vec::create(exec, comm);
    A->copy_from(A_host);
    b->copy_from(b_host);
    x->copy_from(x_host);

    // Take timings.
    comm.synchronize();
    ValueType t_read_setup_end = gko::experimental::mpi::get_walltime();
    std::cout << "rank " << rank << " t_read_setup_end " << t_read_setup_end - t_init << std::endl;

    // @sect3{Solve the Distributed System}
    // Generate the solver, this is the same as in the non-distributed case.
    // with a local block diagonal preconditioner.

    // Setup the local block diagonal solver factory.
    auto local_solver = gko::share(bj::build().on(exec));

    // Setup the stopping criterion and logger
    const gko::remove_complex<ValueType> reduction_factor{relative_tolerance};
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    auto Ainv = solver::build().with_criteria(
                        gko::stop::Iteration::build().with_max_iters(num_iters),
                        gko::stop::ResidualNorm<ValueType>::build()
                            .with_baseline(gko::stop::mode::rhs_norm).with_reduction_factor(reduction_factor))
                    .on(exec)
                    ->generate(A);
    // Add logger to the generated solver to log the iteration count and
    // residual norm
    Ainv->add_logger(logger);

    // Take timings.
    comm.synchronize();
    cudaDeviceSynchronize();
    std::cout << "rank " << rank << "start solve" << std::endl;
    ValueType t_solver_generate_end = gko::experimental::mpi::get_walltime();
    
    // Apply the distributed solver, this is the same as in the non-distributed
    // case.
    Ainv->apply(b, x);

    // Take timings.
    cudaDeviceSynchronize();
    ValueType t_end = gko::experimental::mpi::get_walltime();
    comm.synchronize();
    std::cout << "rank " << rank << "end solve" << std::endl;

    // Get the residual.
    auto res_norm = gko::as<vec>(logger->get_residual_norm());
    // auto initres_vec = gko::initialize<vec>({0.0}, exec);
    ValueType* result_d = x->get_local_values();
    ValueType* result_h = new ValueType[range_end-range_start];
    cudaMemcpy(result_h, result_d, (range_end-range_start)*sizeof(ValueType), cudaMemcpyDeviceToHost);
    // x->compute_norm2(initres_vec);
    
    // write(std::cout, res_norm);
    // write(std::cout, initres_vec);
    // std::cout << result_h[0] << std::endl;
    // double sum_solution = 0.0;
    // double sum_difference = 0.0;
    // for (int i = range_start; i < range_end; i++) {
    //     sum_solution += reference_solution[i];
    //     sum_difference += std::abs(reference_solution[i] - result_h[i]);
    // }
    // MPI_Allreduce(MPI_IN_PLACE, &sum_solution, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    iterations[0] = logger->get_num_iterations();
    time_taken[0] = t_end - t_solver_generate_end;

    // @sect3{Printing Results}
    // Print the achieved residual norm and timings on rank 0.
    if (comm.rank() == 0) {
        // write(std::cout, res_norm);
        // clang-format off
        std::cout << "Num rows in matrix: " << num_rows
                  << "\nNum ranks: " << comm.size()
                  << "\nIteration count: " << logger->get_num_iterations()
                  << "\nInit time: " << t_init_end - t_init
                  << "\nRead time: " << t_read_setup_end - t_init
                  << "\nSolver generate time: " << t_solver_generate_end - t_read_setup_end
                  << "\nSolver apply time: " << t_end - t_solver_generate_end
                  << "\nTotal time: " << t_end - t_init
                  << std::endl;
        // clang-format on
    }

}

} // namespace lib_to_compare