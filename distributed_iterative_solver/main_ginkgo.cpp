#include <iostream>
#include <string>
#include "utils.h"
#include <mpi.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_runtime.h>
#include "utils_gpu.h"
#include <cublas_v2.h>

#include <ginkgo/ginkgo.hpp>
#include <map>


int main(int argc, char* argv[])
{
    // @sect3{Initialize the MPI environment}
    // Since this is an MPI program, we need to initialize and finalize
    // MPI at the begin and end respectively of our program. This can be easily
    // done with the following helper construct that uses RAII to automate the
    // initialization and finalization.
    const gko::experimental::mpi::environment env(argc, argv);
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

    // Create an MPI communicator get the rank of the calling process.
    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();

    // @sect3{User Input Handling}
    // User input settings:
    // - The executor, defaults to reference.
    // - The number of grid points, defaults to 100.
    // - The number of iterations, defaults to 1000.
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " [executor] [num_iterations] "
                      << std::endl;
        }
        std::exit(-1);
    }

    ValueType t_init = gko::experimental::mpi::get_walltime();

    std::string data_path = "/usr/scratch/mont-fort17/almaeder/kmc_80k/system_K";
    int step = 0;
    std::string data_filename = data_path + "/A_data"+std::to_string(step)+".bin";
    std::string row_ptr_filename = data_path + "/A_row_ptr"+std::to_string(step)+".bin";
    std::string col_indices_filename = data_path + "/A_col_indices"+std::to_string(step)+".bin";
    std::string rhs_filename = data_path + "/A_rhs"+std::to_string(step)+".bin";
    std::string solution_filename = data_path + "/solution"+std::to_string(step)+".bin";


    int matrix_size = 70630;
    int nnz = 1719652;    

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const auto grid_dim =
        static_cast<gko::size_type>(matrix_size);
    const auto num_iters =
        static_cast<gko::size_type>(argc >= 3 ? std::atoi(argv[2]) : 1000);

    const std::map<std::string,
                   std::function<std::shared_ptr<gko::Executor>(MPI_Comm)>>
        executor_factory_mpi{
            {"reference",
             [](MPI_Comm) { return gko::ReferenceExecutor::create(); }},
            {"omp", [](MPI_Comm) { return gko::OmpExecutor::create(); }},
            {"cuda",
             [](MPI_Comm comm) {
                 int device_id = gko::experimental::mpi::map_rank_to_device_id(
                     comm, gko::CudaExecutor::get_num_devices());
                 return gko::CudaExecutor::create(
                     device_id, gko::ReferenceExecutor::create());
             }},
            {"hip",
             [](MPI_Comm comm) {
                 int device_id = gko::experimental::mpi::map_rank_to_device_id(
                     comm, gko::HipExecutor::get_num_devices());
                 return gko::HipExecutor::create(
                     device_id, gko::ReferenceExecutor::create());
             }},
            {"dpcpp", [](MPI_Comm comm) {
                 int device_id = 0;
                 if (gko::DpcppExecutor::get_num_devices("gpu")) {
                     device_id = gko::experimental::mpi::map_rank_to_device_id(
                         comm, gko::DpcppExecutor::get_num_devices("gpu"));
                 } else if (gko::DpcppExecutor::get_num_devices("cpu")) {
                     device_id = gko::experimental::mpi::map_rank_to_device_id(
                         comm, gko::DpcppExecutor::get_num_devices("cpu"));
                 } else {
                     throw std::runtime_error("No suitable DPC++ devices");
                 }
                 return gko::DpcppExecutor::create(
                     device_id, gko::ReferenceExecutor::create());
             }}};

    auto exec = executor_factory_mpi.at(executor_string)(MPI_COMM_WORLD);

    // @sect3{Creating the Distributed Matrix and Vectors}
    // As a first step, we create a partition of the rows. The partition
    // consists of ranges of consecutive rows which are assigned a part-id.
    // These part-ids will be used for the distributed data structures to
    // determine which rows will be stored locally. In this example each rank
    // has (nearly) the same number of rows, so we can use the following
    // specialized constructor. See @ref gko::distributed::Partition for other
    // modes of creating a partition.
    const auto num_rows = grid_dim;
    std::cout << "num_rows: " << num_rows << std::endl;
    auto partition = gko::share(part_type::build_from_global_size_uniform(
        exec->get_master(), comm.size(),
        static_cast<GlobalIndexType>(num_rows)));

    // load custom matrix
    double *data = new double[nnz];
    int *row_ptr = new int[matrix_size+1];
    int *col_indices = new int[nnz];
    double *rhs = new double[matrix_size];
    double *reference_solution = new double[matrix_size];

    load_binary_array<double>(data_filename, data, nnz);
    load_binary_array<int>(row_ptr_filename, row_ptr, matrix_size+1);
    load_binary_array<int>(col_indices_filename, col_indices, nnz);
    load_binary_array<double>(rhs_filename, rhs, matrix_size);
    load_binary_array<double>(solution_filename, reference_solution, matrix_size);


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
        // if (i > 0) {
        //     A_data.nonzeros.emplace_back(i, i - 1, -1);
        // }
        // A_data.nonzeros.emplace_back(i, i, 2);
        // if (i < grid_dim - 1) {
        //     A_data.nonzeros.emplace_back(i, i + 1, -1);
        // }
        // b_data.nonzeros.emplace_back(i, 0, std::sin(i * 0.01));
        x_data.nonzeros.emplace_back(i, 0, gko::zero<ValueType>());
    }

    // Take timings.
    comm.synchronize();
    ValueType t_init_end = gko::experimental::mpi::get_walltime();

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


    // @sect3{Solve the Distributed System}
    // Generate the solver, this is the same as in the non-distributed case.
    // with a local block diagonal preconditioner.

    // Setup the local block diagonal solver factory.
    auto local_solver = gko::share(bj::build().on(exec));

    // Setup the stopping criterion and logger
    const gko::remove_complex<ValueType> reduction_factor{1e-16};
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    auto Ainv = solver::build()
                    .with_preconditioner(
                        schwarz::build().with_local_solver(local_solver))
                    .with_criteria(
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
    ValueType t_solver_generate_end = gko::experimental::mpi::get_walltime();

    // Apply the distributed solver, this is the same as in the non-distributed
    // case.
    Ainv->apply(b, x);

    // Take timings.
    comm.synchronize();
    ValueType t_end = gko::experimental::mpi::get_walltime();

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
    double sum_solution = 0.0;
    double sum_difference = 0.0;
    for (int i = range_start; i < range_end; i++) {
        sum_solution += reference_solution[i];
        sum_difference += std::abs(reference_solution[i] - result_h[i]);
    }
    if(rank == 0){
        MPI_Reduce(MPI_IN_PLACE, &sum_solution, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(MPI_IN_PLACE, &sum_difference, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    else{
        MPI_Reduce(&sum_solution, &sum_solution, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&sum_difference, &sum_difference, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }


    // @sect3{Printing Results}
    // Print the achieved residual norm and timings on rank 0.
    if (comm.rank() == 0) {
        // write(std::cout, res_norm);
        // clang-format off
        std::cout << sum_difference/sum_solution << std::endl;
        std::cout << "\nNum rows in matrix: " << num_rows
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

    delete[] data;
    delete[] row_ptr;
    delete[] col_indices;
    delete[] rhs;
    delete[] reference_solution;

}