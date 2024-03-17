#include <iostream>
#include <string>
#include "utils.h"
#include <mpi.h>
#include <cuda_runtime.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "utils_gpu.h"
#include <cublas_v2.h>
#include "../dist_iterative/dist_conjugate_gradient.h"
#include "../dist_iterative/dist_spmv.h"


template <void (*distributed_spmv)(Distributed_matrix&, Distributed_vector&, cusparseDnVecDescr_t&, cudaStream_t&, cusparseHandle_t&)>
void test_preconditioned(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    double *time_taken,
    double *diag_inv_d)
{
    MPI_Barrier(comm);

    std::printf("PCG test starts\n");
    

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    // prepare for allgatherv
    int counts[size];
    int displacements[size];
    int rows_per_rank = matrix_size / size;    
    split_matrix(matrix_size, size, counts, displacements);

    int row_start_index = displacements[rank];
    rows_per_rank = counts[rank];

    int *row_indptr_local_h = new int[rows_per_rank+1];
    double *r_local_h = new double[rows_per_rank];
    for (int i = 0; i < rows_per_rank+1; ++i) {
        row_indptr_local_h[i] = row_indptr_h[i+row_start_index] - row_indptr_h[row_start_index];
    }
    for (int i = 0; i < rows_per_rank; ++i) {
        r_local_h[i] = r_h[i+row_start_index];
    }
    int nnz_local = row_indptr_local_h[rows_per_rank];
    int *col_indices_local_h = new int[nnz_local];
    double *data_local_h = new double[nnz_local];

    for (int i = 0; i < nnz_local; ++i) {
        col_indices_local_h[i] = col_indices_h[i+row_indptr_h[row_start_index]];
        data_local_h[i] = data_h[i+row_indptr_h[row_start_index]];
    }

    // create distributed matrix
    std::printf("Creating distributed matrix\n");
    Distributed_matrix A_distributed(
        matrix_size,
        nnz_local,
        counts,
        displacements,
        col_indices_local_h,
        row_indptr_local_h,
        data_local_h,
        comm
    );
    std::printf("Creating distributed vector\n");
    Distributed_vector p_distributed(
        matrix_size,
        counts,
        displacements,
        A_distributed.number_of_neighbours,
        A_distributed.neighbours,
        comm
    );
    double *r_local_d;
    double *x_local_d;
    cudaMalloc(&r_local_d, rows_per_rank * sizeof(double));
    cudaMalloc(&x_local_d, rows_per_rank * sizeof(double));
    cudaMemcpy(r_local_d, r_local_h, rows_per_rank * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_local_d, starting_guess_h + row_start_index,
        rows_per_rank * sizeof(double), cudaMemcpyHostToDevice);


    MPI_Barrier(comm);
    cudaDeviceSynchronize();
    time_taken[0] = MPI_Wtime();

    double *diag_inv_local_d = diag_inv_d + row_start_index;
    iterative_solver::conjugate_gradient_jacobi<dspmv::gpu_packing>(
        A_distributed,
        p_distributed,
        r_local_d,
        x_local_d,
        diag_inv_local_d,
        relative_tolerance,
        max_iterations,
        comm);

    // iterative_solver::conjugate_gradient<dspmv::gpu_packing>(
    //     A_distributed,
    //     p_distributed,
    //     r_local_d,
    //     x_local_d,
    //     relative_tolerance,
    //     max_iterations,
    //     comm);

    time_taken[0] = MPI_Wtime() - time_taken[0];
    std::cout << "rank " << rank << " time_taken " << time_taken[0] << std::endl;

    //copy solution to host
    cudaErrchk(cudaMemcpy(r_local_h, x_local_d, rows_per_rank * sizeof(double), cudaMemcpyDeviceToHost));


    double difference = 0;
    double sum_ref = 0;
    for (int i = 0; i < rows_per_rank; ++i) {
        difference += std::sqrt( (r_local_h[i] - reference_solution[i+row_start_index]) * (r_local_h[i] - reference_solution[i+row_start_index]) );
        sum_ref += std::sqrt( (reference_solution[i+row_start_index]) * (reference_solution[i+row_start_index]) );
    }
    MPI_Allreduce(MPI_IN_PLACE, &difference, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &sum_ref, 1, MPI_DOUBLE, MPI_SUM, comm);
    if(rank == 0){
        std::cout << "difference/sum_ref " << difference/sum_ref << std::endl;
    }

    delete[] row_indptr_local_h;
    delete[] r_local_h;
    delete[] col_indices_local_h;
    delete[] data_local_h;
    cudaFree(r_local_d);
    cudaFree(x_local_d);

    MPI_Barrier(comm);
}

template 
void test_preconditioned<dspmv::gpu_packing>(
    double *data_h,
    int *col_indices_h,
    int *row_indptr_h,
    double *r_h,
    double *reference_solution,
    double *starting_guess_h,
    int matrix_size,
    double relative_tolerance,
    int max_iterations,
    MPI_Comm comm,
    double *time_taken,
    double *diagonal_d);


int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cudaError_t set_device_error = cudaSetDevice(0);
    std::cout << "rank " << rank << " set_device_error " << set_device_error << std::endl;

    int matsize = 7;
    std::string data_path = "/scratch/snx3000/amaeder/"+std::to_string(matsize)+"k_piz_daint_data";
    //std::string save_path ="/scratch/snx3000/amaeder/measurements/self_preconditioned_scaling_measurement/";
    std::string save_path ="/scratch/snx3000/amaeder/measurements/own_260/";
    data_path = "/usr/scratch/mont-fort17/almaeder/kmc_"+std::to_string(matsize)+"k/system_K";
    // data_path = "/usr/scratch/mont-fort23/almaeder/kmc_matrices";

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

    int start_up_measurements = 2;
    int true_number_of_measurements = 5;
    int number_of_measurements = start_up_measurements + true_number_of_measurements;
    int number_of_kmc_steps = 1;

    int max_iterations = 10000;
    double relative_tolerance = 1e-11;

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



    // int iterations[number_of_kmc_steps];

    double *starting_guess = new double[matrix_size];
    for (int i = 0; i < matrix_size; ++i) {
        starting_guess[i] = 0.0;
    }


    for(int step = 0; step < number_of_kmc_steps; step++){
        std::cout << "rank " << rank << " step " << step << std::endl;
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

        // extract_diagonal_inv_sqrt(
        //     data_d,
        //     col_indices_d,
        //     row_ptr_d,
        //     diagonal_d,
        //     matrix_size
        // );
        // symmetric_precondition_matrix_gpu(
        //     data_d,
        //     col_indices_d,
        //     row_ptr_d,
        //     diagonal_d,
        //     matrix_size
        // );
        // precondition_vector_gpu(
        //     rhs_d,
        //     diagonal_d,
        //     matrix_size
        // );
        // unpreecondition_vector_gpu(
        //     reference_solution_d,
        //     diagonal_d,
        //     matrix_size
        // );

        extract_diagonal_inv(
            data_d,
            col_indices_d,
            row_ptr_d,
            diagonal_d,
            matrix_size
        );

        cudaMemcpy(rhs, rhs_d, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(reference_solution, reference_solution_d, matrix_size * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(data, data_d, nnz * sizeof(double), cudaMemcpyDeviceToHost);


        int iteration;
        std::cout << "rank " << rank << " starting measurement" << std::endl;
        double times_gpu_packing[number_of_measurements];
        double times_gpu_packing_cam[number_of_measurements];


        for(int measurement = 0; measurement < number_of_measurements; measurement++){
            MPI_Barrier(MPI_COMM_WORLD);
            std::cout << "rank " << rank << " gpu_packing " << measurement << std::endl;
            test_preconditioned<dspmv::gpu_packing>(
                data,
                col_indices,
                row_ptr,
                rhs,
                reference_solution,
                starting_guess,
                matrix_size,
                relative_tolerance,
                max_iterations,
                MPI_COMM_WORLD,
                &times_gpu_packing[measurement],
                diagonal_d
            );
        }


        // for(int measurement = 0; measurement < number_of_measurements; measurement++){
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     std::cout << "rank " << rank << " gpu_packing_cam " << measurement << std::endl;
        //     test_preconditioned<dspmv::gpu_packing_cam>(
        //         data,
        //         col_indices,
        //         row_ptr,
        //         rhs,
        //         reference_solution,
        //         starting_guess,
        //         matrix_size,
        //         relative_tolerance,
        //         max_iterations,
        //         MPI_COMM_WORLD,
        //         &times_gpu_packing[measurement]
        //     );
        // }


        // std::string path_solve_gpu_packing = get_filename(save_path, "solve_gpu_packing", number_of_kmc_steps, size, rank);
        // std::string path_solve_gpu_packing_cam = get_filename(save_path, "solve_gpu_packing_cam", number_of_kmc_steps, size, rank);

        // save_measurements(path_solve_gpu_packing,
        //     times_gpu_packing + start_up_measurements,
        //     true_number_of_measurements, true);
        // save_measurements(path_solve_gpu_packing_cam,
        //     times_gpu_packing_cam + start_up_measurements,
        //     true_number_of_measurements, true);

    }


    delete[] data;
    delete[] row_ptr;
    delete[] col_indices;
    delete[] rhs;
    delete[] reference_solution;
    delete[] starting_guess;

    cudaFree(data_d);
    cudaFree(row_ptr_d);
    cudaFree(col_indices_d);
    cudaFree(rhs_d);
    cudaFree(reference_solution_d);
    cudaFree(diagonal_d);

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
