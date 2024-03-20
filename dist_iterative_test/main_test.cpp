#include <iostream>
#include <string>
#include "utils.h"
#include <mpi.h>
#include <hip/hip_runtime.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include "utils_gpu.h"
#include <hipblas.h>
#include "../dist_iterative/dist_conjugate_gradient.h"
#include "../dist_iterative/dist_spmv.h"
#include <pthread.h>
#include "rocsparse.h"

int main(int argc, char **argv) {


    // Init thread multiple
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    std::cout << "provided " << provided << std::endl;


    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char* slurm_localid = getenv("SLURM_LOCALID");
    int localid = atoi(slurm_localid);
    int device_id = localid; 
    std::cout << "rank " << rank << " device_id " << device_id << std::endl;

    hipError_t set_device_error = hipSetDevice(localid);
    std::cout << "rank " << rank << " set_device_error " << set_device_error << std::endl;

    char* rocr_visible_devices = getenv("ROCR_VISIBLE_DEVICES");
    if (rocr_visible_devices != nullptr) {
        std::cout << "Rank " << rank << " ROCR_VISIBLE_DEVICES: " << rocr_visible_devices << std::endl;
    } else {
        std::cerr << "Rank " << rank << " ROCR_VISIBLE_DEVICES not set" << std::endl;
        exit(1);
    }

    std::string data_path = "/scratch/project_465000929/maederal/100/";

    int matrix_size = 102722;
    int nnz_sparse = 1707556;
    int nnz_tot = 95903772;
    int subblock_size = 14854;    
    int indices_offset = 2;
    int nnz_subblock = 94211070;


    if(nnz_subblock + nnz_sparse - subblock_size != nnz_tot){
        std::cout << "nnz_subblock + nnz_sparse - subblock_size != nnz_tot" << std::endl;
        exit(1);
    }

    int max_iterations = 10000;
    double relative_tolerance = 1e-18;

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

    double *data_tot = new double[nnz_tot];
    int *row_ptr_tot = new int[matrix_size+1];
    int *col_indices_tot = new int[nnz_tot];


    std::string data_tot_filename = data_path + "X_data.bin";
    std::string row_ptr_tot_filename = data_path + "X_row_ptr.bin";
    std::string col_indices_tot_filename = data_path + "X_col_indices.bin";

    load_binary_array<double>(data_tot_filename, data_tot, nnz_tot);
    load_binary_array<int>(row_ptr_tot_filename, row_ptr_tot, matrix_size+1);
    load_binary_array<int>(col_indices_tot_filename, col_indices_tot, nnz_tot);

    double *vec_in = new double[matrix_size];
    for(int i = 0; i < matrix_size; i++){
        vec_in[i] = 1.0;
    }

    // create gpu data
    double *data_tot_d;
    int *row_ptr_tot_d;
    int *col_indices_tot_d;
    double *vec_in_d;
    double *vec_out_d;
    double *vec_out_d2;

    hipMalloc((void**)&data_tot_d, nnz_tot * sizeof(double));
    hipMalloc((void**)&row_ptr_tot_d, (matrix_size+1) * sizeof(int));
    hipMalloc((void**)&col_indices_tot_d, nnz_tot * sizeof(int));
    hipMalloc((void**)&vec_in_d, matrix_size * sizeof(double));
    hipMalloc((void**)&vec_out_d, matrix_size * sizeof(double));
    hipMalloc((void**)&vec_out_d2, matrix_size * sizeof(double));

    hipMemcpy(data_tot_d, data_tot, nnz_tot * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(row_ptr_tot_d, row_ptr_tot, (matrix_size+1) * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(col_indices_tot_d, col_indices_tot, nnz_tot * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(vec_in_d, vec_in, matrix_size * sizeof(double), hipMemcpyHostToDevice);

    // create rocsparse handle
    rocsparse_handle default_rocsparseHandle;
    rocsparse_create_handle(&default_rocsparseHandle);


    rocsparse_mat_info info;
    rocsparse_create_mat_info(&info);
    rocsparse_mat_descr descr;
    rocsparse_create_mat_descr(&descr);

    // Perform analysis step to obtain meta data
    rocsparse_dcsrmv_analysis(default_rocsparseHandle,
                            rocsparse_operation_none,
                            matrix_size,
                            matrix_size,
                            nnz_tot,
                            descr,
                            data_tot_d,
                            row_ptr_tot_d,
                            col_indices_tot_d,
                            info);

    double alpha = 1.0;
    double beta = 0.0;

    // Compute y = Ax
    rocsparse_dcsrmv(default_rocsparseHandle,
                    rocsparse_operation_none,
                    matrix_size,
                    matrix_size,
                    nnz_tot,
                    &alpha,
                    descr,
                    data_tot_d,
                    row_ptr_tot_d,
                    col_indices_tot_d,
                    info,
                    vec_in_d,
                    &beta,
                    vec_out_d);

    //create roc dense vector descriptors
    rocsparse_dnvec_descr vec_descr_in;
    rocsparse_create_dnvec_descr(&vec_descr_in, matrix_size, vec_in_d, rocsparse_datatype_f64_r);
    rocsparse_dnvec_descr vec_descr_out;
    rocsparse_create_dnvec_descr(&vec_descr_out, matrix_size, vec_out_d2, rocsparse_datatype_f64_r);

    // create roc sparse matrix descriptors
    rocsparse_spmat_descr csr_descr;
    rocsparse_create_csr_descr(
        &csr_descr,
        matrix_size,
        matrix_size,
        nnz_tot,
        row_ptr_tot_d,
        col_indices_tot_d,
        data_tot_d,
        rocsparse_indextype_i32,
        rocsparse_indextype_i32,
        rocsparse_index_base_zero,
        rocsparse_datatype_f64_r);    

    // create buffer size
    size_t buffer_size;
    rocsparse_spmv(default_rocsparseHandle,
                                rocsparse_operation_none,
                                &alpha,
                                csr_descr,
                                vec_descr_in,
                                &beta,
                                vec_descr_out,
                                rocsparse_datatype_f64_r,
                                rocsparse_spmv_alg_csr_adaptive,
                                &buffer_size,
                                nullptr);
    double *buffer;
    hipMalloc((void**)&buffer, buffer_size);

    // spmv
    rocsparse_spmv(
        default_rocsparseHandle,
        rocsparse_operation_none,
        &alpha,
        csr_descr,
        vec_descr_in,
        &beta,
        vec_descr_out,
        rocsparse_datatype_f64_r,
        rocsparse_spmv_alg_csr_adaptive,
        &buffer_size,
        buffer);

    size_t buffer_size2;
    rocsparse_spmv(default_rocsparseHandle,
                                rocsparse_operation_none,
                                &alpha,
                                csr_descr,
                                vec_descr_in,
                                &beta,
                                vec_descr_out,
                                rocsparse_datatype_f64_r,
                                rocsparse_spmv_alg_csr_stream,
                                &buffer_size2,
                                nullptr);
    double *buffer2;
    hipMalloc((void**)&buffer, buffer_size2);

    // spmv
    rocsparse_spmv(
        default_rocsparseHandle,
        rocsparse_operation_none,
        &alpha,
        csr_descr,
        vec_descr_in,
        &beta,
        vec_descr_out,
        rocsparse_datatype_f64_r,
        rocsparse_spmv_alg_csr_stream,
        &buffer_size2,
        buffer2);



    double *vec_out_h = new double[matrix_size];
    double *vec_out_h2 = new double[matrix_size];
    hipMemcpy(vec_out_h, vec_out_d, matrix_size * sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(vec_out_h2, vec_out_d2, matrix_size * sizeof(double), hipMemcpyDeviceToHost);

    double sum = 0.0;
    double diff = 0.0;
    for(int i = 0; i < matrix_size; i++){
        sum += std::abs(vec_out_h[i]) * std::abs(vec_out_h[i]);
        diff += std::abs(vec_out_h[i] - vec_out_h2[i]) *
            std::abs(vec_out_h[i] - vec_out_h2[i]);
    }
    std::cout <<  diff / sum << std::endl;



    MPI_Finalize();
    return 0;
}
