#pragma once
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <hipsparse.h>
#include <iostream>
#include "cudaerrchk.h"
#include <unistd.h>
#include <rocsparse.h>

class Distributed_vector{
    public:
        int matrix_size;
        int rows_this_rank;
        int size;
        int rank;
        int *counts;
        int *displacements;
        int number_of_neighbours;
        int *neighbours;
        MPI_Comm comm;

        double **vec_h;
        double **vec_d;
        hipsparseDnVecDescr_t *descriptors;

    Distributed_vector(
        int matrix_size,
        int *counts,
        int *displacements,
        int number_of_neighbours,
        int *neighbours,
        MPI_Comm comm);
    ~Distributed_vector();

};

struct Distributed_subblock{
    int *subblock_indices_local_d;
    double *A_subblock_local_d;
    int subblock_size;
    int *count_subblock_h;
    int *displ_subblock_h;
    hipStream_t *streams_recv_subblock;
    hipEvent_t *events_recv_subblock;
    MPI_Request *send_subblock_requests;
    MPI_Request *recv_subblock_requests;
};


struct Distributed_subblock_sparse{
    int *subblock_indices_local_d;
    rocsparse_spmat_descr *descriptor;
    rocsparse_spmv_alg algo;
    size_t *buffersize;
    double *buffer_d;
    int subblock_size;
    int *count_subblock_h;
    int *displ_subblock_h;
    hipStream_t *streams_recv_subblock;
    hipEvent_t *events_recv_subblock;
    MPI_Request *send_subblock_requests;
    MPI_Request *recv_subblock_requests;
};


// assumes that the matrix is symmetric
// does a 1D decomposition over the rows
class Distributed_matrix{
    public:
        int matrix_size;
        int rows_this_rank;
        int nnz;
    
        int size;
        int rank;
        int *counts;
        int *displacements;    
        MPI_Comm comm;

        // includes itself
        int number_of_neighbours;
        // true or false if neighbour
        bool *neighbours_flag;
        // list of neighbour indices 
        // starting from own rank
        int *neighbours;

        // by default, we assume that the matrix is stored in CSR format
        // first matrix is own piece
        double **data_h;
        int **col_indices_h;
        int **row_ptr_h;

        // Data types for cuSPARSE
        size_t *buffer_size;
        double **buffer_d;
        double **data_d;
        int **col_indices_d;
        int **row_ptr_d;
        hipsparseSpMatDescr_t *descriptors;

        // Data types for MPI
        // assumes symmetric matrix
        
        // number of non-zeros per neighbour
        int *nnz_per_neighbour;
        // number of non-zeros columns per neighbour
        int *nnz_cols_per_neighbour;
        // number of non-zeros rows per neighbour
        int *nnz_rows_per_neighbour;

        // indices to fetch from neighbours
        int **cols_per_neighbour_h;
        int **cols_per_neighbour_d;
        // indices to send to neighbours
        int **rows_per_neighbour_h;
        int **rows_per_neighbour_d;
        
        // send and recv buffers
        double **send_buffer_h;
        double **recv_buffer_h;        
        double **send_buffer_d;
        double **recv_buffer_d;

        // MPI data types and requests
        MPI_Datatype *send_types;
        MPI_Datatype *recv_types;
        MPI_Request *send_requests;
        MPI_Request *recv_requests;
        // MPI streams and events
        hipStream_t *streams_recv;
        hipStream_t *streams_send;
        hipEvent_t *events_recv;
        hipEvent_t *events_send;

        hipEvent_t event_default_finished;

    // construct the distributed matrix
    // input is the whol count[rank] * matrix size
    // csr part of the matrix
    Distributed_matrix(
        int matrix_size,
        int nnz,
        int *counts,
        int *displacements,
        int *col_indices_in,
        int *row_ptr_in,
        double *data_in,
        MPI_Comm comm);

    // construct the distributed matrix
    // input is correctly split
    // data is not set
    Distributed_matrix(
        int matrix_size,
        int *counts_in,
        int *displacements_in,
        int number_of_neighbours,
        int *neighbours_in,
        int **col_indices_in_d,
        int **row_ptr_in_d,
        int *nnz_per_neighbour_in,
        MPI_Comm comm);


    ~Distributed_matrix();

    private:
        void find_neighbours(
            int *col_indices_in,
            int *row_ptr_in
        );

        void construct_neighbours_list(
        );

        void construct_nnz_per_neighbour(
            int *col_indices_in,
            int *row_ptr_in
        );

        void split_csr(
            int *col_indices_in,
            int *row_ptr_in,
            double *data_in
        );

        
        void construct_nnz_cols_per_neighbour();

        void construct_nnz_rows_per_neighbour();
        

        void construct_rows_per_neighbour();

        void construct_cols_per_neighbour(); 

        void check_sorted();

        void construct_mpi_data_types();

        void create_events_streams();

        void create_host_memory();

        void create_device_memory(hipsparseHandle_t &cusparseHandle);

};