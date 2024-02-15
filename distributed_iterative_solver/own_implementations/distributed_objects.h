#pragma once
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include "../cudaerrchk.h"
#include <unistd.h>

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
        cusparseDnVecDescr_t *descriptors;
        cusparseHandle_t cusparseHandle;

    Distributed_vector(
        int matrix_size,
        int *counts,
        int *displacements,
        int number_of_neighbours,
        int *neighbours,
        MPI_Comm comm,
        cusparseHandle_t &cusparseHandle);
    ~Distributed_vector();

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
        cusparseSpMatDescr_t *descriptors;
        cusparseHandle_t cusparseHandle;

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
        cudaStream_t *streams_recv;
        cudaStream_t *streams_send;
        cudaEvent_t *events_recv;
        cudaEvent_t *events_send;

    Distributed_matrix(
        int matrix_size,
        int nnz,
        int *counts,
        int *displacements,
        int *col_indices,
        int *row_ptr,
        double *data,
        MPI_Comm comm,
        cusparseHandle_t &cusparseHandle);

    

    ~Distributed_matrix();

    private:
        void find_neighbours(
            int *col_indices,
            int *row_ptr
        );

        void construct_neighbours_list(
        );

        void construct_nnz_per_neighbour(
            int *col_indices,
            int *row_ptr
        );

        void split_csr(
            int *col_indices,
            int *row_ptr,
            double *data
        );

        
        void construct_nnz_cols_per_neighbour();

        void construct_nnz_rows_per_neighbour();
        

        void construct_rows_per_neighbour();

        void construct_cols_per_neighbour(); 

};