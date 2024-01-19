#pragma once
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include "cudaerrchk.h"
#include "utils.h"
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
        cusparseHandle_t &cusparseHandle
    ){
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);
        this->matrix_size = matrix_size;
        this->cusparseHandle = cusparseHandle;
        this->comm = comm;
        this->counts = new int[size];
        this->displacements = new int[size];
        for(int i = 0; i < size; i++){
            this->counts[i] = counts[i];
            this->displacements[i] = displacements[i];
        }
        rows_this_rank = counts[rank];
        this->number_of_neighbours = number_of_neighbours;
        this->neighbours = new int[number_of_neighbours];
        for(int k = 0; k < number_of_neighbours; k++){
            this->neighbours[k] = neighbours[k];
        }
        vec_h = new double*[number_of_neighbours];
        vec_d = new double*[number_of_neighbours];
        descriptors = new cusparseDnVecDescr_t[number_of_neighbours];
        for(int k = 0; k < number_of_neighbours; k++){
            int neighbour_idx = neighbours[k];
            vec_h[k] = new double[counts[neighbour_idx]];
            for(int i = 0; i < counts[neighbour_idx]; i++){
                vec_h[k][i] = 0.0;
            }
            cudaErrchk(cudaMalloc(&vec_d[k], counts[neighbour_idx]*sizeof(double)));
            cudaErrchk(cudaMemset(vec_d[k], 0, counts[neighbour_idx]*sizeof(double)));
            cusparseErrchk(cusparseCreateDnVec(&descriptors[k], counts[neighbour_idx], vec_d[k], CUDA_R_64F));

        }
    }

    ~Distributed_vector(){
        delete[] counts;
        delete[] displacements;
        delete[] neighbours;
        for(int k = 0; k < number_of_neighbours; k++){
            delete[] vec_h[k];
            cudaErrchk(cudaFree(vec_d[k]));
            cusparseErrchk(cusparseDestroyDnVec(descriptors[k]));
        }
        delete[] vec_h;
        delete[] vec_d;
        delete[] descriptors;
    }

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

        // of size number_of_neighbours
        int *nnz_per_neighbour;
        int *nnz_cols_per_neighbour;
        int *nnz_rows_per_neighbour;
    
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
        // indices to recv from neighbours
        int **cols_per_neighbour_h;
        // indices to send to neighbours
        int **rows_per_neighbour_h;
        // indices to fetch from neighbours
        double **send_buffer_h;
        double **recv_buffer_h;

        int **cols_per_neighbour_d;
        // indices to send to neighbours
        int **rows_per_neighbour_d;
        // indices to fetch from neighbours
        double **send_buffer_d;
        double **recv_buffer_d;

        MPI_Datatype *send_types;
        MPI_Datatype *recv_types;

    Distributed_matrix(
        int matrix_size,
        int nnz,
        int *counts,
        int *displacements,
        int *col_indices,
        int *row_ptr,
        double *data,
        MPI_Comm comm,
        cusparseHandle_t &cusparseHandle)
    {
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);

        this->matrix_size = matrix_size;
        this->nnz = nnz;
        this->cusparseHandle = cusparseHandle;
        this->comm = comm;
    
        this->counts = new int[size];
        this->displacements = new int[size];
        for(int i = 0; i < size; i++){
            this->counts[i] = counts[i];
            this->displacements[i] = displacements[i];
        }

        rows_this_rank = counts[rank];

        // find neighbours_flag
        neighbours_flag = new bool[size];
        std::cout << rank << " " << "Finding neighbours" << std::endl;
        find_neighbours(col_indices, row_ptr);
        std::cout << rank << " " << "Number of neighbours: " << number_of_neighbours << std::endl;
        std::cout << rank << " " << "Neighbours Flags: ";
        for(int k = 0; k < size; k++){
            std::cout << neighbours_flag[k] << " ";
        }
        std::cout << std::endl;

        if(number_of_neighbours == 0){
            std::cout << rank << " " << "No neighbours" << std::endl;
        }
        if(!neighbours_flag[rank]){
            std::cout << rank  << "I am not a neighbour" << std::endl;
        } 
        neighbours = new int[number_of_neighbours];
        std::cout << rank << " " << "Constructing neighbours list" << std::endl;
        construct_neighbours_list();
        std::cout << rank << " " << "Neighbours list: ";
        for(int k = 0; k < number_of_neighbours; k++){
            std::cout << neighbours[k] << " ";
        }
        std::cout << std::endl;


        nnz_per_neighbour = new int[number_of_neighbours];
        std::cout << rank << " " << "Constructing nnz per neighbour" << std::endl;
        construct_nnz_per_neighbour(col_indices, row_ptr);
        std::cout << rank << " " << "NNZ per neighbour: ";
        for(int k = 0; k < number_of_neighbours; k++){
            std::cout << nnz_per_neighbour[k] << " ";
        }
        std::cout << std::endl;

        data_h = new double*[number_of_neighbours];
        col_indices_h = new int*[number_of_neighbours];
        row_ptr_h = new int*[number_of_neighbours];
        // allocate memory for data, indices and row_ptr
        for(int k = 0; k < number_of_neighbours; k++){
            data_h[k] = new double[nnz_per_neighbour[k]];
            col_indices_h[k] = new int[nnz_per_neighbour[k]];
            // numbers of rows are constant
            row_ptr_h[k] = new int[rows_this_rank+1];
        }
        std::cout << rank << " " << "Splitting CSR" << std::endl;
        // split data, indices and row_ptr
        split_csr(col_indices, row_ptr, data);


        nnz_cols_per_neighbour = new int[number_of_neighbours];
        std::cout << rank << " " << "Constructing nnz cols per neighbour" << std::endl;
        construct_nnz_cols_per_neighbour();
        std::cout << rank << " " << "NNZ cols per neighbour: ";
        for(int k = 0; k < number_of_neighbours; k++){
            std::cout << nnz_cols_per_neighbour[k] << " ";
        }
        std::cout << std::endl;


        nnz_rows_per_neighbour = new int[number_of_neighbours];
        std::cout << rank << " " << "Constructing nnz rows per neighbour: ";
        construct_nnz_rows_per_neighbour();
        std::cout << rank << " " << "NNZ rows per neighbour: ";
        for(int k = 0; k < number_of_neighbours; k++){
            std::cout << nnz_rows_per_neighbour[k] << " ";
        }
        std::cout << std::endl;
        cols_per_neighbour_h = new int*[number_of_neighbours];
        rows_per_neighbour_h = new int*[number_of_neighbours];
        for(int k = 0; k < number_of_neighbours; k++){
            cols_per_neighbour_h[k] = new int[nnz_cols_per_neighbour[k]];
            rows_per_neighbour_h[k] = new int[nnz_rows_per_neighbour[k]];
        }

        std::cout << rank << " " << "Prepare indices to fetch: " << std::endl;
        construct_cols_per_neighbour();
        construct_rows_per_neighbour();

        cols_per_neighbour_d = new int*[number_of_neighbours];
        rows_per_neighbour_d = new int*[number_of_neighbours];
        for(int k = 0; k < number_of_neighbours; k++){
            cudaErrchk(cudaMalloc(&cols_per_neighbour_d[k], nnz_cols_per_neighbour[k]*sizeof(int)));
            cudaErrchk(cudaMalloc(&rows_per_neighbour_d[k], nnz_rows_per_neighbour[k]*sizeof(int)));
            cudaErrchk(cudaMemcpy(cols_per_neighbour_d[k], cols_per_neighbour_h[k], nnz_cols_per_neighbour[k]*sizeof(int), cudaMemcpyHostToDevice));
            cudaErrchk(cudaMemcpy(rows_per_neighbour_d[k], rows_per_neighbour_h[k], nnz_rows_per_neighbour[k]*sizeof(int), cudaMemcpyHostToDevice));
        }


        std::cout << rank << " " << "Prepare buffers to fetch into: " << std::endl;
        // excluding itself
        send_buffer_h = new double*[number_of_neighbours-1];
        recv_buffer_h = new double*[number_of_neighbours-1];
        send_buffer_d = new double*[number_of_neighbours-1];
        recv_buffer_d = new double*[number_of_neighbours-1];
        for(int k = 0; k < number_of_neighbours-1; k++){
            send_buffer_h[k] = new double[nnz_rows_per_neighbour[k+1]];
            recv_buffer_h[k] = new double[nnz_cols_per_neighbour[k+1]];
            cudaErrchk(cudaMalloc(&send_buffer_d[k], nnz_rows_per_neighbour[k+1]*sizeof(double)));
            cudaErrchk(cudaMalloc(&recv_buffer_d[k], nnz_cols_per_neighbour[k+1]*sizeof(double)));
        }

        std::cout << "Create custom datatypes" << std::endl;
        send_types = new MPI_Datatype[number_of_neighbours-1];
        recv_types = new MPI_Datatype[number_of_neighbours-1];
        for(int k = 0; k < number_of_neighbours-1; k++){
            int *lengths = new int[nnz_rows_per_neighbour[k+1]];
            for(int i = 0; i < nnz_rows_per_neighbour[k+1]; i++){
                lengths[i] = 1;
            }
            MPI_Type_indexed(nnz_rows_per_neighbour[k+1], lengths,
                            rows_per_neighbour_h[k+1], MPI_DOUBLE, &send_types[k]);
            MPI_Type_commit(&send_types[k]);
            delete[] lengths;
        }
        for(int k = 0; k < number_of_neighbours-1; k++){
            int *lengths = new int[nnz_cols_per_neighbour[k+1]];
            for(int i = 0; i < nnz_cols_per_neighbour[k+1]; i++){
                lengths[i] = 1;
            }
            MPI_Type_indexed(nnz_cols_per_neighbour[k+1],lengths,
                            cols_per_neighbour_h[k+1], MPI_DOUBLE, &recv_types[k]);
            MPI_Type_commit(&recv_types[k]);
            delete[] lengths;
        }

        int *blocklengths = new int[number_of_neighbours-1];
        std::memset(blocklengths, 1, (number_of_neighbours-1)*sizeof(int));


        std::cout << rank << " " << "Allocating device memory and copy" << std::endl;
        // allocate device memory

        buffer_size = new size_t[number_of_neighbours];
        buffer_d = new double*[number_of_neighbours];
        data_d = new double*[number_of_neighbours];
        col_indices_d = new int*[number_of_neighbours];
        row_ptr_d = new int*[number_of_neighbours];
        descriptors = new cusparseSpMatDescr_t[number_of_neighbours];
        for(int k = 0; k < number_of_neighbours; k++){
            int neighbour_idx = neighbours[k];
            cudaErrchk(cudaMalloc(&data_d[k], nnz_per_neighbour[k]*sizeof(double)));
            cudaErrchk(cudaMalloc(&col_indices_d[k], nnz_per_neighbour[k]*sizeof(int)));
            cudaErrchk(cudaMalloc(&row_ptr_d[k], (rows_this_rank+1)*sizeof(int)));
            cudaErrchk(cudaMemcpy(data_d[k], data_h[k], nnz_per_neighbour[k]*sizeof(double), cudaMemcpyHostToDevice));
            cudaErrchk(cudaMemcpy(col_indices_d[k], col_indices_h[k], nnz_per_neighbour[k]*sizeof(int), cudaMemcpyHostToDevice));
            cudaErrchk(cudaMemcpy(row_ptr_d[k], row_ptr_h[k], (rows_this_rank+1)*sizeof(int), cudaMemcpyHostToDevice));

            double *vec_in_d;
            double *vec_out_d;
            cusparseDnVecDescr_t vec_in;
            cusparseDnVecDescr_t vec_out;

            cudaErrchk(cudaMalloc(&vec_in_d, counts[neighbour_idx]*sizeof(double)));
            cudaErrchk(cudaMalloc(&vec_out_d, rows_this_rank*sizeof(double)));
            cusparseErrchk(cusparseCreateDnVec(&vec_in, counts[neighbour_idx], vec_in_d, CUDA_R_64F));
            cusparseErrchk(cusparseCreateDnVec(&vec_out, rows_this_rank, vec_out_d, CUDA_R_64F));


            /* Wrap raw data into cuSPARSE generic API objects */
            cusparseErrchk(cusparseCreateCsr(
                &descriptors[k],
                rows_this_rank,
                counts[neighbour_idx],
                nnz_per_neighbour[k],
                row_ptr_d[k],
                col_indices_d[k],
                data_d[k],
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                CUDA_R_64F
            ));

            double alpha = 1.0;
            double beta = 0.0;
            cusparseErrchk(cusparseSpMV_bufferSize(
                cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, descriptors[k], vec_in,
                &beta, vec_out, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size[k]));
            cudaErrchk(cudaMalloc(&buffer_d[k], buffer_size[k]));

            cusparseErrchk(cusparseDestroyDnVec(vec_in));
            cusparseErrchk(cusparseDestroyDnVec(vec_out));
            cudaErrchk(cudaFree(vec_in_d));
            cudaErrchk(cudaFree(vec_out_d));
        }



        std::cout << rank << " " << "Done" << std::endl;
    }

    ~Distributed_matrix(){
        delete[] counts;
        delete[] displacements;
        delete[] neighbours_flag;
        delete[] neighbours;
        delete[] nnz_per_neighbour;
        delete[] nnz_cols_per_neighbour;
        for(int k = 0; k < number_of_neighbours; k++){
            delete[] data_h[k];
            delete[] col_indices_h[k];
            delete[] row_ptr_h[k];
        }
        delete[] data_h;
        delete[] col_indices_h;
        delete[] row_ptr_h;
        delete[] nnz_rows_per_neighbour;

        for(int k = 0; k < number_of_neighbours; k++){
            delete[] cols_per_neighbour_h[k];
            delete[] rows_per_neighbour_h[k];
        }
        delete[] cols_per_neighbour_h;
        delete[] rows_per_neighbour_h;

        for(int k = 0; k < number_of_neighbours; k++){
            cudaErrchk(cudaFree(cols_per_neighbour_d[k]));
            cudaErrchk(cudaFree(rows_per_neighbour_d[k]));
        }
        delete[] cols_per_neighbour_d;
        delete[] rows_per_neighbour_d;

        for(int k = 0; k < number_of_neighbours-1; k++){
            delete[] send_buffer_h[k];
            delete[] recv_buffer_h[k];
            cudaErrchk(cudaFree(send_buffer_d[k]));
            cudaErrchk(cudaFree(recv_buffer_d[k]));
        }
        delete[] send_buffer_h;
        delete[] recv_buffer_h;
        delete[] send_buffer_d;
        delete[] recv_buffer_d;

        for(int k = 0; k < number_of_neighbours-1; k++){
            MPI_Type_free(&send_types[k]);
            MPI_Type_free(&recv_types[k]);
        }
        delete[] send_types;
        delete[] recv_types;

        for(int k = 0; k < number_of_neighbours; k++){
            cudaErrchk(cudaFree(data_d[k]));
            cudaErrchk(cudaFree(col_indices_d[k]));
            cudaErrchk(cudaFree(row_ptr_d[k]));
            cudaErrchk(cudaFree(buffer_d[k]));
            cusparseErrchk(cusparseDestroySpMat(descriptors[k]));
        }
        delete[] buffer_d;
        delete[] data_d;
        delete[] col_indices_d;
        delete[] row_ptr_d;
        delete[] descriptors;
        delete[] buffer_size;
    }

    private:
        void find_neighbours(
            int *col_indices,
            int *row_ptr
        ){

            for(int k = 0; k < size; k++){
                neighbours_flag[k] = false;
            }

            int tmp_number_of_neighbours = 0;

            for(int i = 0; i < rows_this_rank; i++){
                for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
                    int col_idx = col_indices[j];
                    for(int k = 0; k < size; k++){
                        if(col_idx >= displacements[k] && col_idx < displacements[k] + counts[k]){
                            neighbours_flag[k] = true;
                        }
                    }
                }
            }
            for(int k = 0; k < size; k++){
                if(neighbours_flag[k]){
                    tmp_number_of_neighbours++;
                }
            }

            number_of_neighbours = tmp_number_of_neighbours;

        }

        void construct_neighbours_list(
        ){
            int tmp_number_of_neighbours = 0;
            for(int k = 0; k < size; k++){
                int idx = (rank + k) % size;
                if(neighbours_flag[idx]){
                    neighbours[tmp_number_of_neighbours] = idx;
                    tmp_number_of_neighbours++;
                }
            }
        }

        void construct_nnz_per_neighbour(
            int *col_indices,
            int *row_ptr
        )
        {
            for(int k = 0; k < number_of_neighbours; k++){
                nnz_per_neighbour[k] = 0;
            }

            for(int i = 0; i < rows_this_rank; i++){
                for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
                    int col_idx = col_indices[j];
                    for(int k = 0; k < number_of_neighbours; k++){
                        int neighbour_idx = neighbours[k];
                        if(col_idx >= displacements[neighbour_idx] && col_idx < displacements[neighbour_idx] + counts[neighbour_idx]){
                            nnz_per_neighbour[k]++;
                        }
                    }
                }
            }

        }



        void split_csr(
            int *col_indices,
            int *row_ptr,
            double *data
        ){
            int *tmp_nnz_per_neighbour = new int[number_of_neighbours];
            for(int k = 0; k < number_of_neighbours; k++){
                tmp_nnz_per_neighbour[k] = 0;
            }

            for(int i = 0; i < rows_this_rank; i++){
                for(int k = 0; k < number_of_neighbours; k++){
                    row_ptr_h[k][i] = tmp_nnz_per_neighbour[k];
                }

                for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
                    for(int k = 0; k < number_of_neighbours; k++){
                        int neighbour_idx = neighbours[k];
                        int col_idx = col_indices[j];
                        if(col_idx >= displacements[neighbour_idx] && col_idx < displacements[neighbour_idx] + counts[neighbour_idx]){
                            data_h[k][tmp_nnz_per_neighbour[k]] = data[j];
                            col_indices_h[k][tmp_nnz_per_neighbour[k]] = col_idx - displacements[neighbour_idx];
                            tmp_nnz_per_neighbour[k]++;
                        }
                    }
                }
            }

            for(int k = 0; k < number_of_neighbours; k++){
                row_ptr_h[k][rows_this_rank] = tmp_nnz_per_neighbour[k];
            }


            for(int k = 0; k < number_of_neighbours; k++){
                if(tmp_nnz_per_neighbour[k] != nnz_per_neighbour[k]){
                    std::cout << "Error in split_csr" << std::endl;
                }
            }

            delete[] tmp_nnz_per_neighbour;

        }

        // void construct_nnz_cols_per_neighbour(
        //     int *col_indices,
        //     int *row_ptr
        // )
        // {
        //     int *tmp_col_flag = new int[matrix_size];
        //     for(int col_idx = 0; col_idx < matrix_size; col_idx++){
        //         tmp_col_flag[col_idx] = 0;
        //     }

        //     // difficult to do in parallel
        //     for(int i = 0; i < rows_this_rank; i++){
        //         for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
        //             int col_idx = col_indices[j];
        //             tmp_col_flag[col_idx] = 1;
        //         }
        //     }

        //     for(int i = 0; i < number_of_neighbours; i++){
        //         nnz_cols_per_neighbour[i] = 0;
        //     }

        //     // inner loop is technically not needed
        //     // could be done directly in the loop above
        //     // with modulo
        //     for(int col_idx = 0; col_idx < matrix_size; col_idx++){
        //         if(tmp_col_flag[col_idx] == 0){
        //             continue;
        //         }
        //         for(int k = 0; k < number_of_neighbours; k++){
        //             int neighbour_idx = neighbours[k];
        //             if(col_idx >= displacements[neighbour_idx] && col_idx < displacements[neighbour_idx] + counts[neighbour_idx]){
        //                 nnz_cols_per_neighbour[k] += tmp_col_flag[col_idx];
        //             }
        //         }
        //     }

        //     delete[] tmp_col_flag;
        // }

        void construct_nnz_cols_per_neighbour(
        )
        {
            for(int k = 0; k < number_of_neighbours; k++){
                nnz_cols_per_neighbour[k] = 0;
                int neighbour_idx = neighbours[k];
                bool *cols_per_neighbour_flags = new bool[counts[neighbour_idx]];
                for(int i = 0; i < counts[neighbour_idx]; i++){
                    cols_per_neighbour_flags[i] = false;
                }
                for(int i = 0; i < rows_this_rank; i++){
                    for(int j = row_ptr_h[k][i]; j < row_ptr_h[k][i+1]; j++){
                        int col_idx = col_indices_h[k][j];
                        cols_per_neighbour_flags[col_idx] = true;
                    }
                }
                for(int i = 0; i < counts[neighbour_idx]; i++){
                    if(cols_per_neighbour_flags[i]){
                        nnz_cols_per_neighbour[k]++;
                    }
                }
                delete[] cols_per_neighbour_flags;
            }
        }

        void construct_nnz_rows_per_neighbour()
        {
            for(int i = 0; i < number_of_neighbours; i++){
                nnz_rows_per_neighbour[i] = 0;
            }
            for(int i = 0; i < rows_this_rank; i++){
                for(int k = 0; k < number_of_neighbours; k++){
                    if(row_ptr_h[k][i+1] - row_ptr_h[k][i] > 0){
                        nnz_rows_per_neighbour[k]++;
                    }
                }
            }
        }

        void construct_rows_per_neighbour()
        {
            int *tmp_nnz_rows_per_neighbour = new int[number_of_neighbours];
            for(int i = 0; i < number_of_neighbours; i++){
                tmp_nnz_rows_per_neighbour[i] = 0;
            }
            for(int i = 0; i < rows_this_rank; i++){
                for(int k = 0; k < number_of_neighbours; k++){
                    if(row_ptr_h[k][i+1] - row_ptr_h[k][i] > 0){
                        rows_per_neighbour_h[k][tmp_nnz_rows_per_neighbour[k]] = i;
                        tmp_nnz_rows_per_neighbour[k]++;
                    }
                }
            }
            delete[] tmp_nnz_rows_per_neighbour;
        }   


        void construct_cols_per_neighbour()
        {

            for(int k = 0; k < number_of_neighbours; k++){
                int neighbour_idx = neighbours[k];
                bool *cols_per_neighbour_flags = new bool[counts[neighbour_idx]];
                for(int i = 0; i < counts[neighbour_idx]; i++){
                    cols_per_neighbour_flags[i] = false;
                }
                for(int i = 0; i < rows_this_rank; i++){
                    for(int j = row_ptr_h[k][i]; j < row_ptr_h[k][i+1]; j++){
                        int col_idx = col_indices_h[k][j];
                        cols_per_neighbour_flags[col_idx] = true;
                    }
                }
                int tmp_nnz_cols = 0;
                for(int i = 0; i < counts[neighbour_idx]; i++){
                    if(cols_per_neighbour_flags[i]){
                        cols_per_neighbour_h[k][tmp_nnz_cols] = i;
                        tmp_nnz_cols++;
                    }
                }
                delete[] cols_per_neighbour_flags;
            }

        }   

};