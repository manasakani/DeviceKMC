#pragma once
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>

void split_matrix(
    int matrix_size,
    int size,
    int *counts,
    int *displacements
){
    int rows_per_rank = matrix_size / size;    
    for (int i = 0; i < size; ++i) {
        if(i < matrix_size % size){
            counts[i] = rows_per_rank+1;
        }
        else{
            counts[i] = rows_per_rank;
        }
    }
    displacements[0] = 0;
    for (int i = 1; i < size; ++i) {
        displacements[i] = displacements[i-1] + counts[i-1];
    }

}

void find_neighbours(
    int rows_this_rank,
    int *col_indices,
    int *row_ptr,
    int size,
    int *counts,
    int *displacements,
    bool *neighbours_flag,
    int *number_of_neighbours
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

    number_of_neighbours[0] = tmp_number_of_neighbours;

}

void construct_neighbours_list(
    int size,
    int rank,
    bool *neighbours_flag,
    int *neighbours
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
    int rows_this_rank,
    int *col_indices,
    int *row_ptr,
    int *counts,
    int *displacements,
    int number_of_neighbours,
    int *neighbours,
    int *nnz_per_neighbour
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

void construct_nnz_cols_per_neighbour(
    int matrix_size,
    int rows_this_rank,
    int *col_indices,
    int *row_ptr,
    int *counts,
    int *displacements,
    int number_of_neighbours,
    int *neighbours,
    int *nnz_cols_per_neighbour
)
{
    int *tmp_col_flag = new int[matrix_size];
    for(int col_idx = 0; col_idx < matrix_size; col_idx++){
        tmp_col_flag[col_idx] = 0;
    }

    // difficult to do in parallel
    for(int i = 0; i < rows_this_rank; i++){
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            int col_idx = col_indices[j];
            tmp_col_flag[col_idx] = 1;
        }
    }

    for(int i = 0; i < number_of_neighbours; i++){
        nnz_cols_per_neighbour[i] = 0;
    }

    // inner loop is technically not needed
    // could be done directly in the loop above
    // with modulo
    for(int col_idx = 0; col_idx < matrix_size; col_idx++){
        if(tmp_col_flag[col_idx] == 0){
            continue;
        }
        for(int k = 0; k < number_of_neighbours; k++){
            int neighbour_idx = neighbours[k];
            if(col_idx >= displacements[neighbour_idx] && col_idx < displacements[neighbour_idx] + counts[neighbour_idx]){
                nnz_cols_per_neighbour[k] += tmp_col_flag[col_idx];
            }
        }
    }

    delete[] tmp_col_flag;
}

void split_csr(
    int rows_this_rank,
    int *col_indices,
    int *row_ptr,
    double *data,
    int *counts,
    int *displacements,
    int number_of_neighbours,
    int *neighbours,
    int *nnz_per_neighbour,
    int **col_indices_h,
    int **row_ptr_h,
    double **data_h
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

// assumes that the matrix is symmetric
// does a 1D decomposition over the rows
class distributed_matrix{
    public:
        int matrix_size;
        int rows_this_rank;
        int nnz;
    
        int size;
        int rank;
        int *counts;
        int *displacements;    
    
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
    
        // by default, we assume that the matrix is stored in CSR format
        // first matrix is own piece
        double **data_h;
        int **col_indices_h;
        int **row_ptr_h;

        // double **buffer_d;
        // double **data_d;
        // int **col_indices_d;
        // int **row_ptr_d;
        // cusparseSpMatDescr_t *descriptors;

        // Data types for MPI
        //int **cols_per_neighbour;
        //MPI_Datatype *fetch_types;

    distributed_matrix(
        int matrix_size,
        int nnz,
        int *counts,
        int *displacements,
        int *col_indices,
        int *row_ptr,
        double *data,
        MPI_Comm comm
    ){
        // int size;
        // int rank;
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);

        this->matrix_size = matrix_size;
        this->nnz = nnz;
        // this->size = size;
        // this->rank = rank;

        this->counts = new int[size];
        this->displacements = new int[size];
        for(int i = 0; i < size; i++){
            this->counts[i] = counts[i];
            this->displacements[i] = displacements[i];
        }

        // split_matrix(matrix_size, size, counts, displacements);
        rows_this_rank = counts[rank];

        // find neighbours_flag
        neighbours_flag = new bool[size];
        std::cout << rank << " " << "Finding neighbours" << std::endl;
        find_neighbours(rows_this_rank, col_indices, row_ptr, size, counts,
            displacements, neighbours_flag, &number_of_neighbours);
        std::cout << rank  << "Number of neighbours: " << number_of_neighbours << std::endl;
        std::cout << rank  << "Neighbours Flags: ";
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
        construct_neighbours_list(size, rank, neighbours_flag, neighbours);
        std::cout << rank << " " << "Neighbours list: ";
        for(int k = 0; k < number_of_neighbours; k++){
            std::cout << neighbours[k] << " ";
        }
        std::cout << std::endl;


        nnz_per_neighbour = new int[number_of_neighbours];
        nnz_cols_per_neighbour = new int[number_of_neighbours];
        std::cout << rank << " " << "Constructing nnz per neighbour" << std::endl;
        construct_nnz_per_neighbour(rows_this_rank, col_indices, row_ptr, counts, displacements, number_of_neighbours, neighbours, nnz_per_neighbour);
        std::cout << rank << " " << "NNZ per neighbour: ";
        for(int k = 0; k < number_of_neighbours; k++){
            std::cout << nnz_per_neighbour[k] << " ";
        }
        std::cout << std::endl;
        std::cout << rank << " " << "Constructing nnz cols per neighbour" << std::endl;
        construct_nnz_cols_per_neighbour(matrix_size, rows_this_rank, col_indices, row_ptr, counts, displacements, number_of_neighbours, neighbours, nnz_cols_per_neighbour);
        std::cout << rank << " " << "NNZ cols per neighbour: ";
        for(int k = 0; k < number_of_neighbours; k++){
            std::cout << nnz_cols_per_neighbour[k] << " ";
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
            row_ptr_h[k] = new int[counts[rank]+1];
        }
        std::cout << rank << " " << "Splitting CSR" << std::endl;
        // split data, indices and row_ptr
        split_csr(rows_this_rank, col_indices, row_ptr, data, counts,
            displacements, number_of_neighbours, neighbours,
            nnz_per_neighbour, col_indices_h, row_ptr_h, data_h);

    }

    ~distributed_matrix(){
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
    }

};