#include "utils_gpu.h"

__global__ void _extract_diagonal_gpu(
    double *data,
    int *col_indices,
    int *row_indptr,
    double *diagonal_values_inv_sqrt,
    int matrix_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        for(int j = row_indptr[i]; j < row_indptr[i+1]; j++){
            if(col_indices[j] == i){
                diagonal_values_inv_sqrt[i] = 1/std::sqrt(data[j]);
                break;
            }
        }
    }

}
void extract_diagonal_gpu(
    double *data,
    int *col_indices,
    int *row_indptr,
    double *diagonal_values_inv_sqrt,
    int matrix_size
)
{
    int block_size = 1024;
    int num_blocks = (matrix_size + block_size - 1) / block_size;
    _extract_diagonal_gpu<<<num_blocks, block_size>>>(
        data,
        col_indices,
        row_indptr,
        diagonal_values_inv_sqrt,
        matrix_size
    );
}

__global__ void _precondition_vector_gpu(
    double *array,
    double *diagonal_values_inv_sqrt,
    int matrix_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        array[i] = array[i] * diagonal_values_inv_sqrt[i];
    }

}
void precondition_vector_gpu(
    double *array,
    double *diagonal_values_inv_sqrt,
    int matrix_size
)
{
    int block_size = 1024;
    int num_blocks = (matrix_size + block_size - 1) / block_size;
    _precondition_vector_gpu<<<num_blocks, block_size>>>(
        array,
        diagonal_values_inv_sqrt,
        matrix_size
    );
}

__global__ void _unpreecondition_vector_gpu(
    double *array,
    double *diagonal_values_inv_sqrt,
    int matrix_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        array[i] = array[i] * 1/diagonal_values_inv_sqrt[i];
    }

}

void unpreecondition_vector_gpu(
    double *array,
    double *diagonal_values_inv_sqrt,
    int matrix_size
)
{
    int block_size = 1024;
    int num_blocks = (matrix_size + block_size - 1) / block_size;
    _unpreecondition_vector_gpu<<<num_blocks, block_size>>>(
        array,
        diagonal_values_inv_sqrt,
        matrix_size
    );
}


__global__ void _symmetric_precondition_matrix_gpu(
    double *data,
    int *col_indices,
    int *row_indptr,
    double *diagonal_values_inv_sqrt,
    int matrix_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        for(int j = row_indptr[i]; j < row_indptr[i+1]; j++){
            data[j] = data[j] *
            diagonal_values_inv_sqrt[i] * diagonal_values_inv_sqrt[col_indices[j]];
        }
    }
}

void symmetric_precondition_matrix_gpu(
    double *data,
    int *col_indices,
    int *row_indptr,
    double *diagonal_values_inv_sqrt,
    int matrix_size
)
{
    int block_size = 1024;
    int num_blocks = (matrix_size + block_size - 1) / block_size;
    _symmetric_precondition_matrix_gpu<<<num_blocks, block_size>>>(
        data,
        col_indices,
        row_indptr,
        diagonal_values_inv_sqrt,
        matrix_size
    );
}