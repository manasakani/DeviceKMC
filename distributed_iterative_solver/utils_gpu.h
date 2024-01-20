#pragma once
#include <cuda_runtime.h>

void extract_diagonal_gpu(
    double *data,
    int *col_indices,
    int *row_indptr,
    double *diagonal_values_inv_sqrt,
    int matrix_size
);

void precondition_vector_gpu(
    double *array,
    double *diagonal_values_inv_sqrt,
    int matrix_size
);

void unpreecondition_vector_gpu(
    double *array,
    double *diagonal_values_inv_sqrt,
    int matrix_size
);

void symmetric_precondition_matrix_gpu(
    double *data,
    int *col_indices,
    int *row_indptr,
    double *diagonal_values_inv_sqrt,
    int matrix_size
);

void pack_gpu(
    double *packed_buffer,
    double *unpacked_buffer,
    int *indices,
    int number_of_elements);

void unpack_gpu(
    double *unpacked_buffer,
    double *packed_buffer,
    int *indices,
    int number_of_elements);

void unpack_gpu(
    double *unpacked_buffer,
    double *packed_buffer,
    int *indices,
    int number_of_elements,
    cudaStream_t stream);