#pragma once
#include <hip/hip_runtime.h>

void extract_diagonal_inv_sqrt(
    double *data,
    int *col_indices,
    int *row_indptr,
    double *diagonal_values_inv_sqrt,
    int matrix_size
);

void extract_diagonal_inv(
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

void invert_array(
    double *array_in,
    double *array_out,
    int matrix_size
);