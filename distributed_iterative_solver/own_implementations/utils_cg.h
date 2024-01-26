#pragma once
#include <cuda_runtime.h>


void pack_gpu(
    double *packed_buffer,
    double *unpacked_buffer,
    int *indices,
    int number_of_elements);

void pack_gpu(
    double *packed_buffer,
    double *unpacked_buffer,
    int *indices,
    int number_of_elements,
    cudaStream_t stream);

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

void unpack_gpu2(
    double *unpacked_buffer,
    double *packed_buffer,
    int *indices,
    int number_of_elements,
    cudaStream_t stream);

void cg_addvec(
    double *x,
    double beta,
    double *y,
    int n);

void cg_addvec(
    double *x,
    double beta,
    double *y,
    int n,
    cudaStream_t stream);

void fused_daxpy(
    double alpha1,
    double alpha2,
    double *x1,
    double *x2,
    double *y1,
    double *y2,
    int n);

void fused_daxpy(
    double alpha1,
    double alpha2,
    double *x1,
    double *x2,
    double *y1,
    double *y2,
    int n,
    cudaStream_t stream);

void fused_daxpy2(
    double alpha1,
    double alpha2,
    double *x1,
    double *x2,
    double *y1,
    double *y2,
    int n);

void fused_daxpy2(
    double alpha1,
    double alpha2,
    double *x1,
    double *x2,
    double *y1,
    double *y2,
    int n,
    cudaStream_t stream);