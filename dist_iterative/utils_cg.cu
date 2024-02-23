#include "utils_cg.h"

__global__ void _pack_gpu(
    double *packed_buffer,
    double *unpacked_buffer,
    int *indices,
    int number_of_elements
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < number_of_elements; i += blockDim.x * gridDim.x){
        packed_buffer[i] = unpacked_buffer[indices[i]];
    }
}

void pack_gpu(
    double *packed_buffer,
    double *unpacked_buffer,
    int *indices,
    int number_of_elements
)
{
    int block_size = 32;
    int num_blocks = (number_of_elements + block_size - 1) / block_size;
    _pack_gpu<<<num_blocks, block_size>>>(
        packed_buffer,
        unpacked_buffer,
        indices,
        number_of_elements
    );
}

void pack_gpu(
    double *packed_buffer,
    double *unpacked_buffer,
    int *indices,
    int number_of_elements,
    cudaStream_t stream
)
{
    int block_size = 32;
    int num_blocks = (number_of_elements + block_size - 1) / block_size;
    _pack_gpu<<<num_blocks, block_size, 0, stream>>>(
        packed_buffer,
        unpacked_buffer,
        indices,
        number_of_elements
    );
}

__global__ void _unpack_gpu(
    double *unpacked_buffer,
    double *packed_buffer,
    int *indices,
    int number_of_elements
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < number_of_elements; i += blockDim.x * gridDim.x){
        unpacked_buffer[indices[i]] = packed_buffer[i];
    }
}

void unpack_gpu(
    double *unpacked_buffer,
    double *packed_buffer,
    int *indices,
    int number_of_elements
)
{
    int block_size = 32;
    int num_blocks = (number_of_elements + block_size - 1) / block_size;
    _unpack_gpu<<<num_blocks, block_size>>>(
        unpacked_buffer,
        packed_buffer,
        indices,
        number_of_elements
    );
}

void unpack_gpu(
    double *unpacked_buffer,
    double *packed_buffer,
    int *indices,
    int number_of_elements,
    cudaStream_t stream
)
{
    int block_size = 32;
    int num_blocks = (number_of_elements + block_size - 1) / block_size;
    _unpack_gpu<<<num_blocks, block_size, 0, stream>>>(
        unpacked_buffer,
        packed_buffer,
        indices,
        number_of_elements
    );
}


__global__ void _cg_addvec(
    double * __restrict__ x,
    double beta,
    double * __restrict__ y,
    int n
)
{
    // y = x + beta * y
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x){
        y[i] = x[i] + beta * y[i];
    }
}

void cg_addvec(
    double *x,
    double beta,
    double *y,
    int n
)
{
    int block_size = 1024;
    int num_blocks = (n + block_size - 1) / block_size;
    _cg_addvec<<<num_blocks, block_size>>>(x, beta, y, n);
}

void cg_addvec(
    double *x,
    double beta,
    double *y,
    int n,
    cudaStream_t stream
)
{
    int block_size = 1024;
    int num_blocks = (n + block_size - 1) / block_size;
    _cg_addvec<<<num_blocks, block_size, 0, stream>>>(x, beta, y, n);
}

__global__ void _fused_daxpy(
    double alpha1,
    double alpha2,
    double * __restrict__ x1,
    double * __restrict__ x2,
    double * __restrict__ y1,
    double * __restrict__ y2,
    int n
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x){
        y1[i] = y1[i] + alpha1 * x1[i];
        y2[i] = y2[i] + alpha2 * x2[i];
    }
}

void fused_daxpy(
    double alpha1,
    double alpha2,
    double *x1,
    double *x2,
    double *y1,
    double *y2,
    int n
)
{
    int block_size = 1024;
    int num_blocks = (n + block_size - 1) / block_size;
    _fused_daxpy<<<num_blocks, block_size>>>(
        alpha1,
        alpha2,
        x1,
        x2,
        y1,
        y2,
        n
    );
}

void fused_daxpy(
    double alpha1,
    double alpha2,
    double *x1,
    double *x2,
    double *y1,
    double *y2,
    int n,
    cudaStream_t stream
)
{
    int block_size = 1024;
    int num_blocks = (n + block_size - 1) / block_size;
    _fused_daxpy<<<num_blocks, block_size, 0, stream>>>(
        alpha1,
        alpha2,
        x1,
        x2,
        y1,
        y2,
        n
    );
}

__global__ void _fused_daxpy2(
    double alpha1,
    double alpha2,
    double * __restrict__ x1,
    double * __restrict__ x2,
    double * __restrict__ y1,
    double * __restrict__ y2,
    int n
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = idx - blockDim.x * gridDim.x / 2;
    if(idx < n){
        y1[idx] = y1[idx] + alpha1 * x1[idx];
    }
    else if(idx2 >= 0 && idx2 < n){
        y2[idx2] = y2[idx2] + alpha2 * x2[idx2];
    }

}

void fused_daxpy2(
    double alpha1,
    double alpha2,
    double *x1,
    double *x2,
    double *y1,
    double *y2,
    int n
)
{
    int block_size = 1024;
    int num_blocks = (n + block_size - 1) / block_size;
    num_blocks *= 2;
    _fused_daxpy2<<<num_blocks, block_size>>>(
        alpha1,
        alpha2,
        x1,
        x2,
        y1,
        y2,
        n
    );
}

void fused_daxpy2(
    double alpha1,
    double alpha2,
    double *x1,
    double *x2,
    double *y1,
    double *y2,
    int n,
    cudaStream_t stream
)
{
    int block_size = 1024;
    int num_blocks = (n + block_size - 1) / block_size;
    num_blocks *= 2;
    _fused_daxpy2<<<num_blocks, block_size, 0, stream>>>(
        alpha1,
        alpha2,
        x1,
        x2,
        y1,
        y2,
        n
    );
}

__global__ void _elementwise_vector_vector(
    double * __restrict__ array1,
    double * __restrict__ array2,
    double * __restrict__ result,
    int size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < size; i += blockDim.x * gridDim.x){
        result[i] = array1[i] * array2[i];
    }

}

void elementwise_vector_vector(
    double *array1,
    double *array2,
    double *result,
    int size
)
{
    int block_size = 1024;
    int num_blocks = (size + block_size - 1) / block_size;
    _elementwise_vector_vector<<<num_blocks, block_size>>>(
        array1,
        array2,
        result,
        size
    );
}

void elementwise_vector_vector(
    double *array1,
    double *array2,
    double *result,
    int size,
    cudaStream_t stream
)
{
    int block_size = 1024;
    int num_blocks = (size + block_size - 1) / block_size;
    _elementwise_vector_vector<<<num_blocks, block_size, 0, stream>>>(
        array1,
        array2,
        result,
        size
    );
}
