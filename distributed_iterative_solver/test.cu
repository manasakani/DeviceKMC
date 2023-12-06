#include <cuda_runtime.h>

__global__ void naiveDgemm(
    const int m,
    const int n,
    const int k,
    const double alpha,
    const double* const A,
    const double* const B,
    const double beta,
    double* const C)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes one element of C by accumulating results into Cvalue
    double Cvalue = 0;
    for (int e = 0; e < k; ++e)
        Cvalue += A[e*m + row] * B[col*k + e];

    C[col*m + row] = alpha*Cvalue + beta*C[col*m + row];
}
