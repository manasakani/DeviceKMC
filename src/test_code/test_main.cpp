#include <iostream>
#include <cstdio>
#include <cstddef>
#include <stdlib.h>

#include "utils.h"
#include "cuda_wrapper.h"

// #define NUM_THREADS 512

// // Instantiate the template function for double
// template __global__ void reduce<double, NUM_THREADS>(const double* array_to_reduce, double* value, const int N);

// Explicit instantiation of the reduce function
// template __global__ void reduce<double, NUM_THREADS>(const double* array_to_reduce, double* value, const int N);


// // unit test for reduce kernel, checks correctness for large arrays --> move to a .cu file
// void test_reduce()
// {
//     int N = 70000;

//     int num_threads = 512;
//     int num_blocks = (N - 1) / num_threads + 1;

//     double *gpu_test_array;
//     double *gpu_test_sum;
//     double t_test = 0.0;
//     std::vector<double> test_array(N, 1.0);

//     gpuErrchk( cudaMalloc((void**)&gpu_test_array, N * sizeof(double)) );
//     gpuErrchk( cudaMalloc((void**)&gpu_test_sum, 1 * sizeof(double)) );
//     gpuErrchk( cudaMemcpy(gpu_test_array, test_array.data(), N * sizeof(double), cudaMemcpyHostToDevice) );
//     gpuErrchk( cudaMemcpy(gpu_test_sum, &t_test, 1 * sizeof(double), cudaMemcpyHostToDevice) );

//     reduce<<<num_blocks, num_threads, 512*sizeof(double)>>>(gpu_test_array, gpu_test_sum, N);
//     gpuErrchk( cudaGetLastError() );

//     gpuErrchk( cudaMemcpy(&t_test, gpu_test_sum, 1 * sizeof(double), cudaMemcpyDeviceToHost));
//     assert(t_test == 70000.0);
//     std::cout << "--> Ran test for kernel reduce()\n";
// }       

int main(int argc, char **argv)
{

    // Run tests
    std::cout << "unit tests would be run here, if they existed\n";

    // test_reduce();

    return 0;
}
