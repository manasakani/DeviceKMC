#include "cuda_wrapper.h"
#include <stdio.h>

void get_gpu_info(char *gpu_string, int dev){
 struct cudaDeviceProp dprop;
 cudaGetDeviceProperties(&dprop, dev);
 strcpy(gpu_string,dprop.name);
}

void set_gpu(int dev){
 cudaSetDevice(dev);
}

// __global__ void update_m(double *m, long minidx, int np2) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     // int bid = blockIdx.x;

//     if (idx < np2) {
//         double minm = m[minidx];
//         m[idx] += abs(minm);
//     }
// }

__global__ void update_charge_V(const ELEMENT *element, int *charge, const int *neigh_idx, const int N, const int nn){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    int Vnn = 0;

    // each thread gets a different site to evaluate
    for (int idx = tid; idx < N; idx += total_threads) {

        if (tid < N && element[tid] == VACANCY){
            charge[tid] = 2;

            // iterate over the neighbors
            for (int j = tid * nn; j < (tid + 1) * nn; ++j){
                if (element[neigh_idx[j]] == VACANCY){
                    Vnn++;
                }
                if (Vnn > 3){
                    charge[tid] = 0;
                }
            }
        }
    }

    // # if __CUDA_ARCH__>=200
    // printf("%i \n", tid);
    // #endif  

}

void update_charge_gpu(ELEMENT *gpu_site_element, 
                       int *gpu_site_charge,
                       int *gpu_neigh_idx, int N, int nn){

    std::cout << "N: " << N << "\n";
    std::cout << "nn: " << nn << "\n";

    int num_threads = 512;
    int num_blocks = (N * nn - 1) / num_threads + 1;
    num_blocks = min(65535, num_blocks);

    update_charge_V<<<num_blocks, num_threads>>>(gpu_site_element, gpu_site_charge, gpu_neigh_idx, N, nn);

}