#include "cuda_wrapper.h"
#include <stdio.h>
//#include <thrust/reduce.h>
#include <vector>
#include <cassert>


#define NUM_THREADS 1024

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

__global__ void update_charge(const ELEMENT *element, 
                                int *charge, 
                                const int *site_is_metal, 
                                const int *neigh_idx, 
                                const int N, const int nn){

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
                if (site_is_metal[neigh_idx[j]]){
                    charge[tid] = 0;
                }
                if (Vnn > 3){
                    charge[tid] = 0;
                }
            }
        }

        if (tid < N && element[tid] == OXYGEN_DEFECT){
            charge[tid] = -2;

            // iterate over the neighbors
            for (int j = tid * nn; j < (tid + 1) * nn; ++j){
                if (site_is_metal[neigh_idx[j]]){
                    charge[tid] = 0;
                }
            }
        }
    }
     
    // iterate over the site list again to extract the #vacancies and ions to output

    // # if __CUDA_ARCH__>=200
    // printf("%i \n", tid);
    // #endif  

}

void update_charge_gpu(ELEMENT *site_element, 
                       int *site_charge,
                       int *site_is_metal,
                       int *neigh_idx, int N, int nn){

    // std::cout << "N: " << N << "\n";
    // std::cout << "nn: " << nn << "\n";

    int num_threads = 512;
    int num_blocks = (N * nn - 1) / num_threads + 1;
    num_blocks = min(65535, num_blocks);

    update_charge<<<num_blocks, num_threads>>>(site_element, site_charge, site_is_metal, neigh_idx, N, nn);

}


template <typename T, int NTHREADS>
__global__ void reduce(const T* array_to_reduce, T* value, const int N){
    //reduces the array into the value 

    __shared__ T buf[NTHREADS];
    
    int num_threads = blockDim.x; // number of threads in this block
    int blocks_per_row = (N-1)/num_threads + 1; // number of blocks to fit in this array

    int block_id = blockIdx.x; // id of the block
    int tid = threadIdx.x; // local thread id to this block
    int row = block_id / blocks_per_row; // which 'row' of the array to work on, takes care of the overflow

    buf[tid] = 0;

    // overflow for when the number of blocks is too small for the total array
    for (int ridx = row; ridx < N/(blocks_per_row*num_threads) + 1; ridx++){
    
        if (ridx*blocks_per_row*num_threads + block_id * num_threads + tid < N){
            buf[tid] = array_to_reduce[ridx*blocks_per_row*num_threads + block_id * num_threads + tid];
        }
       
        int width = num_threads / 2;

        while (width != 0){
            __syncthreads();
            if (tid < width){
                buf[tid] += buf[tid+width];
            }
            width /= 2;
        }

        if (tid == 0){
           atomicAdd(value, buf[0]);
        }
    }
}

// unit test for reduce kernel
void test_reduce()
{
    int N = 70000;
    int num_threads = 512;
    int num_blocks = (N - 1) / num_threads + 1;
    num_blocks = min(65535, num_blocks);

    double *gpu_test_array;
    double *gpu_test_sum;
    double t_test = 0.0;
    std::vector<double> test_array(N, 1.0);

    gpuErrchk( cudaMalloc((void**)&gpu_test_array, N * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&gpu_test_sum, 1 * sizeof(double)) );
    gpuErrchk( cudaMemcpy(gpu_test_array, test_array.data(), N * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(gpu_test_sum, &t_test, 1 * sizeof(double), cudaMemcpyHostToDevice) );

    reduce<double, NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS*sizeof(double)>>>(gpu_test_array, gpu_test_sum, N);

    gpuErrchk( cudaGetLastError() );
    gpuErrchk( cudaMemcpy(&t_test, gpu_test_sum, 1 * sizeof(double), cudaMemcpyDeviceToHost));
    assert(t_test == 70000.0);
}

__global__ void update_temp_global(double *P_tot, double* T_bg, const double a_coeff, const double b_coeff, const double number_steps, const double C_thermal, const double small_step)
{
    double c_coeff = b_coeff + *P_tot/C_thermal * small_step;
    double T_intermediate = *T_bg;
    int step = number_steps;
    *T_bg = c_coeff*(1.0-pow(a_coeff, (double) step)) / (1.0-a_coeff) + pow(a_coeff, (double) step)* T_intermediate;
}

void update_temperatureglobal_gpu(const double *site_power, double *T_bg, const int N, const double a_coeff, const double b_coeff, const double number_steps, const double C_thermal, const double small_step){

    int num_threads = 512;
    int num_blocks = (N - 1) / num_threads + 1;
    num_blocks = min(65535, num_blocks);

    //collect site_power
    double *P_tot;
    gpuErrchk( cudaMalloc((void**)&P_tot, 1 * sizeof(double)) );
    gpuErrchk( cudaMemset(P_tot, 0, 1 * sizeof(double)) );
    reduce<double, NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS*sizeof(double)>>>(site_power, P_tot, N);

    //update the temperature
    update_temp_global<<<1, 1>>>(P_tot, T_bg, a_coeff, b_coeff, number_steps, C_thermal, small_step);
}