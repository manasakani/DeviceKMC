#include "cuda_wrapper.h"
#include <stdio.h>
//#include <thrust/reduce.h>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include <math_functions.h>

#define NUM_THREADS 1024

void get_gpu_info(char *gpu_string, int dev){
 struct cudaDeviceProp dprop;
 cudaGetDeviceProperties(&dprop, dev);
 strcpy(gpu_string,dprop.name);
}

void set_gpu(int dev){
 cudaSetDevice(dev);
}

// ********************************************************
// *************** GPU HELPER FUNCTIONS *******************
// ********************************************************

// returns true if thing is present in the vector of things
template <typename T>
__device__ int is_in_array_gpu(const T *array, const T element, const int size) {

    for (int i = 0; i < size; ++i) {
        if (array[i] == element) {
        return 1;
        }
    }
    return 0;
}

__device__ double site_dist_gpu(double pos1x, double pos1y, double pos1z,
                                double pos2x, double pos2y, double pos2z,
                                double lattx, double latty, double lattz, bool pbc)
{

    double dist = 0;

    if (pbc == 1)
    {
        double dist_x = pos1x - pos2x;
        double distance_frac[3];

        distance_frac[1] = (pos1y - pos2y) / latty;
        distance_frac[1] -= round(distance_frac[1]);
        distance_frac[2] = (pos1z - pos2z) / lattz;
        distance_frac[2] -= round(distance_frac[2]);

        double dist_xyz[3];
        dist_xyz[0] = dist_x;

        dist_xyz[1] = distance_frac[1] * latty;
        dist_xyz[2] = distance_frac[2] * lattz;

        dist = sqrt(dist_xyz[0] * dist_xyz[0] + dist_xyz[1] * dist_xyz[1] + dist_xyz[2] * dist_xyz[2]);
        
    }
    else
    {
        dist = sqrt(pow(pos2x - pos1x, 2) + pow(pos2y - pos1y, 2) + pow(pos2z - pos1z, 2));
    }

    return dist;
}

__device__ double v_solve_gpu(double r_dist, int charge, const double *sigma, const double *k) { 

    double q = 1.60217663e-19;              // [C]
    double vterm = static_cast<double>(charge) * erfc(r_dist / ((*sigma) * sqrt(2.0))) * (*k) * q / r_dist; 

    return vterm;
}


// ********************************************************
// ******************** KERNELS ***************************
// ********************************************************

// // iterates over every pair of sites, and does an operation based on the distance
template <typename T, int NTHREADS>
__global__ void calculate_pairwise_interaction(const T* posx, const T* posy, const T*posz, 
                                               const double *lattice, const int pbc, 
                                               const int N, const double *sigma, const double *k, 
                                               const int *charge, T* potential){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    double V_temp = 0;
    double dist;

    // each thread gets a different site to evaluate
    for (int i = tid; i < N; i += total_threads) {
        
        // iterate over the neighbors
        for (int j = 0; j < N; j++){
            if (i != j && charge[j] != 0){

                dist = 1e-10 * site_dist_gpu(posx[i], posy[i], posz[i], 
                                    posx[j], posy[j], posz[j], 
                                    lattice[0], lattice[1], lattice[2], pbc);
                V_temp += v_solve_gpu(dist, charge[j], sigma, k);
                
            }
        }
        potential[tid] += V_temp;
    }
}

__global__ void update_charge(const ELEMENT *element, 
                              int *charge, 
                              const int *neigh_idx, 
                              const int N, const int nn, 
                              const ELEMENT* metals, const int num_metals){

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
                if (is_in_array_gpu(metals, element[neigh_idx[j]], num_metals)){
                    charge[tid] = 0;
                }
                if (Vnn >= 3){
                    charge[tid] = 0;
                }
            }
        }

        if (tid < N && element[tid] == OXYGEN_DEFECT){
            charge[tid] = -2;

            // iterate over the neighbors
            for (int j = tid * nn; j < (tid + 1) * nn; ++j){
                
                if (is_in_array_gpu(metals, element[neigh_idx[j]], num_metals)){
                    charge[tid] = 0;
                }
            }
        }
    }
     
    // TODO: iterate over the site list again to extract the #vacancies and ions to output
}

//reduces the array into the value 
template <typename T, int NTHREADS>
__global__ void reduce(const T* array_to_reduce, T* value, const int N){

    __shared__ T buf[NTHREADS];
    
    int num_threads = blockDim.x;                           // number of threads in this block
    int blocks_per_row = (N-1)/num_threads + 1;             // number of blocks to fit in this array
    int block_id = blockIdx.x;                              // id of the block
    int tid = threadIdx.x;                                  // local thread id to this block
    int row = block_id / blocks_per_row;                    // which 'row' of the array to work on, rows are the overflow

    buf[tid] = 0;

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

//called by a single gpu-thread
__global__ void update_temp_global(double *P_tot, double* T_bg, const double a_coeff, const double b_coeff, const double number_steps, const double C_thermal, const double small_step)
{
    double c_coeff = b_coeff + *P_tot/C_thermal * small_step;
    double T_intermediate = *T_bg;
    int step = number_steps;
    *T_bg = c_coeff*(1.0-pow(a_coeff, (double) step)) / (1.0-a_coeff) + pow(a_coeff, (double) step)* T_intermediate;
}

// ********************************************************
// ****************** KERNEL UNIT TESTS *******************
// ********************************************************

// unit test for reduce kernel, checks correctness for large arrays
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
    std::cout << "--> Ran test for kernel reduce()\n";
}                

// ********************************************************
// *************** WRAPPER FUNCTIONS **********************
// ********************************************************

// wrapper function for the update_charge kernel, sets the site_charge array
void update_charge_gpu(ELEMENT *site_element, 
                       int *site_charge,
                       int *neigh_idx, int N, int nn, 
                       const ELEMENT *metals, const int num_metals){

    int num_threads = 512;
    int num_blocks = (N * nn - 1) / num_threads + 1; // revise kernel to distribute pair-resolved work instead of site-resolved work
    num_blocks = min(65535, num_blocks);

    update_charge<<<num_blocks, num_threads>>>(site_element, site_charge, neigh_idx, N, nn, metals, num_metals);
}

void update_temperatureglobal_gpu(const double *site_power, double *T_bg, const int N, const double a_coeff, const double b_coeff, const double number_steps, const double C_thermal, const double small_step){

    int num_threads = 512;
    int num_blocks = (N - 1) / num_threads + 1;
    num_blocks = min(65535, num_blocks);

    double *P_tot;
    gpuErrchk( cudaMalloc((void**)&P_tot, 1 * sizeof(double)) );
    gpuErrchk( cudaMemset(P_tot, 0, 1 * sizeof(double)) );

    //collect site_power
    reduce<double, NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS*sizeof(double)>>>(site_power, P_tot, N);

    //update the temperature
    update_temp_global<<<1, 1>>>(P_tot, T_bg, a_coeff, b_coeff, number_steps, C_thermal, small_step);

    // double p_test = 0.0;
    // gpuErrchk( cudaMemcpy(&p_test, P_tot, 1 * sizeof(double), cudaMemcpyDeviceToHost));
    // std::cout << "power: " << p_test << "\n";
    // double t_test = 0.0;
    // gpuErrchk( cudaMemcpy(&t_test, T_bg, 1 * sizeof(double), cudaMemcpyDeviceToHost));
    // std::cout << "temperature: " << t_test << "\n";

}

void background_potential_gpu(cusolverDnHandle_t handle, const int num_atoms_contact, const double Vd, const double *lattice,
                              const double G_coeff, const double high_G, const double low_G, const int *site_is_metal){

    std::cout << "inside background_potential_gpu\n";
    std::cout << "still need to implement this!\n";

}

void poisson_gridless_gpu(const int num_atoms_contact, const int pbc, const int N, const double *lattice, 
                          const double *sigma, const double *k,
                          const double *posx, const double *posy, const double *posz, 
                          const int *site_charge, double *site_potential){

    int num_threads = 512;
    int num_blocks = (N) / num_threads + 1;
    num_blocks = min(65535, num_blocks);

    calculate_pairwise_interaction<double, NUM_THREADS><<<num_blocks, num_threads>>>(posx, posy, posz, lattice, pbc, N, sigma, k, site_charge, site_potential);
                                                
}

    // # if __CUDA_ARCH__>=200
    // printf("%i \n", tid);
    // #endif  