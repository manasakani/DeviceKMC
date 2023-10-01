#pragma once
#include "utils.h"

// forward declaration of device class for member function declarations
class Device;

class GPUBuffers {

public:
    ELEMENT *gpu_site_element;
    double *gpu_site_x, *gpu_site_y, *gpu_site_z;
    int *gpu_site_charge, *gpu_site_is_metal;
    int *gpu_neigh_idx;
    int N_ = 0;
    int nn_ = 0;

    // uploads the local device attributes into the GPU memory versions
    void upload_HostToGPU(Device &device);

    // downloads the GPU device attributes into the local versions
    void download_GPUToHost(Device &device);
    
    // constructor allocates arrays in GPU memory
    GPUBuffers(int N, int nn) {
        this->N_ = N;
        this->nn_ = nn;

        cudaDeviceSynchronize();
        gpuErrchk( cudaMalloc((void**)&gpu_site_element, N_ * sizeof(ELEMENT)) );
        gpuErrchk( cudaMalloc((void**)&gpu_site_x, N_  * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&gpu_site_y, N_  * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&gpu_site_z, N_  * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&gpu_site_charge, N_ * sizeof(int)) );
        gpuErrchk( cudaMalloc((void**)&gpu_site_is_metal, N_* sizeof(int)) );
        gpuErrchk( cudaMalloc((void**)&gpu_neigh_idx, N_ * nn_ * sizeof(int)) );

        cudaDeviceSynchronize();
    }

    void freeGPUmemory();

};
