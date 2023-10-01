#pragma once
#include "utils.h"

// forward declaration of device class for member function declarations
class Device;

class GPUBuffers {

public:
    ELEMENT *gpu_site_element;
    double *gpu_site_x, *gpu_site_y, *gpu_site_z;
    int *gpu_site_charge, *gpu_site_is_metal;
    int N_ = 0;

    // uploads the local device attributes into the GPU memory versions
    void upload_HostToGPU(Device &device);

    // downloads the GPU device attributes into the local versions
    void download_GPUToHost(Device &device);
    
    // constructor allocates arrays in GPU memory
    GPUBuffers(int N) {
        this->N_ = N;
        cudaDeviceSynchronize();
        gpuErrchk( cudaMalloc((void**)&gpu_site_element, N_ * sizeof(ELEMENT)) );
        gpuErrchk( cudaMalloc((void**)&gpu_site_x, N_  * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&gpu_site_y, N_  * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&gpu_site_z, N_  * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&gpu_site_charge, N_ * sizeof(int)) );
        gpuErrchk( cudaMalloc((void**)&gpu_site_is_metal, N_* sizeof(int)) );
        cudaDeviceSynchronize();
    }

    void freeGPUmemory();

    // ~GPUBuffers() {     
    //     cudaFree(gpu_site_element);
    //     cudaFree(gpu_site_x);
    //     cudaFree(gpu_site_y);
    //     cudaFree(gpu_site_z);
    //     cudaFree(gpu_site_charge);
    //     cudaFree(gpu_site_is_metal);
    // }

};
