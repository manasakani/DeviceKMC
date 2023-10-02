#pragma once
#include "utils.h"

// forward declaration of device class for member function declarations
class Device;

class GPUBuffers {

public:
    ELEMENT *site_element;
    double *site_x, *site_y, *site_z, *site_power;
    int *site_charge, *site_is_metal;
    int *neigh_idx;
    int N_ = 0;
    int nn_ = 0;
    double *T_bg;

    // uploads the local device attributes into the GPU memory versions
    void upload_HostToGPU(Device &device);

    // downloads the GPU device attributes into the local versions
    void download_GPUToHost(Device &device);
    
    // constructor allocates arrays in GPU memory
    GPUBuffers(int N, int nn) {
        this->N_ = N;
        this->nn_ = nn;

        cudaDeviceSynchronize();
        gpuErrchk( cudaMalloc((void**)&site_element, N_ * sizeof(ELEMENT)) );
        gpuErrchk( cudaMalloc((void**)&site_x, N_  * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_y, N_  * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_z, N_  * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_power, N_ * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_charge, N_ * sizeof(int)) );
        gpuErrchk( cudaMalloc((void**)&site_is_metal, N_* sizeof(int)) );
        gpuErrchk( cudaMalloc((void**)&neigh_idx, N_ * nn_ * sizeof(int)) );
        gpuErrchk( cudaMalloc((void**)&T_bg, 1 * sizeof(double)) );

        cudaDeviceSynchronize();
    }

    void freeGPUmemory();

};
