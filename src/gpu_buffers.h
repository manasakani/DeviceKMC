#pragma once
#include "utils.h"

// forward declaration of device class for member function declarations
class Device;

class GPUBuffers {

public:
    ELEMENT *site_element, *metal_types;
    double *site_x, *site_y, *site_z, *site_power, *site_potential;
    int *site_charge, *site_is_metal;
    int *neigh_idx;
    int N_ = 0;
    int nn_ = 0;
    int num_metal_types_ = 0;
    double *T_bg;

    // uploads the local device attributes into the GPU memory versions
    void upload_HostToGPU(Device &device);

    // downloads the GPU device attributes into the local versions
    void download_GPUToHost(Device &device);

    // copy back just the site_power into the power vector
    void copy_power_fromGPU(std::vector<double> &power);
    
    // constructor allocates arrays in GPU memory
    GPUBuffers(int N, int nn, std::vector<ELEMENT> metals, int num_metals_types) {
        this->N_ = N;
        this->nn_ = nn;
        this->num_metal_types_ = num_metals_types;

        cudaDeviceSynchronize();
        gpuErrchk( cudaMalloc((void**)&site_element, N_ * sizeof(ELEMENT)) );
        gpuErrchk( cudaMalloc((void**)&metal_types, num_metal_types_ * sizeof(ELEMENT)) );
        gpuErrchk( cudaMalloc((void**)&site_x, N_  * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_y, N_  * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_z, N_  * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_power, N_ * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_potential, N_ * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_charge, N_ * sizeof(int)) );
        gpuErrchk( cudaMalloc((void**)&site_is_metal, N_* sizeof(int)) );
        gpuErrchk( cudaMalloc((void**)&neigh_idx, N_ * nn_ * sizeof(int)) );
        gpuErrchk( cudaMalloc((void**)&T_bg, 1 * sizeof(double)) );

        // fixed parameters:
        gpuErrchk( cudaMemcpy(metal_types, metals.data(), num_metal_types_ * sizeof(ELEMENT), cudaMemcpyHostToDevice) );

        cudaDeviceSynchronize();
    }

    void freeGPUmemory();

};
