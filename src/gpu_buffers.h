#pragma once
#include "utils.h"

// forward declaration of device class
class Device;

class GPUBuffers {

public:
    // varying parameters
    int *site_charge, *site_is_metal = nullptr;
    double *site_power, *site_potential = nullptr;
    double *T_bg = nullptr;

    // unchanging parameters:
    ELEMENT *site_element = nullptr; 
    double *site_x, *site_y, *site_z = nullptr;
    ELEMENT *metal_types;
    double *sigma, *k, *lattice;
    int *neigh_idx = nullptr;
    int num_metal_types_ = 0;
    int N_ = 0;
    int nn_ = 0;

    // uploads the local device attributes into the GPU memory versions
    void upload_HostToGPU(Device &device);

    // downloads the GPU device attributes into the local versions
    void download_GPUToHost(Device &device);

    // copy back just the site_power into the power vector
    void copy_power_fromGPU(std::vector<double> &power);
    
    // constructor allocates arrays in GPU memory
    GPUBuffers(int N, std::vector<double> site_x_in,  std::vector<double> site_y_in,  std::vector<double> site_z_in,
               int nn, double sigma_in, double k_in, std::vector<double> lattice_in, std::vector<int> neigh_idx_in, std::vector<ELEMENT> metals,
               int num_metals_types) {
                
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
        gpuErrchk( cudaMalloc((void**)&sigma, 1 * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&k, 1 * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&lattice, 3 * sizeof(double)) );

        cudaDeviceSynchronize();

        // fixed parameters which can be copied from the beginning:
        gpuErrchk( cudaMemcpy(site_x, site_x_in.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(site_y, site_y_in.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(site_z, site_z_in.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(metal_types, metals.data(), num_metal_types_ * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(sigma, &sigma_in, 1 * sizeof(double), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(k, &k_in, 1 * sizeof(double), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(lattice, lattice_in.data(), 3 * sizeof(double), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(neigh_idx, neigh_idx_in.data(), N_ * nn_ * sizeof(int), cudaMemcpyHostToDevice) );

        cudaDeviceSynchronize();
    }

    void freeGPUmemory();

};
