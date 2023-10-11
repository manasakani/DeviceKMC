#pragma once
#include "utils.h"
#include "cuda_wrapper.h"

// forward declaration of device class
class Device;

// Member variables are pointers to GPU memory unless specified with _host
class GPUBuffers {

public:
    // varying parameters
    int *site_charge = nullptr;
    double *site_power, *site_potential, *site_temperature = nullptr;
    double *T_bg = nullptr;

    // unchanging parameters:
    ELEMENT *site_element = nullptr; 
    double *site_x, *site_y, *site_z = nullptr;
    ELEMENT *metal_types;
    double *sigma, *k, *lattice, *freq;
    int *neigh_idx, *site_layer = nullptr;

    // NOT gpu pointers, passed by value
    int num_metal_types_ = 0;
    int N_ = 0;
    int nn_ = 0;

    // helper variables stored on host:
    std::vector<double> E_gen_host, E_rec_host, E_Vdiff_host, E_Odiff_host;

    // uploads the local device attributes into the GPU memory versions
    void sync_HostToGPU(Device &device);

    // downloads the GPU device attributes into the local versions
    void sync_GPUToHost(Device &device);

    // copy back just the site_power into the power vector
    void copy_power_fromGPU(std::vector<double> &power);

    void copy_charge_toGPU(std::vector<int> &charge);
    
    // constructor allocates arrays in GPU memory
    GPUBuffers(std::vector<Layer> layers, std::vector<int> site_layer_in, double freq_in, int N,
               std::vector<double> site_x_in,  std::vector<double> site_y_in,  std::vector<double> site_z_in,
               int nn, double sigma_in, double k_in, std::vector<double> lattice_in, std::vector<int> neigh_idx_in, std::vector<ELEMENT> metals,
               int num_metals_types) {
                
        this->N_ = N;
        this->nn_ = nn;
        this->num_metal_types_ = num_metals_types;

        // make layer arrays
        for (auto l : layers){
            E_gen_host.push_back(l.E_gen_0);
            E_rec_host.push_back(l.E_rec_1);
            E_Vdiff_host.push_back(l.E_diff_2);
            E_Odiff_host.push_back(l.E_diff_3);
        }
        int num_layers = layers.size();

        // variables to store in GPU global memory
        copytoConstMemory(E_gen_host, E_rec_host, E_Vdiff_host, E_Odiff_host);

        cudaDeviceSynchronize();
        
        // member variables of the KMCProcess 
        gpuErrchk( cudaMalloc((void**)&site_layer, N_ * sizeof(int)) );

        // member variables of the Device
        gpuErrchk( cudaMalloc((void**)&site_element, N_ * sizeof(ELEMENT)) );
        gpuErrchk( cudaMalloc((void**)&metal_types, num_metal_types_ * sizeof(ELEMENT)) );
        gpuErrchk( cudaMalloc((void**)&site_x, N_  * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_y, N_  * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_z, N_  * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_power, N_ * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_potential, N_ * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_temperature, N_ * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_charge, N_ * sizeof(int)) );
        gpuErrchk( cudaMalloc((void**)&neigh_idx, N_ * nn_ * sizeof(int)) );
        gpuErrchk( cudaMalloc((void**)&T_bg, 1 * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&sigma, 1 * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&k, 1 * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&lattice, 3 * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&freq, 1 * sizeof(double)) );

        cudaDeviceSynchronize();

        // fixed parameters which can be copied from the beginning:
        gpuErrchk( cudaMemcpy(site_layer, site_layer_in.data(), N_ * sizeof(int), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(site_x, site_x_in.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(site_y, site_y_in.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(site_z, site_z_in.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(metal_types, metals.data(), num_metal_types_ * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(sigma, &sigma_in, 1 * sizeof(double), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(k, &k_in, 1 * sizeof(double), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(freq, &freq_in, 1 * sizeof(double), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(lattice, lattice_in.data(), 3 * sizeof(double), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(neigh_idx, neigh_idx_in.data(), N_ * nn_ * sizeof(int), cudaMemcpyHostToDevice) );

        cudaDeviceSynchronize();
    }

    void freeGPUmemory();

};
