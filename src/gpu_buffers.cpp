#include "gpu_buffers.h"
#include "Device.h" 
#include "utils.h"
#include <cassert>

#ifdef USE_CUDA
#include <hip/hip_runtime.h>
#endif

void GPUBuffers::sync_HostToGPU(Device &device){

    assert(N_ > 0);
    assert(nn_ > 0);

    size_t dataSize = N_ * sizeof(ELEMENT);
    if (dataSize != device.site_element.size() * sizeof(ELEMENT)) {
        std::cout << "N_: " << N_ << "\n";
        std::cout << device.site_element.size() << "\n";
        fprintf(stderr, "ERROR: Size mismatch in GPU memory copy.\n");
        exit(EXIT_FAILURE);
    }

#ifdef USE_CUDA
    hipDeviceSynchronize();
    gpuErrchk( hipMemcpy(site_element, device.site_element.data(), N_ * sizeof(ELEMENT), hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(site_charge, device.site_charge.data(), N_ * sizeof(int), hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(site_power, device.site_power.data(), N_ * sizeof(double), hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(site_CB_edge, device.site_CB_edge.data(), N_ * sizeof(double), hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(site_potential_boundary, device.site_potential_boundary.data(), N_ * sizeof(double), hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(site_potential_charge, device.site_potential_charge.data(), N_ * sizeof(double), hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(site_temperature, device.site_temperature.data(), N_ * sizeof(double), hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(atom_CB_edge, device.atom_CB_edge.data(), N_atom_ * sizeof(double), hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(T_bg, &device.T_bg, 1 * sizeof(double), hipMemcpyHostToDevice) );
    hipDeviceSynchronize();
    gpuErrchk(hipGetLastError());
#endif
}

void GPUBuffers::sync_GPUToHost(Device &device){

#ifdef USE_CUDA
    hipDeviceSynchronize();
    gpuErrchk( hipMemcpy(device.site_element.data(), site_element, N_ * sizeof(ELEMENT), hipMemcpyDeviceToHost) );
    gpuErrchk( hipMemcpy(device.site_charge.data(), site_charge, N_ * sizeof(int), hipMemcpyDeviceToHost) );
    gpuErrchk( hipMemcpy(device.site_power.data(), site_power, N_ * sizeof(double), hipMemcpyDeviceToHost) );
    gpuErrchk( hipMemcpy(device.site_CB_edge.data(), site_CB_edge, N_ * sizeof(double), hipMemcpyDeviceToHost) );
    gpuErrchk( hipMemcpy(device.site_potential_boundary.data(), site_potential_boundary, N_ * sizeof(double), hipMemcpyDeviceToHost) );
    gpuErrchk( hipMemcpy(device.site_potential_charge.data(), site_potential_charge, N_ * sizeof(double), hipMemcpyDeviceToHost) );
    gpuErrchk( hipMemcpy(device.site_temperature.data(), site_temperature, N_ * sizeof(double), hipMemcpyDeviceToHost) );
    gpuErrchk( hipMemcpy(device.atom_CB_edge.data(), atom_CB_edge, N_atom_ * sizeof(double), hipMemcpyDeviceToHost) );
    gpuErrchk( hipMemcpy(&device.T_bg, T_bg, 1 * sizeof(double), hipMemcpyDeviceToHost) );
    hipDeviceSynchronize();
    gpuErrchk(hipGetLastError()); 
#endif
}

// copy back just the site_power into the power vector
void GPUBuffers::copy_power_fromGPU(std::vector<double> &power){
    power.resize(N_);
#ifdef USE_CUDA
    gpuErrchk( hipMemcpy(power.data(), site_power, N_ * sizeof(double), hipMemcpyDeviceToHost) );
    // hipDeviceSynchronize();
    // std::cout << "copied\n";
    // double psum = 0.0;
    // for (auto p : power)
    // {
    //     psum += p;
    // }
    // std::cout << "psum*1e9: " << psum*(1e9) << "\n";
#endif

}

// copy the background temperature TO the gpu buffer
void GPUBuffers::copy_Tbg_toGPU(double new_T_bg){
#ifdef USE_CUDA
    gpuErrchk( hipMemcpy(T_bg, &new_T_bg, 1 * sizeof(double), hipMemcpyHostToDevice) );
#endif

}

// void GPU::Buffers::copy_atom_CB_edge_toGPU(std::vector<double> &CB_edge){

// }


void GPUBuffers::copy_charge_toGPU(std::vector<int> &charge){
    charge.resize(N_);
#ifdef USE_CUDA
    gpuErrchk( hipMemcpy(site_charge, charge.data(), N_ * sizeof(int), hipMemcpyHostToDevice) );
#endif
}


void GPUBuffers::freeGPUmemory(){
#ifdef USE_CUDA
    hipFree(site_element);
    hipFree(site_x);
    hipFree(site_y);
    hipFree(site_z);
    hipFree(neigh_idx);
    hipFree(site_layer);
    hipFree(site_charge);
    hipFree(site_power);
    hipFree(site_potential_boundary);
    hipFree(site_potential_charge);
    hipFree(site_temperature);
    hipFree(T_bg);
    hipFree(metal_types);
    hipFree(sigma);
    hipFree(k);
    hipFree(lattice);
    hipFree(freq);
    //... FREE THE REST OF THE MEMORY !!! ...
#endif

//destroy handles!
}
