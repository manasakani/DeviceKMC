#include "gpu_buffers.h"
#include "Device.h" 
#include "utils.h"
#include <cassert>
#include <cuda.h>

void GPUBuffers::upload_HostToGPU(Device &device){

    assert(N_ > 0);
    assert(nn_ > 0);

    size_t dataSize = N_ * sizeof(ELEMENT);
    if (dataSize != device.site_element.size() * sizeof(ELEMENT)) {
        fprintf(stderr, "Size mismatch in GPU memory copy.\n");
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();
    gpuErrchk( cudaMemcpy(site_element, device.site_element.data(), N_ * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(site_x, device.site_x.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(site_y, device.site_y.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(site_z, device.site_z.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(site_charge, device.site_charge.data(), N_ * sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(site_is_metal, device.site_is_metal.data(), N_ * sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(neigh_idx, device.neigh_idx.data(), N_ * nn_ * sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(site_power, device.site_power.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(site_power, device.site_potential.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(T_bg, &device.T_bg, 1 * sizeof(double), cudaMemcpyHostToDevice) );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}

void GPUBuffers::download_GPUToHost(Device &device){

    cudaDeviceSynchronize();
    gpuErrchk( cudaMemcpy(&device.site_element[0], site_element, N_ * sizeof(ELEMENT), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(device.site_x.data(), site_x, N_ * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(device.site_y.data(), site_y, N_ * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(device.site_z.data(), site_z, N_ * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(device.site_charge.data(), site_charge, N_ * sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(device.site_is_metal.data(), site_is_metal, N_ * sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(device.neigh_idx.data(), neigh_idx, N_ * nn_ * sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(device.site_power.data(), site_power, N_ * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(device.site_potential.data(), site_power, N_ * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(&device.T_bg, T_bg, 1 * sizeof(double), cudaMemcpyDeviceToHost) );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

}

void GPUBuffers::freeGPUmemory(){
    cudaFree(site_element);
    cudaFree(site_x);
    cudaFree(site_y);
    cudaFree(site_z);
    cudaFree(site_charge);
    cudaFree(site_is_metal);
    cudaFree(site_power);
    cudaFree(site_potential);
    cudaFree(T_bg);
}