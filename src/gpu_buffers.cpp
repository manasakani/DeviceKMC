#include "gpu_buffers.h"
#include "Device.h" 
#include "utils.h"
#include <cassert>
#include <cuda.h>

// GPUBuffers::GPUBuffers(int N) {
//     this->N_ = N;
//     gpuErrchk( cudaMalloc((void**)&gpu_site_element, N_ * sizeof(ELEMENT)) );
//     gpuErrchk( cudaMalloc((void**)&gpu_site_x, N_  * sizeof(double)) );
//     gpuErrchk( cudaMalloc((void**)&gpu_site_y, N_  * sizeof(double)) );
//     gpuErrchk( cudaMalloc((void**)&gpu_site_z, N_  * sizeof(double)) );
//     gpuErrchk( cudaMalloc((void**)&gpu_site_charge, N_ * sizeof(int)) );
//     gpuErrchk( cudaMalloc((void**)&gpu_site_is_metal, N_* sizeof(int)) );
// }

void GPUBuffers::upload_HostToGPU(Device &device){

    assert(N_ > 0);
    if (gpu_site_element == nullptr || device.site_element.data() == nullptr) {
        fprintf(stderr, "Invalid GPU buffer or device data pointer.\n");
        exit(EXIT_FAILURE);
    }

    size_t dataSize = N_ * sizeof(ELEMENT);
    if (dataSize != device.site_element.size() * sizeof(ELEMENT)) {
        fprintf(stderr, "Size mismatch in GPU memory copy.\n");
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();
    gpuErrchk( cudaMemcpy(gpu_site_element, device.site_element.data(), N_ * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(gpu_site_x, device.site_x.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(gpu_site_y, device.site_y.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(gpu_site_z, device.site_z.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(gpu_site_charge, device.site_charge.data(), N_ * sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(gpu_site_is_metal, device.site_is_metal.data(), N_ * sizeof(int), cudaMemcpyHostToDevice) );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
}

void GPUBuffers::download_GPUToHost(Device &device){

    cudaDeviceSynchronize();
    gpuErrchk( cudaMemcpy(&device.site_element[0], gpu_site_element, N_ * sizeof(ELEMENT), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(device.site_x.data(), gpu_site_x, N_ * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(device.site_y.data(), gpu_site_y, N_ * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(device.site_z.data(), gpu_site_z, N_ * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(device.site_charge.data(), gpu_site_charge, N_ * sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(device.site_is_metal.data(), gpu_site_is_metal, N_ * sizeof(int), cudaMemcpyDeviceToHost) );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

}

void GPUBuffers::freeGPUmemory(){
    cudaFree(gpu_site_element);
    cudaFree(gpu_site_x);
    cudaFree(gpu_site_y);
    cudaFree(gpu_site_z);
    cudaFree(gpu_site_charge);
    cudaFree(gpu_site_is_metal);
}