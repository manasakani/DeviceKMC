#include "cuda_wrapper.h"

void get_gpu_info(char *gpu_string, int dev){
 struct cudaDeviceProp dprop;
 cudaGetDeviceProperties(&dprop, dev);
 strcpy(gpu_string,dprop.name);
}

void set_gpu(int dev){
 cudaSetDevice(dev);
}

void update_charge_gpu(ELEMENT *gpu_site_element, double *gpu_site_x, double *gpu_site_y, double *gpu_site_z, int *gpu_site_charge){
    std::cout << "In updateCharge_gpu\n";
}