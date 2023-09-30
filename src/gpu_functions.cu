#include "cuda_wrapper.h"

extern "C"
void get_gpu_info(char *gpu_string, int dev){
 struct cudaDeviceProp dprop;
 cudaGetDeviceProperties(&dprop, dev);
 strcpy(gpu_string,dprop.name);
}

extern "C"
void set_gpu(int dev){
 cudaSetDevice(dev);
}

