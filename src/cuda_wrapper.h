// gpu functions
#include "utils.h"

extern "C" {
void get_gpu_info(char *gpu_string, int dev);
void set_gpu(int dev);
void update_charge_gpu(ELEMENT *gpu_site_element, 
                       double *gpu_site_x, 
                       double *gpu_site_y, 
                       double *gpu_site_z, 
                       int *gpu_site_charge);
}
