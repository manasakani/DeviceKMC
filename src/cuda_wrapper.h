// gpu functions
#include "utils.h"

extern "C" {
void get_gpu_info(char *gpu_string, int dev);
void set_gpu(int dev);
void update_charge_gpu(ELEMENT *gpu_site_element, 
                       int *gpu_site_charge,
                       int *gpu_neigh_idx, int N, int nn);
}
