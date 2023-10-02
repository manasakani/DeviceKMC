// gpu functions
#include "utils.h"

extern "C" {
void get_gpu_info(char *gpu_string, int dev);
void set_gpu(int dev);
void update_charge_gpu(ELEMENT *gpu_site_element, 
                       int *gpu_site_charge, int *gpu_site_is_metal,
                       int *gpu_neigh_idx, int N, int nn);
void update_temperatureglobal_gpu(const double *site_power, double *T_bg, const int N, const double a_coeff, const double b_coeff, const double number_steps, const double C_thermal, const double small_step);
}
