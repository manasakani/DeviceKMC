#include "utils.h"

extern "C" {
void get_gpu_info(char *gpu_string, int dev);
void set_gpu(int dev);
void update_charge_gpu(ELEMENT *gpu_site_element, 
                       int *gpu_site_charge, int *gpu_site_is_metal,
                       int *gpu_neigh_idx, int N, int nn, 
                       const ELEMENT *metals, const int num_metals);
void update_temperatureglobal_gpu(const double *site_power, 
                                  double *T_bg, const int N, 
                                  const double a_coeff, const double b_coeff, 
                                  const double number_steps, const double C_thermal, 
                                  const double small_step);
void background_potential_gpu(cusolverDnHandle_t handle, const int num_atoms_contact, 
                              const double Vd, const double *lattice,
                              const double G_coeff, const double high_G, 
                              const double low_G, const int *site_is_metal);
void poisson_gridless_gpu(const int num_atoms_contact, const double *lattice, 
                          const int *site_charge, double *site_potential);
}
