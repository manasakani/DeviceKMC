#include "utils.h"
#include "random_num.h"
// #include "gpu_buffers.h"

// forward declaration of gpubuf class          
class GPUBuffers;

extern "C" {

void get_gpu_info(char *gpu_string, int dev);

void set_gpu(int dev);

void update_charge_gpu(ELEMENT *gpu_site_element, 
                       int *gpu_site_charge,
                       int *gpu_neigh_idx, int N, int nn, 
                       const ELEMENT *metals, const int num_metals);

void update_temperatureglobal_gpu(const double *site_power, 
                                  double *T_bg, const int N, 
                                  const double a_coeff, const double b_coeff, 
                                  const double number_steps, const double C_thermal, 
                                  const double small_step);

void background_potential_gpu(cusolverDnHandle_t handle, const GPUBuffers &gpubuf, const int N, const int N_left_tot, const int N_right_tot,
                              const double d_Vd, const int pbc, const double d_high_G, const double d_low_G, const double nn_dist,
                              const int num_metals);

void poisson_gridless_gpu(const int num_atoms_contact, const int pbc, const int N, const double *lattice,
                          const double *sigma, const double *k,
                          const double *posx, const double *posy, const double *posz,
                          const int *site_charge, double *site_potential);

void update_power_gpu(cublasHandle_t handle, cusolverDnHandle_t handle_cusolver, GPUBuffers &gpubuf, const int N, const int num_source_inj, const int num_ground_ext,
                      const double Vd, const int pbc, const double high_G, const double low_G,
                      const double nn_dist, const double m_e, const double V0, int num_metals, const double t_ox, double *imacro);

double execute_kmc_step_gpu(const int N, const int nn, const int *neigh_idx, const int *site_layer,
                          const double *lattice, const int pbc, const double *T_bg, 
                          const double *freq, const double *sigma, const double *k,
                          const double *posx, const double *posy, const double *posz, 
                          const double *site_potential, const double *site_temperature,
                          ELEMENT *site_element, int *site_charge, RandomNumberGenerator &rng, const int *neigh_idx_host);

void copytoConstMemory(std::vector<double> E_gen, std::vector<double> E_rec, std::vector<double> E_Vdiff, std::vector<double> E_Odiff);

}
