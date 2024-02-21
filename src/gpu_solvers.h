#pragma once

#include "utils.h"
#include "random_num.h"
// #define NUM_THREADS 512

// excluded for testing
#ifndef COMPILE_WITH_TESTS
#include "gpu_buffers.h"
#endif 

#include <stdio.h>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include <cmath>
#include <math.h>
#include <chrono>

#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <cusparse_v2.h>

// forward declaration of gpubuf class          
class GPUBuffers;

extern "C" {

//***************************************
// Matrix solver utilities / gpu_utils.cu
//***************************************

// Initialize the buffer and the indices of the non-zeros in the matrix which represent neighbor connectivity
void initialize_sparsity(GPUBuffers &gpubuf, int pbc, const double nn_dist, int num_atoms_contact);

// check that sparse and dense versions are the same
void check_sparse_dense_match(int m, int nnz, double *dense_matrix, int* d_csrRowPtr, int* d_csrColInd, double* d_csrVal);

// dump sparse matrix into a file
void dump_csr_matrix_txt(int m, int nnz, int* d_csrRowPtr, int* d_csrColIndices, double* d_csrValues, int kmc_step_count);

// convert dense matrix to CSR representation
void denseToCSR(cusparseHandle_t handle, double* d_dense, int num_rows, int num_cols,
                double** d_csr_values, int** d_csr_offsets, int** d_csr_columns, int* total_nnz);

// Solution of A*x = y on sparse representation of A using cusolver in host pointer mode
void sparse_system_solve(cusolverSpHandle_t handle, int* d_csrRowPtr, int* d_csrColInd, double* d_csrVal,
                         int nnz, int m, double *d_x, double *d_y);

// Iterative sparse linear solver using CG steps
void solve_sparse_CG(cublasHandle_t handle_cublas, cusparseHandle_t handle, 
					 double* A_data, int* A_row_ptr, int* A_col_indices, const int A_nnz, 
                     int m, double *d_x, double *d_y);

// Iterative sparse linear solver using CG steps on matrix represented in mixed sparse/dense format 
// the insertion indices specify the correspondence between the (dense) submatrix rows/cols and the (sparse) full matrix rows/cols
void solve_sparse_CG_splitmatrix(cublasHandle_t handle_cublas, cusparseHandle_t handle, 
                                 double* M, int msub, double* A_data, int* A_row_ptr, int* A_col_indices, const int A_nnz, 
                                 int m, int *insertion_indices, double *d_x, double *d_y,
                                 double *diagonal_inv_d);

// Iterative sparse linear solver using CG steps and Jacobi preconditioner
void solve_sparse_CG_Jacobi(cublasHandle_t handle_cublas, cusparseHandle_t handle, 
                            double* A_data, int* A_row_ptr, int* A_col_indices,  
                            const int A_nnz, int m, double *d_x, double *d_y);

// Initialize sparsity of the transmission matrix
void Assemble_X_sparsity(int Natom, const double *posx, const double *posy, const double *posz,
                         const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                         const double *lattice, bool pbc, double nn_dist, const double tol,
                         int num_source_inj, int num_ground_ext, const int num_layers_contact,
                         int num_metals, int **X_row_ptr, int **X_col_indices, int *X_nnz);


// populate the values of the transmission matrix
void Assemble_X(int Natom, const double *posx, const double *posy, const double *posz,
                const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                const double *lattice, bool pbc, double nn_dist, const double tol, const double Vd, const double m_e, const double V0,
                const double high_G, const double low_G, const double loop_G,
                int num_source_inj, int num_ground_ext, const int num_layers_contact,
                int num_metals, double **X_data, int **X_row_ptr, int **X_col_indices, int *X_nnz);

// populate the values of the transmission matrix
void Assemble_X2(int Natom, const double *posx, const double *posy, const double *posz,
                const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                const double *lattice, bool pbc, double nn_dist, const double tol, const double Vd, const double m_e, const double V0,
                const double high_G, const double low_G, const double loop_G,
                int num_source_inj, int num_ground_ext, const int num_layers_contact,
                int num_metals, double **X_data, int **X_row_indices, int **X_row_ptr, int **X_col_indices, int *X_nnz);
             

// Initialize sparsity of the background potential solver
void Assemble_K_sparsity(const double *posx, const double *posy, const double *posz,
                         const double *lattice, const bool pbc, const double cutoff_radius,
                         int system_size, int contact_left_size, int contact_right_size,
                         int **A_row_ptr, int **A_col_indices, int *A_nnz, 
                         int **contact_left_col_indices, int **contact_left_row_ptr, int *contact_left_nnz, 
                         int **contact_right_col_indices, int **contact_right_row_ptr, int *contact_right_nnz);

//***************************************************
// Field solver modules on single GPU / gpu_Device.cu
//***************************************************

// GPU id utils
void get_gpu_info(char *gpu_string, int dev);
void set_gpu(int dev);

//*****************************************************
// Potential Solver functions / potential_solver_gpu.cu
//*****************************************************

// Solve the Laplace equation to get the CB edge along the device
void update_CB_edge_gpu_sparse(cublasHandle_t handle_cublas, cusolverDnHandle_t handle, GPUBuffers &gpubuf,
                               const int N, const int N_left_tot, const int N_right_tot,
                               const double d_Vd, const int pbc, const double d_high_G, const double d_low_G, const double nn_dist, 
                               const int num_metals);

// Updates the site-resolved charge (gpu_site_charge) based on a neighborhood condition
void update_charge_gpu(ELEMENT *gpu_site_element, 
                       int *gpu_site_charge,
                       int *gpu_neigh_idx, int N, int nn, 
                       const ELEMENT *metals, const int num_metals);

// Updates the site-resolved potential (gpubuf.site_potential) using a resistive network model 
// dense matrix with LU solver
void background_potential_gpu(cusolverDnHandle_t handle, GPUBuffers &gpubuf, const int N, const int N_left_tot, const int N_right_tot,
                              const double d_Vd, const int pbc, const double d_high_G, const double d_low_G, const double nn_dist,
                              const int num_metals, int kmc_step_count);

// sparse matrix with iterative solver
void background_potential_gpu_sparse(cublasHandle_t handle_cublas, cusolverDnHandle_t handle, GPUBuffers &gpubuf, const int N, const int N_left_tot, const int N_right_tot,
                              const double d_Vd, const int pbc, const double d_high_G, const double d_low_G, const double nn_dist,
                              const int num_metals, int kmc_step_count);

// Updates the site-resolved potential (gpubuf.site_potential) using the short-range Poisson solution summed over charged species
void poisson_gridless_gpu(const int num_atoms_contact, const int pbc, const int N, const double *lattice,
                          const double *sigma, const double *k,
                          const double *posx, const double *posy, const double *posz,
                          const int *site_charge, double *site_potential_charge,
                          const int rank, const int size, const int *count, const int *displ);


//**************************************************
// Current solver functions / current_solver_gpu.cu
//**************************************************

// Updates the site-resolved dissipated power (gpubuf.site_power) using a graph-based current flow solver 

// dense matrix with LU solver
void update_power_gpu(cublasHandle_t handle, cusolverDnHandle_t handle_cusolver, GPUBuffers &gpubuf, 
                      const int num_source_inj, const int num_ground_ext, const int num_layers_contact,
                      const double Vd, const int pbc, 
                      const double high_G, const double low_G, const double loop_G, const double G0, const double tol,
                      const double nn_dist, const double m_e, const double V0, int num_metals, double *imacro,
                      const bool solve_heating_local, const bool solve_heating_global, const double alpha);
                    
// sparse matrix with iterative solver
void update_power_gpu_sparse(cublasHandle_t handle, cusolverDnHandle_t handle_cusolver, GPUBuffers &gpubuf, 
                             const int num_source_inj, const int num_ground_ext, const int num_layers_contact,
                             const double Vd, const int pbc, const double high_G, const double low_G, const double loop_G, const double G0, const double tol,
                             const double nn_dist, const double m_e, const double V0, int num_metals, double *imacro,
                             const bool solve_heating_local, const bool solve_heating_global, const double alpha_disp);

// mixed sparse neighbor matrix + dense tunneling submatrix
void update_power_gpu_split(cublasHandle_t handle, cusolverDnHandle_t handle_cusolver, GPUBuffers &gpubuf, 
                            const int num_source_inj, const int num_ground_ext, const int num_layers_contact,
                            const double Vd, const int pbc, const double high_G, const double low_G, const double loop_G, const double G0, const double tol,
                            const double nn_dist, const double m_e, const double V0, int num_metals, double *imacro,
                            const bool solve_heating_local, const bool solve_heating_global, const double alpha_disp);


//********************************************
// Heat solver functions / heat_solver_gpu.cu
//********************************************

// Updates the global temperature (T_bg) based on a capacitative heat equation
void update_temperatureglobal_gpu(const double *site_power, 
                                  double *T_bg, const int N, 
                                  const double a_coeff, const double b_coeff, 
                                  const double number_steps, const double C_thermal, 
                                  const double small_step);


//************************************
// KMC Event Selection / kmc_events.cu
//************************************

// Selects and executes events, and updates the relevant site attribute (_element, _charge, etc) using the residence time algorithm
double execute_kmc_step_gpu(const int N, const int nn, const int *neigh_idx, const int *site_layer,
                          const double *lattice, const int pbc, const double *T_bg, 
                          const double *freq, const double *sigma, const double *k,
                          const double *posx, const double *posy, const double *posz, 
                          const double *site_potential_boundary, const double *site_potential_charge, const double *site_temperature,
                          ELEMENT *site_element, int *site_charge, RandomNumberGenerator &rng, const int *neigh_idx_host);

// excluded for testing
#ifndef COMPILE_WITH_TESTS
void copytoConstMemory(std::vector<double> E_gen, std::vector<double> E_rec, std::vector<double> E_Vdiff, std::vector<double> E_Odiff);
#endif

}

//******************************************************************************
// Helper functions for the solvers (the inlines are used in multiple .cu files)
//******************************************************************************

template <typename T>
__device__ inline int is_in_array_gpu(const T *array, const T element, const int size) {

    for (int i = 0; i < size; ++i) {
        if (array[i] == element) {
        return 1;
        }
    }
    return 0;
}

__device__ inline double site_dist_gpu(double pos1x, double pos1y, double pos1z,
                                double pos2x, double pos2y, double pos2z,
                                double lattx, double latty, double lattz, bool pbc)
{

    double dist = 0;

    if (pbc == 1)
    {
        double dist_x = pos1x - pos2x;
        double distance_frac[3];

        distance_frac[1] = (pos1y - pos2y) / latty;
        distance_frac[1] -= round(distance_frac[1]);
        distance_frac[2] = (pos1z - pos2z) / lattz;
        distance_frac[2] -= round(distance_frac[2]);

        double dist_xyz[3];
        dist_xyz[0] = dist_x;

        dist_xyz[1] = distance_frac[1] * latty;
        dist_xyz[2] = distance_frac[2] * lattz;

        dist = sqrt(dist_xyz[0] * dist_xyz[0] + dist_xyz[1] * dist_xyz[1] + dist_xyz[2] * dist_xyz[2]);
        
    }
    else
    {
        dist = sqrt(pow(pos2x - pos1x, 2) + pow(pos2y - pos1y, 2) + pow(pos2z - pos1z, 2));
    }

    return dist;
}

__device__ inline double v_solve_gpu(double r_dist, int charge, const double *sigma, const double *k) { 

    double q = 1.60217663e-19;              // [C]
    double vterm = static_cast<double>(charge) * erfc(r_dist / ((*sigma) * sqrt(2.0))) * (*k) * q / r_dist; 

    return vterm;
}


// used to be named 'calc_diagonal_A_gpu' - row reduction into the diagonal elements for a sparse matrix
__global__ void reduce_rows_into_diag( int *col_indices, int *row_ptr, double *data, int matrix_size );

// used to be called 'set_diag' - overwrite the diagonal of matrix A with vector diag
__global__ void write_to_diag(double *A, double *diag, int N);

// used to be named diagonal_sum - sum the rows of A into the vector diag
template <int NTHREADS>
__global__ void row_reduce(double *A, double *diag, int N);