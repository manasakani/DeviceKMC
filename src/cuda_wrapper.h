#pragma once

#include "utils.h"
#include "random_num.h"
#include "gpu_buffers.h"

#include <stdio.h>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include <cmath>
#include <math.h>

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
#include <cusparse_v2.h>

// forward declaration of gpubuf class          
class GPUBuffers;

extern "C" {

//***************************************
// Matrix solver utilities / gpu_utils.cu
//***************************************

// Initialize the buffer and the indices of the non-zeros in the matrix which represent neighbor connectivity
void initialize_sparsity(GPUBuffers &gpubuf, int pbc, const double nn_dist, int num_atoms_contact);

// initialize the handles for CuSparse, CuBlas, and CuSolver to keep in GPU memory
// void initialize_handles();

// check that sparse and dense versions are the same
void check_sparse_dense_match(int m, int nnz, double *dense_matrix, int* d_csrRowPtr, int* d_csrColInd, double* d_csrVal);

// dump sparse matrix into a file
void dump_csr_matrix_txt(int m, int nnz, int* d_csrRowPtr, int* d_csrColIndices, double* d_csrValues, int kmc_step_count);

// Solution of A*x = y on sparse representation of A using cusolver in host pointer mode
void sparse_system_solve(cusolverSpHandle_t handle, int* d_csrRowPtr, int* d_csrColInd, double* d_csrVal,
                         int nnz, int m, double *d_x, double *d_y);

// Iterative sparse linear solver using CG steps
void solve_sparse_CG(cublasHandle_t handle_cublas, cusparseHandle_t handle, 
					 cusparseSpMatDescr_t matA, int m, double *d_x, double *d_y);

// Iterative sparse linear solver using CG steps and Jacobi preconditioner
void solve_sparse_CG_Jacobi(cublasHandle_t handle_cublas, cusparseHandle_t handle, 
                            double* A_data, int* A_row_ptr, int* A_col_indices,  
                            const int A_nnz, int m, double *d_x, double *d_y);

// Initialize sparsity of the background potential solver
void Assemble_K_sparsity(const double *posx, const double *posy, const double *posz,
                         const double *lattice, const bool pbc, const double cutoff_radius,
                         int system_size, int contact_left_size, int contact_right_size,
                         int **A_row_ptr, int **A_col_indices, int *A_nnz, 
                         int **contact_left_col_indices, int **contact_left_row_ptr, int *contact_left_nnz, 
                         int **contact_right_col_indices, int **contact_right_row_ptr, int *contact_right_nnz);

//*************************************
// Field solver modules / gpu_Device.cu
//*************************************

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
                              const int num_metals, int kmc_step_count);
        
void background_potential_gpu_sparse(cublasHandle_t handle_cublas, cusolverDnHandle_t handle, GPUBuffers &gpubuf, const int N, const int N_left_tot, const int N_right_tot,
                              const double d_Vd, const int pbc, const double d_high_G, const double d_low_G, const double nn_dist,
                              const int num_metals, int kmc_step_count);

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
