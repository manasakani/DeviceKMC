#include "cuda_wrapper.h"
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

const double eV_to_J = 1.6e-19;
const double h_bar_sq = 4.3957e-67;
constexpr double kB = 8.617333262e-5;     
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include "gpu_buffers.h"
#define NUM_THREADS 512
#define MAX_NUM_LAYERS 5

// in GPU cache
__constant__ double E_gen_const[MAX_NUM_LAYERS];
__constant__ double E_rec_const[MAX_NUM_LAYERS];
__constant__ double E_Vdiff_const[MAX_NUM_LAYERS];
__constant__ double E_Odiff_const[MAX_NUM_LAYERS];

void get_gpu_info(char *gpu_string, int dev){
 struct cudaDeviceProp dprop;
 cudaGetDeviceProperties(&dprop, dev);
 strcpy(gpu_string,dprop.name);
}

void set_gpu(int dev){
 cudaSetDevice(dev);
}

// ********************************************************
// *************** GPU HELPER FUNCTIONS *******************
// ********************************************************

struct is_defect
{
    __host__ __device__ bool operator()(const ELEMENT element)
    {
        return (element != DEFECT);
    }
};

// returns true if thing is present in the array of things
template <typename T>
__device__ int is_in_array_gpu(const T *array, const T element, const int size) {

    for (int i = 0; i < size; ++i) {
        if (array[i] == element) {
        return 1;
        }
    }
    return 0;
}

__device__ double site_dist_gpu(double pos1x, double pos1y, double pos1z,
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

__device__ double v_solve_gpu(double r_dist, int charge, const double *sigma, const double *k) { 

    double q = 1.60217663e-19;              // [C]
    double vterm = static_cast<double>(charge) * erfc(r_dist / ((*sigma) * sqrt(2.0))) * (*k) * q / r_dist; 

    return vterm;
}


// ********************************************************
// ******************** KERNELS ***************************
// ********************************************************

__global__ void remake_atom_list(int N, double *site_x, double *site_y, double *site_z, ELEMENT *metals, const ELEMENT *element, double *site_potential, double *site_power,
                                 double *atom_x, double *atom_y, double *atom_z, ELEMENT *atom_element, double *atom_potential,
                                 double *atom_power, int *Natoms)
{
    // Removes defects sites from the site list
    // One thread kernel at the moment
    int atom_index = 0;

    for (auto i = 0; i < N; i += 1)
    {
        if (element[i] == DEFECT || element[i] == OXYGEN_DEFECT)
        {
            // do nothing
        }
        else
        {
            atom_x[atom_index] = site_x[i];
            atom_y[atom_index] = site_y[i];
            atom_z[atom_index] = site_z[i];
            atom_element[atom_index] = element[i];
            atom_potential[atom_index] = site_potential[i];
            atom_power[atom_index] = site_power[i];
            ++atom_index;
        }
    }
    Natoms[0] = atom_index;
}

__global__ void set_potential(double *A, double *B, int N)
{
    int didx = blockIdx.x * blockDim.x + threadIdx.x;
    for (auto i = didx; i < N; i += gridDim.x * blockDim.x)
    {
        A[i] = -B[i];
    }
}

__global__ void set_diag_K(double *A, double *diag, int N)
{
    int didx = blockIdx.x * blockDim.x + threadIdx.x;
    for (auto i = didx; i < N; i += gridDim.x * blockDim.x)
    {
        double tmp = A[i];
        A[i] = tmp + diag[i];
    }
}

template <int NTHREADS>
__global__ void diagonal_sum_K(
    double *A,
    double *diag,
    double *V,
    int N, int NI, int NJ)
{

    int num_threads = blockDim.x;
    int blocks_per_row = (NJ - 1) / num_threads + 1;
    int block_id = blockIdx.x;

    int row = block_id / blocks_per_row;
    int scol = (block_id % blocks_per_row) * num_threads;
    int lcol = min(NJ, scol + num_threads);

    int tid = threadIdx.x;

    __shared__ double buf[NTHREADS];

    for (auto ridx = row; ridx < NI; ridx += gridDim.x)
    {

        buf[tid] = 0.0;
        if (tid + scol < lcol)
        {
            buf[tid] = A[ridx * N + scol + tid] * V[tid + scol];
            // if (ridx == 7039) {
            //     printf("Thread %d (%d, %d) A=%E, V=%E, buf=%E\n", tid, ridx, tid + scol, A[ridx * N + tid + scol], V[tid + scol], buf[tid]);
            // }
        }

        int width = num_threads / 2;
        while (width != 0)
        {
            __syncthreads();
            if (tid < width)
            {
                buf[tid] += buf[tid + width];
            }
            width /= 2;
        }

        if (tid == 0)
        {
            atomicAdd(diag + ridx, buf[0]);
        }
    }
}

__global__ void set_diag(double *A, double *diag, int N)
{
    int didx = blockIdx.x * blockDim.x + threadIdx.x;
    if (didx < N)
    {
        double tmp = A[didx * N + didx];
        A[didx * N + didx] = 2 * tmp - diag[didx];
    }
}

__global__ void copy_pdisp(double *site_power, ELEMENT *element, const ELEMENT *metals, double *pdisp, int *atom_gpu_index, int N_atom,
                           const int num_metals)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int idx = tid; idx < N_atom; idx += total_threads)
    {
        bool metal = is_in_array_gpu(metals, element[atom_gpu_index[idx]], num_metals);
        if (!metal)
            site_power[atom_gpu_index[idx]] = -0.1 * pdisp[idx];
    }
}

template <int NTHREADS>
__global__ void get_imacro(const double *x, const double *m, double *imacro, double Vd, int N)
{

    int num_threads = blockDim.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int total_tid = bid * num_threads + tid;

    __shared__ double buf[NTHREADS];

    // total_tid >= 2 && total_tid < N

    buf[tid] = 0.0;
    double tmp = x[(N + 2) * (3) + (total_tid + 2)] * (m[total_tid + 2] - m[3]);
    if ((total_tid > 1 && total_tid < N) && ((tmp > 0 && Vd > 0) || (tmp < 0 && Vd < 0)))
    {
        buf[tid] = tmp;
    }

    int width = num_threads / 2;
    while (width != 0)
    {
        __syncthreads();
        if (tid < width)
        {
            buf[tid] += buf[tid + width];
        }
        width /= 2;
    }

    if (tid == 0)
    {
        atomicAdd(imacro, buf[0]);
    }
}

template <int NTHREADS>
__global__ void diagonal_sum(double *A, double *diag, int N)
{

    int num_threads = blockDim.x;
    int blocks_per_row = (N - 1) / num_threads + 1;
    int block_id = blockIdx.x;

    int tid = threadIdx.x;

    __shared__ double buf[NTHREADS];

    for (auto idx = block_id; idx < N * blocks_per_row; idx += gridDim.x)
    {

        int ridx = idx / blocks_per_row;
        int scol = (idx % blocks_per_row) * num_threads;
        int lcol = min(N, scol + num_threads);

        buf[tid] = 0.0;
        if (tid + scol < lcol)
        {
            buf[tid] = A[ridx * N + scol + tid];
        }

        int width = num_threads / 2;
        while (width != 0)
        {
            __syncthreads();
            if (tid < width)
            {
                buf[tid] += buf[tid + width];
            }
            width /= 2;
        }

        if (tid == 0)
        {
            atomicAdd(diag + ridx, buf[0]);
        }
    }
}

__global__ void create_K(
    double *X,
    const double *posx, const double *posy, const double *posz,
    const ELEMENT *metals, const ELEMENT *element, const int *site_charge,
    const double *lattice, const bool pbc, const double high_G, const double low_G_1,
    const double nn_dist, const int N, const int num_metals)
{

    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;

    for (auto idx = tid_total; idx < (size_t) N * N; idx += num_threads_total)
    {
        int i = idx / N;
        int j = idx % N;

        bool metal1 = is_in_array_gpu(metals, element[i], num_metals);
        bool metal2 = is_in_array_gpu(metals, element[j], num_metals);
        bool ischarged1 = site_charge[i] != 0;
        bool ischarged2 = site_charge[j] != 0;
        bool isVacancy1 = element[i] == VACANCY;
        bool isVacancy2 = element[j] == VACANCY;
        // bool cvacancy1 = isVacancy1 && ischarged1;
        // bool cvacancy2 = isVacancy2 && ischarged2;
        bool cvacancy1 = isVacancy1 && !ischarged1;
        bool cvacancy2 = isVacancy2 && !ischarged2;
        double dist = site_dist_gpu(posx[i], posy[i], posz[i], posx[j], posy[j], posz[j], lattice[0], lattice[1], lattice[2], pbc);

        bool neighbor = false;
        if (dist < nn_dist && i != j)
            neighbor = true;

        // direct terms:
        if (i != j && neighbor)
        {
            if ((metal1 && metal2) || (cvacancy1 && cvacancy2))
            {
                X[N * (i) + (j)] = -high_G;
            }
            else
            {
                X[N * (i) + (j)] = -low_G_1;
            }
        }
    }
}

__global__ void set_ineg(double *ineg, const double *x,
                         const double *m, double Vd, int N, const double *posx, const double *posy, const double *posz,
                         const int pbc, const double *lattice, const double nn_dist,
                         const double V0, const double *atom_potential, const double t_ox)
{
    // ineg is matrix N x N
    // x is matrix (N+2) x (N+2)
    // m is vector (N + 2)

    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;

    for (auto idx = tid_total; idx < N * N; idx += num_threads_total)
    {
        int i = idx / N;
        int j = idx % N;

        double xdiff = (1e-10) * (posx[j] - posx[i]);
        double dist = site_dist_gpu(posx[i], posy[i], posz[i], posx[j], posy[j], posz[j], lattice[0], lattice[1], lattice[2], pbc);
        bool neighbor = false;
        if (dist < nn_dist && i != j)
            neighbor = true;

        ineg[i * N + j] = 0.0;
        double ical = x[(N + 2) * (i + 2) + (j + 2)] * (m[i + 2] - m[j + 2]);
        if (neighbor && i != j && ical < 0 && Vd > 0)
        {
            ineg[i * N + j] = -ical;
        }
        else if (ical < 0 && Vd > 0 && xdiff < t_ox * V0 && xdiff > nn_dist && !neighbor)
        { // excluding Fowler Nordheim tunneling
            ineg[i * N + j] = -ical;
        }
    }
}

// iterates over every pair of sites, and does an operation based on the distance
// NOTE: There is an error in the case of block overflow in the input matrix! But this will never
// be triggered because the number of blocks can reach 2^31.
template <int NTHREADS>
__global__ void calculate_pairwise_interaction(const double* posx, const double* posy, const double*posz, 
                                               const double *lattice, const int pbc, 
                                               const int N, const double *sigma, const double *k, 
                                               const int *charge, double* potential){

    // Version with reduction, where every thread evaluates site-site interaction term
    int num_threads = blockDim.x;
    int blocks_per_row = (N - 1) / num_threads + 1;
    int block_id = blockIdx.x;

    int row = block_id / blocks_per_row;
    int scol = (block_id % blocks_per_row) * num_threads;
    int lcol = min(N, scol + num_threads);

    int tid = threadIdx.x;

    __shared__ double buf[NTHREADS];
    double dist;
    int i, j;

    for (int ridx = row; ridx < N; ridx += gridDim.x) {

        buf[tid] = 0.0;
        if (tid + scol < lcol) {

            i = ridx;
            j = scol+tid;
            if (i != j && charge[j] != 0){
                dist = 1e-10 * site_dist_gpu(posx[i], posy[i], posz[i], 
                                             posx[j], posy[j], posz[j], 
                                             lattice[0], lattice[1], lattice[2], pbc);
                buf[tid] = v_solve_gpu(dist, charge[j], sigma, k);

            }
        }

        int width = num_threads / 2;
        while (width != 0) {
            __syncthreads();
            if (tid < width) {
                buf[tid] += buf[tid + width];
            }
            width /= 2;
        }

        if (tid == 0) {
            atomicAdd(potential + ridx, buf[0]);
        }
    
    }

    // // Version without reduction, where every thread evaluates a site:
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // int total_threads = blockDim.x * gridDim.x;
    // double V_temp = 0;
    // double dist;

    // for (int i = tid; i < N; i += total_threads) {
        
    //     // iterate over the neighbors
    //     for (int j = 0; j < N; j++){
    //         if (i != j && charge[j] != 0){

    //             dist = 1e-10 * site_dist_gpu(posx[i], posy[i], posz[i], 
    //                                          posx[j], posy[j], posz[j], 
    //                                          lattice[0], lattice[1], lattice[2], pbc);
    //             V_temp += v_solve_gpu(dist, charge[j], sigma, k);
                
    //         }
    //     }
    //     potential[tid] += V_temp;
    // }
}

__global__ void update_m(double *m, long minidx, int np2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // int bid = blockIdx.x;

    if (idx < np2)
    {
        double minm = m[minidx];
        m[idx] += abs(minm);
    }
}

__global__ void create_X(
    double *X,
    const double *posx, const double *posy, const double *posz,
    const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_potential,
    const double *lattice, bool pbc, double high_G, double low_G_1,
    double nn_dist, double m_e, double V0, int num_source_inj, int num_ground_ext, int N, int num_metals, const double t_ox, const double Vd)
{

    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;

    int N_full = N + 2;

    // TODO: Does it make sense to restructure for N_full * N_full threads?
    for (auto idx = tid_total; idx < N * N; idx += num_threads_total)
    {
        int i = idx / N;
        int j = idx % N;
        // if (i < N && j < N) {

        bool metal1 = is_in_array_gpu(metals, element[i], num_metals);
        bool metal2 = is_in_array_gpu(metals, element[j], num_metals);

        // bool cvacancy1 = is_vacant[i] && is_charged[i];
        // bool cvacancy2 = is_vacant[j] && is_charged[i];
        // bool vacancy1 = is_vacant[i] && !is_charged[i];
        // bool vacancy2 = is_vacant[j] && !is_charged[i];

        bool ischarged1 = atom_charge[i] != 0;
        bool ischarged2 = atom_charge[j] != 0;

        bool isVacancy1 = element[i] == VACANCY;
        bool isVacancy2 = element[j] == VACANCY;

        bool cvacancy1 = isVacancy1 && !ischarged1;
        bool cvacancy2 = isVacancy2 && !ischarged2;

        bool vacancy1 = isVacancy1 && ischarged1;
        bool vacancy2 = isVacancy2 && ischarged2;
        double dist = site_dist_gpu(posx[i], posy[i], posz[i], posx[j], posy[j], posz[j], lattice[0], lattice[1], lattice[2], pbc);
        
        // dist = abs(dist);
        bool neighbor = false;
        if (dist < nn_dist && i != j)
            neighbor = true;

        // direct terms:
        if (i != j && neighbor)
        {
            if ((metal1 && metal2) || (cvacancy1 && cvacancy2))
            {
                X[N_full * (i + 2) + (j + 2)] = -high_G;
                // X[N_full * (j + 2) + (i + 2)] = -high_G;
            }
            else
            {
                X[N_full * (i + 2) + (j + 2)] = -low_G_1;
                // X[N_full * (j + 2) + (i + 2)] = -low_G_1;
            }
        }

        // tunneling terms
        if (i != j && !neighbor && j > i)
        { 
            bool V_V = (vacancy1 && vacancy2) || (vacancy2 && cvacancy1) || (vacancy1 && cvacancy2) || (cvacancy1 && cvacancy2);

            if (V_V)
            {
                double Vdiff = Vd;
                double xdiff = (1e-10) * (posx[j] - posx[i]); // potential accross the x-direction => if x_j < x_i then Vdiff < 0
                double b = Vdiff / t_ox;
                double a = 1e18; // zero prob
                if (abs(V0 / b - xdiff) < 1e-18 && xdiff > 0)
                {
                    a = 2.0 / 3.0 * sqrt(V0) * xdiff;
                }
                else if (xdiff < V0 / b && xdiff > nn_dist)
                {                                                                     // if Vdiff < 0 then lower prob
                    a = -2.0 / 3.0 * (1 / b) * (pow(V0 - b * xdiff, 1.5) - pow(V0, 1.5)); // always +
                }
                else if (xdiff > V0 / b < 0 && xdiff > 0)
                {
                    a = -2.0 / 3.0 * (1 / b) * (-1) * pow(V0, 3 / 2); // always +
                }
                double T = exp(-2 * sqrt((2 * m_e * eV_to_J) / (h_bar_sq)) * a);
                double G = 2 * 3.8612e-5 * T;
                X[N_full * (i + 2) + (j + 2)] = -G;
                X[N_full * (j + 2) + (i + 2)] = -G;
            }
        }

        // NOTE: Is there a data race here?
        // connect the source/ground nodes to the first/last contact layers
        __syncthreads();
        if (i < num_source_inj && j == 0)
        {
            X[1 * N_full + (i + 2)] = -high_G;
            X[(i + 2) * N_full + 1] = -high_G;
        }

        if (i > (N - num_ground_ext) && j == 0)
        {
            X[0 * N_full + (i + 2)] = -high_G;
            X[(i + 2) * N_full + 0] = -high_G;
        }

        __syncthreads();
        if (i == 0 && j == 0)
        {
            X[0 * N_full + 1] = -high_G;
            X[1 * N_full + 0] = -high_G;
        }
    }
}

__global__ void update_charge(const ELEMENT *element, 
                              int *charge, 
                              const int *neigh_idx, 
                              const int N, const int nn, 
                              const ELEMENT* metals, const int num_metals){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    int Vnn = 0;

    // each thread gets a different site to evaluate
    for (int idx = tid; idx < N; idx += total_threads) {
        
        if (tid < N && element[tid] == VACANCY){
            charge[tid] = 2;

            // iterate over the neighbors
            for (int j = tid * nn; j < (tid + 1) * nn; ++j){
                if (element[neigh_idx[j]] == VACANCY){
                    Vnn++;
                }
                if (is_in_array_gpu(metals, element[neigh_idx[j]], num_metals)){
                    charge[tid] = 0;
                }
                if (Vnn >= 3){
                    charge[tid] = 0;
                }
            }
        }

        if (tid < N && element[tid] == OXYGEN_DEFECT){
            charge[tid] = -2;

            // iterate over the neighbors
            for (int j = tid * nn; j < (tid + 1) * nn; ++j){
                
                if (is_in_array_gpu(metals, element[neigh_idx[j]], num_metals)){
                    charge[tid] = 0;
                }
            }
        }
    }
}

//reduces the array into the value 
template <typename T, int NTHREADS>
__global__ void reduce(const T* array_to_reduce, T* value, const int N){

    __shared__ T buf[NTHREADS];
    
    int num_threads = blockDim.x;                           // number of threads in this block
    int blocks_per_row = (N-1)/num_threads + 1;             // number of blocks to fit in this array
    int block_id = blockIdx.x;                              // id of the block
    int tid = threadIdx.x;                                  // local thread id to this block
    int row = block_id / blocks_per_row;                    // which 'row' of the array to work on, rows are the overflow

    buf[tid] = 0;

    for (int ridx = row; ridx < N/(blocks_per_row*num_threads) + 1; ridx++){
    
        if (ridx*blocks_per_row*num_threads + block_id * num_threads + tid < N){
            buf[tid] = array_to_reduce[ridx*blocks_per_row*num_threads + block_id * num_threads + tid];
        }
       
        int width = num_threads / 2;

        while (width != 0){
            __syncthreads();
            if (tid < width){
                buf[tid] += buf[tid+width];
            }
            width /= 2;
        }

        if (tid == 0){
           atomicAdd(value, buf[0]);
        }
    }
}

//called by a single gpu-thread
__global__ void update_temp_global(double *P_tot, double* T_bg, const double a_coeff, const double b_coeff, const double number_steps, const double C_thermal, const double small_step)
{
    double c_coeff = b_coeff + *P_tot/C_thermal * small_step;
    double T_intermediate = *T_bg;
    int step = number_steps;
    *T_bg = c_coeff*(1.0-pow(a_coeff, (double) step)) / (1.0-a_coeff) + pow(a_coeff, (double) step)* T_intermediate;
}

__global__ void build_event_list(const int N, const int nn, const int *neigh_idx, 
                                 const int *layer, const double *lattice, const int pbc, 
                                 const double *T_bg, const double *freq, const double *sigma, const double *k, 
                                 const double *posx, const double *posy, const double *posz,
                                 const double *potential, const double *temperature,
                                 const ELEMENT *element, const int *charge, EVENTTYPE *event_type, double *event_prob)
{
    int total_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int idx = total_tid; idx < N * nn; idx += total_threads) {

        EVENTTYPE event_type_ = NULL_EVENT;
        double P = 0.0;

        int i = static_cast<int>(floorf(idx / nn));
        int j = neigh_idx[idx];

        // condition for neighbor existing
        if (j >= 0 && j < N) {
            double dist = 1e-10 * site_dist_gpu(posx[i], posy[i], posz[i], 
                                                posx[j], posy[j], posz[j], 
                                                lattice[0], lattice[1], lattice[2], pbc);

            // Generation
            if (element[i] == DEFECT && element[j] == O_EL)
            {

                double E = 2 * (potential[i] - potential[j]);
                double zero_field_energy = E_gen_const[layer[j]]; 
                event_type_ = VACANCY_GENERATION;
                double Ekin = kB * (temperature[j] - temperature[i]);
                double EA = zero_field_energy - E - Ekin;
                P = exp(-1 * EA / (kB * (*T_bg))) * (*freq);
            }

            // Recombination
            else if (element[i] == OXYGEN_DEFECT && element[j] == VACANCY) 
            {
                int charge_abs = 2;
                double self_int_V = v_solve_gpu(dist, charge_abs, sigma, k);

                int charge_state = charge[i] - charge[j];
                double E = charge_state * (potential[i] - potential[j] + (charge_state / 2) * self_int_V);
                double zero_field_energy = E_rec_const[layer[j]];

                event_type_ = VACANCY_RECOMBINATION;
                double Ekin = kB * (temperature[i] - temperature[j]);
                double EA = zero_field_energy - E - Ekin;
                P = exp(-1 * EA / (kB * (*T_bg))) * (*freq);
            }

            // Vacancy diffusion
            if (element[i] == VACANCY && element[j] == O_EL)
            {

                double self_int_V = 0.0;
                if (charge[i] != 0)
                {
                    double self_int_V = v_solve_gpu(dist, charge[i], sigma, k);
                }

                event_type_ = VACANCY_DIFFUSION;
                double E = (charge[i] - charge[j]) * (potential[i] - potential[j] + self_int_V);
                double zero_field_energy = E_Vdiff_const[layer[j]];  
                double Ekin = kB * (temperature[j] - temperature[i]);
                double EA = zero_field_energy - E - Ekin;
                P = exp(-1 * EA / (kB * (*T_bg))) * (*freq);
            }

            // Ion diffusion
            else if (element[i] == OXYGEN_DEFECT && element[j] == DEFECT)
            {
                int charge_abs = 2;
                double self_int_V = 0.0;
                if (charge[i] != 0)
                {                    
                    double self_int_V = v_solve_gpu(dist, charge_abs, sigma, k);
                }

                double E = (charge[i] - charge[j]) * (potential[i] - potential[j] - self_int_V);
                double zero_field_energy = E_Odiff_const[layer[j]];

                event_type_ = ION_DIFFUSION;
                double Ekin = kB * (temperature[i] - temperature[j]);
                double EA = zero_field_energy - E - Ekin;
                P = exp(-1 * EA / (kB * (*T_bg))) * (*freq);
            }
        }
        event_type[idx] = event_type_;
        event_prob[idx] = P;
    }
}

// ********************************************************
// ****************** KERNEL UNIT TESTS *******************
// ********************************************************

// unit test for reduce kernel, checks correctness for large arrays
void test_reduce()
{
    int N = 70000;

    int num_threads = 512;
    int num_blocks = (N - 1) / num_threads + 1;
    // num_blocks = min(65535, num_blocks);

    double *gpu_test_array;
    double *gpu_test_sum;
    double t_test = 0.0;
    std::vector<double> test_array(N, 1.0);

    gpuErrchk( cudaMalloc((void**)&gpu_test_array, N * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&gpu_test_sum, 1 * sizeof(double)) );
    gpuErrchk( cudaMemcpy(gpu_test_array, test_array.data(), N * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(gpu_test_sum, &t_test, 1 * sizeof(double), cudaMemcpyHostToDevice) );

    reduce<double, NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS*sizeof(double)>>>(gpu_test_array, gpu_test_sum, N);
    gpuErrchk( cudaGetLastError() );

    gpuErrchk( cudaMemcpy(&t_test, gpu_test_sum, 1 * sizeof(double), cudaMemcpyDeviceToHost));
    assert(t_test == 70000.0);
    std::cout << "--> Ran test for kernel reduce()\n";
}                

// ********************************************************
// *************** WRAPPER FUNCTIONS **********************
// ********************************************************

void update_charge_gpu(ELEMENT *site_element, 
                       int *site_charge,
                       int *neigh_idx, int N, int nn, 
                       const ELEMENT *metals, const int num_metals){

    int num_threads = 512;
    int num_blocks = (N * nn - 1) / num_threads + 1;

    update_charge<<<num_blocks, num_threads>>>(site_element, site_charge, neigh_idx, N, nn, metals, num_metals);
}

void update_temperatureglobal_gpu(const double *site_power, double *T_bg, const int N, const double a_coeff, const double b_coeff, const double number_steps, const double C_thermal, const double small_step){

    int num_threads = 512;
    int num_blocks = (N - 1) / num_threads + 1;

    double *P_tot;
    gpuErrchk( cudaMalloc((void**)&P_tot, 1 * sizeof(double)) );
    gpuErrchk( cudaMemset(P_tot, 0, 1 * sizeof(double)) );

    //collect site_power
    reduce<double, NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS*sizeof(double)>>>(site_power, P_tot, N);

    //update the temperature
    update_temp_global<<<1, 1>>>(P_tot, T_bg, a_coeff, b_coeff, number_steps, C_thermal, small_step);

    // double p_test = 0.0;
    // gpuErrchk( cudaMemcpy(&p_test, P_tot, 1 * sizeof(double), cudaMemcpyDeviceToHost));
    // std::cout << "power: " << p_test << "\n";
    // double t_test = 0.0;
    // gpuErrchk( cudaMemcpy(&t_test, T_bg, 1 * sizeof(double), cudaMemcpyDeviceToHost));
    // std::cout << "temperature: " << t_test << "\n";
    cudaFree(P_tot);
}

void background_potential_gpu(cusolverDnHandle_t handle, const GPUBuffers &gpubuf, const int N, const int N_left_tot, const int N_right_tot,
                              const double Vd, const int pbc, const double d_high_G, const double d_low_G, const double nn_dist,
                              const int num_metals)
{

    int N_interface = N - (N_left_tot + N_right_tot);

    double *VL, *VR;
    gpuErrchk( cudaMalloc((void **)&VL, N_left_tot * sizeof(double)) );
    gpuErrchk( cudaMalloc((void **)&VR, N_right_tot * sizeof(double)) );

    double *gpu_k;
    double *gpu_diag;
    gpuErrchk( cudaMalloc((void **)&gpu_k, (size_t) N * N * sizeof(double)) );
    gpuErrchk( cudaMalloc((void **)&gpu_diag, N * sizeof(double)) );
    gpuErrchk( cudaMemset(gpu_k, 0, (size_t) N * N * sizeof(double)) );
    gpuErrchk( cudaDeviceSynchronize() );

    // prepare contact potentials
    thrust::device_ptr<double> VL_ptr = thrust::device_pointer_cast(VL);
    thrust::fill(VL_ptr, VL_ptr + N_left_tot, -Vd/2);
    thrust::device_ptr<double> VR_ptr = thrust::device_pointer_cast(VR);
    thrust::fill(VR_ptr, VR_ptr + N_right_tot, Vd/2);

    //  BUILDING THE CONDUCTIVITY MATRIX
    int num_threads = 512;
    int blocks_per_row = (N - 1) / num_threads + 1;
    int num_blocks = blocks_per_row * N;

    // compute the off-diagonal elements of K
    create_K<<<num_blocks, num_threads>>>(
        gpu_k, gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
        gpubuf.metal_types, gpubuf.site_element, gpubuf.site_charge,
        gpubuf.lattice, pbc, d_high_G, d_low_G,
        nn_dist, N, num_metals);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Update the diagonal of K
    gpuErrchk( cudaMemset(gpu_diag, 0, N * sizeof(double)) );
    gpuErrchk( cudaDeviceSynchronize() );
    diagonal_sum<NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS * sizeof(double)>>>(gpu_k, gpu_diag, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    num_blocks = (N - 1) / num_threads + 1;
    set_diag<<<num_blocks, num_threads>>>(gpu_k, gpu_diag, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemset(gpu_diag, 0, N * sizeof(double)) );
    gpuErrchk( cudaDeviceSynchronize() );

    // 
    blocks_per_row = (N_left_tot - 1) / num_threads + 1;
    num_blocks = blocks_per_row * N_interface;
    diagonal_sum_K<NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS * sizeof(double)>>>(&gpu_k[N_left_tot * N], gpu_diag, VL, N, N_interface, N_left_tot);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    blocks_per_row = (N_right_tot - 1) / num_threads + 1;
    num_blocks = blocks_per_row * N_interface;
    diagonal_sum_K<NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS * sizeof(double)>>>(&gpu_k[N_left_tot * N + N - N_right_tot], gpu_diag, VR, N, N_interface, N_right_tot);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    //  SOLVING FOR THE NEGATIVE INTERNAL POTENTIALS (KSUB)

    double *gpu_k_sub;
    gpuErrchk( cudaMalloc((void **)&gpu_k_sub, N_interface * sizeof(double)) ); 
    gpuErrchk( cudaMemset(gpu_k_sub, 0, N_interface * sizeof(double)) );
    num_blocks = (N_interface - 1) / num_threads + 1;
    set_diag_K<<<blocks_per_row, num_threads>>>(gpu_k_sub, gpu_diag, N_interface);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaFree(gpu_diag);

     // ** Solve Ax=B through LU factorization **

    int lwork = 0;              /* size of workspace */
    double *gpu_work = nullptr; /* device workspace for getrf */
    int *gpu_info = nullptr;    /* error info */
    int *gpu_ipiv; // int info;
    gpuErrchk( cudaMalloc((void **)&gpu_ipiv, N_interface * sizeof(int)) ); 
    gpuErrchk( cudaMalloc((void **)(&gpu_info), sizeof(int)) );

    // points to the start of Koxide inside K:
    double* gpu_D = gpu_k + (N_left_tot * N) + N_left_tot;

    CheckCusolverDnError(cusolverDnDgetrf_bufferSize(handle, N_interface, N_interface, gpu_D, N, &lwork));
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMalloc((void **)(&gpu_work), sizeof(double) * lwork) );

    CheckCusolverDnError(cusolverDnDgetrf(handle, N_interface, N_interface, gpu_D, N, gpu_work, gpu_ipiv, gpu_info));
    // cudaMemcpy(&info, gpu_info, sizeof(int), cudaMemcpyDeviceToHost); // printf("info for cusolverDnDgetrf: %i \n", info);
    gpuErrchk( cudaDeviceSynchronize() );

    CheckCusolverDnError(cusolverDnDgetrs(handle, CUBLAS_OP_N, N_interface, 1, gpu_D, N, gpu_ipiv, gpu_k_sub, N_interface, gpu_info));
    // cudaMemcpy(&info, gpu_info, sizeof(int), cudaMemcpyDeviceToHost); // printf("info for cusolverDnDgetrs: %i \n", info);
    gpuErrchk( cudaDeviceSynchronize() );

    cudaFree(gpu_k);

    num_blocks = (N_interface - 1) / num_threads + 1;
    set_potential<<<num_blocks, num_threads>>>(gpubuf.site_potential + N_left_tot, gpu_k_sub, N_interface);
    gpuErrchk( cudaPeekAtLastError() ); 
    gpuErrchk( cudaDeviceSynchronize() ); 
    cudaFree(gpu_k_sub);

    gpuErrchk( cudaMemcpy(gpubuf.site_potential, VL, N_left_tot * sizeof(double), cudaMemcpyDeviceToDevice) );
    gpuErrchk( cudaMemcpy(gpubuf.site_potential + N_left_tot + N_interface, VR, N_right_tot * sizeof(double), cudaMemcpyDeviceToDevice) );

    cudaFree(gpu_ipiv);
    cudaFree(gpu_work);
    cudaFree(gpu_info);
    cudaFree(VL);
    cudaFree(VR);

}

void poisson_gridless_gpu(const int num_atoms_contact, const int pbc, const int N, const double *lattice, 
                          const double *sigma, const double *k,
                          const double *posx, const double *posy, const double *posz, 
                          const int *site_charge, double *site_potential){

    int num_threads = 1024;
    int blocks_per_row = (N - 1) / num_threads + 1; 
    int num_blocks = blocks_per_row * N; // NOTE: fix the kernel for block overflow!

    calculate_pairwise_interaction<NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS * sizeof(double)>>>(posx, posy, posz, lattice, pbc, N, sigma, k, site_charge, site_potential);
}

void update_power_gpu(cublasHandle_t handle, cusolverDnHandle_t handle_cusolver, GPUBuffers &gpubuf, const int N, const int num_source_inj, const int num_ground_ext,
                      const double Vd, const int pbc, const double high_G, const double low_G,
                      const double nn_dist, const double m_e, const double V0, int num_metals, const double t_ox, double *imacro)
{
    int *gpu_index;
    cudaMalloc((void **)&gpu_index, N * sizeof(int)); // indices of the site array
    int *atom_gpu_index;
    cudaMalloc((void **)&atom_gpu_index, N * sizeof(int)); // indices of the atom array

    thrust::device_ptr<int> gpu_index_ptr = thrust::device_pointer_cast(gpu_index);
    thrust::sequence(gpu_index_ptr, gpu_index_ptr + N, 0);

    double *last_atom = thrust::copy_if(thrust::device, gpubuf.site_x, gpubuf.site_x + N, gpubuf.site_element, gpubuf.atom_x, is_defect());
    int N_atom = last_atom - gpubuf.atom_x;
    thrust::copy_if(thrust::device, gpubuf.site_y, gpubuf.site_y + N, gpubuf.site_element, gpubuf.atom_y, is_defect());
    thrust::copy_if(thrust::device, gpubuf.site_z, gpubuf.site_z + N, gpubuf.site_element, gpubuf.atom_z, is_defect());
    thrust::copy_if(thrust::device, gpubuf.site_charge, gpubuf.site_charge + N, gpubuf.site_element, gpubuf.atom_charge, is_defect());
    thrust::copy_if(thrust::device, gpubuf.site_potential, gpubuf.site_potential + N, gpubuf.site_element, gpubuf.atom_potential, is_defect());
    thrust::copy_if(thrust::device, gpubuf.site_element, gpubuf.site_element + N, gpubuf.site_element, gpubuf.atom_element, is_defect());
    thrust::copy_if(thrust::device, gpu_index, gpu_index + N, gpubuf.site_element, atom_gpu_index, is_defect());

    double *gpu_imacro, *gpu_m, *gpu_x, *gpu_ineg, *gpu_diag, *gpu_pdisp, *gpu_A;
    cudaMalloc((void **)&gpu_imacro, 1 * sizeof(double));                      // IMACRO
    cudaMalloc((void **)&gpu_m, (N_atom + 2) * sizeof(double));                // M
    cudaMalloc((void **)&gpu_x, (size_t) (N_atom + 2) * (N_atom + 2) * sizeof(double)); // X
    cudaMalloc((void **)&gpu_ineg, (size_t) N_atom * N_atom * sizeof(double));          // INEG
    cudaMalloc((void **)&gpu_diag, (N_atom + 2) * sizeof(double));             // DIAG
    cudaMalloc((void **)&gpu_pdisp, N_atom * sizeof(double));                  // PDISP
    cudaMalloc((void **)&gpu_A, (size_t) (N_atom + 1) * (N_atom + 1) * sizeof(double)); // A

    cudaMemset(gpu_x, 0, (size_t) (N_atom + 2) * (N_atom + 2) * sizeof(double));
    cudaDeviceSynchronize();

    cudaMemset(gpu_m, 0, (N_atom + 2) * sizeof(double));
    cudaDeviceSynchronize();

    // Make M vector
    cudaMemset(gpu_m, 0, (N_atom + 2) * sizeof(double));
    cudaDeviceSynchronize();

    thrust::device_ptr<double> m_ptr = thrust::device_pointer_cast(gpu_m);
    thrust::fill(m_ptr, m_ptr + 1, -high_G * Vd);
    thrust::fill(m_ptr + 1, m_ptr + 2, high_G * Vd);

    // Create X
    int num_threads = 128;
    int blocks_per_row = (N_atom - 1) / num_threads + 1;
    int num_blocks = blocks_per_row * N;
    create_X<<<num_blocks, num_threads>>>(
        gpu_x, gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z,
        gpubuf.metal_types, gpubuf.atom_element, gpubuf.atom_charge, gpubuf.atom_potential,
        gpubuf.lattice, pbc, high_G, low_G,
        nn_dist, m_e, V0, num_source_inj, num_ground_ext, N_atom, num_metals, t_ox, Vd);
    cudaDeviceSynchronize();

    // Diag X
    cudaMemset(gpu_diag, 0, (N_atom + 2) * sizeof(double));
    cudaDeviceSynchronize();
    num_threads = 512;
    blocks_per_row = (N_atom + 2 - 1) / num_threads + 1;
    num_blocks = blocks_per_row * (N + 2);
    diagonal_sum<NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS * sizeof(double)>>>(gpu_x, gpu_diag, N_atom + 2);
    cudaDeviceSynchronize();

    set_diag<<<blocks_per_row, num_threads>>>(gpu_x, gpu_diag, N_atom + 2);
    cudaDeviceSynchronize();

    // GESV
    int lwork = 0;              /* size of workspace */
    double *gpu_work = nullptr; /* device workspace for getrf */
    int *gpu_info = nullptr;    /* error info */
    int *gpu_ipiv;

    // double *gpu_A = gpu_x + (N_atom + 2) + 1
    //  double* gpu_D = gpu_k + (N_left_tot * N) + N_left_tot;

    cudaMalloc((void **)&gpu_ipiv, (N_atom + 1) * sizeof(int));
    cudaMalloc((void **)(&gpu_info), sizeof(int));
    cudaMemcpy2D(gpu_A, (N_atom + 1) * sizeof(double), gpu_x, (N_atom + 2) * sizeof(double), (N_atom + 1) * sizeof(double), (N_atom + 1), cudaMemcpyDeviceToDevice);

    CheckCusolverDnError(cusolverDnDgetrf_bufferSize(handle_cusolver, N_atom + 1, N_atom + 1, gpu_A, N_atom + 1, &lwork));
    cudaDeviceSynchronize();

    cudaMalloc((void **)(&gpu_work), sizeof(double) * lwork);

    // Solve Ax=B through LU factorization
    CheckCusolverDnError(cusolverDnDgetrf(handle_cusolver, N_atom + 1, N_atom + 1, gpu_A, N_atom + 1, gpu_work, gpu_ipiv, gpu_info));
    cudaDeviceSynchronize();

    CheckCusolverDnError(cusolverDnDgetrs(handle_cusolver, CUBLAS_OP_T, N_atom + 1, 1, gpu_A, N_atom + 1, gpu_ipiv, gpu_m, N_atom + 1, gpu_info));
    cudaDeviceSynchronize();

    // NOTE: M is different from the cpu code in the 4th decimal place!!!
    // Compute I_macro
    cudaMemset(gpu_imacro, 0, sizeof(double));
    cudaDeviceSynchronize();

    num_threads = 512;
    num_blocks = (N_atom - 1) / num_threads + 1;
    // num_blocks = min(65535, num_blocks);
    get_imacro<NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS * sizeof(double)>>>(gpu_x, gpu_m, gpu_imacro, Vd, N_atom);
    cudaDeviceSynchronize();

    // Copy back I_macro to the GPU
    double i_macro = 0;
    cudaMemcpy(&i_macro, gpu_imacro, sizeof(double), cudaMemcpyDeviceToHost);
    *imacro = i_macro;

    // Find index of minimum element in m[2:N+2]
    auto min_index = thrust::min_element(thrust::device, gpu_m + 2, gpu_m + N_atom + 2) - gpu_m;

    num_threads = 512;
    blocks_per_row = (N_atom + 2 - 1) / num_threads + 1;
    num_blocks = blocks_per_row;
    update_m<<<num_blocks, num_threads>>>(gpu_m, min_index, N_atom + 2);
    cudaDeviceSynchronize();

    // Compute I_neg
    num_threads = 128;
    blocks_per_row = (N_atom - 1) / num_threads + 1;
    num_blocks = blocks_per_row * N;

    set_ineg<<<num_blocks, num_threads>>>(gpu_ineg, gpu_x, gpu_m, Vd, N_atom,
                                          gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z,
                                          pbc, gpubuf.lattice, nn_dist,
                                          V0, gpubuf.atom_potential, t_ox);
    cudaDeviceSynchronize();


    // Update I_neg diagonal
    cudaMemset(gpu_diag, 0, (N_atom + 2) * sizeof(double));
    cudaDeviceSynchronize();

    diagonal_sum<NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS * sizeof(double)>>>(gpu_ineg, gpu_diag, N_atom);
    cudaDeviceSynchronize();

    set_diag<<<blocks_per_row, num_threads>>>(gpu_ineg, gpu_diag, N_atom);
    cudaDeviceSynchronize();

    double alpha = 1.0, beta = 0.0;
    double *gpu_alpha, *gpu_beta;
    cudaMalloc((void **)&gpu_alpha, sizeof(double));
    cudaMalloc((void **)&gpu_beta, sizeof(double));
    cudaMemcpy(gpu_alpha, &alpha, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_beta, &beta, sizeof(double), cudaMemcpyHostToDevice);

    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    CheckCublasError( cublasDgemv(handle, CUBLAS_OP_T, N_atom, N_atom, gpu_alpha, gpu_ineg, N_atom, gpu_m + 2, 1, gpu_beta, gpu_pdisp, 1) );
    cudaDeviceSynchronize();

    num_threads = 512;
    num_blocks = (N_atom - 1) / num_threads + 1;
    num_blocks = min(65535, num_blocks);
    copy_pdisp<<<num_blocks, num_threads>>>(gpubuf.site_power, gpubuf.site_element, gpubuf.metal_types, gpu_pdisp, atom_gpu_index, N_atom, num_metals);
    cudaDeviceSynchronize();

    cudaFree(gpu_ipiv);
    cudaFree(gpu_work);
    cudaFree(gpu_imacro);
    cudaFree(gpu_m);
    cudaFree(gpu_x);
    cudaFree(gpu_ineg);
    cudaFree(gpu_diag);
    cudaFree(gpu_pdisp);
    cudaFree(gpu_A);
    cudaFree(gpu_info);
    cudaFree(gpu_alpha);
    cudaFree(gpu_beta);
    cudaFree(gpu_index);
    cudaFree(atom_gpu_index);
}

double execute_kmc_step_gpu(const int N, const int nn, const int *neigh_idx, const int *site_layer,
                            const double *lattice, const int pbc, const double *T_bg, 
                            const double *freq, const double *sigma, const double *k,
                            const double *posx, const double *posy, const double *posz, 
                            const double *site_potential, const double *site_temperature,
                            ELEMENT *site_element, int *site_charge, RandomNumberGenerator &rng, const int *neigh_idx_host){

    // **************************
    // **** Build Event List ****
    // **************************

    // the KMC event list arrays only exist in gpu memory
    EVENTTYPE *event_type; 
    double    *event_prob; 
    gpuErrchk( cudaMalloc((void**)&event_type, N * nn * sizeof(EVENTTYPE)) );
    gpuErrchk( cudaMalloc((void**)&event_prob, N * nn * sizeof(double)) );
                
    int num_threads = 512;
    int num_blocks = (N * nn - 1) / num_threads + 1;

    // populate the event_type and event_prob arrays:
    build_event_list<<<num_blocks, num_threads>>>(N, nn, neigh_idx, 
                                                 site_layer, lattice, pbc,
                                                 T_bg, freq, sigma, k,
                                                 posx, posy, posz, 
                                                 site_potential, site_temperature, 
                                                 site_element, site_charge, event_type, event_prob);

    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );

    // **************************
    // ** Event Execution Loop **
    // **************************

    // helper variables:
    // NOTE: INITIALIZE THESE ON GPU AND USE MEMCPY DEVICETODEVICE INSTEAD
    int two_host = 2;
    int two_neg_host = -2;
    int zero_host = 0;
    ELEMENT defect_element_host = DEFECT;
    ELEMENT O_defect_element_host = OXYGEN_DEFECT;
    ELEMENT vacancy_element_host = VACANCY;
    ELEMENT O_element_host = O_EL;

    double *event_prob_cum;
    gpuErrchk( cudaMalloc((void**)&event_prob_cum, N * nn * sizeof(double)) );
 
    double freq_host;
    gpuErrchk( cudaMemcpy(&freq_host, freq, 1 * sizeof(double), cudaMemcpyDeviceToHost) );

    double event_time = 0.0;
    while (event_time < 1 / freq_host) {

        // get the cumulative sum of the probabilities
        thrust::inclusive_scan(thrust::device, event_prob, event_prob + N * nn, event_prob_cum);

        // select an event
        double Psum_host;
        gpuErrchk( cudaMemcpy(&Psum_host, event_prob_cum + N * nn - 1, sizeof(double), cudaMemcpyDeviceToHost) );
        double number = rng.getRandomNumber() * Psum_host;
        int event_idx = thrust::upper_bound(thrust::device, event_prob_cum, event_prob_cum + N * nn, number) - event_prob_cum;
        // std::cout << "selected event: " << event_idx << "\n";

        EVENTTYPE sel_event_type = NULL_EVENT;
        gpuErrchk( cudaMemcpy(&sel_event_type, event_type + event_idx, sizeof(EVENTTYPE), cudaMemcpyDeviceToHost) );

        // test output:
        // double sel_event_prob;
        // gpuErrchk( cudaMemcpy(&sel_event_prob, event_prob + event_idx, sizeof(double), cudaMemcpyDeviceToHost) );
        // std::cout << "Selected event index: " << event_idx << " with type "
        //           << sel_event_type << " and probability " << sel_event_prob << std::endl;

        // get attributes of the sites involved:
        int i_host = static_cast<int>(floorf(event_idx / nn));
        int j_host;
        ELEMENT element_i_host, element_j_host;
        int charge_i_host, charge_j_host;

        gpuErrchk( cudaMemcpy(&j_host, neigh_idx + event_idx, sizeof(int), cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(&element_i_host, site_element + i_host, sizeof(ELEMENT), cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(&element_j_host, site_element + j_host, sizeof(ELEMENT), cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(&charge_i_host, site_charge + i_host, sizeof(int), cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(&charge_j_host, site_charge + j_host, sizeof(int), cudaMemcpyDeviceToHost) );

        // Event execution loop
        switch (sel_event_type)
        {
        case VACANCY_GENERATION:
        {
            if (element_i_host != DEFECT || element_j_host != O_EL)
            {
                print("Wrong event type - VACANCY_GENERATION!");
                print(return_element(element_i_host) << " and " << return_element(element_j_host));
            }

            // turn the defect (i) into an oxygen ion
            // turn the oxygen (j) into a charged vacancy
            gpuErrchk( cudaMemcpy(site_element + i_host, &O_defect_element_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_element + j_host, &vacancy_element_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + i_host, &two_neg_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + j_host, &two_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );

            break;
        }
        case VACANCY_RECOMBINATION:
        {
            if (element_i_host != OXYGEN_DEFECT || element_j_host != VACANCY)
            {
                print("Wrong event type - VACANCY_RECOMBINATION!");
                print(return_element(element_i_host) << " and " << return_element(element_j_host));
            }

            // turn the oxygen (i) into a defect
            // turn the vacancy (j) into an oxygen atom
            gpuErrchk( cudaMemcpy(site_element + i_host, &defect_element_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_element + j_host, &O_element_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + i_host, &zero_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + j_host, &zero_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );

            break;
        }
        case VACANCY_DIFFUSION:
        {
            if (element_i_host != VACANCY || element_j_host != O_EL)
            {
                print("Wrong event type - VACANCY_DIFFUSION!");
                print(return_element(element_i_host) << " and " << return_element(element_j_host));
            }

            // swap element and charge
            gpuErrchk( cudaMemcpy(site_element + i_host, &element_j_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_element + j_host, &element_i_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + i_host, &charge_j_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + j_host, &charge_i_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );

            break;
        }
        case ION_DIFFUSION:
        {
            if (element_i_host != OXYGEN_DEFECT || element_j_host != DEFECT)
            {
                print("Wrong event type - ION_DIFFUSION!");
                print(return_element(element_i_host) << " and " << return_element(element_j_host));
            }

            // swap element and charge
            gpuErrchk( cudaMemcpy(site_element + i_host, &element_j_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_element + j_host, &element_i_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + i_host, &charge_j_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + j_host, &charge_i_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );

            break;
        }
        default:
            print("error: unidentified event key found: ");
            print(sel_event_type);
        }

        // Deactivate conflicting events

        EVENTTYPE null_event_host = NULL_EVENT;
        double zero_double_host = 0.0;

        // other site's events with i or j
         int i_, j_;
        for (auto idx = 0; idx < N * nn; ++idx){
            i_ = std::floor(idx / nn);
            j_ = neigh_idx_host[idx];

            if (i_host == i_ || j_host == j_ || i_host == j_ || j_host == i_){

                gpuErrchk( cudaMemcpy(event_type + idx, &null_event_host, 1 * sizeof(EVENTTYPE), cudaMemcpyHostToDevice) );
                gpuErrchk( cudaMemcpy(event_prob + idx, &zero_double_host, 1 * sizeof(double), cudaMemcpyHostToDevice) );
            }
        }

        // REPLACE WITH THRUST::FILL AND CUDAMEMSET
        for (int fill_ind = 0; fill_ind < (nn-1); fill_ind++){

            // i's events with its neighbors    
            gpuErrchk( cudaMemcpy(event_type + i_host * nn + fill_ind, &null_event_host, 1 * sizeof(EVENTTYPE), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(event_type + i_host * nn + fill_ind, &zero_double_host, 1 * sizeof(double), cudaMemcpyHostToDevice) );

            // j's events with its neighbors
            gpuErrchk( cudaMemcpy(event_type + j_host * nn + fill_ind, &null_event_host, 1 * sizeof(EVENTTYPE), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(event_type + j_host * nn + fill_ind, &zero_double_host, 1 * sizeof(double), cudaMemcpyHostToDevice) );
        }

        event_time = -log(rng.getRandomNumber()) / Psum_host;
        // std::cout << "event time: " << event_time << "\n";
    }

    gpuErrchk( cudaFree(event_prob_cum) );
    gpuErrchk( cudaFree(event_type) );
    gpuErrchk( cudaFree(event_prob) );

    return event_time;    
}

void copytoConstMemory(std::vector<double> E_gen, std::vector<double> E_rec, std::vector<double> E_Vdiff, std::vector<double> E_Odiff)
{   
    gpuErrchk( cudaMemcpyToSymbol(E_gen_const, E_gen.data(), E_gen.size() * sizeof(double)) );
    gpuErrchk( cudaMemcpyToSymbol(E_rec_const, E_rec.data(), E_rec.size() * sizeof(double)) );
    gpuErrchk( cudaMemcpyToSymbol(E_Vdiff_const, E_Vdiff.data(), E_Vdiff.size() * sizeof(double)) );
    gpuErrchk( cudaMemcpyToSymbol(E_Odiff_const, E_Odiff.data(), E_Odiff.size() * sizeof(double)) );
}

    // # if __CUDA_ARCH__>=200
    // printf("%i \n", tid);
    // #endif  


        // std::cout << "created K"
    //           << "\n";

    // // debug - use floor() for i
    // std::vector<double> new_k(N * N);
    // gpuErrchk( cudaMemcpy(new_k.data(), gpu_k , N * N * sizeof(double), cudaMemcpyDeviceToHost) );
    // std::cout << "copied";
    // std::ofstream fout("new_k.txt");
    // for(int i = 0; i< N*N; i++){
    //     if (new_k[i] != 0){
    //         fout << new_k[i]; 
    //         fout << ' ';
    //     }
    // }
    // exit(1);


    // // debug
    // double *M = (double *)calloc(N, sizeof(double));
    // gpuErrchk( cudaMemcpy(M, gpubuf.site_potential, N * sizeof(double), cudaMemcpyDeviceToHost) );
    // std::cout << "copied\n";
    // std::ofstream fout2("gpu_M.txt");
    // for(int i = 0; i< N; i++){
    //     if (M[i] != 0){
    //         fout2 << M[i]; 
    //         fout2 << ' ';
    //     }
    // }