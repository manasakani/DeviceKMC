#include "utils.h"
#include <cuda_runtime.h>
#include "gpu_buffers.h"
#include <iostream>
#include <omp.h>
#include <cub/cub.cuh>

#define NUM_THREADS 512

// returns true if thing is present in the array of things
template <typename T>
__device__ int is_in_array_gpu_og(const T *array, const T element, const int size) {

    for (int i = 0; i < size; ++i) {
        if (array[i] == element) {
        return 1;
        }
    }
    return 0;
}

template <typename T>
int is_in_array_cpu(const T *array, const T element, const int size) {

    for (int i = 0; i < size; ++i) {
        if (array[i] == element) {
        return 1;
        }
    }
    return 0;
}

__device__ double site_dist_gpu_og(double pos1x, double pos1y, double pos1z,
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

double site_dist_cpu(double pos1x, double pos1y, double pos1z,
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

__global__ void create_K_og(
    double *X,
    const double *posx, const double *posy, const double *posz,
    const ELEMENT *metals, const ELEMENT *element, const int *site_charge,
    const double *lattice, const bool pbc, const double d_high_G, const double d_low_G,
    const double nn_dist, const int N, const int num_metals)
{

    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;

    for (auto idx = tid_total; idx < (size_t) N * N; idx += num_threads_total)
    {
        int i = idx / N;
        int j = idx % N;

        bool metal1 = is_in_array_gpu_og(metals, element[i], num_metals);
        bool metal2 = is_in_array_gpu_og(metals, element[j], num_metals);
        bool ischarged1 = site_charge[i] != 0;
        bool ischarged2 = site_charge[j] != 0;
        bool isVacancy1 = element[i] == VACANCY;
        bool isVacancy2 = element[j] == VACANCY;
        bool cvacancy1 = isVacancy1 && !ischarged1;
        bool cvacancy2 = isVacancy2 && !ischarged2;
        double dist = site_dist_gpu_og(posx[i], posy[i], posz[i], posx[j], posy[j], posz[j], lattice[0], lattice[1], lattice[2], pbc);

        bool neighbor = false;
        if (dist < nn_dist && i != j)
            neighbor = true;

        // direct terms:
        if (i != j && neighbor)
        {
            if ((metal1 && metal2) || (cvacancy1 && cvacancy2))
            {
                X[N * (i) + (j)] = -d_high_G;
            }
            else
            {
                X[N * (i) + (j)] = -d_low_G;
            }
        }
    }
}


template <int NTHREADS>
__global__ void diagonal_sum_og(double *A, double *diag, int N)
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

__global__ void set_diag_og(double *A, double *diag, int N)
{
    int didx = blockIdx.x * blockDim.x + threadIdx.x;
    if (didx < N)
    {
        double tmp = A[didx * N + didx];
        A[didx * N + didx] = 2 * tmp - diag[didx];
    }
}

template<typename T>
void sparse_to_dense(
    T *dense_matrix,
    T *data,
    int *col_indices,
    int *row_ptr,
    int matrix_size)
{

    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            // could not work for complex data type
            dense_matrix[i*matrix_size + j] = T(0);
        }
    }

    for(int i = 0; i < matrix_size; i++){
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            dense_matrix[i*matrix_size + col_indices[j]] = data[j];
        }
    }
}


template<typename T>
bool assert_array_magnitude(
    T *array_test,
    T *array_ref,
    double abstol,
    double reltol,
    int size)
{
    double sum_difference = 0.0;
    double sum_ref = 0.0;
    for (int i = 0; i < size; i++) {
        sum_difference += std::abs(array_test[i] - array_ref[i]) * std::abs(array_test[i] - array_ref[i]);
        sum_ref += std::abs(array_ref[i])*std::abs(array_ref[i]);

    }
    sum_difference = std::sqrt(sum_difference);
    sum_ref = std::sqrt(sum_ref);
    if (sum_difference > reltol * sum_ref + abstol) {
        std::printf("Arrays are in magnitude not the same\n");
        std::cout << "Difference " << sum_difference << std::endl;
        std::cout << "Relative " << sum_difference/sum_ref << std::endl;
        std::cout << "Mixed tolerance " << reltol * sum_ref + abstol << std::endl;
        return false;
    }

    return true;
}


double assemble_K_og(cusolverDnHandle_t handle, const GPUBuffers &gpubuf, const int N, const int N_left_tot, const int N_right_tot,
                              const double Vd, const int pbc, const double d_high_G, const double d_low_G, const double nn_dist,
                              const int num_metals, int kmc_step_count,
                              double *K_h)
{

    double *VL, *VR;
    gpuErrchk( cudaMalloc((void **)&VL, N_left_tot * sizeof(double)) );
    gpuErrchk( cudaMalloc((void **)&VR, N_right_tot * sizeof(double)) );

    double *gpu_k;
    double *gpu_diag;
    gpuErrchk( cudaMalloc((void **)&gpu_k, (size_t) N * N * sizeof(double)) );
    gpuErrchk( cudaMalloc((void **)&gpu_diag, N * sizeof(double)) );
    gpuErrchk( cudaMemset(gpu_k, 0, (size_t) N * N * sizeof(double)) );
    gpuErrchk( cudaDeviceSynchronize() );

    double time = -omp_get_wtime();
    //  BUILDING THE CONDUCTIVITY MATRIX
    int num_threads = 512;
    int blocks_per_row = (N - 1) / num_threads + 1;
    int num_blocks = blocks_per_row * N;

    // compute the off-diagonal elements of K
    create_K_og<<<num_blocks, num_threads>>>(
        gpu_k, gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
        gpubuf.metal_types, gpubuf.site_element, gpubuf.site_charge,
        gpubuf.lattice, pbc, d_high_G, d_low_G,
        nn_dist, N, num_metals);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Update the diagonal of K
    gpuErrchk( cudaMemset(gpu_diag, 0, N * sizeof(double)) );
    gpuErrchk( cudaDeviceSynchronize() );
    diagonal_sum_og<NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS * sizeof(double)>>>(gpu_k, gpu_diag, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    num_blocks = (N - 1) / num_threads + 1;
    set_diag_og<<<num_blocks, num_threads>>>(gpu_k, gpu_diag, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemset(gpu_diag, 0, N * sizeof(double)) );
    gpuErrchk( cudaDeviceSynchronize() );

    time += omp_get_wtime();

    gpuErrchk(cudaMemcpy(K_h, gpu_k, N * N * sizeof(double), cudaMemcpyDeviceToHost));


    cudaFree(gpu_diag);
    cudaFree(VL);
    cudaFree(VR);
    cudaFree(gpu_k);

    return time;
}

int count_nnz(
    double *array,
    int size
)
{
    int count = 0;
    for (int i = 0; i < size; i++) {
        if (array[i] != 0.0) {
            count++;
        }
    }
    return count;
}


int calc_nnz(
    const double *posx, const double *posy, const double *posz,
    const double *lattice, const bool pbc,
    const double cutoff_radius,
    int matrix_size
){
    int nnz = 0;
    for(int i = 0; i < matrix_size; i++){
        for(int j = 0; j < matrix_size; j++){
            double dist = site_dist_cpu(posx[i], posy[i], posz[i], posx[j], posy[j], posz[j], lattice[0], lattice[1], lattice[2], pbc);
            if(dist < cutoff_radius){
                nnz++;
            }
        }
    }
    return nnz;
}

void calc_nnz_per_row(
    const double *posx, const double *posy, const double *posz,
    const double *lattice, const bool pbc,
    const double cutoff_radius,
    int matrix_size,
    int *nnz_per_row
){
    
    for(int i = 0; i < matrix_size; i++){
        int nnz_row = 0;
        for(int j = 0; j < matrix_size; j++){
            double dist = site_dist_cpu(posx[i], posy[i], posz[i], posx[j], posy[j], posz[j], lattice[0], lattice[1], lattice[2], pbc);
            if(dist < cutoff_radius){
                nnz_row++;
            }
        }
        nnz_per_row[i] = nnz_row;
    }
}

__global__ void calc_nnz_per_row_gpu(
    const double *posx_d, const double *posy_d, const double *posz_d,
    const double *lattice_d, const bool pbc,
    const double cutoff_radius,
    int matrix_size,
    int *nnz_per_row_d
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO optimize this with a 2D grid instead of 1D
    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        int nnz_row = 0;
        for(int j = 0; j < matrix_size; j++){
            double dist = site_dist_gpu_og(posx_d[i], posy_d[i], posz_d[i],
                                        posx_d[j], posy_d[j], posz_d[j],
                                        lattice_d[0], lattice_d[1], lattice_d[2], pbc);
            if(dist < cutoff_radius){
                nnz_row++;
            }
        }
        nnz_per_row_d[i] = nnz_row;
    }

}


bool assert_nnz(
    double *matrix,
    int *row_ptr,
    int *col_indices,
    int nnz,
    int matrix_size
)
{
    bool nnz_match = true;

    // match that all the elements in csr indices are no zero
    for(int i = 0; i < matrix_size; i++){
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(matrix[i*matrix_size + col_indices[j]] == 0.0){
                nnz_match = false;
            }
        }
    }

    // match that element not in csr indices are zero
    for(int i = 0; i < matrix_size; i++){
        for(int j = 0; j < matrix_size; j++){

            bool in_csr = false;
            for(int k = row_ptr[i]; k < row_ptr[i+1]; k++){
                if(col_indices[k] == j){
                    in_csr = true;
                }
            }

            if(!in_csr && matrix[i*matrix_size + j] != 0.0){
                nnz_match = false;
            }

        }
    }

    return nnz_match;
}


template<typename T>
T reduce_array(
    T *array,
    int size
)
{
    T reduction = T(0);
    for (int i = 0; i < size; i++) {
        reduction += array[i];
    }
    return reduction;
}

template<typename T>
void modified_exclusive_scan(
    T *array,
    T *excl_scan,
    int size
)
{
    // saves additional the sum of all elements
    T reduction = T(0);
    for (int i = 0; i < size+1; i++) {
        excl_scan[i] = reduction;
        reduction += array[i];
    }
}



void assemble_K_indices(
    const double *posx, const double *posy, const double *posz,
    const double *lattice, const bool pbc,
    const double cutoff_radius,
    int matrix_size,
    int *nnz_per_row,
    int *row_ptr,
    int *col_indices)
{
    // row ptr is already calculated
    // exclusive scam of nnz_per_row

    // loop first over rows, then over columns
    for(int i = 0; i < matrix_size; i++){
        int nnz_row = 0;
        for(int j = 0; j < matrix_size; j++){
        
            double dist = site_dist_cpu(posx[i], posy[i], posz[i], posx[j], posy[j], posz[j], lattice[0], lattice[1], lattice[2], pbc);
            if(dist < cutoff_radius){
                col_indices[row_ptr[i] + nnz_row] = j;
                nnz_row++;
            }
        }
    }
}

__global__ void assemble_K_indices_gpu(
    const double *posx_d, const double *posy_d, const double *posz_d,
    const double *lattice_d, const bool pbc,
    const double cutoff_radius,
    int matrix_size,
    int *nnz_per_row_d,
    int *row_ptr_d,
    int *col_indices_d)
{
    // row ptr is already calculated
    // exclusive scam of nnz_per_row

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    //TODO can be optimized with a 2D grid instead of 1D
    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        int nnz_row = 0;
        for(int j = 0; j < matrix_size; j++){
        
            double dist = site_dist_gpu_og(posx_d[i], posy_d[i], posz_d[i],
                                        posx_d[j], posy_d[j], posz_d[j],
                                        lattice_d[0], lattice_d[1], lattice_d[2], pbc);
            if(dist < cutoff_radius){
                col_indices_d[row_ptr_d[i] + nnz_row] = j;
                nnz_row++;
            }
        }
    }
}




void calc_off_diagonal_K_cpu(
    const ELEMENT *metals, const ELEMENT *element, const int *site_charge,
    int num_metals,
    double d_high_G, double d_low_G,
    int matrix_size,
    int *col_indices,
    int *row_ptr,
    double *data
)
{
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < matrix_size; i++){
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(i != col_indices[j]){
                bool metal1 = is_in_array_cpu(metals, element[i], num_metals);
                bool metal2 = is_in_array_cpu(metals, element[col_indices[j]], num_metals);
                bool ischarged1 = site_charge[i] != 0;
                bool ischarged2 = site_charge[col_indices[j]] != 0;
                bool isVacancy1 = element[i] == VACANCY;
                bool isVacancy2 = element[col_indices[j]] == VACANCY;
                bool cvacancy1 = isVacancy1 && !ischarged1;
                bool cvacancy2 = isVacancy2 && !ischarged2;
                if ((metal1 && metal2) || (cvacancy1 && cvacancy2))
                {
                    data[j] = -d_high_G;
                }
                else
                {
                    data[j] = -d_low_G;
                }
            }
        }
    }
}


__global__ void calc_off_diagonal_K_gpu(
    const ELEMENT *metals, const ELEMENT *element, const int *site_charge,
    int num_metals,
    double d_high_G, double d_low_G,
    int matrix_size,
    int *col_indices,
    int *row_ptr,
    double *data
)
{
    // parallelize over rows
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(i != col_indices[j]){
                bool metal1 = is_in_array_gpu_og(metals, element[i], num_metals);
                bool metal2 = is_in_array_gpu_og(metals, element[col_indices[j]], num_metals);
                bool ischarged1 = site_charge[i] != 0;
                bool ischarged2 = site_charge[col_indices[j]] != 0;
                bool isVacancy1 = element[i] == VACANCY;
                bool isVacancy2 = element[col_indices[j]] == VACANCY;
                bool cvacancy1 = isVacancy1 && !ischarged1;
                bool cvacancy2 = isVacancy2 && !ischarged2;
                if ((metal1 && metal2) || (cvacancy1 && cvacancy2))
                {
                    data[j] = -d_high_G;
                }
                else
                {
                    data[j] = -d_low_G;
                }
            }
        }
    }
}



void calc_diagonal_K_cpu(
    int *col_indices,
    int *row_ptr,
    double *data,
    int matrix_size
)
{
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < matrix_size; i++){
        //reduce the elements in the row
        double tmp = 0.0;
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(i != col_indices[j]){
                tmp += data[j];
            }
        }
        //write the diagonal element
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(i == col_indices[j]){
                data[j] = -tmp;
            }
        }
    }
}

__global__ void calc_diagonal_K_gpu(
    int *col_indices,
    int *row_ptr,
    double *data,
    int matrix_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        //reduce the elements in the row
        double tmp = 0.0;
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(i != col_indices[j]){
                tmp += data[j];
            }
        }
        //write the diagonal element
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(i == col_indices[j]){
                data[j] = -tmp;
            }
        }
    }
}


void assemble_K_cpu(
    const ELEMENT *metals, const ELEMENT *element, const int *site_charge,
    const int num_metals,
    const double d_high_G, const double d_low_G,
    int matrix_size,
    int *col_indices,
    int *row_ptr,
    double *data
)
{

    calc_off_diagonal_K_cpu(
        metals, element, site_charge,
        num_metals,
        d_high_G, d_low_G,
        matrix_size,
        col_indices,
        row_ptr,
        data);

    calc_diagonal_K_cpu(col_indices, row_ptr, data, matrix_size);


}


void assemble_K_gpu(
    const ELEMENT *metals_d, const ELEMENT *element_d, const int *site_charge_d,
    const int num_metals,
    const double d_high_G, const double d_low_G,
    int matrix_size,
    int *col_indices_d,
    int *row_ptr_d,
    double *data_d
)
{
    int threads = 512;
    int blocks = (matrix_size + threads - 1) / threads;

    calc_off_diagonal_K_gpu<<<blocks, threads>>>(
        metals_d, element_d, site_charge_d,
        num_metals,
        d_high_G, d_low_G,
        matrix_size,
        col_indices_d,
        row_ptr_d,
        data_d);

    calc_diagonal_K_gpu<<<blocks, threads>>>(col_indices_d, row_ptr_d, data_d, matrix_size);
}


void indices_creation_cpu(
    const double *site_posx, const double *site_posy, const double *site_posz,
    const double *lattice, const bool pbc,
    const double cutoff_radius,
    const int matrix_size,
    int **col_indices,
    int **row_ptr,
    int *nnz
)
{
    int nnz_per_row[matrix_size];
    *row_ptr = (int *)malloc((matrix_size + 1) * sizeof(int));

    // calculate the nnz per row
    calc_nnz_per_row(site_posx, site_posy, site_posz, lattice, pbc, cutoff_radius, matrix_size, nnz_per_row);
    
    // exclusive sum to get the row ptr
    modified_exclusive_scan<int>(nnz_per_row, (*row_ptr), matrix_size);

    // by convention the last element of the row ptr is the nnz
    nnz[0] = (*row_ptr)[matrix_size];

    *col_indices = (int *)malloc(nnz[0] * sizeof(int));
    

    // assemble the indices of K
    assemble_K_indices(
        site_posx, site_posy, site_posz,
        lattice, pbc,
        cutoff_radius,
        matrix_size,
        nnz_per_row,
        (*row_ptr),
        (*col_indices)
    );
}

void indices_creation_gpu(
    const double *site_posx_d, const double *site_posy_d, const double *site_posz_d,
    const double *lattice_d, const bool pbc,
    const double cutoff_radius,
    const int matrix_size,
    int **col_indices_d,
    int **row_ptr_d,
    int *nnz
)
{
    // parallelize over rows
    int threads = 512;
    int blocks = (matrix_size + threads - 1) / threads;

    int *nnz_per_row_d;
    gpuErrchk( cudaMalloc((void **)row_ptr_d, (matrix_size + 1) * sizeof(int)) );
    gpuErrchk( cudaMalloc((void **)&nnz_per_row_d, matrix_size * sizeof(int)) );
    gpuErrchk(cudaMemset((*row_ptr_d), 0, (matrix_size + 1) * sizeof(int)) );

    // calculate the nnz per row
    calc_nnz_per_row_gpu<<<blocks, threads>>>(site_posx_d, site_posy_d, site_posz_d, lattice_d, pbc, cutoff_radius, matrix_size, nnz_per_row_d);

    void     *temp_storage_d = NULL;
    size_t   temp_storage_bytes = 0;
    // determines temporary device storage requirements for inclusive prefix sum
    cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*row_ptr_d)+1, matrix_size);
    // Allocate temporary storage for inclusive prefix sum
    gpuErrchk(cudaMalloc(&temp_storage_d, temp_storage_bytes));
    // Run inclusive prefix sum
    // inclusive sum starting at second value to get the row ptr
    // which is the same as inclusive sum starting at first value and last value filled with nnz
    cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*row_ptr_d)+1, matrix_size);
    
    // nnz is the same as (*row_ptr_d)[matrix_size]
    gpuErrchk( cudaMemcpy(nnz, (*row_ptr_d) + matrix_size, sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMalloc((void **)col_indices_d, nnz[0] * sizeof(int)) );

    // assemble the indices of K
    assemble_K_indices_gpu<<<blocks, threads>>>(
        site_posx_d, site_posy_d, site_posz_d,
        lattice_d, pbc,
        cutoff_radius,
        matrix_size,
        nnz_per_row_d,
        (*row_ptr_d),
        (*col_indices_d)
    );

    cudaFree(temp_storage_d);
    cudaFree(nnz_per_row_d);
}




void test_assemble_K(cusolverDnHandle_t handle, const GPUBuffers &gpubuf, const int N, const int N_left_tot, const int N_right_tot,
                              const double Vd, const int pbc, const double d_high_G, const double d_low_G, const double nn_dist,
                              const int num_metals, int kmc_step_count)
{


    double *K_og;
    K_og = (double *)malloc(N * N * sizeof(double));
    double *K_sparse_cpu_assemble;
    double *K_sparse_gpu_assemble;
    K_sparse_cpu_assemble = (double *)malloc(N * N * sizeof(double));
    K_sparse_gpu_assemble = (double *)malloc(N * N * sizeof(double));

    double reltol = 1e-12;
    double abstol = 1e-12;

    // original code for dense assemble on the gpu
    assemble_K_og(handle, gpubuf, N, N_left_tot, N_right_tot, Vd, pbc, d_high_G, d_low_G, nn_dist, num_metals, kmc_step_count, K_og);

    // load the data from the gpu for the sparse assemble
    double *site_posx = (double *)malloc(N * sizeof(double));
    double *site_posy = (double *)malloc(N * sizeof(double));
    double *site_posz = (double *)malloc(N * sizeof(double));
    double *lattice = (double *)malloc(3 * sizeof(double));
    //gpubuf.metal_types, gpubuf.site_element, gpubuf.site_charge,
    //const ELEMENT *metals, const ELEMENT *element, const int *site_charge,
    ELEMENT *metals = (ELEMENT *)malloc(num_metals * sizeof(ELEMENT));
    ELEMENT *element = (ELEMENT *)malloc(N * sizeof(ELEMENT));
    int *site_charge = (int *)malloc(N * sizeof(int));

    gpuErrchk(cudaMemcpy(site_posx, gpubuf.site_x, N * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(site_posy, gpubuf.site_y, N * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(site_posz, gpubuf.site_z, N * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(lattice, gpubuf.lattice, 3 * sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(metals, gpubuf.metal_types, num_metals * sizeof(ELEMENT), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(element, gpubuf.site_element, N * sizeof(ELEMENT), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(site_charge, gpubuf.site_charge, N * sizeof(int), cudaMemcpyDeviceToHost));

    // count the nonzero elements of the dense K
    int nnz_og = count_nnz(K_og, N * N);

    // assemble the nonzero elements of K
    int nnz = calc_nnz(site_posx, site_posy, site_posz, lattice, pbc, nn_dist, N);

    double *data_h = (double *)malloc(nnz * sizeof(double));

    int nnz_cpu;
    int *row_ptr_h = NULL;
    int *col_indices_h = NULL;

    indices_creation_cpu(
        site_posx, site_posy, site_posz,
        lattice, pbc,
        nn_dist,
        N,
        &col_indices_h,
        &row_ptr_h,
        &nnz_cpu
    );


    // nnz from counting
    if(nnz_og != nnz){
        std::cout << "nnz mismatch" << std::endl;
        std::cout << "nnz_og " << nnz_og << std::endl;
        std::cout << "nnz " << nnz << std::endl;
    }
    else{
        std::cout << "nnz match" << std::endl;
    }
    // nnz from reducing nnz per row
    if(nnz_og != nnz_cpu){
        std::cout << "nnz_cpu mismatch" << std::endl;
        std::cout << "nnz_og " << nnz_og << std::endl;
        std::cout << "nnz_cpu " << nnz_cpu << std::endl;
    }
    else{
        std::cout << "nnz_cpu match" << std::endl;
    }


    // test if the indices are correct
    // i.e. that in the dense matrix onlz elements
    // at positions given by indices and ptr are non zero
    bool right_indices = assert_nnz(
        K_og,
        row_ptr_h,
        col_indices_h,
        nnz,
        N);
    if(!right_indices){
        std::cout << "indices mismatch" << std::endl;
    }
    else{
        std::cout << "indices match" << std::endl;
    }


    assemble_K_cpu(
        metals, element, site_charge,
        num_metals,
        d_high_G, d_low_G,
        N,
        col_indices_h,
        row_ptr_h,
        data_h
    );
    sparse_to_dense<double>(K_sparse_cpu_assemble, data_h, col_indices_h, row_ptr_h, N);

    if(!assert_array_magnitude(K_sparse_cpu_assemble, K_og, abstol, reltol, N * N)){
        std::cout << "K_sparse_cpu_assemble and K_og mismatch" << std::endl;
    }
    else{
        std::cout << "K_sparse_cpu_assemble and K_og match" << std::endl;
    }

    // assemble the indices on the gpu

    int *col_indices_gpu_assemble_d;
    int *row_ptr_gpu_assemble_d;
    int nnz_gpu_assemble;

    indices_creation_gpu(
        gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
        gpubuf.lattice, pbc,
        nn_dist,
        N,
        &col_indices_gpu_assemble_d,
        &row_ptr_gpu_assemble_d,
        &nnz_gpu_assemble
    );





    // gpuErrchk( cudaMalloc((void **)&row_ptr_gpu_assemble_d, (N + 1) * sizeof(int)) );
    // gpuErrchk( cudaMalloc((void **)&nnz_per_row_d, N * sizeof(int)) );
    // gpuErrchk(cudaMemset(row_ptr_gpu_assemble_d, 0, (N + 1) * sizeof(int)) );

    // // parallelize over rows
    // int threads = 512;
    // int blocks = (N + threads - 1) / threads;

    // // calculate the nnz per row
    // calc_nnz_per_row_gpu<<<blocks, threads>>>(gpubuf.site_x, gpubuf.site_y, gpubuf.site_z, gpubuf.lattice, pbc, nn_dist, N, nnz_per_row_d);



    // void     *temp_storage_d = NULL;
    // size_t   temp_storage_bytes = 0;
    // // determines temporary device storage requirements for inclusive prefix sum
    // cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, row_ptr_gpu_assemble_d+1, N);
    // // Allocate temporary storage for inclusive prefix sum
    // gpuErrchk(cudaMalloc(&temp_storage_d, temp_storage_bytes));
    // // Run inclusive prefix sum
    // // inclusive sum starting at second value to get the row ptr
    // // which is the same as inclusive sum starting at first value and last value filled with nnz
    // cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, row_ptr_gpu_assemble_d+1, N);
    // cudaFree(temp_storage_d);
    // // nnz is the same as row_ptr_gpu_assemble_d[N]
    // int nnz_gpu_assemble = 0;
    // gpuErrchk( cudaMemcpy(&nnz_gpu_assemble, row_ptr_gpu_assemble_d + N, sizeof(int), cudaMemcpyDeviceToHost) );

    if(nnz_gpu_assemble != nnz){
        std::cout << "nnz_gpu_assemble mismatch" << std::endl;
        std::cout << "nnz_gpu_assemble " << nnz_gpu_assemble << std::endl;
        std::cout << "nnz " << nnz << std::endl;
    }
    else{
        std::cout << "nnz_gpu_assemble match" << std::endl;
    }

    // gpuErrchk( cudaMalloc((void **)&col_indices_gpu_assemble_d, nnz_gpu_assemble * sizeof(int)) );

    // // assemble the indices of K
    // assemble_K_indices_gpu<<<blocks, threads>>>(
    //     gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
    //     gpubuf.lattice, pbc,
    //     nn_dist,
    //     N,
    //     nnz_per_row_d,
    //     row_ptr_gpu_assemble_d,
    //     col_indices_gpu_assemble_d
    // );

    // unload the indices and test if they are the same
    int col_indices_gpu_assemble_h[nnz];
    int row_ptr_gpu_assemble_h[N + 1];
    gpuErrchk( cudaMemcpy(col_indices_gpu_assemble_h, col_indices_gpu_assemble_d, nnz * sizeof(int), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(row_ptr_gpu_assemble_h, row_ptr_gpu_assemble_d, (N + 1) * sizeof(int), cudaMemcpyDeviceToHost) );


    if(!assert_array_magnitude(col_indices_gpu_assemble_h, col_indices_h, abstol, reltol, nnz)){
        std::cout << "col_indices_gpu_assemble_h and col_indices_h mismatch" << std::endl;
    }
    else{
        std::cout << "col_indices_gpu_assemble_h and col_indices_h match" << std::endl;
    }
    if(!assert_array_magnitude(row_ptr_gpu_assemble_h, row_ptr_h, abstol, reltol, N + 1)){
        std::cout << "row_ptr_gpu_assemble_h and row_ptr_h mismatch" << std::endl;
    }
    else{
        std::cout << "row_ptr_gpu_assemble_h and row_ptr_h match" << std::endl;
    }


    double *data_d;
    int *col_indices_d;
    int *row_ptr_d;

    gpuErrchk( cudaMalloc((void **)&data_d, nnz * sizeof(double)) );
    gpuErrchk( cudaMalloc((void **)&col_indices_d, nnz * sizeof(int)) );
    gpuErrchk( cudaMalloc((void **)&row_ptr_d, (N + 1) * sizeof(int)) );


    // copy the nonzero elements of K to the gpu
    gpuErrchk( cudaMemcpy(col_indices_d, col_indices_h, nnz * sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(row_ptr_d, row_ptr_h, (N + 1) * sizeof(int), cudaMemcpyHostToDevice) );


    assemble_K_gpu(
        gpubuf.metal_types, gpubuf.site_element, gpubuf.site_charge,
        num_metals,
        d_high_G, d_low_G,
        N,
        col_indices_d,
        row_ptr_d,
        data_d
    );

    // unload sparse matrix
    gpuErrchk( cudaMemcpy(data_h, data_d, nnz * sizeof(double), cudaMemcpyDeviceToHost) );
    sparse_to_dense<double>(K_sparse_gpu_assemble, data_h, col_indices_h, row_ptr_h, N);



    if(!assert_array_magnitude(K_sparse_gpu_assemble, K_og, abstol, reltol, N * N)){
        std::cout << "K_sparse_gpu_assemble and K_og mismatch" << std::endl;
    }
    else{
        std::cout << "K_sparse_gpu_assemble and K_og match" << std::endl;
    }


    // start of the benchmark
    int num_measurents = 10;

    double times_cpu[num_measurents];
    double times_gpu[num_measurents];
    double times_gpu_og[num_measurents];


    for(int i = 0; i < num_measurents; i++){
        times_gpu_og[i] = assemble_K_og(handle, gpubuf, N, N_left_tot, N_right_tot, Vd, pbc, d_high_G, d_low_G, nn_dist, num_metals, kmc_step_count, K_og);
        std::cout << "times_gpu_og " << times_gpu_og[i] << std::endl;
    }

    for(int i = 0; i < num_measurents; i++){
        times_cpu[i] = omp_get_wtime();
        assemble_K_cpu(
            metals, element, site_charge,
            num_metals,
            d_high_G, d_low_G,
            N,
            col_indices_h,
            row_ptr_h,
            data_h
        );
        times_cpu[i] = omp_get_wtime() - times_cpu[i];
        std::cout << "times_cpu " << times_cpu[i] << std::endl;
    }

    for(int i = 0; i < num_measurents; i++){
        times_gpu[i] = omp_get_wtime();
        assemble_K_gpu(
            gpubuf.metal_types, gpubuf.site_element, gpubuf.site_charge,
            num_metals,
            d_high_G, d_low_G,
            N,
            col_indices_d,
            row_ptr_d,
            data_d
        );
        times_gpu[i] = omp_get_wtime() - times_gpu[i];
        std::cout << "times_gpu " << times_gpu[i] << std::endl;
    }




    gpuErrchk( cudaFree(data_d) );
    gpuErrchk( cudaFree(col_indices_d) );
    gpuErrchk( cudaFree(row_ptr_d) );
    gpuErrchk( cudaFree(col_indices_gpu_assemble_d) );
    gpuErrchk( cudaFree(row_ptr_gpu_assemble_d) );
    

    free(K_og);
    free(K_sparse_cpu_assemble);
    free(K_sparse_gpu_assemble);
    free(site_posx);
    free(site_posy);
    free(site_posz);
    free(lattice);
    free(metals);
    free(element);
    free(site_charge);
    free(data_h);
    free(col_indices_h);
    free(row_ptr_h);
    std::cout << "K matrix assembled" << std::endl;




    std::cin.ignore();

    // TODO : calculate the RHS
    //  SOLVING FOR THE NEGATIVE INTERNAL POTENTIALS (KSUB)
    // prepare contact potentials

    // double *gpu_k_sub;
    // gpuErrchk( cudaMalloc((void **)&gpu_k_sub, N_interface * sizeof(double)) ); 
    // gpuErrchk( cudaMemset(gpu_k_sub, 0, N_interface * sizeof(double)) );
    // blocks_per_row = (N_left_tot - 1) / num_threads + 1;
    // num_blocks = blocks_per_row * N_interface;

    // thrust::device_ptr<double> VL_ptr = thrust::device_pointer_cast(VL);
    // thrust::fill(VL_ptr, VL_ptr + N_left_tot, -Vd/2);
    // thrust::device_ptr<double> VR_ptr = thrust::device_pointer_cast(VR);
    // thrust::fill(VR_ptr, VR_ptr + N_right_tot, Vd/2);


    // diagonal_sum_K<NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS * sizeof(double)>>>
    //     (&gpu_k[N_left_tot * N], gpu_diag, VL, N, N_interface, N_left_tot);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    // diagonal_sum_K<NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS * sizeof(double)>>>
    //     (&gpu_k[N_left_tot * N + N - N_right_tot], gpu_diag, VR, N, N_interface, N_right_tot);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    // set_diag_K<<<blocks_per_row, num_threads>>>(gpu_k_sub, gpu_diag, N_interface);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    

    // cudaFree(gpu_k_sub);


}

