#include "hip/hip_runtime.h"
#include "gpu_solvers.h"
#include <hipcub/hipcub.hpp>

// Sparse matrix assembly functions and iterative solvers

#define NUM_THREADS 512
const double eV_to_J = 1.60217663e-19;          // [C]
const double h_bar = 1.054571817e-34;           // [Js]

// used to be named 'calc_diagonal_A_gpu'
__global__ void reduce_rows_into_diag( 
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


// used to be called 'set_diag'
__global__ void write_to_diag(double *A, double *diag, int N)
{
    int didx = blockIdx.x * blockDim.x + threadIdx.x;
    if (didx < N)
    {
        double tmp = A[didx * N + didx];
        A[didx * N + didx] = 2 * tmp - diag[didx];
    }
}

// sum the rows of A into the vector diag
// with an explicit instantiation for the linker
template __global__ void row_reduce<NUM_THREADS>(double *A, double *diag, int N); 

template <int NTHREADS>
__global__ void row_reduce(double *A, double *diag, int N)
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

__global__ void calc_nnz_per_row(
    const double *posx_d, const double *posy_d, const double *posz_d,
    const double *lattice_d, const bool pbc,
    const double cutoff_radius,
    int block_size_i,
    int block_size_j,
    int block_start_i,
    int block_start_j,
    int *nnz_per_row_d
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO optimize this with a 2D grid instead of 1D
    for(int row = idx; row < block_size_i; row += blockDim.x * gridDim.x){
        int nnz_row = 0;
        for(int col = 0; col < block_size_j; col++){
            int i = block_start_i + row;
            int j = block_start_j + col;
            double dist = site_dist_gpu(posx_d[i], posy_d[i], posz_d[i],
                                        posx_d[j], posy_d[j], posz_d[j],
                                        lattice_d[0], lattice_d[1], lattice_d[2], pbc);
            if(dist < cutoff_radius){
                nnz_row++;
            }
        }
        nnz_per_row_d[row] = nnz_row;
    }

}

__global__ void assemble_K_indices_gpu_off_diagonal_block(
    const double *posx_d, const double *posy_d, const double *posz_d,
    const double *lattice_d, const bool pbc,
    const double cutoff_radius,
    int block_size_i,
    int block_size_j,
    int block_start_i,
    int block_start_j,
    int *row_ptr_d,
    int *col_indices_d)
{
    // row ptr is already calculated
    // exclusive scam of nnz_per_row

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    //TODO can be optimized with a 2D grid instead of 1D
    for(int row = idx; row < block_size_i; row += blockDim.x * gridDim.x){
        int nnz_row = 0;
        for(int col = 0; col < block_size_j; col++){
            int i = block_start_i + row;
            int j = block_start_j + col;
            double dist = site_dist_gpu(posx_d[i], posy_d[i], posz_d[i],
                                        posx_d[j], posy_d[j], posz_d[j],
                                        lattice_d[0], lattice_d[1], lattice_d[2], pbc);
            if(dist < cutoff_radius){
                col_indices_d[row_ptr_d[row] + nnz_row] = col;
                nnz_row++;
            }
        }
    }
}

void indices_creation_gpu_off_diagonal_block(
    const double *posx_d, const double *posy_d, const double *posz_d,
    const double *lattice_d, const bool pbc,
    const double cutoff_radius,
    int block_size_i,
    int block_size_j,
    int block_start_i,
    int block_start_j,
    int **col_indices_d,
    int **row_ptr_d,
    int *nnz
)
{
    // parallelize over rows
    int threads = 512;
    int blocks = (block_size_i + threads - 1) / threads;

    int *nnz_per_row_d;
    gpuErrchk( hipMalloc((void **)row_ptr_d, (block_size_i + 1) * sizeof(int)) );
    gpuErrchk( hipMalloc((void **)&nnz_per_row_d, block_size_i * sizeof(int)) );
    gpuErrchk(hipMemset((*row_ptr_d), 0, (block_size_i + 1) * sizeof(int)) );

    // calculate the nnz per row
    hipLaunchKernelGGL(calc_nnz_per_row, blocks, threads, 0, 0, posx_d, posy_d, posz_d, lattice_d, pbc, cutoff_radius,
        block_size_i, block_size_j, block_start_i, block_start_j, nnz_per_row_d);

    void     *temp_storage_d = NULL;
    size_t   temp_storage_bytes = 0;

    // determines temporary device storage requirements for inclusive prefix sum
    hipcub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*row_ptr_d)+1, block_size_i);

    // Allocate temporary storage for inclusive prefix sum
    gpuErrchk(hipMalloc(&temp_storage_d, temp_storage_bytes));

    // Run inclusive prefix sum
    // inclusive sum starting at second value to get the row ptr
    // which is the same as inclusive sum starting at first value and last value filled with nnz
    hipcub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*row_ptr_d)+1, block_size_i);
    
    // nnz is the same as (*row_ptr_d)[block_size_i]
    gpuErrchk( hipMemcpy(nnz, (*row_ptr_d) + block_size_i, sizeof(int), hipMemcpyDeviceToHost) );
    gpuErrchk( hipMalloc((void **)col_indices_d, nnz[0] * sizeof(int)) );

    // assemble the indices of K
    hipLaunchKernelGGL(assemble_K_indices_gpu_off_diagonal_block, blocks, threads, 0, 0, 
        posx_d, posy_d, posz_d,
        lattice_d, pbc,
        cutoff_radius,
        block_size_i,
        block_size_j,
        block_start_i,
        block_start_j,
        (*row_ptr_d),
        (*col_indices_d)
    );

    hipFree(temp_storage_d);
    hipFree(nnz_per_row_d);
}


__global__ void determine_neighbor_nnz(
    const double *posx_i_d, const double *posy_i_d, const double *posz_i_d,
    const double *posx_j_d, const double *posy_j_d, const double *posz_j_d,
    const double *lattice_d, const bool pbc,
    const double cutoff_radius,
    long int size_i,
    long int size_j,
    int *dist_nnz_d,
    int *dist_nnz_per_row_d
){
    // this rank has i sites
    // other rank has j sites
    long long int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(long long int id = idx; id < size_i * size_j; id += blockDim.x * gridDim.x){
        int i = id / size_j;
        int j = id % size_j;
        if(i < size_i && j < size_j){
            double dist = site_dist_gpu(posx_i_d[i], posy_i_d[i], posz_i_d[i],
                                        posx_j_d[j], posy_j_d[j], posz_j_d[j],
                                        lattice_d[0], lattice_d[1], lattice_d[2], pbc);
            if(dist < cutoff_radius){
                atomicAdd(dist_nnz_d, 1);
                atomicAdd(dist_nnz_per_row_d + i, 1);
            }
        }
    }
}

template <typename T>
void writeArrayToBinFile(T* array, int numElements, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<char*>(array), numElements*sizeof(T));
        file.close();
        std::cout << "Array data written to file: " << filename << std::endl;
    } else {
        std::cerr << "Unable to open the file for writing." << std::endl;
    }
}


void initialize_sparsity_K(GPUBuffers &gpubuf, int pbc, const double nn_dist, int num_atoms_contact, KMC_comm &kmc_comm)
{
    

    int rank = kmc_comm.rank_K;
    int size = kmc_comm.size_K;
    int rows_this_rank = kmc_comm.counts_K[rank];
    int disp_this_rank = kmc_comm.displs_K[rank];

    int N_left_tot = num_atoms_contact;
    int N_right_tot = num_atoms_contact;
    int N_interface = gpubuf.N_ - (N_left_tot + N_right_tot);
    
    int *dist_nnz_h = new int[size];
    int *dist_nnz_d;
    int *dist_nnz_per_row_d;

    gpuErrchk( hipMalloc((void **)&dist_nnz_d, size * sizeof(int)) );
    gpuErrchk(hipMemset(dist_nnz_d, 0, size * sizeof(int)));
    gpuErrchk( hipMalloc((void **)&dist_nnz_per_row_d, size * rows_this_rank * sizeof(int)) );
    gpuErrchk(hipMemset(dist_nnz_per_row_d, 0, size * rows_this_rank * sizeof(int)));

    // Assemble the sparsity pattern
    
    // loop over the size to determine neighbours
    for(int i = 0; i < size; i++){
        int rows_other = kmc_comm.counts_K[i];
        int displ_other = kmc_comm.displs_K[i];

        int threads = 1024;
        //start with self
        int blocks = (rows_this_rank - 1) / threads + 1;
        hipLaunchKernelGGL(calc_nnz_per_row, blocks, threads, 0, 0, 
            gpubuf.site_x + N_left_tot,
            gpubuf.site_y + N_left_tot,
            gpubuf.site_z + N_left_tot,
            gpubuf.lattice, pbc, nn_dist,
            rows_this_rank,
            rows_other,
            disp_this_rank,
            displ_other,
            dist_nnz_per_row_d + i * rows_this_rank
        );

        // reduce nnz per row
        void     *temp_storage_d = NULL;
        size_t   temp_storage_bytes = 0;
        hipcub::DeviceReduce::Sum(
        temp_storage_d, temp_storage_bytes, 
            dist_nnz_per_row_d + i * rows_this_rank,
            dist_nnz_d + i, rows_this_rank);

        // Allocate temporary storage
        hipMalloc(&temp_storage_d, temp_storage_bytes);

        // Run sum-reduction
        hipcub::DeviceReduce::Sum(temp_storage_d, temp_storage_bytes,
            dist_nnz_per_row_d + i * rows_this_rank,
            dist_nnz_d + i, rows_this_rank);
    }

    gpuErrchk( hipMemcpy(dist_nnz_h, dist_nnz_d, size * sizeof(int), hipMemcpyDeviceToHost) );
    // counting neighbours
    int neighbor_count = 0;
    for(int i = 0; i < size; i++){
        if(dist_nnz_h[i] > 0){
            neighbor_count++;
        }
    }

    // // print nieghbor count:
    // std::cout << "K neighbor_count = " << neighbor_count << std::endl;

    // get the indices of the neighbours
    int *neighbor_idx = new int[neighbor_count];
    int *neighbor_nnz_h = new int[neighbor_count];
    int *neighbor_nnz_per_row_d;
    gpuErrchk( hipMalloc((void **)&neighbor_nnz_per_row_d, neighbor_count * rows_this_rank * sizeof(int)) );

    // determine neighbours
    neighbor_count = 0;
    for(int i = 0; i < size; i++){
        int neighbor = (i+rank) % size;
        if(dist_nnz_h[neighbor] > 0){
            neighbor_idx[neighbor_count] = neighbor;
            neighbor_count++;
        }
    }        
    // fill the neighbor nnz
    for(int i = 0; i < neighbor_count; i++){
        neighbor_nnz_h[i] = dist_nnz_h[neighbor_idx[i]];
        gpuErrchk( hipMemcpy(neighbor_nnz_per_row_d + i * rows_this_rank,
            dist_nnz_per_row_d + neighbor_idx[i] * rows_this_rank,
            rows_this_rank * sizeof(int), hipMemcpyHostToDevice) );
    }


    // alloc memory
    int **col_indices_d = new int*[neighbor_count];
    int **row_ptr_d = new int*[neighbor_count];
    for(int i = 0; i < neighbor_count; i++){
        gpuErrchk( hipMalloc((void **)&col_indices_d[i], neighbor_nnz_h[i] * sizeof(int)) );
        gpuErrchk( hipMalloc((void **)&row_ptr_d[i], (rows_this_rank + 1) * sizeof(int)) );
    }
    
    // create row ptr
    for(int i = 0; i < neighbor_count; i++){

        gpuErrchk(hipMemset(row_ptr_d[i], 0, (rows_this_rank + 1) * sizeof(int)));
        void     *temp_storage_d = NULL;
        size_t   temp_storage_bytes = 0;
        // determines temporary device storage requirements for inclusive prefix sum
        hipcub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes,
            neighbor_nnz_per_row_d + i * rows_this_rank, (row_ptr_d[i])+1, rows_this_rank);

        // Allocate temporary storage for inclusive prefix sum
        gpuErrchk(hipMalloc(&temp_storage_d, temp_storage_bytes));
        // Run inclusive prefix sum
        // inclusive sum starting at second value to get the row ptr
        // which is the same as inclusive sum starting at first value and last value filled with nnz
        hipcub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes,
            neighbor_nnz_per_row_d + i * rows_this_rank, (row_ptr_d[i])+1, rows_this_rank);

        // Free temporary storage
        gpuErrchk(hipFree(temp_storage_d)); 

    }


    // column indices
    for(int i = 0; i < neighbor_count; i++){
        int neighbour = neighbor_idx[i];
        int rows_neighbour = kmc_comm.counts_K[neighbour];
        int disp_neighbour = kmc_comm.displs_K[neighbour];

        int threads = 1024;
        int blocks = (rows_this_rank + threads - 1) / threads;
       
        hipLaunchKernelGGL(assemble_K_indices_gpu_off_diagonal_block, blocks, threads, 0, 0, 
            gpubuf.site_x,
            gpubuf.site_y,
            gpubuf.site_z,
            gpubuf.lattice, pbc, nn_dist,
            rows_this_rank,
            rows_neighbour,
            N_left_tot + disp_this_rank,
            N_left_tot + disp_neighbour,
            row_ptr_d[i],
            col_indices_d[i]
        );
    }

    // TODO: replace with unique ptrs
    gpubuf.K_distributed = new Distributed_matrix(
        N_interface,
        kmc_comm.counts_K,
        kmc_comm.displs_K,
        neighbor_count,
        neighbor_idx,
        col_indices_d,
        row_ptr_d,
        neighbor_nnz_h,
        rocsparse_spmv_alg_csr_adaptive,
        kmc_comm.comm_K
    );

    gpubuf.K_p_distributed = new Distributed_vector(
        N_interface,
        kmc_comm.counts_K,
        kmc_comm.displs_K,
        gpubuf.K_distributed->number_of_neighbours,
        gpubuf.K_distributed->neighbours,
        kmc_comm.comm_K
    );

    // indices of the off-diagonal leftcontact-A matrix
    indices_creation_gpu_off_diagonal_block(
        gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
        gpubuf.lattice, pbc,
        nn_dist,
        rows_this_rank,
        N_left_tot,
        N_left_tot + disp_this_rank,
        0,
        &gpubuf.left_col_indices_d,
        &gpubuf.left_row_ptr_d,
        &gpubuf.left_nnz
    );

    // indices of the off-diagonal A-rightcontact matrix
    indices_creation_gpu_off_diagonal_block(
        gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
        gpubuf.lattice, pbc,
        nn_dist,
        rows_this_rank,
        N_right_tot,
        N_left_tot + disp_this_rank,
        N_left_tot + N_interface,
        &gpubuf.right_col_indices_d,
        &gpubuf.right_row_ptr_d,
        &gpubuf.right_nnz
    );

    for(int i = 0; i < neighbor_count; i++){
        gpuErrchk( hipFree(col_indices_d[i]) );
        gpuErrchk( hipFree(row_ptr_d[i]) );
    }   
    delete[] col_indices_d;
    delete[] row_ptr_d;
    delete[] neighbor_idx;
    delete[] dist_nnz_h;
    gpuErrchk( hipFree(dist_nnz_d) );    
    gpuErrchk( hipFree(dist_nnz_per_row_d) );
    delete[] neighbor_nnz_h;
    gpuErrchk( hipFree(neighbor_nnz_per_row_d) );
}

void initialize_sparsity_CB(GPUBuffers &gpubuf, int pbc, const double nn_dist, int num_atoms_contact){

    int N_left_tot = num_atoms_contact;
    int N_right_tot = num_atoms_contact;
    int N_interface = gpubuf.N_ - (N_left_tot + N_right_tot);

    // This populates the site_CB_edge vector, and runs once at the beginning (not distributed)
    // needs to be distributed in the future if ever OOM (use a function pointer to the correct kernel for CB edge)    
    Assemble_K_sparsity(gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
                        gpubuf.lattice, pbc, nn_dist,
                        N_interface, N_left_tot, N_right_tot,
                        &gpubuf.Device_row_ptr_d, &gpubuf.Device_col_indices_d, &gpubuf.Device_nnz,
                        &gpubuf.contact_left_col_indices, &gpubuf.contact_left_row_ptr, &gpubuf.contact_left_nnz,
                        &gpubuf.contact_right_col_indices, &gpubuf.contact_right_row_ptr, &gpubuf.contact_right_nnz);
}



// check that sparse and dense versions are the same
void check_sparse_dense_match(int m, int nnz, double *dense_matrix, int* d_csrRowPtr, int* d_csrColInd, double* d_csrVal){
    
    double *h_D = (double *)calloc(m*m, sizeof(double));
    double *h_D_csr = (double *)calloc(nnz, sizeof(double));
    int *h_pointers = (int *)calloc((m + 1), sizeof(int));
    int *h_inds = (int *)calloc(nnz, sizeof(int));

    gpuErrchk( hipMemcpy(h_D, dense_matrix, m*m * sizeof(double), hipMemcpyDeviceToHost) );
    gpuErrchk( hipMemcpy(h_D_csr, d_csrVal, nnz * sizeof(double), hipMemcpyDeviceToHost) );
    gpuErrchk( hipMemcpy(h_pointers, d_csrRowPtr, (m + 1) * sizeof(int), hipMemcpyDeviceToHost) );
    gpuErrchk( hipMemcpy(h_inds, d_csrColInd, nnz * sizeof(int), hipMemcpyDeviceToHost) );

    int nnz_count = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < m; col++) {
            int i = row * m + col;  // Linear index in dense matrix
            // Check if the element in the dense matrix is non-zero
            if (h_D[i] != 0) {
                // Compare the row and column indices
                if (h_D[i] != h_D_csr[nnz_count] || col != h_inds[nnz_count]) {
                    std::cout << "Mismatch found at (row, col) = (" << row << ", " << col << ")\n";
                }
                nnz_count++;
            }
        }
    }
}

// dump sparse matrix into a file
void dump_csr_matrix_txt(int m, int nnz, int* d_csrRowPtr, int* d_csrColIndices, double* d_csrValues, int kmc_step_count){

    // Copy matrix back to host memory
    double *h_csrValues = (double *)calloc(nnz, sizeof(double));
     std::cout << "here1\n";
    int *h_csrRowPtr = (int *)calloc((m + 1), sizeof(int));
     std::cout << "here\n";
    int *h_csrColIndices = (int *)calloc(nnz, sizeof(int));

    // print nnz and m to console
    std::cout << "nnz = " << nnz << ", m = " << m << "\n";

    gpuErrchk( hipMemcpy(h_csrValues, d_csrValues, nnz * sizeof(double), hipMemcpyDeviceToHost) );
    gpuErrchk( hipMemcpy(h_csrRowPtr, d_csrRowPtr, (m + 1) * sizeof(int), hipMemcpyDeviceToHost) );
    gpuErrchk( hipMemcpy(h_csrColIndices, d_csrColIndices, nnz * sizeof(int), hipMemcpyDeviceToHost) );

    std::cout << "memcpy\n";

    // print to file, tagged with the kmc step number
    std::ofstream fout_val("csrValues_step#" + std::to_string(kmc_step_count) + ".txt");
    for(int i = 0; i < nnz; i++){
        fout_val << h_csrValues[i] << " "; 
    }
    std::ofstream fout_row("csrRowPtr_step#" + std::to_string(kmc_step_count) + ".txt");
    for(int i = 0; i < (m + 1); i++){
        fout_row << h_csrRowPtr[i] << " "; 
    }
    std::ofstream fout_col("csrColIndices_step#" + std::to_string(kmc_step_count) + ".txt");
    for(int i = 0; i < nnz; i++){
        fout_col << h_csrColIndices[i] << " "; 
    }

    std::cout << "dumping matrix done\n";
    fflush(stdout);

    free(h_csrValues);
    free(h_csrRowPtr);
    free(h_csrColIndices);
}

// // Solution of A*x = y using cusolver in host pointer mode
// void sparse_system_solve(hipsolverHandle_t handle, int* d_csrRowPtr, int* d_csrColInd, double* d_csrVal,
//                          int nnz, int m, double *d_x, double *d_y){

//     // Ref: https://stackoverflow.com/questions/31840341/solving-general-sparse-linear-systems-in-cuda

//     // cusolverSpDcsrlsvlu only supports the host path
//     int *h_A_RowIndices = (int *)malloc((m + 1) * sizeof(int));
//     int *h_A_ColIndices = (int *)malloc(nnz * sizeof(int));
//     double *h_A_Val = (double *)malloc(nnz * sizeof(double));
//     double *h_x = (double *)malloc(m * sizeof(double));
//     double *h_y = (double *)malloc(m * sizeof(double));
//     gpuErrchk( hipMemcpy(h_A_RowIndices, d_csrRowPtr, (m + 1) * sizeof(int), hipMemcpyDeviceToHost) );
//     gpuErrchk( hipMemcpy(h_A_ColIndices, d_csrColInd, nnz * sizeof(int), hipMemcpyDeviceToHost) );
//     gpuErrchk( hipMemcpy(h_A_Val, d_csrVal, nnz * sizeof(double), hipMemcpyDeviceToHost) );   
//     gpuErrchk( hipMemcpy(h_x, d_x, m * sizeof(double), hipMemcpyDeviceToHost) );
//     gpuErrchk( hipMemcpy(h_y, d_y, m * sizeof(double), hipMemcpyDeviceToHost) );

//     hipsparseMatDescr_t matDescrA;
//     hipsparseCreateMatDescr(&matDescrA);
//     hipsparseSetMatType(matDescrA, HIPSPARSE_MATRIX_TYPE_GENERAL);
//     hipsparseSetMatIndexBase(matDescrA, HIPSPARSE_INDEX_BASE_ZERO);

//     int singularity;
//     double tol = 0.00000001;

//     // Solve with LU
//     // CheckCusolverDnError( cusolverSpDcsrlsvluHost(handle, m, nnz, matDescrA, h_A_Val, h_A_RowIndices, 
//     //                       h_A_ColIndices, h_y, tol, 0, h_x, &singularity) );
    
//     // Solve with QR
//     // CheckCusolverDnError( cusolverSpDcsrlsvqrHost(handle, m, nnz, matDescrA, h_A_Val, h_A_RowIndices, 
//     //                       h_A_ColIndices, h_y, tol, 1, h_x, &singularity) );

//     // Solve with Cholesky
//     CheckCusolverDnError( cusolverSpDcsrlsvcholHost(handle, m, nnz, matDescrA, h_A_Val, h_A_RowIndices,
//                           h_A_ColIndices, h_y, tol, 1, h_x, &singularity) );

//     gpuErrchk( hipDeviceSynchronize() );
//     if (singularity != -1){
//         std::cout << "In sparse_system_solve: Matrix has a singularity at : " << singularity << "\n";
//     }

//     // copy back the solution vector:
//     gpuErrchk( hipMemcpy(d_x, h_x, m * sizeof(double), hipMemcpyHostToDevice) );

//     cusolverSpDestroy(handle);
//     hipsparseDestroyMatDescr(matDescrA);
//     free(h_A_RowIndices);
//     free(h_A_ColIndices);
//     free(h_A_Val);
//     free(h_x);
//     free(h_y);
// }

// Extracts the inverse sqrt of the diagonal values into a vector to use for the preconditioning
__global__ void computeDiagonalInvSqrt(const double* A_data, const int* A_row_ptr,
                                       const int* A_col_indices, double* diagonal_values_inv_sqrt_d,
                                       const int matrix_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < matrix_size) {
        // Find the range of non-zero elements for the current row
        int row_start = A_row_ptr[tid];
        int row_end = A_row_ptr[tid + 1];

        // Initialize the sum for the diagonal element
        double diagonal_sum = 0.0;

        // Loop through the non-zero elements in the current row
        for (int i = row_start; i < row_end; ++i) {
            if (A_col_indices[i] == tid) {
                // Found the diagonal element
                diagonal_sum = A_data[i];
                break;
            }
        }

        double diagonal_inv_sqrt = 1.0 / sqrt(diagonal_sum);

        // Store the result in the output array
        diagonal_values_inv_sqrt_d[tid] = diagonal_inv_sqrt;
    }
}

// apply Jacobi preconditioner to an rhs vector
__global__ void jacobi_precondition_array(
    double *array,
    double *diagonal_values_inv_sqrt,
    int matrix_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        array[i] = array[i] * diagonal_values_inv_sqrt[i];
    }

}

// apply Jacobi preconditioner to matrix
__global__ void jacobi_precondition_matrix(
    double *data,
    const int *col_indices,
    const int *row_indptr,
    double *diagonal_values_inv_sqrt,
    int matrix_size
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        // Iterate over the row elements
        for(int j = row_indptr[i]; j < row_indptr[i+1]; j++){
            // Use temporary variables to store the original values
            double original_value = data[j];

            // Update data with the preconditioned value
            data[j] = original_value * diagonal_values_inv_sqrt[i] * diagonal_values_inv_sqrt[col_indices[j]];
        }
    }
}

// apply Jacobi preconditioner to starting guess
__global__ void jacobi_unprecondition_array(
    double *array,
    double *diagonal_values_inv_sqrt,
    int matrix_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){
        array[i] = array[i] * 1/diagonal_values_inv_sqrt[i];
    }

}

// Iterative sparse linear solver using CG steps
void solve_sparse_CG_Jacobi(hipblasHandle_t handle_cublas, hipsparseHandle_t handle, 
							double* A_data, int* A_row_ptr,
                            int* A_col_indices, const int A_nnz, int m, double *d_x, double *d_y){

    // A is an m x m sparse matrix represented by CSR format
    // - d_x is right hand side vector in gpu memory,
    // - d_y is solution vector in gpu memory.
    // - d_z is intermediate result on gpu memory.

    // Sets the initial guess for the solution vector to zero
    bool zero_guess = 0;    

    // Error tolerance for the norm of the residual in the CG steps
    double tol = 1e-14;  // make this an input, used to be 1e-12

    double one = 1.0;
    double n_one = -1.0;
    double zero = 0.0;
    double *one_d, *n_one_d, *zero_d;
    gpuErrchk( hipMalloc((void**)&one_d, sizeof(double)) );
    gpuErrchk( hipMalloc((void**)&n_one_d, sizeof(double)) );
    gpuErrchk( hipMalloc((void**)&zero_d, sizeof(double)) );
    gpuErrchk( hipMemcpy(one_d, &one, sizeof(double), hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(n_one_d, &n_one, sizeof(double), hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(zero_d, &zero, sizeof(double), hipMemcpyHostToDevice) );
    hipsparseStatus_t status;

    // ************************************
    // ** Initial Guess **

    if (zero_guess)
    {
        // Set the initial guess for the solution vector to zero
        gpuErrchk( hipMemset(d_y, 0, m * sizeof(double)) ); 
        gpuErrchk( hipDeviceSynchronize() );
    }

    // *******************************
    // ** Preconditioner **

    double* diagonal_values_inv_sqrt_d;
    hipMalloc((void**)&diagonal_values_inv_sqrt_d, sizeof(double) * m);

    int block_size = 256;
    int grid_size = (m + block_size - 1) / block_size;

    hipLaunchKernelGGL(computeDiagonalInvSqrt, grid_size, block_size, 0, 0, A_data, A_row_ptr, A_col_indices,
                                                      diagonal_values_inv_sqrt_d, m);
    gpuErrchk( hipPeekAtLastError() );
    gpuErrchk( hipDeviceSynchronize() );

    // scale rhs
    hipLaunchKernelGGL(jacobi_precondition_array, grid_size, block_size, 0, 0, d_x, diagonal_values_inv_sqrt_d, m);
    gpuErrchk( hipPeekAtLastError() );
    gpuErrchk( hipDeviceSynchronize() );
    
    // scale matrix
    hipLaunchKernelGGL(jacobi_precondition_matrix, grid_size, block_size, 0, 0, A_data, A_col_indices, A_row_ptr, 
                                                          diagonal_values_inv_sqrt_d, m);
    gpuErrchk( hipPeekAtLastError() );
    gpuErrchk( hipDeviceSynchronize() );

    // scale starting guess
    hipLaunchKernelGGL(jacobi_unprecondition_array, grid_size, block_size, 0, 0, d_y, diagonal_values_inv_sqrt_d, m);
    gpuErrchk( hipPeekAtLastError() );
    gpuErrchk( hipDeviceSynchronize() );

    hipsparseSpMatDescr_t matA;
    status = hipsparseCreateCsr(&matA, m, m, A_nnz, A_row_ptr, A_col_indices, A_data, 
                               HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
    if (status != HIPSPARSE_STATUS_SUCCESS)
    {
        std::cout << "ERROR: creation of sparse matrix descriptor in solve_sparse_CG_Jacobi() failed!\n";
    }

    // *******************************
    // ** Iterative refinement loop **

    // initialize variables for the residual calculation
    double h_norm;
    double *d_r, *d_p, *d_temp;
    gpuErrchk( hipMalloc((void**)&d_r, m * sizeof(double)) ); 
    gpuErrchk( hipMalloc((void**)&d_p, m * sizeof(double)) ); 
    gpuErrchk( hipMalloc((void**)&d_temp, m * sizeof(double)) ); 

    // for SpMV:
    // - d_x is right hand side vector
    // - d_y is solution vector
    hipsparseDnVecDescr_t vecY, vecR, vecP, vectemp; 
    hipsparseCreateDnVec(&vecY, m, d_y, HIP_R_64F);
    hipsparseCreateDnVec(&vecR, m, d_r, HIP_R_64F);
    hipsparseCreateDnVec(&vecP, m, d_p, HIP_R_64F);
    hipsparseCreateDnVec(&vectemp, m, d_temp, HIP_R_64F);

    size_t MVBufferSize;
    void *MVBuffer = 0;
    status = hipsparseSpMV_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA, 
                          vecY, zero_d, vectemp, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, &MVBufferSize);  
    gpuErrchk( hipMalloc((void**)&MVBuffer, sizeof(double) * MVBufferSize) );

    // Initialize the residual and conjugate vectors
    // r = A*y - x & p = -r
    status = hipsparseSpMV(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA,                         
                          vecY, zero_d, vecR, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, MVBuffer);           // r = A*y
    CheckCublasError( hipblasDaxpy(handle_cublas, m, &n_one, d_x, 1, d_r, 1) );                            // r = -x + r
    CheckCublasError( hipblasDcopy(handle_cublas, m, d_r, 1, d_p, 1) );                                    // p = r
    CheckCublasError( hipblasDscal(handle_cublas, m, &n_one, d_p, 1) );                                    // p = -p

    // calculate the error (norm of the residual)
    CheckCublasError( hipblasDnrm2(handle_cublas, m, d_r, 1, &h_norm) );
    gpuErrchk( hipDeviceSynchronize() );
    
    // Conjugate Gradient steps
    int counter = 0;
    double t, tnew, alpha, beta, alpha_temp;
    while (h_norm > tol*tol){

        // alpha = rT * r / (pT * A * p)
        CheckCublasError( hipblasDdot (handle_cublas, m, d_r, 1, d_r, 1, &t) );                           // t = rT * r
        status = hipsparseSpMV(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA, 
                              vecP, zero_d, vectemp, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, MVBuffer);   // temp = A*p
        CheckCublasError( hipblasDdot (handle_cublas, m, d_p, 1, d_temp, 1, &alpha_temp) );               // alpha = pT*temp = pT*A*p
        alpha = t / alpha_temp; 

        // y = y + alpha * p
        CheckCublasError( hipblasDaxpy(handle_cublas, m, &alpha, d_p, 1, d_y, 1) );                       // y = y + alpha * p

        // r = r + alpha * A * p 
        CheckCublasError( hipblasDaxpy(handle_cublas, m, &alpha, d_temp, 1, d_r, 1) );                    // r = r + alpha * temp

        // beta = (rT * r) / t
        CheckCublasError( hipblasDdot (handle_cublas, m, d_r, 1, d_r, 1, &tnew) );                        // tnew = rT * r
        beta = tnew / t;

        // p = -r + beta * p
        CheckCublasError( hipblasDscal(handle_cublas, m, &beta, d_p, 1) );                                 // p = p * beta
        CheckCublasError( hipblasDaxpy(handle_cublas, m, &n_one, d_r, 1, d_p, 1) );                        // p = p - r

        // calculate the error (norm of the residual)
        CheckCublasError( hipblasDdot(handle_cublas, m, d_r, 1, d_r, 1, &h_norm) );
        // std::cout << h_norm << "\n";

        counter++;
        if (counter > 50000){
            std::cout << "WARNING: might be stuck in diverging CG iterations, check the residual!\n";
        }
    }
    std::cout << "# CG steps: " << counter << "\n";

    // unprecondition the solution vector
    hipLaunchKernelGGL(jacobi_precondition_array, grid_size, block_size, 0, 0, d_y, diagonal_values_inv_sqrt_d, m);
    gpuErrchk(hipPeekAtLastError());
    gpuErrchk(hipDeviceSynchronize());

    // // check solution vector
    // double *copy_back = (double *)calloc(m, sizeof(double));
    // gpuErrchk( hipMemcpy(copy_back, d_y, m * sizeof(double), hipMemcpyDeviceToHost) );
    // for (int i = 0; i < m; i++){
    //     std::cout << copy_back[i] << " ";
    // }
    // std::cout << "\nPrinted solution vector, now exiting\n";
    // exit(1);

    hipFree(diagonal_values_inv_sqrt_d);
    hipFree(MVBuffer); 
    hipFree(one_d);
    hipFree(n_one_d);
    hipFree(zero_d);
    hipFree(d_r);
    hipFree(d_p);
    hipFree(d_temp);
}

// Iterative sparse linear solver using CG steps
void solve_sparse_CG(hipblasHandle_t handle_cublas, hipsparseHandle_t handle, 
					 double* A_data, int* A_row_ptr, int* A_col_indices, const int A_nnz, 
                     int m, double *d_x, double *d_y){

    // A is an m x m sparse matrix represented by CSR format
    // - d_x is right hand side vector in gpu memory,
    // - d_y is solution vector in gpu memory.
    // - d_z is intermediate result on gpu memory.

    // Sets the initial guess for the solution vector to zero
    bool zero_guess = 0;

    // Error tolerance for the norm of the residual in the CG steps
    double tol = 1e-12;

    double one = 1.0;
    double n_one = -1.0;
    double zero = 0.0;
    double *one_d, *n_one_d, *zero_d;
    gpuErrchk( hipMalloc((void**)&one_d, sizeof(double)) );
    gpuErrchk( hipMalloc((void**)&n_one_d, sizeof(double)) );
    gpuErrchk( hipMalloc((void**)&zero_d, sizeof(double)) );
    gpuErrchk( hipMemcpy(one_d, &one, sizeof(double), hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(n_one_d, &n_one, sizeof(double), hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(zero_d, &zero, sizeof(double), hipMemcpyHostToDevice) );
    hipsparseStatus_t status;

    hipsparseSpMatDescr_t matA;
    status = hipsparseCreateCsr(&matA, m, m, A_nnz, A_row_ptr, A_col_indices, A_data, 
                               HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
    if (status != HIPSPARSE_STATUS_SUCCESS)
    {
        std::cout << "ERROR: creation of sparse matrix descriptor in solve_sparse_CG_Jacobi() failed!\n";
    }

    // ************************************
    // ** Set Initial Guess **

    if (zero_guess)
    {
        // Set the initial guess for the solution vector to zero
        gpuErrchk( hipMemset(d_y, 0, m * sizeof(double)) ); 
        gpuErrchk( hipDeviceSynchronize() );
    }

    // *******************************
    // ** Iterative refinement loop **

    // initialize variables for the residual calculation
    double h_norm;
    double *d_r, *d_p, *d_temp;
    gpuErrchk( hipMalloc((void**)&d_r, m * sizeof(double)) ); 
    gpuErrchk( hipMalloc((void**)&d_p, m * sizeof(double)) ); 
    gpuErrchk( hipMalloc((void**)&d_temp, m * sizeof(double)) ); 

    // for SpMV:
    // - d_x is right hand side vector
    // - d_y is solution vector
    hipsparseDnVecDescr_t vecY, vecR, vecP, vectemp; 
    hipsparseCreateDnVec(&vecY, m, d_y, HIP_R_64F);
    hipsparseCreateDnVec(&vecR, m, d_r, HIP_R_64F);
    hipsparseCreateDnVec(&vecP, m, d_p, HIP_R_64F);
    hipsparseCreateDnVec(&vectemp, m, d_temp, HIP_R_64F);

    size_t MVBufferSize;
    void *MVBuffer = 0;
    status = hipsparseSpMV_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA, 
                          vecY, zero_d, vectemp, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, &MVBufferSize);
    gpuErrchk( hipMalloc((void**)&MVBuffer, sizeof(double) * MVBufferSize) );
    
    // Initialize the residual and conjugate vectors
    // r = A*y - x & p = -r
    status = hipsparseSpMV(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA, 
                          vecY, zero_d, vecR, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, MVBuffer);         // r = A*y
    //gpuErrchk( hipDeviceSynchronize() );
    CheckCublasError( hipblasDaxpy(handle_cublas, m, &n_one, d_x, 1, d_r, 1) );                          // r = -x + r
    //gpuErrchk( hipDeviceSynchronize() );
    CheckCublasError(hipblasDcopy(handle_cublas, m, d_r, 1, d_p, 1));                                    // p = r
    //gpuErrchk( hipDeviceSynchronize() );
    CheckCublasError(hipblasDscal(handle_cublas, m, &n_one, d_p, 1));                                    // p = -p
    //gpuErrchk( hipDeviceSynchronize() );

    // calculate the error (norm of the residual)
    CheckCublasError( hipblasDnrm2(handle_cublas, m, d_r, 1, &h_norm) );
    gpuErrchk( hipDeviceSynchronize() );
    
    // Conjugate Gradient steps
    int counter = 0;
    double t, tnew, alpha, beta, alpha_temp;
    while (h_norm > tol){

        // alpha = rT * r / (pT * A * p)
        CheckCublasError( hipblasDdot (handle_cublas, m, d_r, 1, d_r, 1, &t) );                         // t = rT * r
        //gpuErrchk( hipDeviceSynchronize() );
        status = hipsparseSpMV(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA, 
                              vecP, zero_d, vectemp, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, MVBuffer); // temp = A*p
        //gpuErrchk( hipDeviceSynchronize() );
        CheckCublasError( hipblasDdot (handle_cublas, m, d_p, 1, d_temp, 1, &alpha_temp) );             // alpha = pT*temp = pT*A*p
        //gpuErrchk( hipDeviceSynchronize() );
        alpha = t / alpha_temp; 

        // y = y + alpha * p
        CheckCublasError(hipblasDaxpy(handle_cublas, m, &alpha, d_p, 1, d_y, 1));                       // y = y + alpha * p
        //gpuErrchk( hipDeviceSynchronize() );

        // r = r + alpha * A * p 
        CheckCublasError(hipblasDaxpy(handle_cublas, m, &alpha, d_temp, 1, d_r, 1));                    // r = r + alpha * temp
        //gpuErrchk( hipDeviceSynchronize() );

        // beta = (rT * r) / t
        CheckCublasError( hipblasDdot (handle_cublas, m, d_r, 1, d_r, 1, &tnew) );                       // tnew = rT * r
        //gpuErrchk( hipDeviceSynchronize() );
        beta = tnew / t;

        // p = -r + beta * p
        CheckCublasError(hipblasDscal(handle_cublas, m, &beta, d_p, 1));                                  // p = p * beta
        //gpuErrchk( hipDeviceSynchronize() );
        CheckCublasError(hipblasDaxpy(handle_cublas, m, &n_one, d_r, 1, d_p, 1));                         // p = p - r
        //gpuErrchk( hipDeviceSynchronize() );

        // calculate the error (norm of the residual)
        CheckCublasError( hipblasDnrm2(handle_cublas, m, d_r, 1, &h_norm) );
        //gpuErrchk( hipDeviceSynchronize() );
        // std::cout << h_norm << "\n";

        counter++;
        if (counter > 50000){
            std::cout << "WARNING: might be stuck in diverging CG iterations, check the residual!\n";
        }
    }
    std::cout << "# CG steps: " << counter << "\n";

    hipFree(MVBuffer); 
    hipFree(one_d);
    hipFree(n_one_d);
    hipFree(zero_d);
    hipFree(d_r);
    hipFree(d_p);
    hipFree(d_temp);

    // // check solution vector
    // double *copy_back = (double *)calloc(m, sizeof(double));
    // gpuErrchk( hipMemcpy(copy_back, d_y, m * sizeof(double), hipMemcpyDeviceToHost) );
    // for (int i = 0; i < m; i++){
    //     std::cout << copy_back[i] << " ";
    // }
    // exit(1);
    
}

// helper function for the splitmatrix CG implementation, multiplies the compressed submatrix with the solution vector
__global__ void add_submatrix_product(double *M, double *y, double *r, int msub, int *insertion_indices) {

    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;

    // Each thread gets a row of the matrix
    for (int idx = tid_total; idx < msub; idx += num_threads_total)
    {   
        // it needs to multiply it's row by staggered elements of y
        double row_sum = 0.0;
        for ( int j = 0; j < msub; j++ )
        {
            row_sum += M[idx * msub + j] * y[insertion_indices[j] + 2];
        }

        // add this row_sum to index insertion_indices[idx] + 2 of the output vector
        r[insertion_indices[idx] + 2] += row_sum;
    }
}

//remove this as it exists already
__global__ void elementwise_vector_vector_tmp(
    double * __restrict__ array1,
    double * __restrict__ array2,
    double * __restrict__ result,
    int size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = idx; i < size; i += blockDim.x * gridDim.x){
        result[i] = array1[i] * array2[i];
    }

}

// Iterative sparse linear solver using CG steps on matrix represented in mixed sparse/dense format 
// the insertion indices specify the correspondence between the (dense) submatrix rows/cols and the (sparse) full matrix rows/cols
void solve_sparse_CG_splitmatrix(hipblasHandle_t handle_cublas, hipsparseHandle_t handle, 
                                 double* M, int msub, double* A_data, int* A_row_ptr, int* A_col_indices, const int A_nnz, 
                                 int m, int *insertion_indices, double *d_x, double *d_y,
                                 double *diagonal_inv_d){
    // A is an m x m sparse matrix in CSR format
    // M is an msub x msub dense matrix
    // the full system matrix is 

    // - d_x is right hand side vector in gpu memory,
    // - d_y is solution vector in gpu memory.
    // - d_z is intermediate result on gpu memory.

    // Sets the initial guess for the solution vector to zero
    bool zero_guess = 0;

    // Error tolerance for the norm of the residual in the CG steps
    double tol = 1e-5;//1e-12;

    double one = 1.0;
    double n_one = -1.0;
    double zero = 0.0;
    double *one_d, *n_one_d, *zero_d;
    gpuErrchk( hipMalloc((void**)&one_d, sizeof(double)) );
    gpuErrchk( hipMalloc((void**)&n_one_d, sizeof(double)) );
    gpuErrchk( hipMalloc((void**)&zero_d, sizeof(double)) );
    gpuErrchk( hipMemcpy(one_d, &one, sizeof(double), hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(n_one_d, &n_one, sizeof(double), hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(zero_d, &zero, sizeof(double), hipMemcpyHostToDevice) );
    hipsparseStatus_t status;

    hipsparseSpMatDescr_t matA;
    status = hipsparseCreateCsr(&matA, m, m, A_nnz, A_row_ptr, A_col_indices, A_data, 
                               HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
    if (status != HIPSPARSE_STATUS_SUCCESS)
    {
        std::cout << "ERROR: creation of sparse matrix descriptor in solve_sparse_CG_splitmatrix() failed!\n";
    }

    // ************************************
    // ** Set Initial Guess **

    if (zero_guess)
    {
        // Set the initial guess for the solution vector to zero
        gpuErrchk( hipMemset(d_y, 0, m * sizeof(double)) ); 
        gpuErrchk( hipDeviceSynchronize() );
    }

    // *******************************
    // ** Iterative refinement loop **

    // initialize variables for the residual calculation
    double h_norm;
    double *d_r, *d_p, *d_temp, *d_z;
    gpuErrchk( hipMalloc((void**)&d_r, m * sizeof(double)) ); 
    gpuErrchk( hipMalloc((void**)&d_p, m * sizeof(double)) );
    gpuErrchk( hipMalloc((void**)&d_temp, m * sizeof(double)) ); 
    gpuErrchk( hipMalloc((void**)&d_z, m * sizeof(double)) );
    gpuErrchk(hipMemcpy(d_r, d_x, m * sizeof(double), hipMemcpyDeviceToDevice));

    // gpuErrchk(cuda)

    // for SpMV:
    // - d_x is right hand side vector
    // - d_y is solution vector
    hipsparseDnVecDescr_t vecY, vecR, vecP, vectemp; 
    hipsparseCreateDnVec(&vecY, m, d_y, HIP_R_64F);
    hipsparseCreateDnVec(&vecR, m, d_r, HIP_R_64F);
    hipsparseCreateDnVec(&vecP, m, d_p, HIP_R_64F);
    hipsparseCreateDnVec(&vectemp, m, d_temp, HIP_R_64F);

    size_t MVBufferSize;
    void *MVBuffer = 0;
    status = hipsparseSpMV_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA, 
                          vecY, zero_d, vectemp, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, &MVBufferSize);
    gpuErrchk( hipMalloc((void**)&MVBuffer, sizeof(double) * MVBufferSize) );
    
    // Initialize the residual and conjugate vectors
    // r = A*y - x & p = -r

    // r = (A + M)*y = A*y + M*y
    status = hipsparseSpMV(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA, 
                          vecY, zero_d, vectemp, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, MVBuffer);         // r = A*y

    // do r += M*y in a single CUDA kernel by multiplying M by the sub-vector of y 
    // specified by in the indices in 'insertion indices', and adding the result to r. 
    int threads = 1024;
    int blocks = (m + threads - 1) / threads;
    hipLaunchKernelGGL(add_submatrix_product, blocks, threads, 0, 0, M, d_y, d_temp, msub, insertion_indices);                   // r += M*y

    //gpuErrchk( hipDeviceSynchronize() );
    //CheckCublasError( hipblasDaxpy(handle_cublas, m, &n_one, d_x, 1, d_r, 1) );                          // r = -x + r
    
    // r = b - Ax0
    CheckCublasError(hipblasDaxpy(handle_cublas, m, &n_one, d_temp, 1, d_r, 1));

    // Mz0 = r0
    hipLaunchKernelGGL(elementwise_vector_vector_tmp, blocks, threads, 0, 0, 
        d_r,
        diagonal_inv_d,
        d_z,
        m
    ); 


    double r1, r0, b, a, na, dot;

    // calculate the error (norm of the residual)
    CheckCublasError(hipblasDdot(handle_cublas, m, d_r, 1, d_z, 1, &r1));

    int k = 1;
    while (r1 > tol * tol && k <= 1000) {
        
        if(k > 1){
            b = r1 / r0;
            CheckCublasError(hipblasDscal(handle_cublas, m, &b, d_p, 1));

            CheckCublasError(hipblasDaxpy(handle_cublas, m, &one, d_z, 1, d_p, 1));   
        }
        else {
            CheckCublasError(hipblasDcopy(handle_cublas, m, d_z, 1, d_p, 1));
        }
        // temp = (A + M)*p = A*p + M*p
        status = hipsparseSpMV(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, one_d, matA, 
                              vecP, zero_d, vectemp, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, MVBuffer); // temp = A*p
        hipLaunchKernelGGL(add_submatrix_product, blocks, threads, 0, 0, M, d_p, d_temp, msub, insertion_indices);           // temp += M*p
        CheckCublasError(hipblasDdot(handle_cublas, m, d_p, 1, d_temp, 1, &dot));
        a = r1 / dot;
        CheckCublasError(hipblasDaxpy(handle_cublas, m, &a, d_p, 1, d_y, 1));
        na = -a;
        CheckCublasError(hipblasDaxpy(handle_cublas, m, &na, d_temp, 1, d_r, 1));
        // Mz = r
        hipLaunchKernelGGL(elementwise_vector_vector_tmp, blocks, threads, 0, 0, 
            d_r,
            diagonal_inv_d,
            d_z,
            m
        ); 
        r0 = r1;
        CheckCublasError(hipblasDdot(handle_cublas, m, d_r, 1, d_z, 1, &r1));
        // gpuErrchk(hipStreamSynchronize(stream));
        k++;
    }

    double *h_y = (double *)calloc(m, sizeof(double));
    gpuErrchk( hipMemcpy(h_y, d_y, m * sizeof(double), hipMemcpyDeviceToHost) );
    double sum = 0.0;
    for (int i = 0; i < m; i++){
        sum += 1e30*h_y[i]*h_y[i];
    }
    std::cout << "sum of solution vector: " << sum << "\n";

    std::cout << "# CG split steps: " << k << "\n";
    std::cout << "solve_sparse_CG_splitmatrix residual: " << r1 << "\n";

    hipFree(MVBuffer); 
    hipFree(one_d);
    hipFree(n_one_d);
    hipFree(zero_d);
    hipFree(d_r);
    hipFree(d_p);
    hipFree(d_temp);
    hipFree(d_z);

    // // check solution vector
    // double *copy_back = (double *)calloc(m, sizeof(double));
    // gpuErrchk( hipMemcpy(copy_back, d_y, m * sizeof(double), hipMemcpyDeviceToHost) );
    // for (int i = 0; i < m; i++){
    //     std::cout << copy_back[i] << " ";
    // }
    // std::cout << "exiting after printing the solution vector\n";
    // exit(1);
    
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
        
            double dist = site_dist_gpu(posx_d[i], posy_d[i], posz_d[i],
                                        posx_d[j], posy_d[j], posz_d[j],
                                        lattice_d[0], lattice_d[1], lattice_d[2], pbc);
            if(dist < cutoff_radius){
                col_indices_d[row_ptr_d[i] + nnz_row] = j;
                nnz_row++;
            }
        }
    }
}

// Compute the number of nonzeros per row of the matrix including the injection, extraction, and device nodes (excluding the ground). 
// Has dimensions of Nsub by Nsub (by the cpu code)
__global__ void calc_nnz_per_row_X_gpu( const double *posx_d, const double *posy_d, const double *posz_d,
                                        const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                                        const double *lattice, bool pbc, double nn_dist, const double tol,
                                        int num_source_inj, int num_ground_ext, const int num_layers_contact,
                                        int num_metals, int matrix_size, int *nnz_per_row_d){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Natom = matrix_size - 2; 
    
    // TODO optimize this with a 2D grid instead of 1D
    for(int i = idx; i < Natom - 1; i += blockDim.x * gridDim.x){  // N_atom - 1 to exclude the ground node

        int nnz_row = 0;

        for(int j = 0; j < Natom - 1; j++){ // N_atom - 1 to exclude the ground node

            double dist = site_dist_gpu(posx_d[i], posy_d[i], posz_d[i],
                                        posx_d[j], posy_d[j], posz_d[j],
                                        lattice[0], lattice[1], lattice[2], pbc);
            
            // diagonal terms
            if ( i == j )
            {
                nnz_row++;
            }

            // direct terms 
            else if ( i != j && dist < nn_dist )
            {
                nnz_row++;
            }

            // tunneling terms 
            else
            { 
                bool any_vacancy1 = element[i] == VACANCY;
                bool any_vacancy2 = element[j] == VACANCY;

                // contacts, excluding the last layer 
                bool metal1p = is_in_array_gpu(metals, element[i], num_metals) 
                                                && (i > ((num_layers_contact - 1)*num_source_inj))
                                                && (i < (Natom - (num_layers_contact - 1)*num_ground_ext)); 

                bool metal2p = is_in_array_gpu(metals, element[j], num_metals)
                                                && (j > ((num_layers_contact - 1)*num_source_inj))
                                                && (j < (Natom - (num_layers_contact - 1)*num_ground_ext));  

                // types of tunnelling conditions considered
                bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                bool contact_to_contact = (metal1p && metal2p);
                double local_E_drop = atom_CB_edge[i] - atom_CB_edge[j];                

                if ((trap_to_trap || contact_to_trap || contact_to_contact)  && (fabs(local_E_drop) > tol))
                {
                    nnz_row++;
                }
            }
        }

        nnz_per_row_d[i+2] = nnz_row;

        // source/ground connections
        if ( i < num_source_inj )
        {
            atomicAdd(&nnz_per_row_d[1], 1);
            nnz_per_row_d[i+2]++;
        }
        if ( i > (Natom - num_ground_ext) )
        {
            atomicAdd(&nnz_per_row_d[0], 1);
            nnz_per_row_d[i+2]++;
        }
        if ( i == 0 )
        {
            atomicAdd(&nnz_per_row_d[0], 2); // loop connection and diagonal element
            atomicAdd(&nnz_per_row_d[1], 2); // loop connection and diagonal element
        }

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
            double dist = site_dist_gpu(posx_d[i], posy_d[i], posz_d[i],
                                        posx_d[j], posy_d[j], posz_d[j],
                                        lattice_d[0], lattice_d[1], lattice_d[2], pbc);
            if(dist < cutoff_radius){
                nnz_row++;
            }
        }
        nnz_per_row_d[i] = nnz_row;
    }

}

void indices_creation_gpu(
    const double *posx_d, const double *posy_d, const double *posz_d,
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
    gpuErrchk( hipMalloc((void **)row_ptr_d, (matrix_size + 1) * sizeof(int)) );
    gpuErrchk( hipMalloc((void **)&nnz_per_row_d, matrix_size * sizeof(int)) );
    gpuErrchk(hipMemset((*row_ptr_d), 0, (matrix_size + 1) * sizeof(int)) );

    // calculate the nnz per row
    hipLaunchKernelGGL(calc_nnz_per_row_gpu, blocks, threads, 0, 0, posx_d, posy_d, posz_d, lattice_d, pbc, cutoff_radius, matrix_size, nnz_per_row_d);

    void     *temp_storage_d = NULL;
    size_t   temp_storage_bytes = 0;
    // determines temporary device storage requirements for inclusive prefix sum
    hipcub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*row_ptr_d)+1, matrix_size);

    // Allocate temporary storage for inclusive prefix sum
    gpuErrchk(hipMalloc(&temp_storage_d, temp_storage_bytes));
    // Run inclusive prefix sum
    // inclusive sum starting at second value to get the row ptr
    // which is the same as inclusive sum starting at first value and last value filled with nnz
    hipcub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*row_ptr_d)+1, matrix_size);
    
    // nnz is the same as (*row_ptr_d)[matrix_size]
    gpuErrchk( hipMemcpy(nnz, (*row_ptr_d) + matrix_size, sizeof(int), hipMemcpyDeviceToHost) );
    gpuErrchk( hipMalloc((void **)col_indices_d, nnz[0] * sizeof(int)) );

    // assemble the indices of K
    hipLaunchKernelGGL(assemble_K_indices_gpu, blocks, threads, 0, 0, 
        posx_d, posy_d, posz_d,
        lattice_d, pbc,
        cutoff_radius,
        matrix_size,
        nnz_per_row_d,
        (*row_ptr_d),
        (*col_indices_d)
    );

    hipFree(temp_storage_d);
    hipFree(nnz_per_row_d);
}

__global__ void assemble_X_indices_gpu(const double *posx_d, const double *posy_d, const double *posz_d,
                                        const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                                        const double *lattice, bool pbc, double nn_dist, const double tol,
                                        int num_source_inj, int num_ground_ext, const int num_layers_contact,
                                        int num_metals, int matrix_size, int *nnz_per_row_d, int *row_ptr_d, int *col_indices_d)
{
    // row ptr is already calculated

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Natom = matrix_size - 2;
    int N_full = matrix_size;
    
    // TODO can be optimized with a 2D grid instead of 1D
    // INDEXED OVER NFULL
    for(int i = idx; i < N_full - 1; i += blockDim.x * gridDim.x){                      // exclude ground node with Nfull - 1

        int nnz_row = 0;

        // loop connection and injection row
        if ( i == 0 )
        {
            for (int j = 0; j < N_full - 1; j++)                                        // exclude ground node with Nfull - 1
            {
                if ( (j < 2) || j > (N_full - num_ground_ext) )
                {
                    col_indices_d[row_ptr_d[i] + nnz_row] = j;
                    nnz_row++;
                }
            }
        }
        // loop connection and extraction row
        if ( i == 1 )
        {
            for (int j = 0; j < num_source_inj + 2; j++)
            {
                col_indices_d[row_ptr_d[i] + nnz_row] = j;
                nnz_row++;
            }
        }

        // inner matrix terms
        if (i >= 2)
        {
            for(int j = 0; j < N_full - 1; j++){                                        // exclude ground node with Nfull - 1

                // add injection term for this row
                if ( (j == 1) && (i < num_source_inj + 2) )
                {
                    col_indices_d[row_ptr_d[i] + nnz_row] = 1;
                    nnz_row++;
                }

                // add extraction term for this row
                if ( (j == 0) && (i > N_full - num_ground_ext) )
                {
                    col_indices_d[row_ptr_d[i] + nnz_row] = 0;
                    nnz_row++;
                }

                if ( j >= 2 ) 
                {
                    double dist = site_dist_gpu(posx_d[i - 2], posy_d[i - 2], posz_d[i - 2],
                                                posx_d[j - 2], posy_d[j - 2], posz_d[j - 2],
                                                lattice[0], lattice[1], lattice[2], pbc);
                    
                    // diagonal terms
                    if ( i == j )
                    {
                        col_indices_d[row_ptr_d[i] + nnz_row] = j;
                        nnz_row++;
                    }

                    // direct terms 
                    else if ( i != j && dist < nn_dist )
                    {
                        col_indices_d[row_ptr_d[i] + nnz_row] = j;
                        nnz_row++;
                    }

                    // tunneling terms 
                    else
                    { 
                        bool any_vacancy1 = element[i - 2] == VACANCY;
                        bool any_vacancy2 = element[j - 2] == VACANCY;

                        // contacts, excluding the last layer 
                        bool metal1p = is_in_array_gpu(metals, element[i - 2], num_metals) 
                                                    && ((i - 2) > ((num_layers_contact - 1)*num_source_inj))
                                                    && ((i - 2) < (Natom - (num_layers_contact - 1)*num_ground_ext)); 

                        bool metal2p = is_in_array_gpu(metals, element[j - 2], num_metals)
                                                    && ((j - 2) > ((num_layers_contact - 1)*num_source_inj))
                                                    && ((j - 2) < (Natom - (num_layers_contact - 1)*num_ground_ext));  

                        // types of tunnelling conditions considered
                        bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                        bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                        bool contact_to_contact = (metal1p && metal2p);
                        double local_E_drop = atom_CB_edge[i - 2] - atom_CB_edge[j - 2];                

                        if ((trap_to_trap || contact_to_trap || contact_to_contact)  && (fabs(local_E_drop) > tol))
                        {
                            col_indices_d[row_ptr_d[i] + nnz_row] = j;
                            nnz_row++;
                        }
                    }
                }
            }
        }

    }
}

// // assemble the data for the X matrix - 2D distribution over nonzeros
// __global__ void populate_sparse_X_gpu(const double *posx_d, const double *posy_d, const double *posz_d,
//                                       const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
//                                       const double *lattice, bool pbc, double nn_dist, const double tol,
//                                       const double high_G, const double low_G, const double loop_G, 
//                                       const double Vd, const double m_e, const double V0,
//                                       int num_source_inj, int num_ground_ext, const int num_layers_contact,
//                                       int num_metals, int matrix_size, int *row_ptr_d, int *col_indices_d, double *data_d)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int total_tid = blockIdx.x * blockDim.x + threadIdx.x;
//     int total_threads = blockDim.x * gridDim.x;

//     // thread idx works on this element
//     for (int idx = total_tid; idx < row_ptr_d[matrix_size-1]; idx += total_threads) 
//     {
//         // find the row_idx and col_idx for thread idx
//         int row_idx = 0; 
//         while (row_idx < matrix_size && idx >= row_ptr_d[row_idx + 1]){
//             row_idx++;
//         }
//         int col_idx = col_indices_d[idx];

//         int N_full = matrix_size;
//         int N_atom = matrix_size - 2;

//         // if dealing with a diagonal element, we add the positive value from i = i and j = N_full to include the ground node

//         // extraction boundary (row)
//         if (row_idx == 0) {
//             // diagonal element (0, 0) --> add the value from (0, N_full)
//             if (col_idx == 0)
//             {
//                 data_d[idx] = +high_G;
//             }
//             // loop connection (0, 1)
//             if (col_idx == 1)
//             {
//                 data_d[idx] = -loop_G;
//             }
//             // extraction connections from the device
//             if ( col_idx > N_full - num_ground_ext )
//             {
//                 data_d[idx] = -high_G;
//             } 
//         }

//         // injection boundary (row)
//         if (row_idx == 1) {
//             // loop connection (1, 0)
//             if (col_idx == 0)
//             {
//                 data_d[idx] = -loop_G;
//             }
//             // injection connections to the device
//             if ( col_idx >= 2 || (col_idx > N_full - num_ground_ext) )
//             {
//                 data_d[idx] = -high_G;
//             } 
//         }

//         // inner matrix terms
//         if (row_idx >= 2) 
//         {
//             // diagonal elements --> add the value from (i - 2, N_full - 2) if site i - 2 neighbors the ground node
//             if (row_idx == col_idx)
//             {
//                 double dist_angstrom = site_dist_gpu_2(posx_d[row_idx - 2], posy_d[row_idx - 2], posz_d[row_idx - 2],
//                                                        posx_d[N_atom-1], posy_d[N_atom-1], posz_d[N_atom-1], 
//                                                        lattice[0], lattice[1], lattice[2], pbc);                                   
//                 bool neighboring_ground = (dist_angstrom < nn_dist);
                    
//                 if (neighboring_ground) 
//                 {
//                     data_d[idx] = +high_G;     // assuming all the connections to ground come from the right contact
//                 } 
//             }

//             // extraction boundary (column)
//             if ( (col_idx == 0) && (row_idx > N_full - num_ground_ext) )
//             {
//                 data_d[idx] = -high_G;
//             }

//             // injection boundary (column)
//             if ( (col_idx == 1) && (row_idx < num_source_inj + 2) )
//             {
//                 data_d[idx] = -high_G;
//             }

//             // off-diagonal inner matrix elements
//             if ( (col_idx >= 2) && (col_idx != row_idx)) 
//             {

//                 double dist_angstrom = site_dist_gpu_2(posx_d[row_idx - 2], posy_d[row_idx - 2], posz_d[row_idx - 2],
//                                                        posx_d[col_idx - 2], posy_d[col_idx - 2], posz_d[col_idx - 2], 
//                                                        lattice[0], lattice[1], lattice[2], pbc);                                       
                        
//                 bool neighbor = (dist_angstrom < nn_dist);                                                      

//                 // non-neighbor connections
//                 if (!neighbor)
//                 {
//                     bool any_vacancy1 = element[row_idx - 2] == VACANCY;
//                     bool any_vacancy2 = element[col_idx - 2] == VACANCY;

//                     // contacts, excluding the last layer 
//                     bool metal1p = is_in_array_gpu(metals, element[row_idx - 2], num_metals) 
//                                                 && ((row_idx - 2) > ((num_layers_contact - 1)*num_source_inj))
//                                                 && ((row_idx - 2) < (N_full - (num_layers_contact - 1)*num_ground_ext)); 

//                     bool metal2p = is_in_array_gpu(metals, element[col_idx - 2], num_metals)
//                                                 && ((col_idx - 2) > ((num_layers_contact - 1)*num_source_inj))
//                                                 && ((col_idx - 2) < (N_full - (num_layers_contact - 1)*num_ground_ext));  

//                     // types of tunnelling conditions considered
//                     bool trap_to_trap = (any_vacancy1 && any_vacancy2);
//                     bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
//                     bool contact_to_contact = (metal1p && metal2p);

//                     double local_E_drop = atom_CB_edge[row_idx - 2] - atom_CB_edge[col_idx - 2];                // [eV] difference in energy between the two atoms

//                     // compute the WKB tunneling coefficients for all the tunnelling conditions
//                     if ((trap_to_trap || contact_to_trap || contact_to_contact)  && (fabs(local_E_drop) > tol))
//                     {
                                
//                         double prefac = -(sqrt( 2 * m_e ) / h_bar) * (2.0 / 3.0);           // [s/(kg^1/2 * m^2)] coefficient inside the exponential
//                         double dist = (1e-10)*dist_angstrom;                                // [m] 3D distance between atoms i and j

//                         if (contact_to_trap)
//                         {
//                             double energy_window = fabs(local_E_drop);                      // [eV] energy window for tunneling from the contacts
//                             double dV = 0.01;                                               // [V] energy spacing for numerical integration
//                             double dE = eV_to_J * dV;                                       // [eV] energy spacing for numerical integration
                                        
//                             // integrate over all the occupied energy levels in the contact
//                             double T = 0.0;
//                             for (double iv = 0; iv < energy_window; iv += dE)
//                             {
//                                 double E1 = eV_to_J * V0 + iv;                                  // [J] Energy distance to CB before tunnelling
//                                 double E2 = E1 - fabs(local_E_drop);                            // [J] Energy distance to CB after tunnelling

//                                 if (E2 > 0)                                                     // trapezoidal potential barrier (low field)                 
//                                 {                                                           
//                                     T += exp(prefac * (dist / fabs(local_E_drop)) * ( pow(E1, 1.5) - pow(E2, 1.5) ) );
//                                 }

//                                 if (E2 < 0)                                                      // triangular potential barrier (high field)                               
//                                 {
//                                     T += exp(prefac * (dist / fabs(local_E_drop)) * ( pow(E1, 1.5) )); 
//                                 } 
//                             }
//                             data_d[idx] = -T;
//                         } 
//                         else 
//                         {
//                             double E1 = eV_to_J * V0;                                        // [J] Energy distance to CB before tunnelling
//                             double E2 = E1 - fabs(local_E_drop);                             // [J] Energy distance to CB after tunnelling
                                        
//                             if (E2 > 0)                                                      // trapezoidal potential barrier (low field)
//                             {                                                           
//                                 double T = exp(prefac * (dist / fabs(E1 - E2)) * ( pow(E1, 1.5) - pow(E2, 1.5) ) );
//                                 data_d[idx] = -T;
//                             }

//                             if (E2 < 0)                                                        // triangular potential barrier (high field)
//                             {
//                                 double T = exp(prefac * (dist / fabs(E1 - E2)) * ( pow(E1, 1.5) ));
//                                 data_d[idx] = -T;
//                             }
//                         }
//                     }
//                 }

//                 // direct terms
//                 if ( neighbor )
//                 {
//                     // contacts
//                     bool metal1 = is_in_array_gpu<ELEMENT>(metals, element[row_idx - 2], num_metals);
//                     bool metal2 = is_in_array_gpu<ELEMENT>(metals, element[col_idx - 2], num_metals);

//                     // conductive vacancy sites
//                     bool cvacancy1 = (element[row_idx - 2] == VACANCY) && (atom_charge[row_idx - 2] == 0);
//                     bool cvacancy2 = (element[col_idx - 2] == VACANCY) && (atom_charge[col_idx - 2] == 0);
                        
//                     if ((metal1 && metal2) || (cvacancy1 && cvacancy2))
//                     {
//                         data_d[idx] = -high_G;
//                     }
//                     else
//                     {
//                         data_d[idx] = -low_G;
//                     }
//                 }
//             } // off-diagonal inner matrix elements
//         } // inner matrix elements

//     } // tid loop
// }

// assemble the data for the X matrix - 1D distribution over rows
__global__ void populate_sparse_X_gpu(const double *posx_d, const double *posy_d, const double *posz_d,
                                        const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                                        const double *lattice, bool pbc, double nn_dist, const double tol,
                                        const double high_G, const double low_G, const double loop_G, 
                                        const double Vd, const double m_e, const double V0,
                                        int num_source_inj, int num_ground_ext, const int num_layers_contact,
                                        int num_metals, int matrix_size, int *row_ptr_d, int *col_indices_d, double *data_d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N_full = matrix_size;
    int N_atom = matrix_size - 2;
    
    for(int i = idx; i < N_full - 1; i += blockDim.x * gridDim.x){

        for( int j = row_ptr_d[i]; j < row_ptr_d[i+1]; j++ )
        {
            // col_indices_d[j] is the index of j in the matrix. j is the index of the data vector
            // if dealing with a diagonal element, we add the positive value from i = i and j = N_full to include the ground node

            // extraction boundary (row)
            if(i == 0)
            {
                // diagonal element (0, 0) --> add the value from (0, N_full)
                if (col_indices_d[j] == 0)
                {
                    data_d[j] = +high_G;
                }
                // loop connection (0, 1)
                if (col_indices_d[j] == 1)
                {
                    data_d[j] = -loop_G;
                }
                // extraction connections from the device
                if ( col_indices_d[j] > N_full - num_ground_ext )
                {
                    data_d[j] = -high_G;
                } 
            }

            // injection boundary (row)
            if(i == 1)
            {
                // loop connection (1, 0)
                if (col_indices_d[j] == 0)
                {
                    data_d[j] = -loop_G;
                }
                // injection connections to the device
                if ( col_indices_d[j] >= 2 || (col_indices_d[j] > N_full - num_ground_ext) )
                {
                    data_d[j] = -high_G;
                } 
            }

            // inner matrix terms
            if (i >= 2)
            {
                // diagonal elements --> add the value from (i - 2, N_full - 2) if site i - 2 neighbors the ground node
                if (i == col_indices_d[j])
                {
                    double dist_angstrom = site_dist_gpu(posx_d[i - 2], posy_d[i - 2], posz_d[i - 2],
                                                           posx_d[N_atom-1], posy_d[N_atom-1], posz_d[N_atom-1], 
                                                           lattice[0], lattice[1], lattice[2], pbc);                                   
                    bool neighboring_ground = (dist_angstrom < nn_dist);
                    
                    if (neighboring_ground) 
                    {
                        data_d[j] = +high_G;     // assuming all the connections to ground come from the right contact
                    } 
                }

                // extraction boundary (column)
                if ( (col_indices_d[j] == 0) && (i > N_full - num_ground_ext) )
                {
                    data_d[j] = -high_G;
                }

                // injection boundary (column)
                if ( (col_indices_d[j] == 1) && (i < num_source_inj + 2) )
                {
                    data_d[j] = -high_G;
                }

                // off-diagonal inner matrix elements
                if ( (col_indices_d[j] >= 2) && (col_indices_d[j] != i)) 
                {

                    double dist_angstrom = site_dist_gpu(posx_d[i - 2], posy_d[i - 2], posz_d[i - 2],
                                                           posx_d[col_indices_d[j] - 2], posy_d[col_indices_d[j] - 2], posz_d[col_indices_d[j] - 2], 
                                                           lattice[0], lattice[1], lattice[2], pbc);                                       
                        
                    bool neighbor = (dist_angstrom < nn_dist);                                                      

                    // non-neighbor connections
                    if (!neighbor)
                    {
                        bool any_vacancy1 = element[i - 2] == VACANCY;
                        bool any_vacancy2 = element[col_indices_d[j] - 2] == VACANCY;

                        // contacts, excluding the last layer 
                        bool metal1p = is_in_array_gpu(metals, element[i - 2], num_metals) 
                                                    && ((i - 2) > ((num_layers_contact - 1)*num_source_inj))
                                                    && ((i - 2) < (N_full - (num_layers_contact - 1)*num_ground_ext)); 

                        bool metal2p = is_in_array_gpu(metals, element[col_indices_d[j] - 2], num_metals)
                                                    && ((col_indices_d[j] - 2) > ((num_layers_contact - 1)*num_source_inj))
                                                    && ((col_indices_d[j] - 2) < (N_full - (num_layers_contact - 1)*num_ground_ext));  

                        // types of tunnelling conditions considered
                        bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                        bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                        bool contact_to_contact = (metal1p && metal2p);

                        double local_E_drop = atom_CB_edge[i - 2] - atom_CB_edge[col_indices_d[j] - 2];                // [eV] difference in energy between the two atoms

                        // compute the WKB tunneling coefficients for all the tunnelling conditions
                        if ((trap_to_trap || contact_to_trap || contact_to_contact)  && (fabs(local_E_drop) > tol))
                        {
                            double prefac = -(sqrt( 2 * m_e ) / h_bar) * (2.0 / 3.0);           // [s/(kg^1/2 * m^2)] coefficient inside the exponential
                            double dist = (1e-10)*dist_angstrom;                                // [m] 3D distance between atoms i and j

                            if (contact_to_trap)
                            {
                                double energy_window = fabs(local_E_drop);                      // [eV] energy window for tunneling from the contacts
                                double dV = 0.01;                                               // [V] energy spacing for numerical integration
                                double dE = eV_to_J * dV;                                       // [eV] energy spacing for numerical integration
                                        
                                // integrate over all the occupied energy levels in the contact
                                double T = 0.0;
                                for (double iv = 0; iv < energy_window; iv += dE)
                                {
                                    double E1 = eV_to_J * V0 + iv;                                  // [J] Energy distance to CB before tunnelling
                                    double E2 = E1 - fabs(local_E_drop);                            // [J] Energy distance to CB after tunnelling

                                    if (E2 > 0)                                                     // trapezoidal potential barrier (low field)                 
                                    {                                                           
                                        T += exp(prefac * (dist / fabs(local_E_drop)) * ( pow(E1, 1.5) - pow(E2, 1.5) ) );
                                    }

                                    if (E2 < 0)                                                      // triangular potential barrier (high field)                               
                                    {
                                        T += exp(prefac * (dist / fabs(local_E_drop)) * ( pow(E1, 1.5) )); 
                                    } 
                                }
                                data_d[j] = -T;
                            } 
                            else 
                            {
                                double E1 = eV_to_J * V0;                                        // [J] Energy distance to CB before tunnelling
                                double E2 = E1 - fabs(local_E_drop);                             // [J] Energy distance to CB after tunnelling
                                        
                                if (E2 > 0)                                                      // trapezoidal potential barrier (low field)
                                {                                                           
                                    double T = exp(prefac * (dist / fabs(E1 - E2)) * ( pow(E1, 1.5) - pow(E2, 1.5) ) );
                                    data_d[j] = -T;
                                }

                                if (E2 < 0)                                                        // triangular potential barrier (high field)
                                {
                                    double T = exp(prefac * (dist / fabs(E1 - E2)) * ( pow(E1, 1.5) ));
                                    data_d[j] = -T;
                                }
                            }
                        }
                    }

                    // direct terms
                    if ( neighbor )
                    {
                        // contacts
                        bool metal1 = is_in_array_gpu<ELEMENT>(metals, element[i - 2], num_metals);
                        bool metal2 = is_in_array_gpu<ELEMENT>(metals, element[col_indices_d[j] - 2], num_metals);

                        // conductive vacancy sites
                        bool cvacancy1 = (element[i - 2] == VACANCY) && (atom_charge[i - 2] == 0);
                        bool cvacancy2 = (element[col_indices_d[j] - 2] == VACANCY) && (atom_charge[col_indices_d[j] - 2] == 0);
                        
                        if ((metal1 && metal2) || (cvacancy1 && cvacancy2))
                        {
                            data_d[j] = -high_G;
                        }
                        else
                        {
                            data_d[j] = -low_G;
                        }
                    }

                }
            }
        }
    }
}


__global__ void populate_sparse_X_gpu2(const double *posx_d, const double *posy_d, const double *posz_d,
                                        const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                                        const double *lattice, bool pbc, double nn_dist, const double tol,
                                        const double high_G, const double low_G, const double loop_G, 
                                        const double Vd, const double m_e, const double V0,
                                        int num_source_inj, int num_ground_ext, const int num_layers_contact,
                                        int num_metals, int matrix_size, int *row_indices_d, int *col_indices_d, double *data_d, int X_nnz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N_full = matrix_size;
    int N_atom = matrix_size - 2;
    
    for(int id = idx; id < X_nnz; id += blockDim.x * gridDim.x){

        int i = row_indices_d[id];
        int j = id;

        if(i >= N_full-1){
            continue;
        }

        // col_indices_d[j] is the index of j in the matrix. j is the index of the data vector
        // if dealing with a diagonal element, we add the positive value from i = i and j = N_full to include the ground node

        // extraction boundary (row)
        if(i == 0)
        {
            // diagonal element (0, 0) --> add the value from (0, N_full)
            if (col_indices_d[j] == 0)
            {
                data_d[j] = +high_G;
            }
            // loop connection (0, 1)
            if (col_indices_d[j] == 1)
            {
                data_d[j] = -loop_G;
            }
            // extraction connections from the device
            if ( col_indices_d[j] > N_full - num_ground_ext )
            {
                data_d[j] = -high_G;
            } 
        }

        // injection boundary (row)
        if(i == 1)
        {
            // loop connection (1, 0)
            if (col_indices_d[j] == 0)
            {
                data_d[j] = -loop_G;
            }
            // injection connections to the device
            if ( col_indices_d[j] >= 2 || (col_indices_d[j] > N_full - num_ground_ext) )
            {
                data_d[j] = -high_G;
            } 
        }

        // inner matrix terms
        if (i >= 2)
        {
            // diagonal elements --> add the value from (i - 2, N_full - 2) if site i - 2 neighbors the ground node
            if (i == col_indices_d[j])
            {
                double dist_angstrom = site_dist_gpu(posx_d[i - 2], posy_d[i - 2], posz_d[i - 2],
                                                        posx_d[N_atom-1], posy_d[N_atom-1], posz_d[N_atom-1], 
                                                        lattice[0], lattice[1], lattice[2], pbc);                                   
                bool neighboring_ground = (dist_angstrom < nn_dist);
                
                if (neighboring_ground) 
                {
                    data_d[j] = +high_G;     // assuming all the connections to ground come from the right contact
                } 
            }

            // extraction boundary (column)
            if ( (col_indices_d[j] == 0) && (i > N_full - num_ground_ext) )
            {
                data_d[j] = -high_G;
            }

            // injection boundary (column)
            if ( (col_indices_d[j] == 1) && (i < num_source_inj + 2) )
            {
                data_d[j] = -high_G;
            }

            // off-diagonal inner matrix elements
            if ( (col_indices_d[j] >= 2) && (col_indices_d[j] != i)) 
            {

                double dist_angstrom = site_dist_gpu(posx_d[i - 2], posy_d[i - 2], posz_d[i - 2],
                                                        posx_d[col_indices_d[j] - 2], posy_d[col_indices_d[j] - 2], posz_d[col_indices_d[j] - 2], 
                                                        lattice[0], lattice[1], lattice[2], pbc);                                       
                    
                bool neighbor = (dist_angstrom < nn_dist);                                                      

                // non-neighbor connections
                if (!neighbor)
                {
                    bool any_vacancy1 = element[i - 2] == VACANCY;
                    bool any_vacancy2 = element[col_indices_d[j] - 2] == VACANCY;

                    // contacts, excluding the last layer 
                    bool metal1p = is_in_array_gpu(metals, element[i - 2], num_metals) 
                                                && ((i - 2) > ((num_layers_contact - 1)*num_source_inj))
                                                && ((i - 2) < (N_full - (num_layers_contact - 1)*num_ground_ext)); 

                    bool metal2p = is_in_array_gpu(metals, element[col_indices_d[j] - 2], num_metals)
                                                && ((col_indices_d[j] - 2) > ((num_layers_contact - 1)*num_source_inj))
                                                && ((col_indices_d[j] - 2) < (N_full - (num_layers_contact - 1)*num_ground_ext));  

                    // types of tunnelling conditions considered
                    bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                    bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                    bool contact_to_contact = (metal1p && metal2p);

                    double local_E_drop = atom_CB_edge[i - 2] - atom_CB_edge[col_indices_d[j] - 2];                // [eV] difference in energy between the two atoms

                    // compute the WKB tunneling coefficients for all the tunnelling conditions
                    if ((trap_to_trap || contact_to_trap || contact_to_contact)  && (fabs(local_E_drop) > tol))
                    {
                            
                        double prefac = -(sqrt( 2 * m_e ) / h_bar) * (2.0 / 3.0);           // [s/(kg^1/2 * m^2)] coefficient inside the exponential
                        double dist = (1e-10)*dist_angstrom;                                // [m] 3D distance between atoms i and j

                        if (contact_to_trap)
                        {
                            double energy_window = fabs(local_E_drop);                      // [eV] energy window for tunneling from the contacts
                            double dV = 0.01;                                               // [V] energy spacing for numerical integration
                            double dE = eV_to_J * dV;                                       // [eV] energy spacing for numerical integration
                                    
                            // integrate over all the occupied energy levels in the contact
                            double T = 0.0;
                            for (double iv = 0; iv < energy_window; iv += dE)
                            {
                                double E1 = eV_to_J * V0 + iv;                                  // [J] Energy distance to CB before tunnelling
                                double E2 = E1 - fabs(local_E_drop);                            // [J] Energy distance to CB after tunnelling

                                if (E2 > 0)                                                     // trapezoidal potential barrier (low field)                 
                                {                                                           
                                    T += exp(prefac * (dist / fabs(local_E_drop)) * ( pow(E1, 1.5) - pow(E2, 1.5) ) );
                                }

                                if (E2 < 0)                                                      // triangular potential barrier (high field)                               
                                {
                                    T += exp(prefac * (dist / fabs(local_E_drop)) * ( pow(E1, 1.5) )); 
                                } 
                            }
                            data_d[j] = -T;
                        } 
                        else 
                        {
                            double E1 = eV_to_J * V0;                                        // [J] Energy distance to CB before tunnelling
                            double E2 = E1 - fabs(local_E_drop);                             // [J] Energy distance to CB after tunnelling
                                    
                            if (E2 > 0)                                                      // trapezoidal potential barrier (low field)
                            {                                                           
                                double T = exp(prefac * (dist / fabs(E1 - E2)) * ( pow(E1, 1.5) - pow(E2, 1.5) ) );
                                data_d[j] = -T;
                            }

                            if (E2 < 0)                                                        // triangular potential barrier (high field)
                            {
                                double T = exp(prefac * (dist / fabs(E1 - E2)) * ( pow(E1, 1.5) ));
                                data_d[j] = -T;
                            }
                        }
                    }
                }

                // direct terms
                if ( neighbor )
                {
                    // contacts
                    bool metal1 = is_in_array_gpu<ELEMENT>(metals, element[i - 2], num_metals);
                    bool metal2 = is_in_array_gpu<ELEMENT>(metals, element[col_indices_d[j] - 2], num_metals);

                    // conductive vacancy sites
                    bool cvacancy1 = (element[i - 2] == VACANCY) && (atom_charge[i - 2] == 0);
                    bool cvacancy2 = (element[col_indices_d[j] - 2] == VACANCY) && (atom_charge[col_indices_d[j] - 2] == 0);
                    
                    if ((metal1 && metal2) || (cvacancy1 && cvacancy2))
                    {
                        data_d[j] = -high_G;
                    }
                    else
                    {
                        data_d[j] = -low_G;
                    }
                }

            }
        }
    }
}




// Function to convert dense matrix to CSR format using cuSPARSE
void denseToCSR(hipsparseHandle_t handle, double* d_dense, int num_rows, int num_cols,
                double** d_csr_values, int** d_csr_offsets, int** d_csr_columns, int* total_nnz)
{
    hipsparseSpMatDescr_t matB;
    hipsparseDnMatDescr_t matA;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    int                          ld = num_cols;

    // Create dense matrix A
    hipsparseCreateDnMat(&matA, num_rows, num_cols, ld, d_dense,
                        HIP_R_64F, HIPSPARSE_ORDER_ROW);

    // Create sparse matrix B in CSR format
    hipsparseCreateCsr(&matB, num_rows, num_cols, 0,
                      *d_csr_offsets, NULL, NULL,
                      HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                      HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);

    // allocate an external buffer if needed
    hipsparseDenseToSparse_bufferSize(handle, matA, matB,
                                     HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                     &bufferSize);
    hipMalloc(&dBuffer, bufferSize);

    // execute Sparse to Dense conversion
    hipsparseDenseToSparse_analysis(handle, matA, matB,
                                   HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                   dBuffer);
                            
    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    hipsparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp, &nnz); 
    *total_nnz = static_cast<int>(nnz);

    // allocate CSR column indices and values
    hipMalloc((void**) d_csr_columns, nnz * sizeof(int));
    hipMalloc((void**) d_csr_values,  nnz * sizeof(double));

    // reset offsets, column indices, and values pointers
    hipsparseStatus_t status = hipsparseCsrSetPointers(matB, *d_csr_offsets, *d_csr_columns, *d_csr_values);
    if (status != HIPSPARSE_STATUS_SUCCESS)
    {
        std::cerr << "hipsparseCsrSetPointers failed." << std::endl;
        return;
    }

    // execute Sparse to Dense conversion
    hipsparseDenseToSparse_convert(handle, matA, matB,
                                  HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                  dBuffer);

    // destroy matrix/vector descriptors
    hipsparseDestroyDnMat(matA);
    hipsparseDestroySpMat(matB);
}




// populates the row pointers and column indices of X
void Assemble_X_sparsity(int Natom, const double *posx, const double *posy, const double *posz,
                         const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                         const double *lattice, bool pbc, double nn_dist, const double tol,
                         int num_source_inj, int num_ground_ext, const int num_layers_contact,
                         int num_metals, int **X_row_ptr, int **X_col_indices, int *X_nnz)
{

    // number of atoms + ground node + driver nodes 
    int Nfull = Natom + 2;
    int matrix_size = Nfull; 

    // Compute the number of nonzeros per row of the matrix (Nsub x Nsub)
    int *nnz_per_row_d;
    gpuErrchk( hipMalloc((void **)&nnz_per_row_d, matrix_size * sizeof(int)) );
    gpuErrchk( hipMemset(nnz_per_row_d, 0, matrix_size * sizeof(int)) );

    std::cout << "og N_sub: " << Nfull - 1 << "\n";

    int threads = 512;
    int blocks = (matrix_size + threads - 1) / threads;
    hipLaunchKernelGGL(calc_nnz_per_row_X_gpu, blocks, threads, 0, 0, posx, posy, posz,
                       metals, element, atom_charge, atom_CB_edge,
                       lattice, pbc, nn_dist, tol,
                       num_source_inj, num_ground_ext, num_layers_contact,
                       num_metals, matrix_size, nnz_per_row_d);
    gpuErrchk( hipPeekAtLastError() );
    hipDeviceSynchronize();

    std::cout << "matrix size: " << matrix_size << std::endl;

    //debug
    // print nnz per row to file
    // std::ofstream fout("nnz_per_row.txt");
    // int *nnz_per_row_h = (int*)malloc((matrix_size-1) * sizeof(int));
    // gpuErrchk(hipMemcpy(nnz_per_row_h, nnz_per_row_d, (matrix_size-1) * sizeof(int), hipMemcpyDeviceToHost));
    // for (int i = 0; i < (matrix_size-1); i++) {
    //     std::cout << nnz_per_row_h[i] << " ";
    //     fout << nnz_per_row_h[i]; 
    //     fout << ' ';
    // } 
    // fout.close();
    // std::cout << "wrote nnz correct to file\n";
    //debug

    // debug
    // int *nnz_per_row_h = new int[Nfull-1];
    // gpuErrchk(hipMemcpy(nnz_per_row_h, nnz_per_row_d, (Nfull-1) * sizeof(int), hipMemcpyDeviceToHost));
    // int total_nnz = 0;
    // for (int i = 0; i < (Nfull-1); ++i) {
    //     total_nnz += nnz_per_row_h[i];
    // }
    // std::cout << "total nnz: " << total_nnz << std::endl;
    // std::cout << "exiting\n";
    // exit(1);

    // Set the row pointers according to the cumulative sum of the nnz per row (total nnz is the last element of the row pointer)
    gpuErrchk( hipMalloc((void **)X_row_ptr, (matrix_size + 1 - 1) * sizeof(int)) );   // subtract 1 to ignore the ground node
    gpuErrchk( hipMemset((*X_row_ptr), 0, (matrix_size + 1 - 1) * sizeof(int)) );      // subtract 1 to ignore the ground node

    void     *temp_storage_d = NULL;                                                    // determines temporary device storage requirements for inclusive prefix sum
    size_t   temp_storage_bytes = 0;
    hipcub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*X_row_ptr)+1, matrix_size - 1); // subtract 1 to ignore the ground node
    gpuErrchk( hipMalloc(&temp_storage_d, temp_storage_bytes) );                             // inclusive sum starting at second value to get the row ptr, which is the same as inclusive sum starting at first value and last value filled with nnz
    hipcub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*X_row_ptr)+1, matrix_size - 1);
    gpuErrchk( hipMemcpy(X_nnz, (*X_row_ptr) + matrix_size - 1, sizeof(int), hipMemcpyDeviceToHost) );
    // std::cout << "\nsparse nnz: " << *X_nnz << std::endl;

    // assemble the column indices from 0 to Nsub (excluding the ground node)
    gpuErrchk( hipMalloc((void **)X_col_indices, X_nnz[0] * sizeof(int)) );
    hipLaunchKernelGGL(assemble_X_indices_gpu, blocks, threads, 0, 0, posx, posy, posz,
                         metals, element, atom_charge, atom_CB_edge,
                         lattice, pbc, nn_dist, tol,
                         num_source_inj, num_ground_ext, num_layers_contact,
                         num_metals, matrix_size, nnz_per_row_d,
                        (*X_row_ptr),
                        (*X_col_indices));

    // debug
    // std::ofstream fout2("col_sparse.txt");
    // int *X_col_indices_host = (int*)malloc(X_nnz[0] * sizeof(int));
    // gpuErrchk(hipMemcpy(X_col_indices_host, *X_col_indices, X_nnz[0] * sizeof(int), hipMemcpyDeviceToHost));
    // int sum_of_indices = 0;
    // for (int i = 0; i < X_nnz[0]; i++) {
    //     sum_of_indices += X_col_indices_host[i];
    //     fout2 << X_col_indices_host[i]; 
    //     fout2 << ' ';
    // }
    // fout2.close();
    // std::cout << "\nSum of all column indices sparse: " << sum_of_indices << std::endl;
    // exit(1);
    // debug

    hipFree(temp_storage_d);
    hipFree(nnz_per_row_d);

}


// populates the row pointers and column indices of X
void Assemble_X_sparsity2(int Natom, const double *posx, const double *posy, const double *posz,
                         const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                         const double *lattice, bool pbc, double nn_dist, const double tol,
                         int num_source_inj, int num_ground_ext, const int num_layers_contact,
                         int num_metals,
                         int **X_row_ptr,
                         int **X_row_indices,
                         int **X_col_indices,
                         int *X_nnz)
{

    // number of atoms + ground node + driver nodes 
    int Nfull = Natom + 2;
    int matrix_size = Nfull; 

    // Compute the number of nonzeros per row of the matrix (Nsub x Nsub)
    int *nnz_per_row_d;
    gpuErrchk( hipMalloc((void **)&nnz_per_row_d, matrix_size * sizeof(int)) );
    gpuErrchk( hipMemset(nnz_per_row_d, 0, matrix_size * sizeof(int)) );

    int threads = 512;
    int blocks = (matrix_size + threads - 1) / threads;
    hipLaunchKernelGGL(calc_nnz_per_row_X_gpu, blocks, threads, 0, 0, posx, posy, posz,
                         metals, element, atom_charge, atom_CB_edge,
                         lattice, pbc, nn_dist, tol,
                         num_source_inj, num_ground_ext, num_layers_contact,
                         num_metals, matrix_size, nnz_per_row_d);
    gpuErrchk( hipPeekAtLastError() );
    hipDeviceSynchronize();

    // diagonal and 0/1 and connections to ground and source
    int nnz2 = 4 + num_source_inj + (matrix_size - Natom + num_ground_ext);



    // Set the row pointers according to the cumulative sum of the nnz per row (total nnz is the last element of the row pointer)
    gpuErrchk( hipMalloc((void **)X_row_ptr, (matrix_size + 1 - 1) * sizeof(int)) );   // subtract 1 to ignore the ground node
    gpuErrchk( hipMemset((*X_row_ptr), 0, (matrix_size + 1 - 1) * sizeof(int)) );      // subtract 1 to ignore the ground node

    void     *temp_storage_d = NULL;                                                    // determines temporary device storage requirements for inclusive prefix sum
    size_t   temp_storage_bytes = 0;
    hipcub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*X_row_ptr)+1, matrix_size - 1); // subtract 1 to ignore the ground node
    gpuErrchk( hipMalloc(&temp_storage_d, temp_storage_bytes) );                             // inclusive sum starting at second value to get the row ptr, which is the same as inclusive sum starting at first value and last value filled with nnz
    hipcub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, nnz_per_row_d, (*X_row_ptr)+1, matrix_size - 1);
    gpuErrchk( hipMemcpy(X_nnz, (*X_row_ptr) + matrix_size - 1, sizeof(int), hipMemcpyDeviceToHost) );
    std::cout << "\nsparse nnz: " << *X_nnz << std::endl;



    // assemble the column indices from 0 to Nsub (excluding the ground node)
    gpuErrchk( hipMalloc((void **)X_col_indices, X_nnz[0] * sizeof(int)) );
    hipLaunchKernelGGL(assemble_X_indices_gpu, blocks, threads, 0, 0, posx, posy, posz,
                         metals, element, atom_charge, atom_CB_edge,
                         lattice, pbc, nn_dist, tol,
                         num_source_inj, num_ground_ext, num_layers_contact,
                         num_metals, matrix_size, nnz_per_row_d,
                        (*X_row_ptr),
                        (*X_col_indices));

    hipFree(temp_storage_d);
    hipFree(nnz_per_row_d);

}



__global__ void calc_diagonal_X_gpu(
    int *col_indices,
    int *row_ptr,
    double *data,
    int matrix_size
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < matrix_size - 1; i += blockDim.x * gridDim.x){ // MINUS ONE
        //reduce the elements in the row
        double tmp = 0.0;
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(i != col_indices[j]){
                tmp += data[j];
            }
        }
        //write the sum of the off-diagonals onto the existing diagonal element
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(i == col_indices[j]){
                data[j] += -tmp;
            }
        }
    }
}

void Assemble_X(int Natom, const double *posx, const double *posy, const double *posz,
                const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                const double *lattice, bool pbc, double nn_dist, const double tol, const double Vd, const double m_e, const double V0,
                const double high_G, const double low_G, const double loop_G,
                int num_source_inj, int num_ground_ext, const int num_layers_contact,
                int num_metals, double **X_data, int **X_row_ptr, int **X_col_indices, int *X_nnz){

    // parallelize over rows
    int Nfull = Natom + 2;
    int threads = 512;
    int blocks = (Nfull + threads - 1) / threads;

    // allocate the data array and initialize it to zeros
    gpuErrchk(hipMalloc((void **)X_data, X_nnz[0] * sizeof(double)));
    gpuErrchk(hipMemset((*X_data), 0, X_nnz[0] * sizeof(double)));

    // assemble the off-diagonal values of X
    hipLaunchKernelGGL(populate_sparse_X_gpu, blocks, threads, 0, 0, posx, posy, posz,
                                               metals, element, atom_charge, atom_CB_edge,
                                               lattice, pbc, nn_dist, tol, high_G, low_G, loop_G,
                                               Vd, m_e, V0,
                                               num_source_inj, num_ground_ext, num_layers_contact,
                                               num_metals, Nfull, *X_row_ptr, *X_col_indices, *X_data);
    gpuErrchk( hipPeekAtLastError() );
    // gpuErrchk( hipDeviceSynchronize() );

    // add the off diagonals onto the diagonal
    hipLaunchKernelGGL(calc_diagonal_X_gpu, blocks, threads, 0, 0, *X_col_indices, *X_row_ptr, *X_data, Nfull);
    gpuErrchk( hipPeekAtLastError() );
    gpuErrchk( hipDeviceSynchronize() );

}



void Assemble_X2(int Natom, const double *posx, const double *posy, const double *posz,
                const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                const double *lattice, bool pbc, double nn_dist, const double tol, const double Vd, const double m_e, const double V0,
                const double high_G, const double low_G, const double loop_G,
                int num_source_inj, int num_ground_ext, const int num_layers_contact,
                int num_metals, double **X_data, int **X_row_indices,
                int **X_row_ptr, int **X_col_indices, int *X_nnz){

    // parallelize over rows
    int Nfull = Natom + 2;
    int threads2 = 512;
    int blocks2 = (X_nnz[0] + threads2 - 1) / threads2;
    int threads = 512;
    int blocks = (Nfull + threads - 1) / threads;

    // allocate the data array and initialize it to zeros
    gpuErrchk(hipMalloc((void **)X_data, X_nnz[0] * sizeof(double)));
    gpuErrchk(hipMemset((*X_data), 0, X_nnz[0] * sizeof(double)));


    // assemble the off-diagonal values of X
    hipLaunchKernelGGL(populate_sparse_X_gpu2, blocks2, threads2, 0, 0, posx, posy, posz,
                                               metals, element, atom_charge, atom_CB_edge,
                                               lattice, pbc, nn_dist, tol, high_G, low_G, loop_G,
                                               Vd, m_e, V0,
                                               num_source_inj, num_ground_ext, num_layers_contact,
                                               num_metals, Nfull, *X_row_indices, *X_col_indices, *X_data, X_nnz[0]);
    gpuErrchk( hipPeekAtLastError() );
    // hipLaunchKernelGGL(populate_sparse_X_gpu, blocks, threads, 0, 0, posx, posy, posz,
    //                                            metals, element, atom_charge, atom_CB_edge,
    //                                            lattice, pbc, nn_dist, tol, high_G, low_G, loop_G,
    //                                            Vd, m_e, V0,
    //                                            num_source_inj, num_ground_ext, num_layers_contact,
    //                                            num_metals, Nfull, *X_row_ptr, *X_col_indices, *X_data);

    
    // gpuErrchk( hipDeviceSynchronize() );

    // add the off diagonals onto the diagonal
    hipLaunchKernelGGL(calc_diagonal_X_gpu, blocks, threads, 0, 0, *X_col_indices, *X_row_ptr, *X_data, Nfull);
    gpuErrchk( hipPeekAtLastError() );
    gpuErrchk( hipDeviceSynchronize() );

}

void Assemble_K_sparsity(const double *posx, const double *posy, const double *posz,
                         const double *lattice, const bool pbc,
                         const double cutoff_radius,
                         int system_size, int contact_left_size, int contact_right_size,
                         int **A_row_ptr, int **A_col_indices, int *A_nnz, 
                         int **contact_left_col_indices, int **contact_left_row_ptr, int *contact_left_nnz, 
                         int **contact_right_col_indices, int **contact_right_row_ptr, int *contact_right_nnz){

    // indices of A (the device submatrix)
    indices_creation_gpu(
        posx + contact_left_size,
        posy + contact_left_size,
        posz + contact_left_size,
        lattice, pbc,
        cutoff_radius,
        system_size,
        A_col_indices,
        A_row_ptr,
        A_nnz
    );

    // indices of the off-diagonal leftcontact-A matrix
    indices_creation_gpu_off_diagonal_block(
        posx, posy, posz,
        lattice, pbc,
        cutoff_radius,
        system_size,
        contact_left_size,
        contact_left_size,
        0,
        contact_left_col_indices,
        contact_left_row_ptr,
        contact_left_nnz
    );
    // std::cout << "contact_left_nnz " << *contact_left_nnz << std::endl;

    // indices of the off-diagonal A-rightcontact matrix
    indices_creation_gpu_off_diagonal_block(
        posx, posy, posz,
        lattice, pbc,
        cutoff_radius,
        system_size,
        contact_right_size,
        contact_left_size,
        contact_left_size + system_size,
        contact_right_col_indices,
        contact_right_row_ptr,
        contact_right_nnz
    );
    // std::cout << "contact_right_nnz " << *contact_right_nnz << std::endl;
}