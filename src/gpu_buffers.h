#pragma once
#include "utils.h"
#include <mpi.h>
#include "../dist_iterative/dist_objects.h"

#ifdef USE_CUDA
#include "gpu_solvers.h"
#endif

// forward declaration of device class
class Device;

// Member variables are pointers to GPU memory unless specified with _host (or integers passed by value)
class GPUBuffers {

public:
    // varying parameters:
    int *site_charge = nullptr;
    double *site_power, *site_potential_boundary, *site_potential_charge, *site_temperature = nullptr;
    double *site_CB_edge = nullptr;
    double *T_bg = nullptr;
    double *atom_power = nullptr;//, *atom_potential = nullptr;
    double *atom_CB_edge = nullptr;
    double *atom_virtual_potentials = nullptr;
    int *atom_charge = nullptr;

    // unchanging parameters (site_) and ones which aren't copied back (atom_):
    ELEMENT *site_element = nullptr;
    ELEMENT *atom_element = nullptr;
    double *site_x, *site_y, *site_z = nullptr;
    double *atom_x, *atom_y, *atom_z = nullptr;
    ELEMENT *metal_types;
    double *sigma, *k, *lattice, *freq;
    int *neigh_idx, *site_layer = nullptr;

    // CUDA library handles
    // cublasHandle_t cublas_handle;
    // cusolverDnHandle_t  cusolver_handle;
    // cusparseHandle_t cusparse_handle;

    // CSR indices pre-computation for K
    int *Device_row_ptr_d = nullptr;                // CSR representation of the matrix which represents connectivity in the device
    int *Device_col_indices_d = nullptr;            
    int *contact_left_row_ptr = nullptr;            // CSR representation of the matrix which represents connectivity of the left contact
    int *contact_left_col_indices = nullptr;
    int *contact_right_row_ptr = nullptr;           // CSR representation of the matrix which represents connectivity of the right contact
    int *contact_right_col_indices = nullptr;       
    int Device_nnz, contact_left_nnz, contact_right_nnz;



    // NOT gpu pointers, passed by value
    int num_metal_types_ = 0;
    int N_ = 0;                                     // number of sites in the device
    int nn_ = 0;                                    // maximum number of neighbors in the device
    int N_atom_ = 0;                                // number of atomic sites in the device

    // helper variables stored on host:
    std::vector<double> E_gen_host, E_rec_host, E_Vdiff_host, E_Odiff_host;

    // uploads the local device attributes into the GPU memory versions
    void sync_HostToGPU(Device &device);

    // downloads the GPU device attributes into the local versions
    void sync_GPUToHost(Device &device);

    // copy back just some device attribute vectors:
    void copy_power_fromGPU(std::vector<double> &power);
    void copy_charge_toGPU(std::vector<int> &charge);
    // void copy_atom_CB_edge_to_GPU();        ///IMPLEMENT THIS BEFORE DOING MULTIPLE V POINTS
    void copy_Tbg_toGPU(double new_T_bg);

    // MPI data split
    MPI_Comm comm;
    int rank, size;
    int *count_K_device = nullptr;
    int *displ_K_device = nullptr;
    int *count_sites = nullptr;
    int *displ_sites = nullptr;
    Distributed_matrix *K_distributed = nullptr;
    Distributed_vector *K_p_distributed = nullptr; // vector for SPMV of K*p
    int *left_row_ptr_d = nullptr;            // CSR representation of the matrix which represents connectivity of the left contact
    int *left_col_indices_d = nullptr;
    int *right_row_ptr_d = nullptr;           // CSR representation of the matrix which represents connectivity of the right contact
    int *right_col_indices_d = nullptr; 
    int left_nnz, right_nnz;

    // constructor allocates nothing (used for CPU-only code):
    GPUBuffers(){};

    // constructor allocates arrays in GPU memory
    GPUBuffers(std::vector<Layer> layers, std::vector<int> site_layer_in, double freq_in, int N, int N_atom,
               std::vector<double> site_x_in,  std::vector<double> site_y_in,  std::vector<double> site_z_in,
               int nn, double sigma_in, double k_in, std::vector<double> lattice_in, std::vector<int> neigh_idx_in, std::vector<ELEMENT> metals,
               int num_metals_types, MPI_Comm comm, int N_contact) {
            
        this->N_ = N;
        this->N_atom_ = N_atom;
        this->nn_ = nn;
        this->num_metal_types_ = num_metals_types;

        // make layer arrays
        for (auto l : layers){
            E_gen_host.push_back(l.E_gen_0);
            E_rec_host.push_back(l.E_rec_1);
            E_Vdiff_host.push_back(l.E_diff_2);
            E_Odiff_host.push_back(l.E_diff_3);
        }
        int num_layers = layers.size();

        // allocate MPI data split for sites
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        this->comm = comm;

        count_K_device = new int[size];
        displ_K_device = new int[size];
        int rows_per_rank_K_device = (N - 2*N_contact) / size;
        for (int i = 0; i < size; ++i) {
            if(i < (N - 2*N_contact) % size){
                count_K_device[i] = rows_per_rank_K_device+1;
            }
            else{
                count_K_device[i] = rows_per_rank_K_device;
            }
        }
        displ_K_device[0] = 0;
        for (int i = 1; i < size; ++i) {
            displ_K_device[i] = displ_K_device[i-1] + count_K_device[i-1];
        }

        // TODO better split
        count_sites = new int[size];
        displ_sites = new int[size];
        for(int i = 0; i < size; ++i){
            count_sites[i] = count_K_device[i];
        }
        count_sites[0] += N_contact;
        count_sites[size-1] += N_contact;
        displ_sites[0] = 0;
        for (int i = 1; i < size; ++i) {
            displ_sites[i] = displ_sites[i-1] + count_sites[i-1];
        }
        // initialize CUDA library handles:
        // cusparseCreate(&cusparse_handle);
        // cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);
        // CreateCublasHandle(cublas_handle, 0);
        // cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
        // CreateCusolverDnHandle(cusolver_handle, 0);

#ifdef USE_CUDA

        // small lists and variables to store in GPU cache
        copytoConstMemory(E_gen_host, E_rec_host, E_Vdiff_host, E_Odiff_host);
        cudaDeviceSynchronize();
        
        // member variables of the KMCProcess 
        gpuErrchk( cudaMalloc((void**)&site_layer, N_ * sizeof(int)) );

        // member variables of the Device
        gpuErrchk( cudaMalloc((void**)&site_element, N_ * sizeof(ELEMENT)) );
        gpuErrchk( cudaMalloc((void**)&metal_types, num_metal_types_ * sizeof(ELEMENT)) );
        gpuErrchk( cudaMalloc((void**)&site_x, N_  * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_y, N_  * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_z, N_  * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_power, N_ * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_CB_edge, N_ * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_potential_boundary, N_ * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_potential_charge, N_ * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_temperature, N_ * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&site_charge, N_ * sizeof(int)) );
        gpuErrchk( cudaMalloc((void**)&neigh_idx, N_ * nn_ * sizeof(int)) );
        gpuErrchk( cudaMalloc((void**)&T_bg, 1 * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&sigma, 1 * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&k, 1 * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&lattice, 3 * sizeof(double)) );
        gpuErrchk( cudaMalloc((void**)&freq, 1 * sizeof(double)) );
        gpuErrchk( cudaMalloc((void **)&atom_element, N_ * sizeof(ELEMENT)) );
        gpuErrchk( cudaMalloc((void **)&atom_x, N_ * sizeof(double)) );
        gpuErrchk( cudaMalloc((void **)&atom_y, N_ * sizeof(double)) );             // these have length N_ since it's a maximum
        gpuErrchk( cudaMalloc((void **)&atom_z, N_ * sizeof(double)) );
        gpuErrchk( cudaMalloc((void **)&atom_power, N_ * sizeof(double)) );
        gpuErrchk( cudaMalloc((void **)&atom_CB_edge, N_atom_ * sizeof(double)) );
        gpuErrchk( cudaMalloc((void **)&atom_charge, N_ * sizeof(int)) );

        cudaDeviceSynchronize();

        // virtual potentials initial guess to store (solution vector for dissipated power solver):
        gpuErrchk( cudaMalloc((void **)&atom_virtual_potentials, (N_atom_ + 2) * sizeof(double)) );
        gpuErrchk( cudaMemset(atom_virtual_potentials, 0, (N_atom_ + 2) * sizeof(double)) );                          // initialize the solution vector for the dissipated power                                 

        cudaDeviceSynchronize();

        // fixed parameters which can be copied from the beginning:
        gpuErrchk( cudaMemcpy(site_layer, site_layer_in.data(), N_ * sizeof(int), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(site_x, site_x_in.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(site_y, site_y_in.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(site_z, site_z_in.data(), N_ * sizeof(double), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(metal_types, metals.data(), num_metal_types_ * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(sigma, &sigma_in, 1 * sizeof(double), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(k, &k_in, 1 * sizeof(double), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(freq, &freq_in, 1 * sizeof(double), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(lattice, lattice_in.data(), 3 * sizeof(double), cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(neigh_idx, neigh_idx_in.data(), N_ * nn_ * sizeof(int), cudaMemcpyHostToDevice) );
    
        cudaDeviceSynchronize();
        
#endif

    }

    // ~GPUBuffers() {
    //     delete[] count_sites;
    //     delete[] displ_sites;
    // }

    void freeGPUmemory();

};
