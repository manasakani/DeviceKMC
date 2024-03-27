#pragma once
#include "utils.h"
#include <mpi.h>
#include "../dist_iterative/dist_objects.h"

#include "gpu_solvers.h"

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
    int *neigh_idx, *cutoff_window, *cutoff_idx, *site_layer = nullptr;

    // host vectors used for the collection and sum of the distributed potential
    double *potential_local_h = nullptr; // = (double *)calloc(gpubuf.count_sites[gpubuf.rank], sizeof(double));
    double *potential_h = nullptr; // (double *)calloc(gpubuf.N_, sizeof(double));

    // CUDA library handles
    // hipblasHandle_t cublas_handle;
    // cusolverDnHandle_t  cusolver_handle;
    // hipsparseHandle_t cusparse_handle;

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
    int N_sub_ = 0;                                // size of the T_matrix (Natom + 1)
    int N_cutoff_ = 0;

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

    Distributed_matrix *K_distributed = nullptr;
    Distributed_vector *K_p_distributed = nullptr;            // vector for SPMV of K*p
    int *left_row_ptr_d = nullptr;                            // CSR representation of the matrix which represents connectivity of the left contact
    int *left_col_indices_d = nullptr;
    int *right_row_ptr_d = nullptr;                           // CSR representation of the matrix which represents connectivity of the right contact
    int *right_col_indices_d = nullptr; 
    int left_nnz, right_nnz;
    

    // buffers used for the T matrix:
    Distributed_matrix *T_distributed = nullptr;
    Distributed_vector *T_p_distributed = nullptr;            // vector for SPMV of T*p

    // constructor allocates nothing (used for CPU-only code):
    GPUBuffers(){};

    // constructor allocates arrays in GPU memory
    GPUBuffers(std::vector<Layer> layers, std::vector<int> site_layer_in, double freq_in, int N, int N_atom,
               std::vector<double> site_x_in,  std::vector<double> site_y_in,  std::vector<double> site_z_in,
               int nn, double sigma_in, double k_in, std::vector<double> lattice_in, 
               std::vector<int> neigh_idx_in, std::vector<int> cutoff_window_in, std::vector<int> cutoff_idx_in,
               std::vector<ELEMENT> metals, int num_metals_types, int N_contact) {
            
        this->N_ = N;
        this->N_atom_ = N_atom;
        this->N_sub_ = N_atom + 1;                          // size of matrix T
        this->nn_ = nn;
        this->num_metal_types_ = num_metals_types;
        this->N_cutoff_ = cutoff_idx_in.size()/N_;  

        // make layer arrays 
        for (auto l : layers){
            E_gen_host.push_back(l.E_gen_0);
            E_rec_host.push_back(l.E_rec_1);
            E_Vdiff_host.push_back(l.E_diff_2); 
            E_Odiff_host.push_back(l.E_diff_3);
        }
        int num_layers = layers.size();

        // member variables of the KMCProcess 
        gpuErrchk( hipMalloc((void**)&site_layer, N_ * sizeof(int)) );

        // member variables of the Device
        gpuErrchk( hipMalloc((void**)&site_element, N_ * sizeof(ELEMENT)) );
        gpuErrchk( hipMalloc((void**)&metal_types, num_metal_types_ * sizeof(ELEMENT)) );
        gpuErrchk( hipMalloc((void**)&site_x, N_  * sizeof(double)) );
        gpuErrchk( hipMalloc((void**)&site_y, N_  * sizeof(double)) );
        gpuErrchk( hipMalloc((void**)&site_z, N_  * sizeof(double)) );
        gpuErrchk( hipMalloc((void**)&site_power, N_ * sizeof(double)) );
        gpuErrchk( hipMalloc((void**)&site_CB_edge, N_ * sizeof(double)) );
        gpuErrchk( hipMalloc((void**)&site_potential_boundary, N_ * sizeof(double)) );
        gpuErrchk( hipMalloc((void**)&site_potential_charge, N_ * sizeof(double)) );
        gpuErrchk( hipMalloc((void**)&site_temperature, N_ * sizeof(double)) );
        gpuErrchk( hipMalloc((void**)&site_charge, N_ * sizeof(int)) );
        gpuErrchk( hipMalloc((void**)&neigh_idx, N_ * nn_ * sizeof(int)) );
        gpuErrchk( hipMalloc((void**)&cutoff_window, N_ * 2 * sizeof(int)) );
        // gpuErrchk( hipMalloc((void**)&cutoff_idx, (size_t)N_ * (size_t)N_cutoff_ * sizeof(int)) );
        gpuErrchk( hipMalloc((void**)&T_bg, 1 * sizeof(double)) );
        gpuErrchk( hipMalloc((void**)&sigma, 1 * sizeof(double)) );
        gpuErrchk( hipMalloc((void**)&k, 1 * sizeof(double)) );
        gpuErrchk( hipMalloc((void**)&lattice, 3 * sizeof(double)) );
        gpuErrchk( hipMalloc((void**)&freq, 1 * sizeof(double)) );
        gpuErrchk( hipMalloc((void **)&atom_element, N_ * sizeof(ELEMENT)) );
        gpuErrchk( hipMalloc((void **)&atom_x, N_ * sizeof(double)) );
        gpuErrchk( hipMalloc((void **)&atom_y, N_ * sizeof(double)) );             // these have length N_ since it's a maximum
        gpuErrchk( hipMalloc((void **)&atom_z, N_ * sizeof(double)) );
        gpuErrchk( hipMalloc((void **)&atom_power, N_ * sizeof(double)) );
        gpuErrchk( hipMalloc((void **)&atom_CB_edge, N_atom_ * sizeof(double)) );
        gpuErrchk( hipMalloc((void **)&atom_charge, N_ * sizeof(int)) );

        // virtual potentials initial guess to store (solution vector for dissipated power solver):
        gpuErrchk( hipMalloc((void **)&atom_virtual_potentials, (N_atom_ + 2) * sizeof(double)) );
        gpuErrchk( hipMemset(atom_virtual_potentials, 0, (N_atom_ + 2) * sizeof(double)) );                          // initialize the solution vector for the dissipated power                                 

        // fixed parameters which can be copied from the beginning:
        gpuErrchk( hipMemcpy(site_layer, site_layer_in.data(), N_ * sizeof(int), hipMemcpyHostToDevice) );
        gpuErrchk( hipMemcpy(site_x, site_x_in.data(), N_ * sizeof(double), hipMemcpyHostToDevice) );
        gpuErrchk( hipMemcpy(site_y, site_y_in.data(), N_ * sizeof(double), hipMemcpyHostToDevice) );
        gpuErrchk( hipMemcpy(site_z, site_z_in.data(), N_ * sizeof(double), hipMemcpyHostToDevice) );
        gpuErrchk( hipMemcpy(metal_types, metals.data(), num_metal_types_ * sizeof(ELEMENT), hipMemcpyHostToDevice) );
        gpuErrchk( hipMemcpy(sigma, &sigma_in, 1 * sizeof(double), hipMemcpyHostToDevice) );
        gpuErrchk( hipMemcpy(k, &k_in, 1 * sizeof(double), hipMemcpyHostToDevice) );
        gpuErrchk( hipMemcpy(freq, &freq_in, 1 * sizeof(double), hipMemcpyHostToDevice) );
        gpuErrchk( hipMemcpy(lattice, lattice_in.data(), 3 * sizeof(double), hipMemcpyHostToDevice) );
        gpuErrchk( hipMemcpy(neigh_idx, neigh_idx_in.data(), N_ * nn_ * sizeof(int), hipMemcpyHostToDevice) );
        gpuErrchk( hipMemcpy(cutoff_window, cutoff_window_in.data(), N_ * 2 * sizeof(int), hipMemcpyHostToDevice) );
        // gpuErrchk( hipMemcpy(cutoff_idx, cutoff_idx_in.data(), (size_t)N_ * (size_t)N_cutoff_ * sizeof(int), hipMemcpyHostToDevice) );
    }

    void freeGPUmemory();

};
