#include "gpu_solvers.h"
#define NUM_THREADS 512

// Constants needed:
const double eV_to_J = 1.60217663e-19;          // [C]
const double h_bar = 1.054571817e-34;           // [Js]

struct is_defect
{
    __host__ __device__ bool operator()(const ELEMENT element)
    {
        return ((element != DEFECT) && (element != OXYGEN_DEFECT));
    }
};

struct is_not_zero
{
    __host__ __device__ bool operator()(const int integer)
    {
        return (integer != 0);
    }
};

// Collect the indices of the contacts and the vacancies
__global__ void get_is_tunnel(int *is_tunnel, int *tunnel_indices, const ELEMENT *element, 
                              int N_atom, int num_layers_contact, int num_source_inj, int num_ground_ext)
{
    int total_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int idx = total_tid; idx < N_atom; idx += total_threads)
    {
        int yes = 0; 

        // vacancies and contacts have states within the bandgap which are included in the tunneling model
        // include the first layer of the contacts, as the rest are directly connected to it
        // METALS ARE HARDCODED

        if ( element[idx] == VACANCY || 
           ( (element[idx] == Ti_EL || element[idx] == N_EL) &&  (idx > (num_layers_contact - 1)*num_source_inj) && (idx < (N_atom - (num_layers_contact - 1)*num_ground_ext)) )) 
        {
            yes = 1;
        }

        is_tunnel[idx] = yes;
        tunnel_indices[idx] = yes * idx;
    }
}

// Compute the number of nonzeros per row of the matrix including the injection, extraction, and device nodes (excluding the ground). 
// Has dimensions of Nsub by Nsub (by the cpu code)
__global__ void calc_nnz_per_row_T_neighbor( const double *posx_d, const double *posy_d, const double *posz_d,
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
        }

        // this can be memset outside the kernel instead
        // source/ground connections
        if ( i < num_source_inj )
        {
            atomicAdd(&nnz_per_row_d[1], 1);
            nnz_row++;
        }
        if ( i > (Natom - num_ground_ext) )
        {
            atomicAdd(&nnz_per_row_d[0], 1);
            nnz_row++;
        }

        nnz_per_row_d[i+2] = nnz_row;

        if ( i == 0 )
        {
            atomicAdd(&nnz_per_row_d[0], 2); // loop connection and diagonal element
            atomicAdd(&nnz_per_row_d[1], 2); // loop connection and diagonal element
        }
    }

}

__global__ void calc_col_idx_T_neighbor(const double *posx_d, const double *posy_d, const double *posz_d,
                                        const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                                        const double *lattice, bool pbc, double nn_dist, const double tol,
                                        int num_source_inj, int num_ground_ext, const int num_layers_contact,
                                        int num_metals, int matrix_size, int *nnz_per_row_d, int *row_ptr_d, int *col_indices_d)
{
    // row ptr is already calculated
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N_full = matrix_size;
    
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
                }
            }
        }

    }
}


// assemble the data for the T matrix - 1D distribution over rows
__global__ void populate_data_T_neighbor(const double *posx_d, const double *posy_d, const double *posz_d,
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

                    // direct terms (neighbor connections)
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


__global__ void populate_data_T_tunnel(double *X, const double *posx, const double *posy, const double *posz,
                                       const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                                       const double *lattice, bool pbc, double high_G, double low_G, double loop_G,
                                       double nn_dist, double m_e, double V0, int num_source_inj, int num_ground_ext, const int num_layers_contact,
                                       int N_atom, int num_tunnel_points, const int *tunnel_indices, int num_metals, const double Vd, const double tol)
{

    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;

    int N = num_tunnel_points;

    for (auto idx = tid_total; idx < N * N; idx += num_threads_total)
    {    

        int i = idx / N;
        int j = idx % N;

        double posx_i = posx[tunnel_indices[i]];
        double posx_j = posx[tunnel_indices[j]];

        double posy_i = posy[tunnel_indices[i]];
        double posy_j = posy[tunnel_indices[j]];

        double posz_i = posz[tunnel_indices[i]];
        double posz_j = posz[tunnel_indices[j]];

        double atom_CB_edge_i = atom_CB_edge[tunnel_indices[i]];
        double atom_CB_edge_j = atom_CB_edge[tunnel_indices[j]];

        ELEMENT element_i = element[tunnel_indices[i]];
        ELEMENT element_j = element[tunnel_indices[j]];

        double dist_angstrom = site_dist_gpu(posx_i, posy_i, posz_i, 
                                             posx_j, posy_j, posz_j, 
                                             lattice[0], lattice[1], lattice[2], pbc);

        bool neighbor = (dist_angstrom < nn_dist) && (i != j);

        // tunneling terms occur between not-neighbors
        if (i != j && !neighbor)
        { 
            bool any_vacancy1 = element_i == VACANCY;
            bool any_vacancy2 = element_j == VACANCY;

            // contacts, the last layer has already been excluded when creating the tunnel indices
            bool metal1p = is_in_array_gpu(metals, element_i, num_metals);
            bool metal2p = is_in_array_gpu(metals, element_j, num_metals);

            // types of tunnelling conditions considered
            bool trap_to_trap = (any_vacancy1 && any_vacancy2);
            bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
            bool contact_to_contact = (metal1p && metal2p);

            double local_E_drop = atom_CB_edge_i - atom_CB_edge_j;                // [eV] difference in energy between the two atoms

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
                    X[N * i + j] = -T;      
                } 
                else 
                {
                    double E1 = eV_to_J * V0;                                        // [J] Energy distance to CB before tunnelling
                    double E2 = E1 - fabs(local_E_drop);                             // [J] Energy distance to CB after tunnelling
                          
                    if (E2 > 0)                                                      // trapezoidal potential barrier (low field)
                    {                                                           
                        double T = exp(prefac * (dist / fabs(E1 - E2)) * ( pow(E1, 1.5) - pow(E2, 1.5) ) );
                        X[N * i + j] = -T; 
                    }

                    if (E2 < 0)                                                        // triangular potential barrier (high field)
                    {
                        double T = exp(prefac * (dist / fabs(E1 - E2)) * ( pow(E1, 1.5) ));
                        X[N * i + j] = -T; 
                    }
                }
            }
        }
        
    }
}


__global__ void calc_diagonal_T_gpu( int *col_indices, int *row_ptr, double *data, int matrix_size, double *diagonal)
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
        diagonal[i] = -tmp;
        //write the sum of the off-diagonals onto the existing diagonal element
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(i == col_indices[j]){
                data[j] += -tmp;
            }
        }
    }
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


__global__ void copy_pdisp(double *site_power, ELEMENT *element, const ELEMENT *metals, double *pdisp, int *atom_gpu_index, int N_atom,
                           const int num_metals, const double alpha)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int idx = tid; idx < N_atom; idx += total_threads)
    {
        bool metal = is_in_array_gpu(metals, element[atom_gpu_index[idx]], num_metals);
        if (!metal)
            site_power[atom_gpu_index[idx]] = -1 * alpha * pdisp[idx];
    }
}

//extracts the diagonal of the dense submatrix into a global vector
__global__ void extract_diag_tunnel(
    double *tunnel_matrix,
    int *tunnel_indices, 
    int num_tunnel_points,
    double *diagonal
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < num_tunnel_points; i += blockDim.x * gridDim.x)
    {
        // +2 since first two indices are the ground and injection nodes
        diagonal[tunnel_indices[i] + 2] += tunnel_matrix[i * num_tunnel_points + i];
    }
}

__global__ void inverse_vector(double *vec, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x)
    {
        vec[i] = 1.0 / vec[i];
    }
}

template <int NTHREADS>
__global__ void get_imacro_sparse(const double *x_values, const int *x_row_ptr, const int *x_col_ind,
                                  const double *m, double *imacro)
{
    int num_threads = blockDim.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int total_tid = bid * num_threads + tid;
    int total_threads = num_threads * gridDim.x;

    int row_start = x_row_ptr[1] + 2;
    int row_end = x_row_ptr[2];

    __shared__ double buf[NTHREADS];
    buf[tid] = 0.0;
 
    for (int idx = row_start + total_tid; idx < row_end; idx += total_threads)
    {
        int col_index = x_col_ind[idx];
        if (col_index >= 2) 
        {
            // buf[tid] += x_values[idx] * (m[0] - m[col_index]);               // extracted (= injected when including ground node)
            buf[tid] += x_values[idx] * (m[col_index] - m[1]);                  // injected
        }
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

// new version with split matrix for neighbor/tunnel connections
void update_power_gpu_split(cublasHandle_t handle, cusolverDnHandle_t handle_cusolver, GPUBuffers &gpubuf, 
                            const int num_source_inj, const int num_ground_ext, const int num_layers_contact,
                            const double Vd, const int pbc, const double high_G, const double low_G, const double loop_G, const double G0, const double tol,
                            const double nn_dist, const double m_e, const double V0, int num_metals, double *imacro,
                            const bool solve_heating_local, const bool solve_heating_global, const double alpha_disp)
{
    auto t0 = std::chrono::steady_clock::now();

    // ***************************************************************************************
    // 1. Update the atoms array from the sites array using copy_if with is_defect as a filter
    int *gpu_index;
    int *atom_gpu_index;
    gpuErrchk( cudaMalloc((void **)&gpu_index, gpubuf.N_ * sizeof(int)) );                                           // indices of the site array
    gpuErrchk( cudaMalloc((void **)&atom_gpu_index, gpubuf.N_ * sizeof(int)) );                                      // indices of the atom array

    thrust::device_ptr<int> gpu_index_ptr = thrust::device_pointer_cast(gpu_index);
    thrust::sequence(gpu_index_ptr, gpu_index_ptr + gpubuf.N_, 0);

    // do these in parallel with a kernel! - check that the positions dont change
    // check if there's some buffer which can be allocated and reused for all of these
    double *last_atom = thrust::copy_if(thrust::device, gpubuf.site_x, gpubuf.site_x + gpubuf.N_, gpubuf.site_element, gpubuf.atom_x, is_defect());
    int N_atom = last_atom - gpubuf.atom_x;
    thrust::copy_if(thrust::device, gpubuf.site_y, gpubuf.site_y + gpubuf.N_, gpubuf.site_element, gpubuf.atom_y, is_defect());
    thrust::copy_if(thrust::device, gpubuf.site_z, gpubuf.site_z + gpubuf.N_, gpubuf.site_element, gpubuf.atom_z, is_defect());
    thrust::copy_if(thrust::device, gpubuf.site_charge, gpubuf.site_charge + gpubuf.N_, gpubuf.site_element, gpubuf.atom_charge, is_defect());
    thrust::copy_if(thrust::device, gpubuf.site_element, gpubuf.site_element + gpubuf.N_, gpubuf.site_element, gpubuf.atom_element, is_defect());
    thrust::copy_if(thrust::device, gpubuf.site_CB_edge, gpubuf.site_CB_edge + gpubuf.N_, gpubuf.site_element, gpubuf.atom_CB_edge, is_defect());
    thrust::copy_if(thrust::device, gpu_index, gpu_index + gpubuf.N_, gpubuf.site_element, atom_gpu_index, is_defect());

    // auto t1 = std::chrono::steady_clock::now();
    // std::chrono::duration<double> dt = t1 - t0;
    // std::cout << "time to update atom arrays: " << dt.count() << "\n";

    // ***************************************************************************************
    // 2. Collect the indices of the contacts and the vacancies    
    int num_threads = 1024;
    int blocks_per_row = (N_atom - 1) / num_threads + 1;
    int num_blocks = blocks_per_row * N_atom;

    // indices of the tunneling connections (contacts and vacancies) in the Natom array
    int *is_tunnel; // [0, 1, 0, 0, 1...] where 1 indicates a tunnel connection
    int *is_tunnel_indices; // [0, 1, 0, 0, 4...] storing the indices of the tunnel connections
    gpuErrchk( cudaMalloc((void **)&is_tunnel, N_atom * sizeof(int)) );    
    gpuErrchk( cudaMalloc((void **)&is_tunnel_indices, N_atom * sizeof(int)) );                                         
    get_is_tunnel<<<num_blocks, num_threads>>>(is_tunnel, is_tunnel_indices, gpubuf.atom_element, N_atom, num_layers_contact, num_source_inj, num_ground_ext);
    gpuErrchk( cudaPeekAtLastError() );
    // check if global counter could be faster

    // auto tx1 = std::chrono::steady_clock::now();
    // std::chrono::duration<double> dtx1 = tx1 - t1;
    // std::cout << "getting the bool array: " << dtx1.count() << "\n";

    // boolean array of whether this location in Natoms is a tunnel connection or not
    int num_tunnel_points = thrust::reduce(thrust::device, is_tunnel, is_tunnel + N_atom, 0); // sum([0, 1, 0, 0, 1...])
    gpuErrchk( cudaPeekAtLastError() );
    std::cout << "size of tunneling submatrix: " << num_tunnel_points << "\n";

    int *tunnel_indices; // [1, 4...]
    gpuErrchk( cudaMalloc((void **)&tunnel_indices, num_tunnel_points * sizeof(int)) ); 
    thrust::copy_if(thrust::device, is_tunnel_indices, is_tunnel_indices + gpubuf.N_, tunnel_indices, is_not_zero());
    
    // // debug
    // int *check_tunnel_inds = new int[num_tunnel_points];
    // gpuErrchk( cudaMemcpy(check_tunnel_inds, tunnel_indices, num_tunnel_points * sizeof(int), cudaMemcpyDeviceToHost) );
    // for (int i = 0; i < num_tunnel_points; i++)
    // {
    //     std::cout << check_tunnel_inds[i] << " ";
    // }
    // exit(1);
    // // end debug

    // **************************************************************************
    // 3. Assemble the sparsity pattern of the sparse neighbor matrix
    int Nfull = N_atom + 2;
    int matrix_size = Nfull; 

    // get the number of nonzeros per row
    int *neighbor_nnz_per_row_d;
    gpuErrchk( cudaMalloc((void **)&neighbor_nnz_per_row_d, matrix_size * sizeof(int)) );
    gpuErrchk( cudaMemset(neighbor_nnz_per_row_d, 0, matrix_size * sizeof(int)) );

    num_threads = 512;
    num_blocks = (matrix_size + num_threads - 1) / num_threads;
    calc_nnz_per_row_T_neighbor<<<num_blocks, num_threads>>>(gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z,
                                                             gpubuf.metal_types, gpubuf.atom_element, gpubuf.atom_charge, gpubuf.atom_CB_edge,
                                                             gpubuf.lattice, pbc, nn_dist, tol,
                                                             num_source_inj, num_ground_ext, num_layers_contact,
                                                             num_metals, matrix_size, neighbor_nnz_per_row_d);
    gpuErrchk( cudaPeekAtLastError() );

    // compute the row pointers with an inclusive sum:
    int *neighbor_row_ptr_d;
    gpuErrchk( cudaMalloc((void **)&neighbor_row_ptr_d, (matrix_size + 1 - 1) * sizeof(int)) );
    gpuErrchk( cudaMemset(neighbor_row_ptr_d, 0, (matrix_size + 1 - 1) * sizeof(int)) );
    
    void     *temp_storage_d = NULL;                                                          // determines temporary device storage requirements for inclusive prefix sum
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, neighbor_nnz_per_row_d, neighbor_row_ptr_d+1, matrix_size - 1); // subtract 1 to ignore the ground node
    gpuErrchk( cudaMalloc(&temp_storage_d, temp_storage_bytes) );                             // inclusive sum starting at second value to get the row ptr, which is the same as inclusive sum starting at first value and last value filled with nnz
    cub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes, neighbor_nnz_per_row_d, neighbor_row_ptr_d+1, matrix_size - 1);
    
    // get the number of nonzero elements:
    int neighbor_nnz;
    gpuErrchk( cudaMemcpy(&neighbor_nnz, neighbor_row_ptr_d + matrix_size - 1, sizeof(int), cudaMemcpyDeviceToHost) );
    std::cout << "\nsparse nnz: " << neighbor_nnz << std::endl;

    // assemble the column indices from 0 to Nsub (excluding the ground node)
    int *neighbor_col_indices_d;
    gpuErrchk( cudaMalloc((void **)&neighbor_col_indices_d, neighbor_nnz * sizeof(int)) );
    calc_col_idx_T_neighbor<<<num_blocks, num_threads>>>(gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z,
                                                         gpubuf.metal_types, gpubuf.atom_element, gpubuf.atom_charge, gpubuf.atom_CB_edge,
                                                         gpubuf.lattice, pbc, nn_dist, tol,
                                                         num_source_inj, num_ground_ext, num_layers_contact,
                                                         num_metals, matrix_size, neighbor_nnz_per_row_d,
                                                         neighbor_row_ptr_d, neighbor_col_indices_d);
    gpuErrchk( cudaPeekAtLastError() );

     // **************************************************************************
    // 4. Populate the entries of the sparse Natom matrix

    double *neighbor_data_d;
    gpuErrchk(cudaMalloc((void **)&neighbor_data_d, neighbor_nnz * sizeof(double)));
    gpuErrchk(cudaMemset(neighbor_data_d, 0, neighbor_nnz * sizeof(double)));

    num_threads = 512;
    num_blocks = (Nfull + num_threads - 1) / num_threads;
    populate_data_T_neighbor<<<num_blocks, num_threads>>>(gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z,
                                                          gpubuf.metal_types, gpubuf.atom_element, gpubuf.atom_charge, gpubuf.atom_CB_edge,
                                                          gpubuf.lattice, pbc, nn_dist, tol, high_G, low_G, loop_G,
                                                          Vd, m_e, V0,
                                                          num_source_inj, num_ground_ext, num_layers_contact,
                                                          num_metals, Nfull, neighbor_row_ptr_d, neighbor_col_indices_d, neighbor_data_d);
    gpuErrchk( cudaPeekAtLastError() );

    // output sparsity of neighbor connections
    int submatrix_size = Nfull - 1;
    dump_csr_matrix_txt(submatrix_size, neighbor_nnz, neighbor_row_ptr_d, neighbor_col_indices_d, neighbor_data_d, 0);
    std::cout << "dumped sparse neighbor matrix\n";

    // the Nsub matrix of just the sparse neighbor connections is contained in [neighbor_row_ptr_d, neighbor_col_indices_d, neighbor_data_d]

    // *************************************************************************************************************************************
    // 5. Populate the dense matrix corresponding to all of the tunnel connections, using tunnel_indices to index the atom attributes arrays

    double *tunnel_matrix_d;
    gpuErrchk(cudaMalloc((void **)&tunnel_matrix_d, num_tunnel_points * num_tunnel_points * sizeof(double)));
    gpuErrchk(cudaMemset(tunnel_matrix_d, 0, num_tunnel_points * num_tunnel_points * sizeof(double)));

    num_threads = 512;
    num_blocks = (num_tunnel_points + num_threads - 1) / num_threads;
    populate_data_T_tunnel<<<num_blocks, num_threads>>>(tunnel_matrix_d, gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z,
                                                        gpubuf.metal_types, gpubuf.atom_element, gpubuf.atom_charge, gpubuf.atom_CB_edge,
                                                        gpubuf.lattice, pbc, high_G, low_G, loop_G, nn_dist, m_e, V0,
                                                        num_source_inj, num_ground_ext, num_layers_contact, N_atom, num_tunnel_points, tunnel_indices,
                                                        num_metals, Vd, tol);
    gpuErrchk( cudaPeekAtLastError() );

    // **************************************************************************
    // 6. Reduce the diagonals
    double *diagonal_d;
    gpuErrchk( cudaMalloc((void **)&diagonal_d, Nfull * sizeof(double)) );
    gpuErrchk( cudaMemset(diagonal_d, 0, Nfull * sizeof(double) ) );


    // reduce the diagonal for the sparse banded matrix
    num_threads = 512;
    num_blocks = (Nfull + num_threads - 1) / num_threads;
    calc_diagonal_T_gpu<<<num_blocks, num_threads>>>(neighbor_col_indices_d, neighbor_row_ptr_d, neighbor_data_d, Nfull, diagonal_d);
    gpuErrchk( cudaPeekAtLastError() );

    // reduce the diagonal for the dense tunnel matrix
    double *tunnel_diag_d;
    gpuErrchk( cudaMalloc((void **)&tunnel_diag_d, num_tunnel_points * sizeof(double)) );                              // diagonal elements of the transmission matrix
    gpuErrchk( cudaMemset(tunnel_diag_d, 0, num_tunnel_points * sizeof(double)) );

    num_threads = 512;
    blocks_per_row = (num_tunnel_points - 1) / num_threads + 1;
    num_blocks = blocks_per_row * (N_atom + 2);

    row_reduce<NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS * sizeof(double)>>>(tunnel_matrix_d, tunnel_diag_d, num_tunnel_points);
    gpuErrchk( cudaPeekAtLastError() );

    write_to_diag<<<blocks_per_row, num_threads>>>(tunnel_matrix_d, tunnel_diag_d, num_tunnel_points);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();

    //diagonal_d contains already the diagonal of the neighbor matrix
    extract_diag_tunnel<<<blocks_per_row, num_threads>>>(
        tunnel_matrix_d,
        tunnel_indices, 
        num_tunnel_points,
        diagonal_d);
    inverse_vector<<<blocks_per_row, num_threads>>>(diagonal_d, Nfull);

    double *diagonal_inv_d = diagonal_d;


    // double *diagonal_inv_h = (double *)calloc(Nfull, sizeof(double));
    // gpuErrchk( cudaMemcpy(diagonal_inv_h, diagonal_inv_d, Nfull * sizeof(double), cudaMemcpyDeviceToHost) );
    // for (int i = 0; i < Nfull; i++){
    //     std::cout << diagonal_inv_h[i] << " ";
    // }   
    // std::cout << "\n";
    // exit(1);


    // the sparse matrix of the neighbor connectivity is contained in [neighbor_row_ptr_d, neighbor_col_indices_d, neighbor_data_d]
    // the dense matrix of the non-neighbor connectivity is contained in [tunnel_matrix_d] with size num_tunnel_points
    // To build the full matrix, row i and column i of tunnel_matrix_d should be added to row tunnel_indices[i] and col tunnel_indices[i] of the neighbor matrix

    //debug
    // double *cpu_T = new double[num_tunnel_points * num_tunnel_points];
    // cudaMemcpy(cpu_T, tunnel_matrix_d, sizeof(double) * num_tunnel_points * num_tunnel_points, cudaMemcpyDeviceToHost);
    // std::cout << "printing tunnel matrix\n";
    // std::ofstream fout2("T.txt");
    // int row, col;
    // for (row = 0; row < num_tunnel_points; row++) {
    // for (col = 0; col < num_tunnel_points; col++) {
    //     fout2 << cpu_T[row * num_tunnel_points + col] << ' ';
    // }
    // fout2 << '\n';
    // }
    // fout2.close(); 
    //debug end

    //debug
    // int *check_tunnel_inds = new int[num_tunnel_points];
    // gpuErrchk( cudaMemcpy(check_tunnel_inds, tunnel_indices, num_tunnel_points * sizeof(int), cudaMemcpyDeviceToHost) );
    // std::cout << "printing tunnel indices\n";
    // std::ofstream fout("insertion_indices.txt");
    // for (int i = 0; i < num_tunnel_points; i++)
    // {
    //     fout << check_tunnel_inds[i] << ' ';
    // }
    // fout.close(); 
    //debug end

    // results of debug: checked against the full sparse assembly by reassembling the matrix in a python script 

    std::cout << "matrix population is done\n";

    // **************************************************************************
    // 7. Prepare the RHS vector

    double *gpu_m;
    gpuErrchk( cudaMalloc((void **)&gpu_m, (N_atom + 2) * sizeof(double)) );                                 // [] current injection vector
    gpuErrchk( cudaMemset(gpu_m, 0, (N_atom + 2) * sizeof(double)) );                                                                         
    thrust::device_ptr<double> m_ptr = thrust::device_pointer_cast(gpu_m);
    thrust::fill(m_ptr, m_ptr + 1, -loop_G * Vd);                                                            // max Current extraction (ground)                          
    thrust::fill(m_ptr + 1, m_ptr + 2, loop_G * Vd);                                                         // max Current injection (source)
    cudaDeviceSynchronize();

    // ************************************************************
    // 8. Solve the system of linear equations 
    
    // the initial guess for the solution is the current site-resolved potential inside the device
    double *gpu_virtual_potentials;
    gpuErrchk( cudaMalloc((void **)&gpu_virtual_potentials, (N_atom + 2) * sizeof(double)) );                   // [V] Virtual potential vector  
    gpuErrchk( cudaMemset(gpu_virtual_potentials, 0, (N_atom + 2) * sizeof(double)) );                          // initialize the rhs for solving the system                                    
    
    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);
    cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_DEVICE);

    // sparse solver without preconditioning:
    int Nsub = Nfull - 1;
    solve_sparse_CG_splitmatrix(handle, cusparseHandle, tunnel_matrix_d, num_tunnel_points, 
                                neighbor_data_d, neighbor_row_ptr_d, neighbor_col_indices_d, neighbor_nnz, 
                                Nsub, tunnel_indices, gpu_m, gpu_virtual_potentials, diagonal_inv_d);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    double check_element;
    gpuErrchk( cudaMemcpy(&check_element, gpu_virtual_potentials + num_source_inj, sizeof(double), cudaMemcpyDeviceToHost) );
    if (std::abs(check_element - Vd) > 0.1)
    {
        std::cout << "WARNING: non-negligible potential drop of " << std::abs(check_element - Vd) <<
                    " across the contact at VD = " << Vd << "\n";
    }

    std::cout << "done system solve\n";
    // exit(1);

    // auto t4 = std::chrono::steady_clock::now();
    // std::chrono::duration<double> dt3 = t4 - t3;
    // std::cout << "time to solve linear system: " << dt3.count() << "\n";


    // // ****************************************************
    // // 3. Calculate the net current flowing into the device
    double *gpu_imacro;
    gpuErrchk( cudaMalloc((void **)&gpu_imacro, 1 * sizeof(double)) );                                       // [A] The macroscopic device current
    cudaDeviceSynchronize();


    // // scale the virtual potentials by G0 (conductance quantum) instead of multiplying inside the X matrix
    thrust::device_ptr<double> gpu_virtual_potentials_ptr = thrust::device_pointer_cast(gpu_virtual_potentials);
    thrust::transform(gpu_virtual_potentials_ptr, gpu_virtual_potentials_ptr + N_atom + 2, gpu_virtual_potentials_ptr, thrust::placeholders::_1 * G0);

    // // macroscopic device current
    gpuErrchk( cudaMemset(gpu_imacro, 0, sizeof(double)) ); 
    cudaDeviceSynchronize();

    // // dot product of first row of X[i] times M[0] - M[i]
    num_threads = 512;
    num_blocks = (N_atom - 1) / num_threads + 1;
    get_imacro_sparse<NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS * sizeof(double)>>>(
        neighbor_data_d, neighbor_row_ptr_d, neighbor_col_indices_d, gpu_virtual_potentials, gpu_imacro);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();

    gpuErrchk( cudaMemcpy(imacro, gpu_imacro, sizeof(double), cudaMemcpyDeviceToHost) );

    std::cout << solve_heating_local  << "\n";
    std::cout << solve_heating_global << "\n";

    //TODO does not work now
    // implement the heating calculation (possible from the splitting)
    // ineg would be possible the following way: -aij*xij so -aij xsparseij - aij xdenseij

    exit(1);

    // std::cout << "I_macro: " << *imacro * (1e6) << "\n";
    // std::cout << "exiting after I_macro\n"; exit(1);

    // cudaFree(X_data);
    // cudaFree(X_data_copy);
    // cudaFree(X_row_ptr);
    // cudaFree(X_col_indices);
    // cudaFree(gpu_virtual_potentials);
    // cudaFree(gpu_imacro);
    // cudaFree(gpu_m);
    // cudaFree(gpu_index);
    // cudaFree(atom_gpu_index);
}

// *** FULL SPARSE MATRIX VERSION ***



// does not assume that the column indices are sorted
__global__ void set_ineg_sparse(double *ineg_values, int *ineg_row_ptr, int *ineg_col_indices, const double *x_values, const int *x_row_ptr, const int *x_col_indices, const double *m, double Vd, int N)
{
    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;

    for (auto i = tid_total; i < N; i += num_threads_total)
    {
        for( int j = ineg_row_ptr[i]; j < ineg_row_ptr[i+1]; j++ )
        {
            if (ineg_col_indices[j] >= 2)
            {
                ineg_values[j] = 0.0;

                double ical = x_values[j] * (m[i + 2] - m[ineg_col_indices[j] + 2]);

                if (ical < 0 && Vd > 0)
                {
                    ineg_values[j] = -ical;
                }
                else if (ical > 0 && Vd < 0)
                {
                    ineg_values[j] = -ical;
                }
            }
        }
    }
}


// full sparse matrix assembly
void update_power_gpu_sparse(cublasHandle_t handle, cusolverDnHandle_t handle_cusolver, GPUBuffers &gpubuf, 
                             const int num_source_inj, const int num_ground_ext, const int num_layers_contact,
                             const double Vd, const int pbc, const double high_G, const double low_G, const double loop_G, const double G0, const double tol,
                             const double nn_dist, const double m_e, const double V0, int num_metals, double *imacro,
                             const bool solve_heating_local, const bool solve_heating_global, const double alpha_disp)
{
    auto t0 = std::chrono::steady_clock::now();

    // ***************************************************************************************
    // 1. Update the atoms array from the sites array using copy_if with is_defect as a filter
    int *gpu_index;
    int *atom_gpu_index;
    gpuErrchk( cudaMalloc((void **)&gpu_index, gpubuf.N_ * sizeof(int)) );                                           // indices of the site array
    gpuErrchk( cudaMalloc((void **)&atom_gpu_index, gpubuf.N_ * sizeof(int)) );                                      // indices of the atom array

    thrust::device_ptr<int> gpu_index_ptr = thrust::device_pointer_cast(gpu_index);
    thrust::sequence(gpu_index_ptr, gpu_index_ptr + gpubuf.N_, 0);

    double *last_atom = thrust::copy_if(thrust::device, gpubuf.site_x, gpubuf.site_x + gpubuf.N_, gpubuf.site_element, gpubuf.atom_x, is_defect());
    int N_atom = last_atom - gpubuf.atom_x;
    thrust::copy_if(thrust::device, gpubuf.site_y, gpubuf.site_y + gpubuf.N_, gpubuf.site_element, gpubuf.atom_y, is_defect());
    thrust::copy_if(thrust::device, gpubuf.site_z, gpubuf.site_z + gpubuf.N_, gpubuf.site_element, gpubuf.atom_z, is_defect());
    thrust::copy_if(thrust::device, gpubuf.site_charge, gpubuf.site_charge + gpubuf.N_, gpubuf.site_element, gpubuf.atom_charge, is_defect());
    thrust::copy_if(thrust::device, gpubuf.site_element, gpubuf.site_element + gpubuf.N_, gpubuf.site_element, gpubuf.atom_element, is_defect());
    thrust::copy_if(thrust::device, gpubuf.site_CB_edge, gpubuf.site_CB_edge + gpubuf.N_, gpubuf.site_element, gpubuf.atom_CB_edge, is_defect());
    thrust::copy_if(thrust::device, gpu_index, gpu_index + gpubuf.N_, gpubuf.site_element, atom_gpu_index, is_defect());

    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "time to update atom arrays: " << dt.count() << "\n";

    // ***************************************************************************************
    // 2. Assemble the transmission matrix (X) with both direct and tunnel connections and the
    // solution vector (M) which represents the current inflow/outflow
    // int N_full = N_atom + 2;                                                                               // number of atoms + injection node + extraction node
    int Nsub = N_atom + 1;                                                                                 // N_full minus the ground node which is cut from the graph

    // compute the index arrays to build the CSR representation of X (from 0 to Nsub):
    int *X_row_ptr;
    int *X_col_indices;
    int X_nnz = 0;
    Assemble_X_sparsity(N_atom, gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z,
                        gpubuf.metal_types, gpubuf.atom_element, gpubuf.atom_charge, gpubuf.atom_CB_edge,
                        gpubuf.lattice, pbc, nn_dist, tol, 
                        num_source_inj, num_ground_ext, num_layers_contact,
                        num_metals, &X_row_ptr, &X_col_indices, &X_nnz);

    // get the row indices for COO
    int *X_row_indices_h = new int[X_nnz];
    int *X_row_ptr_h = new int[N_atom + 2];

    gpuErrchk( cudaMemcpy(X_row_ptr_h, X_row_ptr, (N_atom + 2) * sizeof(int), cudaMemcpyDeviceToHost) );
    for(int i = 0; i < N_atom + 1; i++){
        for(int j = X_row_ptr_h[i]; j < X_row_ptr_h[i+1]; j++){
            X_row_indices_h[j] = i;
        }
    }
    int *X_row_indices;
    gpuErrchk( cudaMalloc((void **)&X_row_indices, X_nnz * sizeof(int)) );
    gpuErrchk( cudaMemcpy(X_row_indices, X_row_indices_h, X_nnz * sizeof(int), cudaMemcpyHostToDevice) );
    free(X_row_indices_h);
    free(X_row_ptr_h);
    
    auto t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt1 = t2 - t1;
    std::cout << "time to assemble X sparsity: " << dt1.count() << "\n";

    // Assemble the nonzero value array of X in CSR (from 0 to Nsub):
    double *X_data;                                                                                             // [1] Transmission matrix
    // Assemble_X(N_atom, gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z,
    //            gpubuf.metal_types, gpubuf.atom_element, gpubuf.atom_charge, gpubuf.atom_CB_edge,
    //            gpubuf.lattice, pbc, nn_dist, tol, Vd, m_e, V0, high_G, low_G, loop_G,
    //            num_source_inj, num_ground_ext, num_layers_contact,
    //            num_metals, &X_data, &X_row_ptr, &X_col_indices, &X_nnz);

    // double *X_data2;                                                                                          // [1] Transmission matrix
    Assemble_X2(N_atom, gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z,
                gpubuf.metal_types, gpubuf.atom_element, gpubuf.atom_charge, gpubuf.atom_CB_edge,
                gpubuf.lattice, pbc, nn_dist, tol, Vd, m_e, V0, high_G, low_G, loop_G,
                num_source_inj, num_ground_ext, num_layers_contact,
                num_metals, &X_data, &X_row_indices, &X_row_ptr, &X_col_indices, &X_nnz);

    // dump_csr_matrix_txt(Nsub, X_nnz, X_row_ptr, X_col_indices, X_data, 0); // figure out why the vector lengths are wrong according to the python script
    // std::cout << "dumped sparse matrix\n";
    // exit(1);
    
    // gpuErrchk( cudaFree(X_row_indices) );
    // double *X_data_h = new double[X_nnz];
    // double *X_data2_h = new double[X_nnz];
    // gpuErrchk( cudaMemcpy(X_data_h, X_data, X_nnz * sizeof(double), cudaMemcpyDeviceToHost) );
    // gpuErrchk( cudaMemcpy(X_data2_h, X_data2, X_nnz * sizeof(double), cudaMemcpyDeviceToHost) );

    // for (int i = 0; i < X_nnz; i++)
    // {

    //     // if (X_data_h[i] == X_data2_h[i])
    //     // {
    //     //     std::cout << "X_data match at index " << i << " with value " << X_data_h[i] << "\n";
    //     // }
    //     if (X_data_h[i] != X_data2_h[i])
    //     {
    //         std::cout << "X_data mismatch at index " << i << " with values " << X_data_h[i] << " and " << X_data2_h[i] << "\n";
    //     }
    // }

    auto t3 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt2 = t3 - t2;
    std::cout << "time to assemble X data: " << dt2.count() << "\n";

    double *gpu_imacro, *gpu_m;
    gpuErrchk( cudaMalloc((void **)&gpu_imacro, 1 * sizeof(double)) );                                       // [A] The macroscopic device current
    gpuErrchk( cudaMalloc((void **)&gpu_m, (N_atom + 2) * sizeof(double)) );                                 // [V] Virtual potential vector    
    cudaDeviceSynchronize();

    gpuErrchk( cudaMemset(gpu_m, 0, (N_atom + 2) * sizeof(double)) );                                        // initialize the rhs for solving the system                                    
    thrust::device_ptr<double> m_ptr = thrust::device_pointer_cast(gpu_m);
    thrust::fill(m_ptr, m_ptr + 1, -loop_G * Vd);                                                            // max Current extraction (ground)                          
    thrust::fill(m_ptr + 1, m_ptr + 2, loop_G * Vd);                                                         // max Current injection (source)
    cudaDeviceSynchronize();

    // ************************************************************
    // 2. Solve system of linear equations 
    
    // the initial guess for the solution is the current site-resolved potential inside the device
    double *gpu_virtual_potentials = gpubuf.atom_virtual_potentials;                                               // [V] Virtual potential vector  
    
    // making a copy so the original version won't be preconditioned inside the iterative solver
    double *X_data_copy;
    gpuErrchk( cudaMalloc((void **)&X_data_copy, X_nnz * sizeof(double)) );
    gpuErrchk( cudaMemcpyAsync(X_data_copy, X_data, X_nnz * sizeof(double), cudaMemcpyDeviceToDevice) );
    gpuErrchk( cudaDeviceSynchronize() );

    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);
    cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_DEVICE);

    // sparse solver with Jacobi preconditioning:
    // std::cout << "\n\n";
    solve_sparse_CG_Jacobi(handle, cusparseHandle, X_data_copy, X_row_ptr, X_col_indices, X_nnz, Nsub, gpu_m, gpu_virtual_potentials);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // std::cout << "exiting after system solve\n";
    // exit(1);

    double check_element;
    gpuErrchk( cudaMemcpy(&check_element, gpu_virtual_potentials + num_source_inj, sizeof(double), cudaMemcpyDeviceToHost) );
    if (std::abs(check_element - Vd) > 0.1)
    {
        std::cout << "WARNING: non-negligible potential drop of " << std::abs(check_element - Vd) <<
                    " across the contact at VD = " << Vd << "\n";
    }

    auto t4 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt3 = t4 - t3;
    std::cout << "time to solve linear system: " << dt3.count() << "\n";


    // ****************************************************
    // 3. Calculate the net current flowing into the device

    // scale the virtual potentials by G0 (conductance quantum) instead of multiplying inside the X matrix
    thrust::device_ptr<double> gpu_virtual_potentials_ptr = thrust::device_pointer_cast(gpu_virtual_potentials);
    thrust::transform(gpu_virtual_potentials_ptr, gpu_virtual_potentials_ptr + N_atom + 2, gpu_virtual_potentials_ptr, thrust::placeholders::_1 * G0);

    // macroscopic device current
    gpuErrchk( cudaMemset(gpu_imacro, 0, sizeof(double)) ); 
    cudaDeviceSynchronize();

    // dot product of first row of X[i] times M[0] - M[i]
    int num_threads = 512;
    int num_blocks = (N_atom - 1) / num_threads + 1;
    get_imacro_sparse<NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS * sizeof(double)>>>(X_data, X_row_ptr, X_col_indices, gpu_virtual_potentials, gpu_imacro);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();

    gpuErrchk( cudaMemcpy(imacro, gpu_imacro, sizeof(double), cudaMemcpyDeviceToHost) );

    auto t5 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt4 = t5 - t4;
    std::cout << "time to compute current: " << dt4.count() << "\n";

    // std::cout << "I_macro: " << *imacro * (1e6) << "\n";
    // std::cout << "exiting after I_macro\n"; exit(1);

    // **********************************************
    // 4. Calculate the dissipated power at each atom

if (solve_heating_local || solve_heating_global)
{    
    // Shift the virtual potential so that it is all positive, as we will take differences
    double min_index = thrust::min_element(thrust::device, gpu_virtual_potentials + 2, gpu_virtual_potentials + N_atom + 2) - gpu_virtual_potentials;
    num_threads = 512;
    num_blocks = (N_atom + 2 - 1) / num_threads + 1;
    update_m<<<num_blocks, num_threads>>>(gpu_virtual_potentials, min_index, N_atom + 2);
    gpuErrchk( cudaPeekAtLastError() );

    // Collect the forward currents into I_neg, the diagonals are once again the sum of each row
    int *ineg_row_ptr;                                                                                          // [A] Current inflow matrix
    int *ineg_col_indices;
    double *ineg_data;
    gpuErrchk( cudaMalloc((void**) &ineg_row_ptr, (N_atom + 1 + 1) * sizeof(int)) );
    gpuErrchk( cudaMalloc((void**) &ineg_col_indices, X_nnz * sizeof(int)) );
    gpuErrchk( cudaMalloc((void **)&ineg_data, X_nnz * sizeof(double)) );
    gpuErrchk( cudaMemcpyAsync(ineg_row_ptr, X_row_ptr, (N_atom + 1 + 1) * sizeof(int), cudaMemcpyDeviceToDevice) );
    gpuErrchk( cudaMemcpyAsync(ineg_col_indices, X_col_indices, X_nnz * sizeof(int), cudaMemcpyDeviceToDevice) );
    gpuErrchk( cudaMemset(ineg_data, 0, X_nnz*sizeof(double)) ); 
    cudaDeviceSynchronize();

    num_threads = 512;
    num_blocks = (Nsub - 1) / num_threads + 1;
    int N_atomsub = N_atom - 1;
    set_ineg_sparse<<<num_blocks, num_threads>>>(ineg_data, ineg_row_ptr, ineg_col_indices, X_data, X_row_ptr, X_col_indices, gpu_virtual_potentials, Vd, N_atomsub);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();

    // sum off-diagonals into diagonal:
    num_threads = 512;
    num_blocks = (Nsub - 1) / num_threads + 1;
    reduce_rows_into_diag<<<num_blocks, num_threads>>>(ineg_col_indices, ineg_row_ptr, ineg_data, Nsub);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();

    // Compute the dissipated power at each atom with [P]_Nx1 = [I]_NxN * [V]_Nx1 (gemv --> spmv)
    double *gpu_pdisp;
    gpuErrchk( cudaMalloc((void **)&gpu_pdisp, N_atom * sizeof(double)) );                                   // [W] Dissipated power vector
    gpuErrchk( cudaMemset(gpu_pdisp, 0, N_atom*sizeof(double)) ); 

    cusparseStatus_t status;
    cusparseSpMatDescr_t mat_ineg;
    status = cusparseCreateCsr(&mat_ineg, Nsub, Nsub, X_nnz, ineg_row_ptr, ineg_col_indices, ineg_data, 
                               CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout << "ERROR: creation of sparse matrix descriptor in update_power_gpu_sparse() failed!\n";
    }
    cusparseDnVecDescr_t vec_virtual_potentials, vec_pdisp;
    cusparseCreateDnVec(&vec_virtual_potentials, Nsub, gpu_virtual_potentials, CUDA_R_64F);
    cusparseCreateDnVec(&vec_pdisp, Nsub, gpu_pdisp, CUDA_R_64F);

    size_t MVBufferSize;
    void *MVBuffer = 0;
    double *one_d, *zero_d;
    double one = 1.0;
    double zero = 0.0;
    gpuErrchk( cudaMalloc((void**)&one_d, sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&zero_d, sizeof(double)) );
    gpuErrchk( cudaMemcpy(one_d, &one, sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(zero_d, &zero, sizeof(double), cudaMemcpyHostToDevice) );

    status = cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, one_d, mat_ineg, 
                                     vec_virtual_potentials, zero_d, vec_pdisp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &MVBufferSize);  
    gpuErrchk( cudaMalloc((void**)&MVBuffer, sizeof(double) * MVBufferSize) );
    status = cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, one_d, mat_ineg,                         
                          vec_virtual_potentials, zero_d, vec_pdisp, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, MVBuffer);          
    
    // copy the dissipated power into the site attributes
    num_threads = 512;
    num_blocks = (N_atom - 1) / num_threads + 1;
    num_blocks = min(65535, num_blocks);
    copy_pdisp<<<num_blocks, num_threads>>>(gpubuf.site_power, gpubuf.site_element, gpubuf.metal_types, gpu_pdisp, atom_gpu_index, N_atom, num_metals, alpha_disp);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();

    // !!! the dissipated power does not yet perfectly match the dense version !!!
    // !!! there is probably a small expected change due to removing the ground node, but this should be double checked !!!
    
    // double *host_pdisp = new double[N_atom];
    // cudaMemcpy(host_pdisp, gpu_pdisp, N_atom * sizeof(double), cudaMemcpyDeviceToHost);
    // double sum = 0.0;
    // for (int i = 0; i < N_atom; ++i) {
    //     sum += host_pdisp[i];
    // }
    // std::cout << "Sum of atom-resolved power: " << sum << std::endl;
    // exit(1);

    cudaFree(ineg_row_ptr);
    cudaFree(ineg_col_indices);
    cudaFree(ineg_data);
    cudaFree(gpu_pdisp);
    cudaFree(MVBuffer); 
    cudaFree(one_d);
    cudaFree(zero_d);
}

    cudaFree(X_data);
    cudaFree(X_data_copy);
    cudaFree(X_row_ptr);
    cudaFree(X_row_indices);
    cudaFree(X_col_indices);
    cudaFree(gpu_imacro);
    cudaFree(gpu_m);
    cudaFree(gpu_index);
    cudaFree(atom_gpu_index);
}

// *** DENSE MATRIX VERSION ***

__global__ void create_X(
    double *X,
    const double *posx, const double *posy, const double *posz,
    const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
    const double *lattice, bool pbc, double high_G, double low_G, double loop_G,
    double nn_dist, double m_e, double V0, int num_source_inj, int num_ground_ext, const int num_layers_contact,
    int N, int num_metals, const double Vd, const double tol)
{

    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;

    int N_full = N + 2;

    // TODO: Does it make sense to restructure for N_full * N_full threads?
    for (auto idx = tid_total; idx < N * N; idx += num_threads_total)
    {
        int i = idx / N;
        int j = idx % N;

        double dist_angstrom = site_dist_gpu(posx[i], posy[i], posz[i], 
                                             posx[j], posy[j], posz[j], 
                                             lattice[0], lattice[1], lattice[2], pbc);

        bool neighbor = (dist_angstrom < nn_dist) && (i != j);

        // tunneling terms occur between not-neighbors
        if (i != j && !neighbor)
        { 
            bool any_vacancy1 = element[i] == VACANCY;
            bool any_vacancy2 = element[j] == VACANCY;

            // contacts, excluding the last layer 
            bool metal1p = is_in_array_gpu(metals, element[i], num_metals) 
                                       && (i > ((num_layers_contact - 1)*num_source_inj))
                                       && (i < (N - (num_layers_contact - 1)*num_ground_ext)); 

            bool metal2p = is_in_array_gpu(metals, element[j], num_metals)
                                       && (j > ((num_layers_contact - 1)*num_source_inj))
                                       && (j < (N - (num_layers_contact - 1)*num_ground_ext));  

            // types of tunnelling conditions considered
            bool trap_to_trap = (any_vacancy1 && any_vacancy2);
            bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
            bool contact_to_contact = (metal1p && metal2p);

            double local_E_drop = atom_CB_edge[i] - atom_CB_edge[j];                // [eV] difference in energy between the two atoms

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
                    X[N_full * (i + 2) + (j + 2)] = -T;      
                } 
                else 
                {
                    double E1 = eV_to_J * V0;                                        // [J] Energy distance to CB before tunnelling
                    double E2 = E1 - fabs(local_E_drop);                             // [J] Energy distance to CB after tunnelling
                          
                    if (E2 > 0)                                                      // trapezoidal potential barrier (low field)
                    {                                                           
                        double T = exp(prefac * (dist / fabs(E1 - E2)) * ( pow(E1, 1.5) - pow(E2, 1.5) ) );
                        X[N_full * (i + 2) + (j + 2)] = -T; 
                    }

                    if (E2 < 0)                                                        // triangular potential barrier (high field)
                    {
                        double T = exp(prefac * (dist / fabs(E1 - E2)) * ( pow(E1, 1.5) ));
                        X[N_full * (i + 2) + (j + 2)] = -T; 
                    }
                }
            }
        }

        // direct terms occur between neighbors 
        if (i != j && neighbor)
        {
            // contacts
            bool metal1 = is_in_array_gpu(metals, element[i], num_metals);
            bool metal2 = is_in_array_gpu(metals, element[j], num_metals);

            // conductive vacancy sites
            bool cvacancy1 = (element[i] == VACANCY) && (atom_charge[i] == 0);
            bool cvacancy2 = (element[j] == VACANCY) && (atom_charge[j] == 0);

            if ((metal1 && metal2) || (cvacancy1 && cvacancy2))
            {
                X[N_full * (i + 2) + (j + 2)] = -high_G;
            }
            else
            {
                X[N_full * (i + 2) + (j + 2)] = -low_G;
            }
        }

        // NOTE: Is there a data race here?
        // connect the source/ground nodes to the first/last contact layers
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

        if (i == 0 && j == 0)
        {
            X[0 * N_full + 1] = -loop_G;
            X[1 * N_full + 0] = -loop_G;
        }
    }
}


template <int NTHREADS>
__global__ void get_imacro(const double *x, const double *m, double *imacro, int N)
{
    int num_threads = blockDim.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int total_tid = bid * num_threads + tid;

    __shared__ double buf[NTHREADS];

    buf[tid] = 0.0;

    if ((total_tid >= 0 && total_tid < N) && (total_tid >= 2)) 
    {
        buf[tid] = x[(N + 2) * 0 + (total_tid + 2)] * (m[0] - m[total_tid + 2]);            // extracted (M[0] = 0)
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

__global__ void set_ineg(double *ineg, const double *x, const double *m, double Vd, int N)
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

        ineg[i * N + j] = 0.0;
        double ical = x[(N + 2) * (i + 2) + (j + 2)] * (m[i + 2] - m[j + 2]);
        
        if (ical < 0 && Vd > 0)
        {
            ineg[i * N + j] = -ical;
        }
        else if (ical > 0 && Vd < 0)
        { 
            ineg[i * N + j] = -ical;
        }
    }
}


void update_power_gpu(cublasHandle_t handle, cusolverDnHandle_t handle_cusolver, GPUBuffers &gpubuf, 
                      const int num_source_inj, const int num_ground_ext, const int num_layers_contact,
                      const double Vd, const int pbc, const double high_G, const double low_G, const double loop_G, const double G0, const double tol,
                      const double nn_dist, const double m_e, const double V0, int num_metals, double *imacro,
                      const bool solve_heating_local, const bool solve_heating_global, const double alpha_disp)
{

    // ***************************************************************************************
    // 1. Update the atoms array from the sites array using copy_if with is_defect as a filter
    int *gpu_index;
    int *atom_gpu_index;
    gpuErrchk( cudaMalloc((void **)&gpu_index, gpubuf.N_ * sizeof(int)) );                                           // indices of the site array
    gpuErrchk( cudaMalloc((void **)&atom_gpu_index, gpubuf.N_ * sizeof(int)) );                                      // indices of the atom array

    thrust::device_ptr<int> gpu_index_ptr = thrust::device_pointer_cast(gpu_index);
    thrust::sequence(gpu_index_ptr, gpu_index_ptr + gpubuf.N_, 0);

    double *last_atom = thrust::copy_if(thrust::device, gpubuf.site_x, gpubuf.site_x + gpubuf.N_, gpubuf.site_element, gpubuf.atom_x, is_defect());
    int N_atom = last_atom - gpubuf.atom_x;
    thrust::copy_if(thrust::device, gpubuf.site_y, gpubuf.site_y + gpubuf.N_, gpubuf.site_element, gpubuf.atom_y, is_defect());
    thrust::copy_if(thrust::device, gpubuf.site_z, gpubuf.site_z + gpubuf.N_, gpubuf.site_element, gpubuf.atom_z, is_defect());
    thrust::copy_if(thrust::device, gpubuf.site_charge, gpubuf.site_charge + gpubuf.N_, gpubuf.site_element, gpubuf.atom_charge, is_defect());
    thrust::copy_if(thrust::device, gpubuf.site_element, gpubuf.site_element + gpubuf.N_, gpubuf.site_element, gpubuf.atom_element, is_defect());
    thrust::copy_if(thrust::device, gpubuf.site_CB_edge, gpubuf.site_CB_edge + gpubuf.N_, gpubuf.site_element, gpubuf.atom_CB_edge, is_defect());
    thrust::copy_if(thrust::device, gpu_index, gpu_index + gpubuf.N_, gpubuf.site_element, atom_gpu_index, is_defect());

    // ***************************************************************************************
    // 2. Assemble the transmission matrix (X) with both direct and tunnel connections and the
    // solution vector (M) which represents the current inflow/outflow

    // USE SIZE_T FOR ALLOCATIONS
    double *gpu_imacro, *gpu_m, *gpu_x, *gpu_ineg, *gpu_diag, *gpu_pdisp, *gpu_A;
    gpuErrchk( cudaMalloc((void **)&gpu_imacro, 1 * sizeof(double)) );                                       // [A] The macroscopic device current
    gpuErrchk( cudaMalloc((void **)&gpu_m, (N_atom + 2) * sizeof(double)) );                                 // [V] Virtual potential vector    
    gpuErrchk( cudaMalloc((void **)&gpu_x, (N_atom + 2) * (N_atom + 2) * sizeof(double)) );                  // [1] Transmission matrix
    gpuErrchk( cudaMalloc((void **)&gpu_ineg, N_atom * N_atom * sizeof(double)) );                           // [A] Current inflow matrix
    gpuErrchk( cudaMalloc((void **)&gpu_diag, (N_atom + 2) * sizeof(double)) );                              // diagonal elements of the transmission matrix
    gpuErrchk( cudaMalloc((void **)&gpu_pdisp, N_atom * sizeof(double)) );                                   // [W] Dissipated power vector
    gpuErrchk( cudaMalloc((void **)&gpu_A, (N_atom + 1) * (N_atom + 1) * sizeof(double)) );                  // A - copy buffer for the transmission matrix
    cudaDeviceSynchronize();

    gpuErrchk( cudaMemset(gpu_x, 0, (N_atom + 2) * (N_atom + 2) * sizeof(double)) );                         // initialize the transmission matrix to zeros
    gpuErrchk( cudaMemset(gpu_m, 0, (N_atom + 2) * sizeof(double)) );                                        // initialize the rhs for solving the system                                    
    thrust::device_ptr<double> m_ptr = thrust::device_pointer_cast(gpu_m);
    thrust::fill(m_ptr, m_ptr + 1, -loop_G * Vd);                                               // max Current extraction (ground)                          
    thrust::fill(m_ptr + 1, m_ptr + 2, loop_G * Vd);                                            // max Current injection (source)
    cudaDeviceSynchronize();

    int num_threads = 128;
    int blocks_per_row = (N_atom - 1) / num_threads + 1;
    int num_blocks = blocks_per_row * gpubuf.N_;

    // fill off diagonals of X
    create_X<<<num_blocks, num_threads>>>(
        gpu_x, gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z,
        gpubuf.metal_types, gpubuf.atom_element, gpubuf.atom_charge, gpubuf.atom_CB_edge,
        gpubuf.lattice, pbc, high_G, low_G, loop_G,
        nn_dist, m_e, V0, num_source_inj, num_ground_ext, num_layers_contact,
        N_atom, num_metals, Vd, tol);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();

    // fill diagonal of X (all rows sum to zero)
    gpuErrchk( cudaMemset(gpu_diag, 0, (N_atom + 2) * sizeof(double)) );
    num_threads = 512;
    blocks_per_row = (N_atom + 2 - 1) / num_threads + 1;
    num_blocks = blocks_per_row * (gpubuf.N_ + 2);
    row_reduce<NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS * sizeof(double)>>>(gpu_x, gpu_diag, N_atom + 2);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();
    write_to_diag<<<blocks_per_row, num_threads>>>(gpu_x, gpu_diag, N_atom + 2);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();

    // ************************************************************
    // 2. Solve system of linear equations using LU (direct solver)

    int lwork = 0;              /* size of workspace */
    double *gpu_work = nullptr; /* device workspace for getrf */
    int *gpu_info = nullptr;    /* error info */
    int *gpu_ipiv;

    gpuErrchk( cudaMalloc((void **)&gpu_ipiv, (N_atom + 1) * sizeof(int)) );
    gpuErrchk( cudaMalloc((void **)(&gpu_info), sizeof(int)) );
    gpuErrchk( cudaMemcpy2D(gpu_A, (N_atom + 1) * sizeof(double), gpu_x, (N_atom + 2) * sizeof(double), (N_atom + 1) * sizeof(double), (N_atom + 1), cudaMemcpyDeviceToDevice) );
    cudaDeviceSynchronize();

    // Solve Ax=B through LU factorization
    CheckCusolverDnError(cusolverDnDgetrf_bufferSize(handle_cusolver, N_atom + 1, N_atom + 1, gpu_A, N_atom + 1, &lwork));
    gpuErrchk( cudaMalloc((void **)(&gpu_work), sizeof(double) * lwork) );
    cudaDeviceSynchronize();
    CheckCusolverDnError(cusolverDnDgetrf(handle_cusolver, N_atom + 1, N_atom + 1, gpu_A, N_atom + 1, gpu_work, gpu_ipiv, gpu_info));
    cudaDeviceSynchronize();
    CheckCusolverDnError(cusolverDnDgetrs(handle_cusolver, CUBLAS_OP_T, N_atom + 1, 1, gpu_A, N_atom + 1, gpu_ipiv, gpu_m, N_atom + 1, gpu_info));
    cudaDeviceSynchronize();

    int host_info;
    gpuErrchk( cudaMemcpy(&host_info, gpu_info, sizeof(int), cudaMemcpyDeviceToHost) );
    if (host_info)
    {
        std::cout << "WARNING: Info for gesv in update_power is " << host_info << "\n";
    }

    double check_element;
    gpuErrchk( cudaMemcpy(&check_element, gpu_m + num_source_inj, sizeof(double), cudaMemcpyDeviceToHost) );
    if (std::abs(check_element - Vd) > 0.1)
    {
        std::cout << "WARNING: non-negligible potential drop of " << std::abs(check_element - Vd) <<
                    " across the contact at VD = " << Vd << "\n";
    }

    // scale the virtual potentials by G0 (conductance quantum) instead of multiplying inside the X matrix
    thrust::device_ptr<double> gpu_m_ptr = thrust::device_pointer_cast(gpu_m);
    thrust::transform(gpu_m_ptr, gpu_m_ptr + N_atom + 1, gpu_m_ptr, thrust::placeholders::_1 * G0);

    // ****************************************************
    // 3. Calculate the net current flowing into the device

    num_threads = 512;
    num_blocks = (N_atom - 1) / num_threads + 1;
    gpuErrchk( cudaMemset(gpu_imacro, 0, sizeof(double)) ); 
    get_imacro<NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS * sizeof(double)>>>(gpu_x, gpu_m, gpu_imacro, N_atom);
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();
    gpuErrchk( cudaMemcpy(imacro, gpu_imacro, sizeof(double), cudaMemcpyDeviceToHost) );
    std::cout << "I_macro: " << *imacro * (1e6) << "\n";

    // **********************************************
    // 4. Calculate the dissipated power at each atom

if (solve_heating_local || solve_heating_global)
{   
        // Shift the virtual potential so that it is all positive, as we will take differences
        double min_index = thrust::min_element(thrust::device, gpu_m + 2, gpu_m + N_atom + 2) - gpu_m;
        num_threads = 512;
        blocks_per_row = (N_atom + 2 - 1) / num_threads + 1;
        num_blocks = blocks_per_row;
        update_m<<<num_blocks, num_threads>>>(gpu_m, min_index, N_atom + 2);
        gpuErrchk( cudaPeekAtLastError() );

        // Collect the forward currents into I_neg, the diagonals are once again the sum of each row
        num_threads = 512;
        blocks_per_row = (N_atom - 1) / num_threads + 1;
        num_blocks = blocks_per_row * gpubuf.N_;
        set_ineg<<<num_blocks, num_threads>>>(gpu_ineg, gpu_x, gpu_m, Vd, N_atom);
        gpuErrchk( cudaPeekAtLastError() );
        cudaDeviceSynchronize();
        gpuErrchk( cudaMemset(gpu_diag, 0, (N_atom + 2) * sizeof(double)) );
        cudaDeviceSynchronize();
        row_reduce<NUM_THREADS><<<num_blocks, num_threads, NUM_THREADS * sizeof(double)>>>(gpu_ineg, gpu_diag, N_atom);
        gpuErrchk( cudaPeekAtLastError() );
        cudaDeviceSynchronize();
        write_to_diag<<<blocks_per_row, num_threads>>>(gpu_ineg, gpu_diag, N_atom);
        gpuErrchk( cudaPeekAtLastError() );
        cudaDeviceSynchronize();

        // Compute the dissipated power at each atom with [P]_Nx1 = [I]_NxN * [V]_Nx1 (gemv)
        double alpha = 1.0, beta = 0.0;
        CheckCublasError( cublasDgemv(handle, CUBLAS_OP_T, N_atom, N_atom, &alpha, gpu_ineg, N_atom, gpu_m + 2, 1, &beta, gpu_pdisp, 1) );
        cudaDeviceSynchronize();

        // Extract the power dissipated between the contacts
        num_threads = 512;
        num_blocks = (N_atom - 1) / num_threads + 1;
        num_blocks = min(65535, num_blocks);
        copy_pdisp<<<num_blocks, num_threads>>>(gpubuf.site_power, gpubuf.site_element, gpubuf.metal_types, gpu_pdisp, atom_gpu_index, N_atom, num_metals, alpha_disp);
        gpuErrchk( cudaPeekAtLastError() );
        cudaDeviceSynchronize();

        double *host_pdisp = new double[N_atom];
        cudaMemcpy(host_pdisp, gpu_pdisp, N_atom * sizeof(double), cudaMemcpyDeviceToHost);
        double sum = 0.0;
        for (int i = 0; i < N_atom; ++i) {
            sum += host_pdisp[i];
        }
        std::cout << "Sum of atom-resolved power * 1e9: " << sum*(1e9) << std::endl;
        // exit(1);
} // if (solve_heating_local || solve_heating_global)

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
    cudaFree(gpu_index);
    cudaFree(atom_gpu_index);
}