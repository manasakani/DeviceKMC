#include "gpu_solvers.h"
#include <hip/hip_runtime.h>
#include <stdio.h>

const double eV_to_J = 1.60217663e-19;          // [C]
const double h_bar = 1.054571817e-34;           // [Js]

// Compute the number of nonzeros per row of the matrix including the injection, extraction, and device nodes (excluding the ground). 
// Has dimensions of Nsub by Nsub (by the cpu code)
__global__ void calc_nnz_per_row_T( const double *posx_d, const double *posy_d, const double *posz_d,
                                    const ELEMENT *metals, const ELEMENT *element, const double *atom_CB_edge, const double *lattice, bool pbc,
                                    double nn_dist, const double tol,
                                    int num_source_inj, int num_ground_ext, const int num_layers_contact,
                                    int num_metals, int N_sub,
                                    int block_size_i, int block_size_j, int block_start_i, int block_start_j, int *nnz_per_row_d){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Natom = N_sub - 1; 

    for(int row = idx; row < block_size_i; row += blockDim.x * gridDim.x){  // Nsub x Nsub matrix divided into blocks by rank
        int nnz_row = 0;
        for(int col = 0; col < block_size_j; col++){

            int i = block_start_i + row; // i indexes the matrix and i-2 indexes the atoms
            int j = block_start_j + col; 

            if ( i == j ) // all diagonal terms
            {
                nnz_row++;
            }

            if ( (i == 0 && j == 1)  || (i == 1 && j == 0) ) // loop connection
            {
                nnz_row++;
            }

            if ( i == 0 && ( j > ((N_sub+1) - num_ground_ext) )) // extraction terms minus ground node
            {
                nnz_row++;
            }

            if ( i == 1 && (j > 1) && (j < num_source_inj+2) ) // injection terms minus ground node
            {
                nnz_row++;
            }

            if (i > 1 && i != j)
            {
                // inj/ext terms
                if ( (j == 0) && ( i > ((N_sub+1) - num_ground_ext) ) )
                {
                    nnz_row++;
                }

                if ( (j == 1) && (i > 1) && (i < num_source_inj + 2) ) 
                {
                    nnz_row++;
                }

                double dist = site_dist_gpu(posx_d[i-2], posy_d[i-2], posz_d[i-2],
                                            posx_d[j-2], posy_d[j-2], posz_d[j-2],
                                            lattice[0], lattice[1], lattice[2], pbc);

                // direct terms 
                if ( j > 1 && i != j && dist < nn_dist )
                {
                    nnz_row++;
                }

                // // tunneling terms 
                // if ( i != j && dist > nn_dist )
                // { 
                //     bool any_vacancy1 = element[i-2] == VACANCY;
                //     bool any_vacancy2 = element[j-2] == VACANCY;

                //     // contacts, excluding the last layer 
                //     bool metal1p = is_in_array_gpu(metals, element[i-2], num_metals) 
                //                                     && (i-2 > ((num_layers_contact - 1)*num_source_inj))
                //                                     && (i-2 < (Natom - (num_layers_contact - 1)*num_ground_ext)); 

                //     bool metal2p = is_in_array_gpu(metals, element[j-2], num_metals)
                //                                     && (j-2 > ((num_layers_contact - 1)*num_source_inj))
                //                                     && (j-2 < (Natom - (num_layers_contact - 1)*num_ground_ext));  

                //     // types of tunnelling conditions considered
                //     bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                //     bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                //     bool contact_to_contact = (metal1p && metal2p);
                //     double local_E_drop = atom_CB_edge[i-2] - atom_CB_edge[j-2];                

                //     if ((trap_to_trap || contact_to_trap || contact_to_contact)  && (fabs(local_E_drop) > tol))
                //     {
                //         nnz_row++;
                //     }
                // }
            }
        }
        __syncthreads();
        atomicAdd(&nnz_per_row_d[row], nnz_row); 
    }

}


__global__ void assemble_T_col_indices(const double *posx_d, const double *posy_d, const double *posz_d,
                                        const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                                        const double *lattice, bool pbc, double nn_dist, const double tol,
                                        int num_source_inj, int num_ground_ext, const int num_layers_contact,
                                        int num_metals, int N_sub, int *row_ptr_d, int *col_indices_d,
                                        int block_size_i, int block_size_j, int block_start_i, int block_start_j)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Natom = N_sub - 1; 

    for(int row = idx; row < block_size_i; row += blockDim.x * gridDim.x){  // Nsub x Nsub matrix divided into blocks by rank
        int nnz_row = 0;
        for(int col = 0; col < block_size_j; col++){

            int i = block_start_i + row; // this is where this rank's rows start, i indexes the matrix and i-2 indexes the atoms
            int j = block_start_j + col; 

            if ( i == j ) // all diagonal terms
            {
                col_indices_d[row_ptr_d[row] + nnz_row] = col;
                nnz_row++;
            }

            if ( (i == 0 && j == 1)  || (i == 1 && j == 0) ) // loop connection
            {
                col_indices_d[row_ptr_d[row] + nnz_row] = col;
                nnz_row++;
            }

            if ( i == 0 && ( j > ((N_sub+1) - num_ground_ext) )) // extraction terms minus ground node
            {
                col_indices_d[row_ptr_d[row] + nnz_row] = col;
                nnz_row++;
            }

            if ( i == 1 && (j > 1) && (j < num_source_inj+2) ) // injection terms minus ground node
            {
                col_indices_d[row_ptr_d[row] + nnz_row] = col;
                nnz_row++;
            }

            if (i > 1 && i != j)
            {
                // source/ground terms
                if ( (j == 1) && (i > 1) && (i < num_source_inj + 2) ) 
                {
                    col_indices_d[row_ptr_d[row] + nnz_row] = col;
                    nnz_row++;
                }
                if ( (j == 0) && ( i > ((N_sub+1) - num_ground_ext) ) )
                {
                    col_indices_d[row_ptr_d[row] + nnz_row] = col;
                    nnz_row++;
                }

                double dist = site_dist_gpu(posx_d[i-2], posy_d[i-2], posz_d[i-2],
                                            posx_d[j-2], posy_d[j-2], posz_d[j-2]);

                // direct terms 
                if ( i != j && j > 1 && dist < nn_dist )
                {
                    col_indices_d[row_ptr_d[row] + nnz_row] = col;
                    nnz_row++;
                }

                // tunneling terms 
                // if ( i != j && dist > nn_dist )
                // { 
                //     bool any_vacancy1 = element[i-2] == VACANCY;
                //     bool any_vacancy2 = element[j-2] == VACANCY;

                //     // contacts, excluding the last layer 
                //     bool metal1p = is_in_array_gpu(metals, element[i-2], num_metals) 
                //                                     && (i-2 > ((num_layers_contact - 1)*num_source_inj))
                //                                     && (i-2 < (Natom - (num_layers_contact - 1)*num_ground_ext)); 

                //     bool metal2p = is_in_array_gpu(metals, element[j-2], num_metals)
                //                                     && (j-2 > ((num_layers_contact - 1)*num_source_inj))
                //                                     && (j-2 < (Natom - (num_layers_contact - 1)*num_ground_ext));  

                //     // types of tunnelling conditions considered
                //     bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                //     bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                //     bool contact_to_contact = (metal1p && metal2p);
                //     double local_E_drop = atom_CB_edge[i-2] - atom_CB_edge[j-2];                

                //     if ((trap_to_trap || contact_to_trap || contact_to_contact)  && (fabs(local_E_drop) > tol))
                //     {
                //         col_indices_d[row_ptr_d[row] + nnz_row] = col;
                //         nnz_row++;
                //     }
                // }
            }
        }
    }
}


__global__ void calc_nnz_per_row_tunnel(const double *posx, const double *posy, const double *posz,
                                        const double *atom_CB_edge, const ELEMENT *element, const int *atom_charge,
                                        const double nn_dist, const double tol, int *tunnel_indices_global, 
                                        int num_layers_contact, int num_source_inj, int num_ground_ext,
                                        const ELEMENT *metals, int num_metals, int num_tunnel_points, int Natom,
                                        int block_size, int block_start, int *nnz_per_row_d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;
    int Nsub = Natom + 1;

    for(int row_id = idx; row_id < block_size; row_id += blockDim.x * gridDim.x){  // Nsub x Nsub matrix divided into blocks by rank
        int nnz_row = 0;
        for(int col_id = 0; col_id < num_tunnel_points; col_id++){

                int i = block_start + row_id; // this is where this rank's rows start, i indexes the matrix 
                int j = col_id; 

                int ind_i = tunnel_indices_global[i];
                int ind_j = tunnel_indices_global[j];

                //  ACCCESS INDIRECTLY THOUGH TUNNEL INDICES
                double posx_i = posx[ind_i];
                double posx_j = posx[ind_j];

                double posy_i = posy[ind_i];
                double posy_j = posy[ind_j];

                double posz_i = posz[ind_i];
                double posz_j = posz[ind_j];

                double atom_CB_edge_i = atom_CB_edge[ind_i];
                double atom_CB_edge_j = atom_CB_edge[ind_j];

                ELEMENT element_i = element[ind_i];
                ELEMENT element_j = element[ind_j];



                double dist = site_dist_gpu(posx_i, posy_i, posz_i,
                                            posx_j, posy_j, posz_j);

                // all diagonal terms
                if ( i == j )
                {
                    nnz_row++;
                }

                // tunneling terms 
                if ( i != j && dist > nn_dist )
                { 
                    bool any_vacancy1 = element_i == VACANCY;
                    bool any_vacancy2 = element_j == VACANCY;

                    // contacts, excluding the last layer 
                    bool metal1p = is_in_array_gpu(metals, element_i, num_metals)
                                                && (ind_i > ((num_layers_contact - 1)*num_source_inj))
                                                && (ind_i < (Natom - (num_layers_contact - 1)*num_ground_ext)); 

                    bool metal2p = is_in_array_gpu(metals, element_j, num_metals)
                                                && (ind_j > ((num_layers_contact - 1)*num_source_inj))
                                                && (ind_j < (Natom - (num_layers_contact - 1)*num_ground_ext));  

                    // types of tunnelling conditions considered
                    bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                    bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                    bool contact_to_contact = (metal1p && metal2p);
                    double local_E_drop = atom_CB_edge_i - atom_CB_edge_j;                

                    if ((trap_to_trap || contact_to_trap || contact_to_contact)  && (fabs(local_E_drop) > tol))
                    {
                        nnz_row++;
                    }
                }
        }
        nnz_per_row_d[row_id] = nnz_row; 
    }

}

__global__ void assemble_tunnel_col_indices(const double *posx, const double *posy, const double *posz,
                                            const double *atom_CB_edge, const ELEMENT *element, const int *atom_charge,
                                            const double nn_dist, const double tol, int *tunnel_indices_global, 
                                            int num_layers_contact, int num_source_inj, int num_ground_ext,
                                            const ELEMENT *metals, int num_metals, int num_tunnel_points, int Natom,
                                            int block_size, int block_start, int *row_ptr_d, int *col_indices_d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;
    int Nsub = Natom + 1;

    for(int row_id = idx; row_id < block_size; row_id += blockDim.x * gridDim.x){  // Nsub x Nsub matrix divided into blocks by rank
        int nnz_row = 0;
        for(int col_id = 0; col_id < num_tunnel_points; col_id++){

                int i = block_start + row_id; // this is where this rank's rows start, i indexes the matrix 
                int j = col_id; 

                int ind_i = tunnel_indices_global[i];
                int ind_j = tunnel_indices_global[j];

                //  ACCCESS INDIRECTLY THOUGH TUNNEL INDICES
                double posx_i = posx[ind_i];
                double posx_j = posx[ind_j];

                double posy_i = posy[ind_i];
                double posy_j = posy[ind_j];

                double posz_i = posz[ind_i];
                double posz_j = posz[ind_j];

                double atom_CB_edge_i = atom_CB_edge[ind_i];
                double atom_CB_edge_j = atom_CB_edge[ind_j];

                ELEMENT element_i = element[ind_i];
                ELEMENT element_j = element[ind_j];



                double dist = site_dist_gpu(posx_i, posy_i, posz_i,
                                            posx_j, posy_j, posz_j);

                // all diagonal terms
                if ( i == j )
                {
                    col_indices_d[row_ptr_d[row_id] + nnz_row] = col_id;
                    nnz_row++;
                }

                // tunneling terms 
                if ( i != j && dist > nn_dist )
                { 
                    bool any_vacancy1 = element_i == VACANCY;
                    bool any_vacancy2 = element_j == VACANCY;

                    // contacts, excluding the last layer 
                    bool metal1p = is_in_array_gpu(metals, element_i, num_metals)
                                                && (ind_i > ((num_layers_contact - 1)*num_source_inj))
                                                && (ind_i < (Natom - (num_layers_contact - 1)*num_ground_ext)); 

                    bool metal2p = is_in_array_gpu(metals, element_j, num_metals)
                                                && (ind_j > ((num_layers_contact - 1)*num_source_inj))
                                                && (ind_j < (Natom - (num_layers_contact - 1)*num_ground_ext));  

                    // types of tunnelling conditions considered
                    bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                    bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                    bool contact_to_contact = (metal1p && metal2p);
                    double local_E_drop = atom_CB_edge_i - atom_CB_edge_j;                

                    if ((trap_to_trap || contact_to_trap || contact_to_contact)  && (fabs(local_E_drop) > tol))
                    {
                        col_indices_d[row_ptr_d[row_id] + nnz_row] = col_id;
                        nnz_row++;
                    }
                }
        }
    }

}


__global__ void populate_T_tunnel_dist(const double *posx, const double *posy, const double *posz,
                                       const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge, 
                                       const double nn_dist, const double tol, const double high_G, const double low_G, const double loop_G, 
                                       const double Vd, const double m_e, const double V0,
                                       int num_layers_contact, int num_source_inj, int num_ground_ext,
                                       int num_metals, const int num_tunnel_points, int *tunnel_indices, 
                                       int *row_ptr_d, int *col_indices_d, double *data_d, int Natom,
                                       int block_size, int block_start)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;
    int Nsub = Natom + 1;

    for(int row_id = idx; row_id < block_size; row_id += blockDim.x * gridDim.x){  // Nsub x Nsub matrix divided into blocks by rank
        int nnz_row = 0;
        for(int col_id = row_ptr_d[row_id]; col_id < row_ptr_d[row_id+1]; col_id++){ 
        

            int i = block_start + row_id; // this is where this rank's rows start, i indexes the matrix 
            int j = col_indices_d[col_id]; 

            //  ACCCESS INDIRECTLY THOUGH TUNNEL INDICES
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

            int ind_i = tunnel_indices[i];
            int ind_j = tunnel_indices[j];

            double dist_angstrom = site_dist_gpu(posx_i, posy_i, posz_i, 
                                                 posx_j, posy_j, posz_j);

                bool neighbor = (dist_angstrom < nn_dist) && (i != j);

                // tunneling terms occur between not-neighbors
                if (i != j && !neighbor)
                {  
                bool any_vacancy1 = element_i == VACANCY;
                bool any_vacancy2 = element_j == VACANCY;

                // contacts, excluding the last layer 
                bool metal1p = is_in_array_gpu(metals, element_i, num_metals)
                                            && (ind_i > ((num_layers_contact - 1)*num_source_inj))
                                            && (ind_i < (Natom - (num_layers_contact - 1)*num_ground_ext)); 

                bool metal2p = is_in_array_gpu(metals, element_j, num_metals)
                                            && (ind_j > ((num_layers_contact - 1)*num_source_inj))
                                            && (ind_j < (Natom - (num_layers_contact - 1)*num_ground_ext));  

                // types of tunnelling conditions considered
                bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                bool contact_to_contact = (metal1p && metal2p);
                double local_E_drop = atom_CB_edge_i - atom_CB_edge_j;                

                if ((trap_to_trap || contact_to_trap || contact_to_contact)  && (fabs(local_E_drop) > tol))
                {
                    double prefac = -(sqrt( 2 * m_e ) / h_bar) * (2.0 / 3.0);           // [s/(kg^1/2 * m^2)] coefficient inside the exponential
                    double dist = (1e-10)*dist_angstrom;                                // [m] 3D distance between atoms i and j

                    if (contact_to_trap)
                    {
                        double energy_window = fabs(local_E_drop);                      // [eV] energy window for tunneling from the contacts
                        double dV = 0.01;                                               // [V] energy spacing for numerical integration
                        double dE = eV_to_J * dV;                                       // [eV] energy spacing for numerical integration
                        // double dE = eV_to_J * dV * 10; // NOTE: @Manasa this is a temporary fix to avoid MPI issues!


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
                        data_d[col_id] = -T;
                    } 
                    else 
                    {
                        double E1 = eV_to_J * V0;                                        // [J] Energy distance to CB before tunnelling
                        double E2 = E1 - fabs(local_E_drop);                             // [J] Energy distance to CB after tunnelling
                                        
                        if (E2 > 0)                                                      // trapezoidal potential barrier (low field)
                        {                                                           
                            double T = exp(prefac * (dist / fabs(E1 - E2)) * ( pow(E1, 1.5) - pow(E2, 1.5) ) );
                            data_d[col_id] = -T;
                        }

                        if (E2 < 0)                                                        // triangular potential barrier (high field)
                        {
                            double T = exp(prefac * (dist / fabs(E1 - E2)) * ( pow(E1, 1.5) ));
                            data_d[col_id] = -T;
                        }
                    }
                }
            }
        }
    }
}


// Collect the indices of the contacts and the vacancies
__global__ void get_is_tunnel_mpi(int *is_tunnel, int *tunnel_indices, const ELEMENT *element, 
                                  int Nsub, int num_layers_contact, int num_source_inj, int num_ground_ext, int counts, int displacements)
{
    int total_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    int N_atom = Nsub - 1;

    for (int idx_local = total_tid; idx_local < counts; idx_local += total_threads)
    {
        int yes = 0; 
        int idx = idx_local + displacements - 2;

        if (idx >= 0 && idx < Nsub-1)
        {
            // vacancies and contacts have states within the bandgap which are included in the tunneling model
            // include the first layer of the contacts, as the rest are directly connected to it
            // METALS ARE HARDCODED

            if ( element[idx] == VACANCY || 
            ( (element[idx] == Ti_EL || element[idx] == N_EL) &&  (idx > (num_layers_contact - 1)*num_source_inj) && (idx < (N_atom - (num_layers_contact - 1)*num_ground_ext)) )) 
            {
                yes = 1;
            }

            is_tunnel[idx_local] = yes;
            tunnel_indices[idx_local] = yes * idx;
        }

    }
}

// template <typename T>
__global__ void shift_vector_by_constant(int *vec, int constant, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += total_threads)
    {
        vec[i] += constant;
    }
}


__global__ void calc_diagonal_T_tunnel( int *col_indices, int *row_ptr, double *data, int matrix_size, int displacement, double *diagonal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < matrix_size; i += blockDim.x * gridDim.x){ 
        //reduce the elements in the row
        double tmp = 0.0;
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(i + displacement != col_indices[j]){
                tmp += data[j];
            }
        }
        //write the sum of the off-diagonals onto the existing diagonal element
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(i + displacement == col_indices[j]){
                data[j] = -tmp;
                diagonal[i] = data[j];
            }
        }
    }
}


int assemble_sparse_T_submatrix(GPUBuffers &gpubuf, const int N_atom, const double nn_dist, int num_source_inj, int num_ground_ext, int num_layers_contact, 
                                 const double high_G, const double low_G, const double loop_G, const double Vd, const double m_e, const double V0,
                                 Distributed_subblock_sparse &T_tunnel, Distributed_matrix *T_neighbor, double *&diag_tunnel_local,
                                 int *&tunnel_indices_local_d)
{
    // The tunnel submatrix has size num_tunnel_points x num_tunnel_points
    // it is distributed over rows, NOT over blocks

    // The tunnel indices have the size of Nsub
    int Nsub = N_atom + 1;
    int rank = T_neighbor->rank;
    int size = T_neighbor->size;
    int counts_this_rank = T_neighbor->counts[rank];
    int disp_this_rank = T_neighbor->displacements[rank];

    // Collect the indices of the contacts and the vacancies    
    int num_threads = 1024;
    int num_blocks = (counts_this_rank - 1) / num_threads + 1;

    // indices of the tunneling connections (contacts and vacancies) in the Natom array
    int *is_tunnel; // [0, 1, 0, 0, 1...] where 1 indicates a tunnel connection
    int *is_tunnel_indices; // [0, 1, 0, 0, 4...] storing the indices of the tunnel connections
    
    gpuErrchk( hipMalloc((void **)&is_tunnel, counts_this_rank * sizeof(int)) );    
    gpuErrchk( hipMalloc((void **)&is_tunnel_indices, counts_this_rank * sizeof(int)) );  
    hipMemset(is_tunnel, 0, counts_this_rank * sizeof(int));
    hipMemset(is_tunnel_indices, 0, counts_this_rank * sizeof(int));

    hipLaunchKernelGGL(get_is_tunnel_mpi, num_blocks, num_threads, 0, 0, is_tunnel, is_tunnel_indices, gpubuf.atom_element, Nsub, num_layers_contact, num_source_inj, num_ground_ext,
                       counts_this_rank, disp_this_rank);
    gpuErrchk( hipPeekAtLastError() );

    int num_tunnel_points_local;
    num_tunnel_points_local = thrust::reduce(thrust::device, is_tunnel, is_tunnel + counts_this_rank, 0); // sum([0, 1, 0, 0, 1...])

    // allreduce num_tunnel_points_local
    int num_tunnel_points_global;
    MPI_Allreduce(&num_tunnel_points_local, &num_tunnel_points_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    std::cout << "size of tunneling submatrix: " << num_tunnel_points_global << "\n";

    // allgather the num_tunnel_points_local for every rank
    int *counts_subblock = new int[size];
    int *displ_subblock = new int[size];
    MPI_Allgather(&num_tunnel_points_local, 1, MPI_INT, counts_subblock, 1, MPI_INT, MPI_COMM_WORLD);
    displ_subblock[0] = 0;
    for(int i = 1; i < size; i++){
        displ_subblock[i] = displ_subblock[i-1] + counts_subblock[i-1];
    }

    // assemble the local indices into the atoms array for the peice owned by each rank
    // int *tunnel_indices_local_d; // [1, 4...]
    gpuErrchk( hipMalloc((void **)&tunnel_indices_local_d, num_tunnel_points_local * sizeof(int)) );    
    thrust::copy_if(thrust::device, is_tunnel_indices, is_tunnel_indices + counts_this_rank, tunnel_indices_local_d, is_not_zero());
    
    // for the row-wise kernel, every rank needs to iterate over all columns, so we allgather the tunnel indices
    // make global tunnel indices device poinwe
    int *tunnel_indices_global_d;
    gpuErrchk( hipMalloc((void **)&tunnel_indices_global_d, num_tunnel_points_global * sizeof(int)) );
    hipDeviceSynchronize();
    MPI_Allgatherv(tunnel_indices_local_d, num_tunnel_points_local, MPI_INT, tunnel_indices_global_d, counts_subblock, displ_subblock, MPI_INT, MPI_COMM_WORLD);

    int *tunnel_indices_global_h = new int[num_tunnel_points_global];
    // sum them up
    gpuErrchk( hipMemcpy(tunnel_indices_global_h, tunnel_indices_global_d, num_tunnel_points_global * sizeof(int), hipMemcpyDeviceToHost) );
    int sum_indices = 0;
    for(int i = 0; i < num_tunnel_points_global; i++){
        sum_indices += tunnel_indices_global_h[i];
    }
    std::cout << "rank: " << rank << " sum_indices: " << sum_indices << std::endl;

    // make the nnz vector for each rank:    
    // loop over the size to determine neighbours
    int *dist_nnz_per_row_d;

    //sync device
    gpuErrchk( hipDeviceSynchronize() );
    gpuErrchk( hipMalloc((void **)&dist_nnz_per_row_d, counts_subblock[rank] * sizeof(int)) );
    gpuErrchk(hipMemset(dist_nnz_per_row_d, 0, counts_subblock[rank] * sizeof(int)));

    int threads = 1024;
    //start with self
    int blocks = (counts_subblock[rank] - 1) / threads + 1;
    double tol = eV_to_J * 0.01;                                                                // [eV] tolerance after which the barrier slope is considered
    int num_metals = 2;
    MPI_Barrier(MPI_COMM_WORLD);
    hipLaunchKernelGGL(calc_nnz_per_row_tunnel, blocks, threads, 0, 0, 
                        gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z, gpubuf.atom_CB_edge,
                        gpubuf.atom_element, gpubuf.atom_charge,
                        nn_dist, tol,
                        tunnel_indices_global_d,
                        num_layers_contact, num_source_inj, num_ground_ext,
                        gpubuf.metal_types, num_metals,
                        num_tunnel_points_global,
                        N_atom,
                        counts_subblock[rank],
                        displ_subblock[rank],
                        dist_nnz_per_row_d);
    gpuErrchk( hipPeekAtLastError() );

    // int *dist_nnz_per_row_global_d;
    // gpuErrchk( hipMalloc((void **)&dist_nnz_per_row_global_d, num_tunnel_points_global * sizeof(int)) );
    // std::cout << counts_subblock[rank] << std::endl;
    // std::cout << displ_subblock[rank] << std::endl;

    // hipDeviceSynchronize();
    // MPI_Allgatherv(dist_nnz_per_row_d, counts_subblock[rank], MPI_INT, dist_nnz_per_row_global_d, counts_subblock, displ_subblock, MPI_INT, MPI_COMM_WORLD);

    // if(rank == 0){
    //     int *dist_nnz_per_row_global_h = new int[num_tunnel_points_global];
    //     gpuErrchk( hipMemcpy(dist_nnz_per_row_global_h, dist_nnz_per_row_global_d, num_tunnel_points_global * sizeof(int), hipMemcpyDeviceToHost) );
    //     std::string name3 = "dist_nnz_per_row_global_h_" + std::to_string(size) + ".txt";
    //     std::ofstream file3(name3);
    //     for (int i = 0; i < num_tunnel_points_global; i++){
    //         file3 << dist_nnz_per_row_global_d[i] << " ";
    //     }
    //     file3.close();
    // }


    MPI_Barrier(MPI_COMM_WORLD);
    // ** Verified that the nnz_per_row is correct for 1 rank and 3 ranks

    int *row_ptr_subblock_d;
    gpuErrchk( hipMalloc((void **)&row_ptr_subblock_d, (counts_subblock[rank] + 1) * sizeof(int)) );    

    // // create row ptr    
    gpuErrchk(hipMemset(row_ptr_subblock_d, 0, (counts_subblock[rank] + 1) * sizeof(int)));
    void     *temp_storage_d = NULL;
    size_t   temp_storage_bytes = 0;
    hipcub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes,
        dist_nnz_per_row_d, row_ptr_subblock_d+1, counts_subblock[rank]);
    gpuErrchk(hipMalloc(&temp_storage_d, temp_storage_bytes));
    hipcub::DeviceScan::InclusiveSum(temp_storage_d, temp_storage_bytes,
        dist_nnz_per_row_d, row_ptr_subblock_d+1, counts_subblock[rank]);
    gpuErrchk(hipFree(temp_storage_d));

    // ** Verified that the row ptrs are correct for 1 rank and 3 ranks

    // copy the last element of row_ptr_subblock_d back to host
    int nnz_subblock_local;
    gpuErrchk( hipMemcpy(&nnz_subblock_local, row_ptr_subblock_d + counts_subblock[rank], sizeof(int), hipMemcpyDeviceToHost) );

    int nnz_subblock[1];
    hipDeviceSynchronize();
    MPI_Allreduce(&nnz_subblock_local, nnz_subblock, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    std::cout << "rank: " << rank << " nnz_subblock: " << nnz_subblock[0] << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);        // remove

    // make col indices
    int *col_indices_subblock_d;
    gpuErrchk( hipMalloc((void **)&col_indices_subblock_d, nnz_subblock_local * sizeof(int)) );
    int *tmp = new int[nnz_subblock_local];
    for(int i = 0; i < nnz_subblock_local; i++){
        tmp[i] = i+8;
    }
    hipMemcpy(col_indices_subblock_d, tmp, nnz_subblock_local * sizeof(int), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(assemble_tunnel_col_indices, blocks, threads, 0, 0, 
                        gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z, gpubuf.atom_CB_edge,
                        gpubuf.atom_element, gpubuf.atom_charge,
                        nn_dist, tol,
                        tunnel_indices_global_d,
                        num_layers_contact, num_source_inj, num_ground_ext,
                        gpubuf.metal_types, num_metals,
                        num_tunnel_points_global,
                        N_atom,
                        counts_subblock[rank],
                        displ_subblock[rank],
                        row_ptr_subblock_d, col_indices_subblock_d);

    gpuErrchk( hipPeekAtLastError() );
    hipDeviceSynchronize();

    // make the values (remmember the x10 for numerical stability)
    double *data_d;
    gpuErrchk( hipMalloc((void **)&data_d, nnz_subblock_local * sizeof(double)) );
    hipLaunchKernelGGL(populate_T_tunnel_dist, blocks, threads, 0, 0, gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z,
                        gpubuf.metal_types, gpubuf.atom_element, gpubuf.atom_charge, gpubuf.atom_CB_edge,
                        nn_dist, tol, high_G, low_G, loop_G, Vd, m_e, V0,
                        num_layers_contact, num_source_inj, num_ground_ext,
                        num_metals, num_tunnel_points_local,
                        tunnel_indices_global_d,
                        row_ptr_subblock_d, col_indices_subblock_d, data_d, N_atom,
                        counts_subblock[rank], displ_subblock[rank]);

    // add + 2 to tunnel inds before calling constructor, so now tunnel-indices will index the full matrix of Nsub x Nsub
    // int two = 2;
    hipLaunchKernelGGL(shift_vector_by_constant, blocks, threads, 0, 0, tunnel_indices_local_d,
        2-disp_this_rank, counts_subblock[rank]);

    // // dump the tunnel_indices_local_d to file 
    // int *tunnel_indices_local_h = (int *)calloc(counts_subblock[rank], sizeof(int));
    // gpuErrchk( hipMemcpy(tunnel_indices_local_h, tunnel_indices_local_d, counts_subblock[rank] * sizeof(int), hipMemcpyDeviceToHost) );
    // std::string name2 = "tunnel_indices_local_d_" + std::to_string(gpubuf.rank) + ".txt";
    // std::ofstream file2(name2);
    // for (int i = 0; i < counts_subblock[rank]; i++){
    //     file2 << tunnel_indices_local_h[i] << " ";
    // }
    // file2.close();

    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(1);


    std::cout << "shifted tunnel indices" << std::endl;

    // row reduce the diagonals - diag_tunnel_local is passed in
    gpuErrchk( hipMalloc((void **)&diag_tunnel_local, counts_subblock[rank] * sizeof(double)) );
    hipLaunchKernelGGL(calc_diagonal_T_tunnel,
        blocks, threads, 0, 0, col_indices_subblock_d, row_ptr_subblock_d, data_d,
        counts_subblock[rank], displ_subblock[rank], diag_tunnel_local);


    // load data into subblock struct:
    rocsparse_spmat_descr subblock_descriptor;
    rocsparse_dnvec_descr subblock_vector_descriptor_in;
    rocsparse_dnvec_descr subblock_vector_descriptor_out;
    rocsparse_spmv_alg algo = rocsparse_spmv_alg_csr_adaptive;
    size_t subblock_buffersize;

    std::cout << "going to descr" << std::endl;

    rocsparse_create_csr_descr(&subblock_descriptor,
                                counts_subblock[rank],
                                num_tunnel_points_local,
                                nnz_subblock_local,
                                row_ptr_subblock_d,
                                col_indices_subblock_d,
                                data_d,
                                rocsparse_indextype_i32,
                                rocsparse_indextype_i32,
                                rocsparse_index_base_zero,
                                rocsparse_datatype_f64_r);


    rocsparse_handle rocsparse_handle;
    rocsparse_create_handle(&rocsparse_handle);

    std::cout << "going to spmv" << std::endl;

    double alpha = 1.0;
    double beta = 0.0;
    rocsparse_spmv(rocsparse_handle,
                    rocsparse_operation_none,
                    &alpha,
                    subblock_descriptor,
                    subblock_vector_descriptor_in,
                    &beta,
                    subblock_vector_descriptor_out,
                    rocsparse_datatype_f64_r,
                    algo,
                    &subblock_buffersize,
                    nullptr);
    double *subblock_buffer_d;
    hipMalloc(&subblock_buffer_d, subblock_buffersize);

    std::cout << "going to subblock" << std::endl;

    // Distributed_subblock_sparse A_subblock;
    T_tunnel.subblock_indices_local_d = tunnel_indices_local_d;
    T_tunnel.descriptor = &subblock_descriptor;
    T_tunnel.algo = algo;
    T_tunnel.buffersize = &subblock_buffersize;
    T_tunnel.buffer_d = subblock_buffer_d;
    T_tunnel.subblock_size = num_tunnel_points_local;
    T_tunnel.count_subblock_h = counts_subblock;
    T_tunnel.displ_subblock_h = displ_subblock;
    T_tunnel.send_subblock_requests = new MPI_Request[size-1];
    T_tunnel.recv_subblock_requests = new MPI_Request[size-1];
    T_tunnel.streams_recv_subblock = new hipStream_t[size-1];
    for(int i = 0; i < size-1; i++){
        hipStreamCreate(&T_tunnel.streams_recv_subblock[i]);
    }

    // dump_csr_matrix_txt(num_tunnel_points_local, nnz_subblock_local, row_ptr_subblock_d, col_indices_subblock_d, data_d, size+rank);
    // MPI_Barrier(MPI_COMM_WORLD);
    // exit(1); 



    // delete stuff
    // TODO delete outside
    // delete[] counts_subblock;
    // delete[] displ_subblock;
    // gpuErrchk( hipFree(tunnel_indices_local_d) );
    // gpuErrchk( hipFree(row_ptr_subblock_d) );
    // gpuErrchk( hipFree(col_indices_subblock_d) );
    // gpuErrchk( hipFree(data_d) );

    gpuErrchk( hipFree(is_tunnel) );
    gpuErrchk( hipFree(is_tunnel_indices) );
    gpuErrchk( hipFree(tunnel_indices_global_d) );
    gpuErrchk( hipFree(dist_nnz_per_row_d) );

    rocsparse_destroy_handle(rocsparse_handle);

    std::cout << "freed memory inside sparsity T" << std::endl;

    return num_tunnel_points_local;
}

void initialize_sparsity_T(GPUBuffers &gpubuf,
    int pbc, const double nn_dist, int num_source_inj, int num_ground_ext, int num_layers_contact, KMC_comm &kmc_comm)
{
    // copy atom arrays:
    int *gpu_index;
    int *atom_gpu_index;
    gpuErrchk( hipMalloc((void **)&gpu_index, gpubuf.N_ * sizeof(int)) );                                           // indices of the site array
    gpuErrchk( hipMalloc((void **)&atom_gpu_index, gpubuf.N_ * sizeof(int)) );                                      // indices of the atom array
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

    int N_sub = N_atom + 1;
    int rank = kmc_comm.rank_T;
    int size = kmc_comm.size_T;
    int rows_this_rank = kmc_comm.counts_T[rank];
    int disp_this_rank = kmc_comm.displs_T[rank];
    
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
        int rows_other = kmc_comm.counts_T[i];
        int displ_other = kmc_comm.displs_T[i];

        int threads = 1024;
        //start with self
        int blocks = (rows_this_rank - 1) / threads + 1;

        double tol = eV_to_J * 0.01;                                                                // [eV] tolerance after which the barrier slope is considered
        int num_metals = 2;
        hipLaunchKernelGGL(calc_nnz_per_row_T, blocks, threads, 0, 0, gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z,
                           gpubuf.metal_types, gpubuf.atom_element, gpubuf.atom_CB_edge, gpubuf.lattice, pbc,
                           nn_dist, tol, num_source_inj, num_ground_ext, num_layers_contact,
                           num_metals, N_sub, rows_this_rank, rows_other, disp_this_rank, displ_other, dist_nnz_per_row_d + i * rows_this_rank);

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

    std::cout << "got neighbor nnz " << std::endl;

    // print dist_nnz_h:
    for (int i = 0; i < size; i++)
    {
        std::cout << "rank " << rank << "T dist_nnz_h[" << i << "] = " << dist_nnz_h[i] << std::endl;
    }
    std::cout << "rank " << rank <<  "T neighbor_count = " << neighbor_count << std::endl;

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

    std::cout << "rank " << rank <<  "T neighbor_idx = " << neighbor_count << std::endl;   

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
        int rows_neighbour = kmc_comm.counts_T[neighbour];
        int disp_neighbour = kmc_comm.displs_T[neighbour];

        int threads = 1024;
        int blocks = (rows_this_rank + threads - 1) / threads;
       
        double tol = eV_to_J * 0.01;                                                                // [eV] tolerance after which the barrier slope is considered
        int num_metals = 2;
        hipLaunchKernelGGL(assemble_T_col_indices, blocks, threads, 0, 0, gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z,
                           gpubuf.metal_types, gpubuf.atom_element, gpubuf.atom_charge, gpubuf.atom_CB_edge,
                           gpubuf.lattice, pbc, nn_dist, tol,
                           num_source_inj, num_ground_ext, num_layers_contact,
                           num_metals, N_sub,
                           row_ptr_d[i],
                           col_indices_d[i], rows_this_rank, rows_neighbour, disp_this_rank, disp_neighbour);
        gpuErrchk( hipDeviceSynchronize() );
    }

    gpubuf.T_distributed = new Distributed_matrix(
        N_sub,
        kmc_comm.counts_T,
        kmc_comm.displs_T,
        neighbor_count,
        neighbor_idx,
        col_indices_d,
        row_ptr_d,
        neighbor_nnz_h,
        rocsparse_spmv_alg_csr_adaptive,
        kmc_comm.group_T
    );

    gpubuf.T_p_distributed = new Distributed_vector(
        N_sub,
        kmc_comm.counts_T,
        kmc_comm.displs_T,
        gpubuf.T_distributed->number_of_neighbours,
        gpubuf.T_distributed->neighbours,
        kmc_comm.group_T
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
    gpuErrchk( hipFree(neighbor_nnz_per_row_d) );    //FREE MEMORY
    std::cout << "rank " << rank <<  "initialized sparsity of peice of T " << neighbor_count << std::endl;   

}