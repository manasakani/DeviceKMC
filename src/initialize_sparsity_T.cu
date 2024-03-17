#include "gpu_solvers.h"
#include <hip/hip_runtime.h>
#include <stdio.h>

const double eV_to_J = 1.60217663e-19;          // [C]

// Compute the number of nonzeros per row of the matrix including the injection, extraction, and device nodes (excluding the ground). 
// Has dimensions of Nsub by Nsub (by the cpu code)
__global__ void calc_nnz_per_row_T( const double *posx_d, const double *posy_d, const double *posz_d,
                                    const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                                    const double *lattice, bool pbc, double nn_dist, const double tol,
                                    int num_source_inj, int num_ground_ext, const int num_layers_contact,
                                    int num_metals, int N_sub, int *nnz_per_row_d, 
                                    int block_size_i, int block_size_j, int block_start_i, int block_start_j){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Natom = N_sub - 1; 
    
    for(int row = idx; row < block_size_i; row += blockDim.x * gridDim.x){  // Nsub x Nsub matrix divided into blocks by rank
        
        int i = block_start_i + row; // this is where this rank's rows start, i indexes the matrix and i-2 indexes the atoms
        nnz_per_row_d[i] = 0;

        if ( i == 0 )
        {
            atomicAdd(&nnz_per_row_d[i], 1); // loop connection 
            atomicAdd(&nnz_per_row_d[i], num_ground_ext); // extraction terms
        }
        if ( i == 1 )
        {
            atomicAdd(&nnz_per_row_d[i], 1); // loop connection 
            atomicAdd(&nnz_per_row_d[i], num_source_inj); // injection terms
        }

        int nnz_row = 0;

        if (i > 1)
        {
            for(int col = 0; col < block_size_j; col++){

                int j = block_start_j + col; 

                // source/ground terms
                if ( (j == 1) && (i < num_source_inj - 2) )
                {
                    nnz_row++;
                }
                if ( (j == 0) && (i > (N_sub - num_ground_ext)) )
                {
                    nnz_row++;
                }

                double dist = site_dist_gpu(posx_d[i-2], posy_d[i-2], posz_d[i-2],
                                            posx_d[j-2], posy_d[j-2], posz_d[j-2],
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
                    bool any_vacancy1 = element[i-2] == VACANCY;
                    bool any_vacancy2 = element[j-2] == VACANCY;

                    // contacts, excluding the last layer 
                    bool metal1p = is_in_array_gpu(metals, element[i-2], num_metals) 
                                                    && (i-2 > ((num_layers_contact - 1)*num_source_inj))
                                                    && (i-2 < (Natom - (num_layers_contact - 1)*num_ground_ext)); 

                    bool metal2p = is_in_array_gpu(metals, element[j-2], num_metals)
                                                    && (j-2 > ((num_layers_contact - 1)*num_source_inj))
                                                    && (j-2 < (Natom - (num_layers_contact - 1)*num_ground_ext));  

                    // types of tunnelling conditions considered
                    bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                    bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                    bool contact_to_contact = (metal1p && metal2p);
                    double local_E_drop = atom_CB_edge[i-2] - atom_CB_edge[j-2];                

                    if ((trap_to_trap || contact_to_trap || contact_to_contact)  && (fabs(local_E_drop) > tol))
                    {
                        nnz_row++;
                    }
                }
            }
        }
        atomicAdd(&nnz_per_row_d[i], nnz_row); 
    }

}


__global__ void assemble_T_col_indices(const double *posx_d, const double *posy_d, const double *posz_d,
                                        const ELEMENT *metals, const ELEMENT *element, const int *atom_charge, const double *atom_CB_edge,
                                        const double *lattice, bool pbc, double nn_dist, const double tol,
                                        int num_source_inj, int num_ground_ext, const int num_layers_contact,
                                        int num_metals, int N_atom, int *row_ptr_d, int *col_indices_d,
                                        int block_size_i, int block_size_j, int block_start_i, int block_start_j)
{
    // row ptr is already calculated

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // int Natom = matrix_size - 2;
    int N_full = N_atom+2;
    
    // TODO can be optimized with a 2D grid instead of 1D
    // INDEXED OVER NFULL
    for(int row = idx; row < block_size_i; row += blockDim.x * gridDim.x){                      

        int nnz_row = 0;

        // loop connection and injection row
        if ( row == 0 )
        {
            for (int col = 0; col < block_size_j; col++)                                        
            {
                if ( (col < 2) || col > (N_full - num_ground_ext) )
                {
                    col_indices_d[row_ptr_d[row] + nnz_row] = col;
                    nnz_row++;
                }
            }
        }

        // loop connection and extraction row
        if ( row == 1 )
        {
            for (int col = 0; col < num_source_inj + 2; col++)
            {
                col_indices_d[row_ptr_d[row] + nnz_row] = col;
                nnz_row++;
            }
        }

        // inner matrix terms
        if (row >= 2)
        {
            int i = block_start_i + row - 2; // i and j index the atom arrays

            for(int col = 0; col < block_size_j; col++){                                     

                // add injection term for this row
                if ( (col == 1) && (row < num_source_inj + 2) )
                {
                    col_indices_d[row_ptr_d[row] + nnz_row] = 1;
                    nnz_row++;
                }

                // add extraction term for this row
                if ( (col == 0) && (row > N_full - num_ground_ext) )
                {
                    col_indices_d[row_ptr_d[row] + nnz_row] = 0;
                    nnz_row++;
                }

                if ( col >= 2 ) 
                {
                    int j = block_start_j + col - 2;
                    double dist = site_dist_gpu(posx_d[i], posy_d[i], posz_d[i],
                                                posx_d[j], posy_d[j], posz_d[j],
                                                lattice[0], lattice[1], lattice[2], pbc);
                    
                    // diagonal terms
                    if ( i == j )
                    {
                        col_indices_d[row_ptr_d[row] + nnz_row] = j;
                        nnz_row++;
                    }

                    // direct terms 
                    else if ( i != j && dist < nn_dist )
                    {
                        col_indices_d[row_ptr_d[row] + nnz_row] = j;
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
                                                    && (i < (N_atom - (num_layers_contact - 1)*num_ground_ext)); 

                        bool metal2p = is_in_array_gpu(metals, element[j], num_metals)
                                                    && (j > ((num_layers_contact - 1)*num_source_inj))
                                                    && (j < (N_atom - (num_layers_contact - 1)*num_ground_ext));  

                        // types of tunnelling conditions considered
                        bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                        bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                        bool contact_to_contact = (metal1p && metal2p);
                        double local_E_drop = atom_CB_edge[i] - atom_CB_edge[j];                

                        if ((trap_to_trap || contact_to_trap || contact_to_contact)  && (fabs(local_E_drop) > tol))
                        {
                            col_indices_d[row_ptr_d[row] + nnz_row] = j;
                            nnz_row++;
                        }
                    }
                }
            }
        }

    }
}


void initialize_sparsity_T(GPUBuffers &gpubuf, int pbc, const double nn_dist, int num_source_inj, int num_ground_ext, int num_layers_contact)
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
    int rank = gpubuf.rank;
    int size = gpubuf.size;
    int rows_this_rank = gpubuf.count_T_device[rank];
    int disp_this_rank = gpubuf.displ_T_device[rank];

    std::cout << "Nsub inside sparsity: " << N_sub << std::endl;
    std::cout << "N_atom inside sparsity: " << N_atom << std::endl;
    
    int *dist_nnz_h = new int[gpubuf.size];
    int *dist_nnz_d;
    int *dist_nnz_per_row_d;
    gpuErrchk( hipMalloc((void **)&dist_nnz_d, gpubuf.size * sizeof(int)) );
    gpuErrchk(hipMemset(dist_nnz_d, 0, gpubuf.size * sizeof(int)));
    gpuErrchk( hipMalloc((void **)&dist_nnz_per_row_d, gpubuf.size * rows_this_rank * sizeof(int)) );
    gpuErrchk(hipMemset(dist_nnz_per_row_d, 0, gpubuf.size * rows_this_rank * sizeof(int)));

    // Assemble the sparsity pattern

    // loop over the size to determine neighbours
    for(int i = 0; i < size; i++){
        int rows_other = gpubuf.count_T_device[i];
        int displ_other = gpubuf.displ_T_device[i];

        int threads = 1024;
        //start with self
        int blocks = (rows_this_rank - 1) / threads + 1;
        // hipLaunchKernelGGL(calc_nnz_per_row, blocks, threads, 0, 0, 
        //     gpubuf.site_x,
        //     gpubuf.site_y,
        //     gpubuf.site_z,
        //     gpubuf.lattice, pbc, nn_dist,
        //     rows_this_rank,
        //     rows_other,
        //     disp_this_rank,
        //     displ_other,
        //     dist_nnz_per_row_d + i * rows_this_rank
        // );

        //print  rows_this_rank, rows_other, disp_this_rank, displ_other:
        std::cout << "rank " << gpubuf.rank << " rows_this_rank = " << rows_this_rank << std::endl;
        std::cout << "rank " << gpubuf.rank << " rows_other = " << rows_other << std::endl;
        std::cout << "rank " << gpubuf.rank << " disp_this_rank = " << disp_this_rank << std::endl;
        std::cout << "rank " << gpubuf.rank << " displ_other = " << displ_other << std::endl;

        double tol = eV_to_J * 0.01;                                                                // [eV] tolerance after which the barrier slope is considered
        int num_metals = 2;
        hipLaunchKernelGGL(calc_nnz_per_row_T, blocks, threads, 0, 0, gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z,
                         gpubuf.metal_types, gpubuf.atom_element, gpubuf.atom_charge, gpubuf.atom_CB_edge,
                         gpubuf.lattice, pbc, nn_dist, tol,
                         num_source_inj, num_ground_ext, num_layers_contact,
                         num_metals, N_sub, dist_nnz_per_row_d + i * rows_this_rank, rows_this_rank, rows_other, disp_this_rank, displ_other);
        gpuErrchk( hipDeviceSynchronize() );

        //debug
        // copy back to host and print nnz per row dist_nnz_per_row_d:
        int *dist_nnz_per_row_h = new int[1000];
        gpuErrchk( hipMemcpy(dist_nnz_per_row_h, dist_nnz_per_row_d, 1000 * sizeof(int), hipMemcpyDeviceToHost) );
        for(int i = 0; i < 1000; i++){
            std::cout << "rank " << gpubuf.rank << " dist_nnz_per_row_h[" << i << "] = " << dist_nnz_per_row_h[i] << std::endl;
        }
        //debug
        //debug
        // // print nnz per row to file
        // std::ofstream fout("nnz_per_row_testing.txt");
        // int *dist_nnz_per_row_h = new int[rows_this_rank];
        // gpuErrchk( hipMemcpy(dist_nnz_per_row_h, dist_nnz_per_row_d, rows_this_rank * sizeof(int), hipMemcpyDeviceToHost) );
        // for (int i = 0; i < rows_this_rank; i++) {
        //     fout << dist_nnz_per_row_h[i]; 
        //     fout << ' ';
        // }
        // //debug
        exit(1);

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

    //print dist_nnz_h:
    for (int i = 0; i < size; i++)
    {
        std::cout << "rank " << gpubuf.rank << "T dist_nnz_h[" << i << "] = " << dist_nnz_h[i] << std::endl;
    }
    std::cout << "rank " << gpubuf.rank <<  "T neighbor_count = " << neighbor_count << std::endl;

    // CHECK NNZ PER ROW

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
        int rows_neighbour = gpubuf.count_T_device[neighbour];
        int disp_neighbour = gpubuf.displ_T_device[neighbour];

        int threads = 1024;
        int blocks = (rows_this_rank + threads - 1) / threads;
       
        // hipLaunchKernelGGL(assemble_K_indices_gpu_off_diagonal_block, blocks, threads, 0, 0, 
        //     gpubuf.site_x,
        //     gpubuf.site_y,
        //     gpubuf.site_z,
        //     gpubuf.lattice, pbc, nn_dist,
        //     rows_this_rank,
        //     rows_neighbour,
        //     N_left_tot + disp_this_rank,
        //     N_left_tot + disp_neighbour,
        //     row_ptr_d[i],
        //     col_indices_d[i]
        // );
        double tol = eV_to_J * 0.01;                                                                // [eV] tolerance after which the barrier slope is considered
        int num_metals = 2;
        hipLaunchKernelGGL(assemble_T_col_indices, blocks, threads, 0, 0, gpubuf.atom_x, gpubuf.atom_y, gpubuf.atom_z,
                         gpubuf.metal_types, gpubuf.atom_element, gpubuf.atom_charge, gpubuf.atom_CB_edge,
                         gpubuf.lattice, pbc, nn_dist, tol,
                         num_source_inj, num_ground_ext, num_layers_contact,
                         num_metals, N_atom,
                         row_ptr_d[i],
                         col_indices_d[i], rows_this_rank, rows_neighbour, disp_this_rank, disp_neighbour);
        gpuErrchk( hipDeviceSynchronize() );
    }

    std::cout << "finished computing col indices, now filling fake data" << std::endl;

    //debug - fill data of the sparsity of row_ptr_d and col_indices_d with 1s
    double *X_data = new double[dist_nnz_h[0]];
    for(int i = 0; i < dist_nnz_h[0]; i++){
        X_data[i] = 1.0;
    }
    double *X_data_d;
    gpuErrchk( hipMalloc((void **)&X_data_d, dist_nnz_h[0] * sizeof(double)) );
    gpuErrchk( hipMemcpy(X_data_d, X_data, dist_nnz_h[0] * sizeof(double), hipMemcpyHostToDevice) );
    //debug - fill data of the sparsity of row_ptr_d and col_indices_d with 1s

    std::cout << "filled sparsity" << std::endl;
    dump_csr_matrix_txt(N_sub, dist_nnz_h[0], row_ptr_d[0], col_indices_d[0], X_data_d, 1);
    std::cout << "dumped sparse matrix\n";
    exit(1);

    // TODO: replace with unique ptrs
    gpubuf.T_distributed = new Distributed_matrix(
        N_sub,
        gpubuf.count_T_device,
        gpubuf.displ_T_device,
        neighbor_count,
        neighbor_idx,
        col_indices_d,
        row_ptr_d,
        neighbor_nnz_h,
        gpubuf.comm
    );

    std::cout << "made distributed T\n";

    gpubuf.T_p_distributed = new Distributed_vector(
        N_sub,
        gpubuf.count_T_device,
        gpubuf.displ_T_device,
        gpubuf.T_distributed->number_of_neighbours,
        gpubuf.T_distributed->neighbours,
        gpubuf.comm
    );

    std::cout << "made distributed Tp\n";

    // // indices of the off-diagonal leftcontact-A matrix
    // indices_creation_gpu_off_diagonal_block(
    //     gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
    //     gpubuf.lattice, pbc,
    //     nn_dist,
    //     rows_this_rank,
    //     N_left_tot,
    //     N_left_tot + disp_this_rank,
    //     0,
    //     &gpubuf.left_col_indices_d,
    //     &gpubuf.left_row_ptr_d,
    //     &gpubuf.left_nnz
    // );

    // // indices of the off-diagonal A-rightcontact matrix
    // indices_creation_gpu_off_diagonal_block(
    //     gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
    //     gpubuf.lattice, pbc,
    //     nn_dist,
    //     rows_this_rank,
    //     N_right_tot,
    //     N_left_tot + disp_this_rank,
    //     N_left_tot + N_interface,
    //     &gpubuf.right_col_indices_d,
    //     &gpubuf.right_row_ptr_d,
    //     &gpubuf.right_nnz
    // );

    // for(int i = 0; i < neighbor_count; i++){
    //     gpuErrchk( hipFree(col_indices_d[i]) );
    //     gpuErrchk( hipFree(row_ptr_d[i]) );
    // }   
    // delete[] col_indices_d;
    // delete[] row_ptr_d;
    // delete[] neighbor_idx;
    // delete[] dist_nnz_h;
    // gpuErrchk( hipFree(dist_nnz_d) );    
    // gpuErrchk( hipFree(dist_nnz_per_row_d) );
    // delete[] neighbor_nnz_h;
    // gpuErrchk( hipFree(neighbor_nnz_per_row_d) );

    std::cout << "created sparsity of T\n";
    exit(1);

}