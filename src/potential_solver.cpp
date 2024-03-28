#include "Device.h"

// Solve the Laplace equation to get the CB edge along the device
void Device::setLaplacePotential(hipblasHandle_t handle_cublas, hipsolverHandle_t handle_cusolver, GPUBuffers gpubuf, 
                                 KMCParameters &p, double Vd)
{
    size_t N_left_tot = p.num_atoms_first_layer; 
    size_t N_right_tot = p.num_atoms_first_layer;     
    size_t N_interface = N - N_left_tot - N_right_tot;

    gpubuf.sync_HostToGPU(*this); // this one is needed, it's done before the first hostToGPU sync for a given bias point

    update_CB_edge_gpu_sparse(handle_cublas, handle_cusolver, gpubuf, N, N_left_tot, N_right_tot,
                              Vd, pbc, p.high_G, p.low_G, nn_dist, p.metals.size());

    gpubuf.sync_GPUToHost(*this); 

}

// // update the charge of each vacancy and ion
// std::map<std::string, double> Device::updateCharge(GPUBuffers gpubuf, std::vector<ELEMENT> metals)
// {
//     std::map<std::string, double> result;

//     // TODO if charge diistributed, change comm
//     MPI_Barrier(MPI_COMM_WORLD);
//     auto t0 = std::chrono::steady_clock::now();

//     // gpubuf.sync_HostToGPU(*this); // remove once full while loop is completed

//     update_charge_gpu(gpubuf.site_element,
//                       gpubuf.site_charge,
//                       gpubuf.neigh_idx,
//                       gpubuf.N_, gpubuf.nn_, gpubuf.metal_types, gpubuf.num_metal_types_);

//     // gpubuf.sync_GPUToHost(*this); // remove once full while loop is completed

//     auto t1 = std::chrono::steady_clock::now();
//     MPI_Barrier(MPI_COMM_WORLD);
//     std::chrono::duration<double> dt = t1 - t0;

//     result["Z - calculation time - charge [s]"] = dt.count();
//     return result;
// }


// void Device::poisson_gridless_indexed(int num_atoms_contact, std::vector<double> lattice)
// {
    
// #pragma omp parallel for
//     for (int i = 0; i < N; i++)
//     {
//         double V_temp = 0;
//         double r_dist;

//         int j;

//         for (int j_idx = 0; j_idx < N_cutoff; j_idx++)
//         {
//             j = cutoff_idx[i*N_cutoff + j_idx];
//             if (j != -1)
//             {
//                 r_dist = (1e-10) * site_dist(site_x[i], site_y[i], site_z[i],
//                                              site_x[j], site_y[j], site_z[j], lattice, pbc);
                                            
//                 V_temp += v_solve(r_dist, site_charge[j], sigma, k, q);
//             }
//         }
//         this->site_potential_charge[i] = V_temp;
//     }
// }


void Device::poisson_gridless(int num_atoms_contact, std::vector<double> lattice)
{
    
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        double V_temp = 0;
        double r_dist;

        for (int j = 0; j < N; j++)
        {
            if (i != j && site_charge[j] != 0)
            {
                r_dist = (1e-10) * site_dist(site_x[i], site_y[i], site_z[i],
                                             site_x[j], site_y[j], site_z[j], lattice, pbc);
                V_temp += v_solve(r_dist, site_charge[j], sigma, k, q);
            }
        }
        site_potential_charge[i] = V_temp;
    }
}
