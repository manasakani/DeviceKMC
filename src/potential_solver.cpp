#include "Device.h"

// Solve the Laplace equation to get the CB edge along the device
void Device::setLaplacePotential(cublasHandle_t handle_cublas, cusolverDnHandle_t handle_cusolver, GPUBuffers gpubuf, 
                                 KMCParameters &p, double Vd)
{
    size_t N_left_tot = p.num_atoms_first_layer; 
    size_t N_right_tot = p.num_atoms_first_layer;     
    size_t N_interface = N - N_left_tot - N_right_tot;

// #ifdef USE_CUDA

//     gpubuf.sync_HostToGPU(*this); // this one is needed, it's done before the first hostToGPU sync for a given bias point

//     update_CB_edge_gpu_sparse(handle_cublas, handle_cusolver, gpubuf, N, N_left_tot, N_right_tot,
//                               Vd, pbc, p.high_G, p.low_G, nn_dist, p.metals.size());

//     gpubuf.sync_GPUToHost(*this); 

// #else

    double *K = (double *)calloc(N * N, sizeof(double));
    double *VL = (double *)malloc(N_left_tot * sizeof(double));
    double *VR = (double *)malloc(N_right_tot * sizeof(double));
    double *Ksub = (double *)calloc(N_interface, sizeof(double));
    int *ipiv = (int *)calloc(N_interface, sizeof(int));
    int one = 1;
    int info, cntr;

#pragma omp parallel
    {

#pragma omp for
        for (size_t i = 0; i < N_left_tot; i++)
        {
            VL[i] = p.q*Vd/2;
        }

#pragma omp for
        for (size_t i = 0; i < N_right_tot; i++)
        {
            VR[i] = -p.q*Vd/2;
        }

#pragma omp for
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = i; j < N; j++) 
            {
                
                double dist_angstrom = site_dist(site_x[i], site_y[i], site_z[i],
                                                 site_x[j], site_y[j], site_z[j], 
                                                 lattice, pbc);                              // [Angstrom] 3D distance between sites i and j

                bool neighbor = (dist_angstrom < p.nn_dist) && (i != j);
                if (i != j && neighbor)
                {
                    bool metal1 = is_in_vector<ELEMENT>(p.metals, site_element[i]);
                    bool metal2 = is_in_vector<ELEMENT>(p.metals, site_element[j]);

                    if (metal1 || metal2)
                    {
                        K[N * i + j] = -p.high_G;
                        K[N * j + i] = -p.high_G;
                    }
                    else
                    {
                        K[N * i + j] = -p.low_G;
                        K[N * j + i] = -p.low_G;
                    }
                }
            }
        }
    
#pragma omp for
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                if (i != j)
                {
                    K[i * N + i] += -1 * K[i * N + j];
                }
            }
        }

#pragma omp for
        for (size_t i = N_left_tot; i < N - N_right_tot; i++)
        {
            for (size_t j = 0; j < N_left_tot; j++)
            {
                Ksub[i - N_left_tot] += K[i * N + j] * VL[j];
            }
            for (size_t j = N - N_right_tot; j < N; j++)
            {
                Ksub[i - N_left_tot] += K[i * N + j] * VR[j - (N - N_right_tot)];
            }
        }

    } // thread meetup
    // gesv(&N_interface, &one, D, &N_atom, ipiv, Ksub, &N_interface, &info);
    // gesv(static_cast<int*>(&N_interface), &one, D, static_cast<int*>(&N_atom), ipiv, Ksub, static_cast<int*>(&N_interface), &info);
 
    // Ax = b -> VSW = -inv(D)*Ksub -> -D*VSW = Ksub
    double* D = K + (N_left_tot * N) + N_left_tot;
    int N_interface_int = static_cast<int>(N_interface);
    int N_int = static_cast<int>(N);
    gesv(&N_interface_int, &one, D, &N_int, ipiv, Ksub, &N_interface_int, &info);
    if (info)
    {
        std::cout << "WARNING: Info for gesv in setLaplacePotential is " << info << "\n";
    }


#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
    {
        if (i < N_left_tot)
        {
            site_CB_edge[i] = VL[i];
        }
        else if ((i >= N_left_tot) && (i < (N - N_right_tot)))
        {
            site_CB_edge[i] = -1 * Ksub[i - N_left_tot];
        }
        else if (i >= (N - N_right_tot))
        {
            site_CB_edge[i] = VR[i - (N - N_right_tot)];
        }
    }

    free(K);
    free(VL);
    free(VR);
    free(Ksub);
    free(ipiv);

// #endif
}

// update the charge of each vacancy and ion
std::map<std::string, double> Device::updateCharge(GPUBuffers gpubuf, std::vector<ELEMENT> metals)
{
    std::map<std::string, double> result;

#ifdef USE_CUDA

    auto t0 = std::chrono::steady_clock::now();

    // gpubuf.sync_HostToGPU(*this); // remove once full while loop is completed

    update_charge_gpu(gpubuf.site_element,
                      gpubuf.site_charge,
                      gpubuf.neigh_idx,
                      gpubuf.N_, gpubuf.nn_, gpubuf.metal_types, gpubuf.num_metal_types_);

    // gpubuf.sync_GPUToHost(*this); // remove once full while loop is completed

    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;

#else
    auto t0 = std::chrono::steady_clock::now();
    int Vnn;
    int uncharged_V_counter = 0;
    int V_counter = 0;
    int Od_counter = 0;
    int uncharged_Od_counter = 0;
    int nn_cond = 2;
    bool metal_neighbor;

#pragma omp parallel for private(Vnn, metal_neighbor) reduction(+ : uncharged_V_counter, uncharged_Od_counter, V_counter, Od_counter)
    for (int i = 0; i < N; i++)
    {
        if (site_element[i] == VACANCY)
        {
            V_counter++;
            Vnn = 0;
            site_charge[i] = 2;
            for (int j : site_neighbors.l[i])
            {
                metal_neighbor = is_in_vector<ELEMENT>(metals, site_element[j]);
                if (site_element[j] == VACANCY)
                {
                    Vnn++;
                };
                if (metal_neighbor) 
                {
                    site_charge[i] = 0;
                    uncharged_V_counter++;
                    break;
                }
                if (Vnn >= nn_cond)
                {
                    site_charge[i] = 0;
                    uncharged_V_counter++;
                    break;
                }
            }
        }

        if (site_element[i] == OXYGEN_DEFECT)
        {
            Od_counter++;
            site_charge[i] = -2;
            for (int j : site_neighbors.l[i])
            {
                metal_neighbor = is_in_vector<ELEMENT>(metals, site_element[j]);
                if (metal_neighbor) 
                {
                    site_charge[i] = 0;
                    uncharged_Od_counter++;
                    break;
                }
            }
        }
    }

    result["Uncharged vacancies"] = (double) uncharged_V_counter;
    result["Charged vacancies"] = (double) V_counter - (double) uncharged_V_counter;
    result["Uncharged oxygen ions"] = (double) uncharged_Od_counter;
    result["Charged oxygen ions"] = (double) Od_counter - (double) uncharged_Od_counter;
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;

#endif
    result["Z - calculation time - charge [s]"] = dt.count();
    return result;
}

// update the potential of each site
std::map<std::string, double> Device::updatePotential(cublasHandle_t handle_cublas, cusolverDnHandle_t handle_cusolver, 
                                                      GPUBuffers &gpubuf, KMCParameters &p, double Vd, int kmc_step_count)
{
    std::map<std::string, double> result;

#ifdef USE_CUDA
    bool sparse_iterative_solver = 1;

    int N_left_tot = p.num_atoms_first_layer; 
    int N_right_tot = p.num_atoms_first_layer; 

    auto t0 = std::chrono::steady_clock::now();

    // gpubuf.sync_HostToGPU(*this); // comment out to avoid memory copy in GPU-only implementation
    
    if (sparse_iterative_solver) 
    {
        background_potential_gpu_sparse(handle_cublas, handle_cusolver, gpubuf, N, N_left_tot, N_right_tot,
                                        Vd, pbc, p.high_G, p.low_G, nn_dist, p.metals.size(), kmc_step_count);
    } else {
        background_potential_gpu(handle_cusolver, gpubuf, N, N_left_tot, N_right_tot,
                                 Vd, pbc, p.high_G, p.low_G, nn_dist, p.metals.size(), kmc_step_count);
    }
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt1 = t1 - t0;

    poisson_gridless_gpu(p.num_atoms_contact, pbc, gpubuf.N_, gpubuf.lattice, gpubuf.sigma, gpubuf.k,
                         gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
                         gpubuf.site_charge, gpubuf.site_potential_charge); 

    // gpubuf.sync_GPUToHost(*this); // comment out to avoid memory copy in GPU-only implementation

    auto t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt2 = t2 - t1;

#else
    auto t0 = std::chrono::steady_clock::now();

    // circuit-model-based potential solver
    background_potential(handle_cusolver, p.num_atoms_contact, Vd, lattice, p.G_coeff, p.high_G, p.low_G, p.metals, kmc_step_count);
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt1 = t1 - t0;

    // gridless Poisson equation solver (using sum of gaussian charge distribution solutions)
    poisson_gridless(p.num_atoms_contact, p.lattice);
    auto t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt2 = t2 - t1;
    
#endif

    result["Z - calculation time - potential from boundaries [s]"] = dt1.count();
    result["Z - calculation time - potential from charges [s]"] = dt2.count();
    return result;
}


// update the potential of each site
void Device::background_potential(cusolverDnHandle_t handle, int num_atoms_contact, double Vd, std::vector<double> lattice,
                                  double G_coeff, double high_G, double low_G, std::vector<ELEMENT> metals, int kmc_step_num)
{

    std::map<std::string, int> result;
    int N_left_tot = num_atoms_contact; // num_atoms_first_layer
    int N_right_tot = num_atoms_contact;  // num_atoms_first_layer
    int N_interface = N - N_left_tot - N_right_tot;

    int one = 1;
    int info, cntr;

    double *K = (double *)calloc(N * N, sizeof(double));
    double *VL = (double *)malloc(N_left_tot * sizeof(double));
    double *VR = (double *)malloc(N_right_tot * sizeof(double));
    double *Ksub = (double *)calloc(N_interface, sizeof(double));
    int *ipiv = (int *)calloc(N_interface, sizeof(int));

#pragma omp parallel
    {

#pragma omp for
        for (int i = 0; i < N_left_tot; i++)
        {
            VL[i] = -Vd/2; // split voltage across contacts
        }

#pragma omp for
        for (int i = 0; i < N_right_tot; i++)
        {
            VR[i] = Vd/2;
        }

// construct parametrized conductivity matrix (populates the K array)
// neighbor conductances - off-diagonals of K
#pragma omp for
        for (int i = 0; i < N; i++)
        {
            bool metal1 = is_in_vector<ELEMENT>(metals, site_element[i]);
            bool cvacancy1 = site_element[i] == VACANCY && site_charge[i] == 0;

            for (int j : site_neighbors.l[i])
            {
                bool metal2 = is_in_vector<ELEMENT>(metals, site_element[j]);
                bool cvacancy2 = site_element[j] == VACANCY && site_charge[j] == 0;

                if ((metal1 && metal2) || (cvacancy1 && cvacancy2))
                {
                    K[N * i + j] = -high_G;
                    K[N * j + i] = -high_G;
                }
                else
                {
                    K[N * i + j] = -low_G;
                    K[N * j + i] = -low_G;
                }
            }
        }

// diagonals of K:
#pragma omp for
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                if (i != j)
                {
                    K[i * N + i] += -1 * K[i * N + j];
                }
            }
        }

#pragma omp for
        for (int i = N_left_tot; i < N - N_right_tot; i++)
        {
            for (int j = 0; j < N_left_tot; j++)
            {
                Ksub[i - N_left_tot] += K[i * N + j] * VL[j];
            }
            for (int j = N - N_right_tot; j < N; j++)
            {
                Ksub[i - N_left_tot] += K[i * N + j] * VR[j - (N - N_right_tot)];
            }
        }

    } // thread meetup

    double* D = K + (N_left_tot * N) + N_left_tot;

    // do Ax = b -> VSW = -inv(D)*Ksub -> -D*VSW = Ksub
    gesv(&N_interface, &one, D, &N, ipiv, Ksub, &N_interface, &info);
    // the negative internal voltages are now contained in Ksub

    if (info)
    {
        std::cout << "WARNING: Info for gesv in background_potential is " << info << "\n";
    }

// assign potentials to sites:
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        if (i < N_left_tot)
        {
            site_potential_boundary[i] = VL[i];
        }
        else if ((i >= N_left_tot) && (i < (N - N_right_tot)))
        {
            site_potential_boundary[i] = -1 * Ksub[i - N_left_tot];
        }
        else if (i >= (N - N_right_tot))
        {
            site_potential_boundary[i] = VR[i - (N - N_right_tot)];
        }
    }

    free(K);
    free(VL);
    free(VR);
    free(Ksub);
    free(ipiv);
}

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
