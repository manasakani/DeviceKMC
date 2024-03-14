#include "Device.h"

// update the power of each site
std::map<std::string, double> Device::updatePower(hipblasHandle_t handle, hipsolverDnHandle_t handle_cusolver, GPUBuffers &gpubuf, KMCParameters &p, double Vd)
{
    std::map<std::string, double> result;

    double loop_G = p.high_G*10000000;                                                      // 'conductance' of the driver term for the NESS (converge to high value)
    double high_G = p.high_G*100000;                                                        // 'conductance' between metallic connections
    double low_G = p.low_G;  
    double scale = 1e-5;

    double G0 = 2 * 3.8612e-5 * scale;                                                      // G0 = (q^2 / h_bar), G = G0 * Tij
    double tol = p.q * 0.01;                                                                // [eV] tolerance after which the barrier slope is considered
    int num_source_inj = p.num_atoms_first_layer;                                           // number of injection nodes (tied to source)
    int num_ground_ext = p.num_atoms_first_layer;                                           // number of extraction nodes (tied to ground)
    double alpha = 1;                                                                       // [1] fraction of power dissipated as heat

#ifdef USE_CUDA
    auto t0 = std::chrono::steady_clock::now();
    bool sparse_iterative_solver = 1;

    // gpubuf.sync_HostToGPU(*this); // remove once full while loop is completed

    if (sparse_iterative_solver) 
    {
        // one sparse matrix for T:
        // update_power_gpu_sparse(handle, handle_cusolver, gpubuf, num_source_inj, num_ground_ext, p.num_layers_contact,
        //                        Vd, pbc, high_G, low_G, loop_G, G0, tol,
        //                        nn_dist, p.m_e, p.V0, p.metals.size(), &imacro, p.solve_heating_local, p.solve_heating_global, alpha);

        // T seperated into a sparse neighbor matrix and a dense tunnel matrix
        update_power_gpu_split(handle, handle_cusolver, gpubuf, num_source_inj, num_ground_ext, p.num_layers_contact,
                               Vd, pbc, high_G, low_G, loop_G, G0, tol,
                               nn_dist, p.m_e, p.V0, p.metals.size(), &imacro, p.solve_heating_local, p.solve_heating_global, alpha);
    } else {
        update_power_gpu(handle, handle_cusolver, gpubuf, num_source_inj, num_ground_ext, p.num_layers_contact,
                        Vd, pbc, high_G, low_G, loop_G, G0, tol,
                        nn_dist, p.m_e, p.V0, p.metals.size(), &imacro, p.solve_heating_local, p.solve_heating_global, alpha);
    }

    // gpubuf.sync_GPUToHost(*this); // remove once full while loop is completed
    
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;

    result["Current [uA]"] = imacro * 1e6;
    
#else
    auto t0 = std::chrono::steady_clock::now();

    // Re-identify the atomic sites (differentiate from the vacancy sites and oxygen ions)
    updateAtomLists();

    int N_full = N_atom + 1 + 1;                                                            // total number of nodes, including injection and extraction
    int Nsub = N_full - 1;              
    double *M = (double *)calloc(N_full, sizeof(double));                                   // [V] Virtual potential vector
    double *X = (double *)calloc(N_full * N_full, sizeof(double));                          // [1] Transmission matrix
    double *D_T = (double *)malloc(Nsub * Nsub * sizeof(double));                           // copy buffer for transmission matrix

    M[0] = -loop_G * Vd;                                                                    // max Current extraction (ground)
    M[1] = loop_G * Vd;                                                                     // max Current injection (source)

#pragma omp parallel
{

// *** Build the X conductivity matrix ***

#pragma omp for schedule(dynamic)
        for (int i = 0; i < N_atom; i++)
        {
            for (int j = i; j < N_atom; j++)  
            {

                double dist_angstrom = site_dist(atom_x[i], atom_y[i], atom_z[i],
                                                 atom_x[j], atom_y[j], atom_z[j], 
                                                 lattice, pbc);                              // [Angstrom] 3D distance between atoms i and j          
                bool neighbor = (dist_angstrom < nn_dist) && (i != j);                       // whether this is a neighbor pair

                // tunneling terms occur between not-neighbors
                if (i != j && !neighbor)
                {
                    bool any_vacancy1 =  atom_element[i] == VACANCY;
                    bool any_vacancy2 =  atom_element[j] == VACANCY;

                    bool metal1p = is_in_vector<ELEMENT>(p.metals, atom_element[i]) 
                                                     && (i > ((p.num_layers_contact - 1)*num_source_inj))
                                                     && (i < (N_atom - (p.num_layers_contact - 1)*num_ground_ext)); 

                    bool metal2p = is_in_vector<ELEMENT>(p.metals, atom_element[j])
                                                     && (j > ((p.num_layers_contact - 1)*num_source_inj))
                                                     && (j < (N_atom - (p.num_layers_contact - 1)*num_ground_ext));  

                    // types of tunnelling conditions considered
                    bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                    bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                    bool contact_to_contact = (metal1p && metal2p);

                    double local_E_drop = atom_CB_edge[i] - atom_CB_edge[j];                    // [eV] difference in energy between the two atoms
                    
                    // compute the WKB tunneling coefficients for all the tunnelling conditions
                    if ((trap_to_trap || contact_to_trap || contact_to_contact) && (std::abs(local_E_drop) > tol))  
                    {
                        
                        double prefac = -(sqrt( 2 * p.m_e ) / p.h_bar) * (2.0 / 3.0);           // [s/(kg^1/2 * m^2)] coefficient inside the exponential
                        double dist = (1e-10)*dist_angstrom;                                    // [m] 3D distance between atoms i and j

                        if (contact_to_trap)
                        { 
                            double energy_window = std::abs(local_E_drop);                      // [eV] energy window for tunneling from the contacts
                            double dV = 0.01;                                                   // [V] energy spacing for numerical integration
                            double dE = p.q * dV;                                               // [eV] energy spacing for numerical integration
                            
                            // integrate over all the occupied energy levels in the contact
                            double T = 0.0;
                            for (double iv = 0; iv < energy_window; iv += dE)
                            {
                                double E1 = p.q * p.V0 + iv;                                    // [J] Energy distance to CB before tunnelling
                                double E2 = E1 - std::abs(local_E_drop);                        // [J] Energy distance to CB after tunnelling

                                if (E2 > 0)                                                     // trapezoidal potential barrier (low field)                 
                                {                                                           
                                    T += exp(prefac * (dist / std::abs(local_E_drop)) * ( pow(E1, 1.5) - pow(E2, 1.5) ) );
                                }

                                if (E2 < 0)                                                      // triangular potential barrier (high field)                               
                                {
                                    T += exp(prefac * (dist / std::abs(local_E_drop)) * ( pow(E1, 1.5) )); 
                                } 
                            }
                            X[N_full * (i + 2) + (j + 2)] = -T;      
                            X[N_full * (j + 2) + (i + 2)] = -T; 
                        } 

                        else 
                        {
                            double E1 = p.q * p.V0;                                                 // [J] Energy distance to CB before tunnelling
                            double E2 = E1 - std::abs(local_E_drop);                                // [J] Energy distance to CB after tunnelling
                            
                            if (E2 > 0)                                                             // trapezoidal potential barrier (low field)                                            
                            {           
                                double T = exp(prefac * (dist / std::abs(local_E_drop)) * ( pow(E1, 1.5) - pow(E2, 1.5) ) );
                                X[N_full * (i + 2) + (j + 2)] = -T; 
                                X[N_full * (j + 2) + (i + 2)] = -T; 
                            }

                            if (E2 < 0)                                                             // triangular potential barrier (high field)                                            
                            {
                                double T = exp(prefac * (dist / std::abs(local_E_drop)) * ( pow(E1, 1.5) ));
                                X[N_full * (i + 2) + (j + 2)] = -T;       
                                X[N_full * (j + 2) + (i + 2)] = -T;                                
                            }
                        }
                    }
                }

                // direct terms occur between neighbors 
                if (i != j && neighbor)
                {
                    // contacts
                    bool metal1 = is_in_vector<ELEMENT>(p.metals, atom_element[i]);
                    bool metal2 = is_in_vector<ELEMENT>(p.metals, atom_element[j]);

                    // conductive vacancy sites
                    bool cvacancy1 = (atom_element[i] == VACANCY) && (atom_charge[i] == 0);
                    bool cvacancy2 = (atom_element[j] == VACANCY) && (atom_charge[j] == 0);
                    
                    if ((metal1 && metal2) || (cvacancy1 && cvacancy2)) 
                    {
                        X[N_full * (i + 2) + (j + 2)] = -high_G;
                        X[N_full * (j + 2) + (i + 2)] = -high_G;
                    }
                    else
                    {
                        X[N_full * (i + 2) + (j + 2)] = -low_G;
                        X[N_full * (j + 2) + (i + 2)] = -low_G;
                    }
                }
            } // j

            // connect the source/ground nodes to the first/last contact layers
            if (i < num_source_inj)
            {
                X[1 * N_full + (i + 2)] = -high_G;
                X[(i + 2) * N_full + 1] = -high_G;
            }
            if (i > (N_atom - num_ground_ext))
            {
                X[0 * N_full + (i + 2)] = -high_G;
                X[(i + 2) * N_full + 0] = -high_G;
            }

        } // i

// Connect the source node to the ground node
#pragma omp single
        {
            X[0 * N_full + 1] = -loop_G;
            X[1 * N_full + 0] = -loop_G; 
        }

// diagonals of X
#pragma omp for
        for (int i = 0; i < N_full; i++)
        {
            for (int j = 0; j < N_full; j++)
            {
                if (i != j)
                {
                    X[i * N_full + i] += -1 * X[i * N_full + j];
                }
            }
        }

// Prepare D_T to solve for the virtual potentials
#pragma omp for collapse(2)
        for (int i = 0; i < Nsub; i++)
        {
            for (int j = 0; j < Nsub; j++)
            {
                D_T[i * Nsub + j] = X[i * N_full + j];
            }
        }
} // #pragma omp parallel region 

    // Solve system of linear equations to get a virtual potential distribution which corresponds to this conductance matrix
    // (X is being stored into D_T because we don't want to overwrite it with the LU factors)
    // D_T(NsubxNsub) * x = M(Nsubx1) --> (solve for x)
    int info;
    int one = 1;
    int *ipiv_T = (int *)malloc(Nsub * sizeof(int));                                       
    gesv(&Nsub, &one, D_T, &Nsub, ipiv_T, M, &Nsub, &info);
    // M now contains the virtual potentials

    if (info)
    {
        std::cout << "WARNING: Info for gesv in update_power is " << info << "\n";
    }
    if (std::abs(M[num_source_inj] - Vd) > 0.1)
    {
        std::cout << "WARNING: non-negligible potential drop of " << std::abs(M[num_source_inj] - Vd) <<
                     " across the contact at VD = " << Vd << "\n";
    }

    // scale the virtual potentials by G0 (conductance quantum) instead of multiplying inside the X matrix
    for (int j = 0; j < Nsub; j++){
       M[j] *= G0;
    }

// *** Calculate the net current flowing into the device ***

    double I_macro = 0.0;                                                                   // net injected/extracted current

#pragma omp parallel
{
#pragma omp for reduction(+ : I_macro)
        for (int i = 0; i < N_atom; i++) 
        {
            // double I_pos_current = X[1 * N_full + (i + 2)] * (M[1] - M[i + 2]);             // injected  (M[1] = Vd), i < (N_atom - 1)
            double I_pos_current = X[0 * N_full + (i + 2)] * (M[0] - M[i + 2]);             // extracted (M[0] = 0), i < N_atom
            I_macro += I_pos_current;
        }
}

    // debug:
    // std::cout << "Max Current: " << X[0 * N_full + 1] * (M[0] - M[1]) * 1e6 << "\n";

    this->imacro = I_macro;
    double Geq = std::abs(I_macro / Vd);
    result["Current [uA]"] = I_macro * 1e6;
    result["Conductance [uS]"] = Geq * 1e6;
    std::cout << std::fixed << std::setprecision(16) << I_macro * 1e6 << "\n";
    // std::cout << std::fixed << std::setprecision(16) << Geq * 1e6 << "\n";
    // std::cout << "exiting after Imacro\n";
    // exit(1);

// *** Calculate the dissipated power ***

if (p.solve_heating_local || p.solve_heating_global)
{

        double *I_neg = (double *)malloc(N_atom * N_atom * sizeof(double));                     // [A] "inflow" current matrix
        double P_disp[N_atom];                                                                  // [W] atom-resolved dissipated Power
        double I_cal;                                                                           // single bond current
        int i, j;
        double one_d = 1.0;                 
        
        double min_V = *std::min_element(M + 2, M + N_full);            

    #pragma omp parallel private(I_cal, i, j)
    {
        // shifting the virtual potential so that it is all positive, as we will take differences
        #pragma omp for
        for (i = 0; i < N_full; i++)
        {
            M[i] += std::abs(min_V);
        }

        // Collect the forward currents into I_neg
        #pragma omp for collapse(2)
        for (i = 0; i < N_atom; i++)
        {
            for (j = 0; j < N_atom; j++)
            {
                I_neg[i * N_atom + j] = 0;
                I_cal = X[N_full * (i + 2) + (j + 2)] * (M[i + 2] - M[j + 2]); // current flows from j to i

                if (I_cal < 0 && Vd > 0)
                {
                    I_neg[i * N_atom + j] = -I_cal;
                }
                else if (I_cal > 0 && Vd < 0)
                {
                    I_neg[i * N_atom + j] = -I_cal;
                }
            }
        }

    // diagonals of I_neg
    #pragma omp for
        for (i = 0; i < N_atom; i++)
        {
            for (j = 0; j < N_atom; j++)
            {
                if (i != j)
                {
                    I_neg[i * N_atom + i] += -1 * I_neg[i * N_atom + j]; // sum row
                }
            }
        }
    }

        // Compute the dissipated power at each atom with [P]_Nx1 = [I]_NxN * [V]_Nx1 
        char transa = 'T';
        char transb = 'N';
        double zero = 0.0;
        gemm(handle, &transa, &transb, &N_atom, &one, &N_atom, &one_d, I_neg, &N_atom, &M[2], &N_atom, &zero, P_disp, &N_atom);
        //char       *transa, *transb,      *m,   *n,      *k, *alpha,    *A,    *lda,    *B,    *ldb, *beta,     *C,    *ldc

    // Extract the power dissipated between the contacts
    #pragma omp parallel for
    for (i = num_source_inj; i < N_atom - num_source_inj; i++)
    {
        // double alpha;
        bool metal = is_in_vector<ELEMENT>(p.metals, atom_element[i]);

        if (metal)
        {
            alpha = 0.0;
        }
        else
        {
            alpha = 1;
        }

        site_power[atom_ind[i]] = -1 * alpha * P_disp[i];
    }

    // Calculate total dissipate power
    double P_disp_tot = 0.0;
    #pragma omp parallel for reduction(+ : P_disp_tot)
    for (i = 0; i < N; i++)
    {
        P_disp_tot += site_power[i];
    }
    // std::cout << "P_disp_tot*1e9: " << P_disp_tot*1e9 << "\n";

    result["Total dissipated power [mW]"] = P_disp_tot*1e3;
    free(I_neg);

} // if (p.solve_heating_local || p.solve_heating_global)

    free(D_T);
    free(M);
    free(X);
    free(ipiv_T);

    // To do: put alpha in the parameter file
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
#endif

    result["Z - calculation time - dissipated power [s]"] = dt.count();
    return result;
}