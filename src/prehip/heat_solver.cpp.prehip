#include "Device.h"


// find the number of site objects located in layer
int Device::get_num_in_contacts(int num_atoms_contact, std::string contact_name_)
{
    int count = 0;

    if (contact_name_ == "left")
    {
        int i = 0;
        while (i < num_atoms_contact)
        {
            if (site_element[count] != DEFECT)
            {
                i++;
            }
            count++;
        }
    }
    else
    {
        int i = N;
        count = N;
        while (i > N - num_atoms_contact)
        {
            if (site_element[count-1] != DEFECT) 
            {
                i--;
            }
            count--;
        }
        count = N - count;
    }

    return count;
}

// construct laplacian and steady state laplacian
void Device::constructLaplacian(KMCParameters &p)
{
    print("Constructing graph Laplacian");

    int N_left_tot = get_num_in_contacts(p.num_atoms_contact, "left");                      // # atoms in the left contact
    int N_right_tot = get_num_in_contacts(N_metals - p.num_atoms_contact, "right");         // # atoms in the left contact (including Ti)
    N_interface = N - N_left_tot - N_right_tot; 

    // Initialize the laplacian and inverse of the laplacian
    laplacian.resize(N_interface * N_interface);                                            // Laplacian (connectivity) matrix 
    laplacian_ss.resize(N_interface * N_interface);                                         // Laplacian modified for steady state
    index_mapping.resize(N * N);                                                    

    // Inverse laplacian
    int *ipiv_L_T = (int *)malloc(N_interface * sizeof(int));
    int lwork = N_interface;
    double *work = (double *)malloc(N_interface * sizeof(double));
    int N_test = N_interface;

    // Inverse steady state laplacian
    int *ipiv_L_ss_T = (int *)malloc(N_interface * sizeof(int));
    int lwork_ss = N_interface;
    double *work_ss = (double *)malloc(N_interface * sizeof(double));

    double *L = (double *)calloc(N_interface * N_interface, sizeof(double));     // Laplacian
    double *L_ss = (double *)calloc(N_interface * N_interface, sizeof(double));  // Steady state lapalacian
    double *L_inv = (double *)calloc(N_interface * N_interface, sizeof(double)); // Inverse of the laplacian

    // Initialize B for to calculate the inverse
    double *B_L = (double *)calloc(N_interface * N_interface, sizeof(double));

    // Initialize B for to calculate the inverse
    double *B_L_ss = (double *)calloc(N_interface * N_interface, sizeof(double));

    for (int i = 0; i < N_interface; i++)
    {
        B_L[i * N_interface + i] = 1.0;    // setting the diagonal elements to 1 to make it an identity matrix
        B_L_ss[i * N_interface + i] = 1.0; // setting the diagonal elements to 1 to make it an identity matrix
    }

    int info;

    // Map the index to a new array
    int new_index = 0;

    // Calculate constants
    double gamma = 1 / (p.delta * ((p.k_th_interface / p.k_th_metal) + 1)); // [a.u.]
    double step_time = p.delta_t * p.tau;                                   // [a.u.]

    // Map to new index
    for (int i = 0; i < N; i++)
    {
        if (i >= N_left_tot && i < N - N_right_tot)
        {

            index_mapping[i] = new_index;
            new_index += 1;
        }
        else
        {
            index_mapping[i] = -1;                                  // index of this atom is in the contacts
        }
    }

// Build laplacian matrix
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {

        int index_i, index_j;
        index_i = index_mapping[i];

        if (index_i != -1) // Not in the contacts
        {

            for (int j : site_neighbors.l[i])
            {

                index_j = index_mapping[j];

                if (i != j && index_j != -1) // Neighbouring site not in the contacts
                {
                    L[N_interface * index_i + index_j] = 1;
                }

                bool metal_atom2;
                metal_atom2 = is_in_vector<ELEMENT>(p.metals, site_element[j]);

                if (metal_atom2) // Boundary atom iff connected to a metallic site
                {
                    // sites[i].element = "B";
                    L[N_interface * index_i + index_i] = -gamma;
                }

            } // j
        }
    } // i

// Construct (I-L*time_step)
#pragma omp parallel for
    for (int i = 0; i < N_interface; i++)
    {
        for (int j = 0; j < N_interface; j++)
        {
            if (i != j)
            {
                L[N_interface * i + i] += -L[N_interface * i + j];
            }
        }
    }

// Prepare L_T to solve for the inverse of the unity - laplacian matrix (I-delta_t*L)
#pragma omp parallel for collapse(2)
    for (int i = 0; i < N_interface; i++)
    {
        for (int j = 0; j < N_interface; j++)
        {
            L_inv[i * N_interface + j] = -step_time * L[i * N_interface + j];
        }
    }

// Subtract the Laplacian matrix from the unitary matrix
#pragma omp parallel for
    for (int i = 0; i < N_interface; i++)
    {
        L_inv[N_interface * i + i] += 1;
    }

// Prepare Lss to solve for the inverse of the laplacian matrix (Lss)
#pragma omp parallel for collapse(2) num_threads(1)
    for (int i = 0; i < N_interface; i++)
    {
        for (int j = 0; j < N_interface; j++)
        {
            L_ss[i * N_interface + j] = L[i * N_interface + j];
        }
    }

#ifdef USE_CUDA

    gesv(&N_interface, &N_interface, L_inv, &N_interface, ipiv_L_T, B_L, &N_interface, &info);
    gesv(&N_interface, &N_interface, L_ss, &N_interface, ipiv_L_ss_T, B_L_ss, &N_interface, &info);

#else
    //  LU factorization of (I-L) (overwrite L_T with the factorization)
    dgetrf_(&N_interface, &N_interface, L_inv, &N_interface, ipiv_L_T, &info);

    // LU factorization of (L) (overwrite L_T with the factorization)
    dgetrf_(&N_interface, &N_interface, L_ss, &N_interface, ipiv_L_ss_T, &info);

    // Compute the inverse of the matrix L_T using the LU factorization (overwrite A with the factorization)
    dgetri_(&N_interface, L_inv, &N_interface, ipiv_L_T, work, &lwork, &info);

    // Prepare Lss to solve for the inverse of the laplacian matrix (Lss)
#pragma omp parallel for collapse(2) num_threads(1)
    for (int i = 0; i < N_interface; i++)
    {
        for (int j = 0; j < N_interface; j++)
        {
            B_L[i * N_interface + j] = L_inv[i * N_interface + j];
        }
    }

    // Compute the inverse of the matrix L_T using the LU factorization (overwrite A with the factorization)
    dgetri_(&N_interface, L_ss, &N_interface, ipiv_L_ss_T, work_ss, &lwork_ss, &info);

    // Prepare Lss to solve for the inverse of the laplacian matrix (Lss)
#pragma omp parallel for collapse(2) num_threads(1)
    for (int i = 0; i < N_interface; i++)
    {
        for (int j = 0; j < N_interface; j++)
        {
            B_L_ss[i * N_interface + j] = L_ss[i * N_interface + j];
        }
    }
    print("Assembling Laplacian on the CPU");

#endif

    // Update the inverse of the laplacian and steady state laplacian
#pragma omp parallel for collapse(2)
    for (int i = 0; i < N_interface; i++)
    {
        for (int j = 0; j < N_interface; j++)
        {
            laplacian[i * N_interface + j] = B_L[i * N_interface + j];
        }
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < N_interface; i++)
    {
        for (int j = 0; j < N_interface; j++)
        {
            laplacian_ss[i * N_interface + j] = B_L_ss[i * N_interface + j];
        }
    }

    free(ipiv_L_T);
    free(L);
    free(ipiv_L_ss_T);
    free(work_ss);
    free(work);
    free(L_ss);
    free(B_L);
    free(B_L_ss);
}


// update temperature on the CPU
std::map<std::string, double> Device::updateTemperature(GPUBuffers &gpubuf, KMCParameters &p, double step_time)
{
    std::map<std::string, double> result;

// #ifdef USE_CUDA

//     gpubuf.sync_HostToGPU(*this); // remove eventually
//     if (p.solve_heating_global)
//     {
//         double C_thermal = p.A * p.t_ox * p.c_p * (1e6); // [J/K]
//         double number_steps = step_time / p.small_step;
//         double a_coeff = -p.dissipation_constant*1/C_thermal*p.small_step + 1;
//         double b_coeff = p.dissipation_constant*1/C_thermal*p.small_step*p.background_temp; 

//         // call CUDA implementation
//         update_temperatureglobal_gpu(gpubuf.site_power, gpubuf.T_bg, gpubuf.N_, a_coeff, b_coeff, number_steps, C_thermal, p.small_step);
//     }
//     gpubuf.sync_GPUToHost(*this); // remove eventually

// #else

    // result["Global temperature [K]"] = p.background_temp;

    if (p.solve_heating_global)
    {
        // get the site power vector from the gpu
        gpubuf.copy_power_fromGPU(site_power);

        // compute the global temperature on the cpu
        updateTemperatureGlobal(step_time, p.small_step, p.dissipation_constant,
                                p.background_temp, p.t_ox, p.A, p.c_p, result);
        
        // update the global temperature in gpu memory
        gpubuf.copy_Tbg_toGPU(this->T_bg);
        result["Global temperature [K]"] = this->T_bg;
    }
    else if (p.solve_heating_local)
    {

        // use this to modify the rates
        if (step_time > 1e3 * p.delta_t)
        {
            // use steady state solution
            std::map<std::string, double> localTemperatureMap = updateLocalTemperatureSteadyState(p.background_temp, p.delta_t, p.tau, p.power_adjustment_term, p.k_th_interface,
                                                                                                  p.k_th_vacancies, p.num_atoms_contact, p.metals);

            return localTemperatureMap;
        }
        else
        {
            std::map<std::string, double> localTemperatureMap;
            for (int i = 0; i <= int(step_time / p.delta_t); ++i)
            {
                localTemperatureMap = updateLocalTemperature(p.background_temp, p.delta_t, p.tau, p.power_adjustment_term, p.k_th_interface,
                                                             p.k_th_vacancies, p.num_atoms_contact, p.metals);
            }
            return localTemperatureMap;
        }
    }

// #endif
    return result;
}


// update the global temperature using the global temperature model
void Device::updateTemperatureGlobal(double event_time, double small_step, double dissipation_constant,
                                     double background_temp, double t_ox, double A, double c_p, std::map<std::string, double> result)
{
    
    // std::map<std::string, double> result;

    double C_thermal = A * t_ox * c_p * (1e6);                                              // [J/K]
    double P_tot = 0.0;                                                                     // total dissipated power
    #pragma omp parallel for reduction(+ : P_tot)
    for (int i = 0; i < N; i++)
    {
        P_tot += site_power[i];
    }

    // Solve the capacitative heat equation (analytical)
    double a_coeff = dissipation_constant/C_thermal;                      
    double c_coeff = (dissipation_constant/C_thermal)*T_bg + (1/C_thermal) * P_tot; 
    double T_intermediate = (c_coeff/a_coeff) + (T_bg - c_coeff/a_coeff)*exp(-a_coeff * event_time);
    this->T_bg = T_intermediate;

    // // Solve the capacitative heat equation numerically (geometric sum)
    // double number_steps = event_time / small_step;                                          // number of discrete steps within the KMC event time
    // double a_coeff = -dissipation_constant*1/C_thermal*small_step + 1;                      // geometric series coefficients
    // double b_coeff = dissipation_constant*1/C_thermal*small_step*background_temp; 
    // double c_coeff = b_coeff + P_tot/C_thermal * small_step;
    // double T_intermediate = T_bg;
    // double T_bg_test = c_coeff * (1.0 - pow(a_coeff, number_steps)) / (1.0-a_coeff) + pow(a_coeff, number_steps) * T_intermediate;
    // this->T_bg = T_bg_test;

    // result["Global temperature [K]"] = T_bg;
    result["Total dissipated power [mW]"] = P_tot*1e3;
    std::cout << std::fixed << std::setprecision(16) << "Global temperature [K]: " << T_bg << "\n";

    // return result;
}


// update the local and global temperature
std::map<std::string, double> Device::updateLocalTemperature(double background_temp, double t, double tau, double power_adjustment_term, double k_th_interface,
                                                             double k_th_vacancies, double num_atoms_contact, std::vector<ELEMENT> metals)
{

    // Map
    std::map<std::string, double> result;

    double T_tot = 0.0;                                  // [K] Background temperature
    double T_0 = background_temp;                        // [K] Temperature scale
    double *T_vec = (double *)calloc(N, sizeof(double)); // Normalized temperatures

    // Map the index to a new array
    int index_i, index_j;
    double T_transf;

    // Calculate constants
    double step_time = t * tau;                                                                                                       // [a.u.]                                                               // [a.u.]
    const double p_transfer_vacancies = 1 / ((nn_dist * (1e-10) * k_th_interface) * (T_1 - background_temp));                         // [a.u.]
    const double p_transfer_non_vacancies = 1 / ((nn_dist * (1e-10) * k_th_vacancies) * (T_1 - background_temp));                     // [a.u.]

// Transform background temperatures
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {

        index_i = index_mapping[i];

        if (index_i != -1)
        {

            T_vec[i] = (site_temperature[i] - T_0) / (T_1 - T_0);
        }
    }

// Iterate through all the sites
#pragma omp parallel for private(T_transf, index_i, index_j)
    for (int i = 0; i < N; i++)
    {

        T_transf = 0;
        index_i = index_mapping[i];

        if (index_i != -1)
        {

            for (int j = 0; j < N; j++)
            {

                index_j = index_mapping[j];

                if (index_j != -1)
                {

                    double factor = laplacian[index_i * N_interface + index_j] * T_vec[j];

                    if (site_element[j] == VACANCY) 
                    {

                        T_transf += factor + laplacian[index_i * N_interface + index_j] * (site_power[j]) * p_transfer_vacancies * step_time;
                    }
                    else
                    {

                        T_transf += factor + laplacian[index_i * N_interface + index_j] * (site_power[j]) * p_transfer_non_vacancies * step_time;
                    }
                }

            } // j

            // Update the temperature at the specific site
            site_temperature[i] = T_transf * (T_1 - T_0) + T_0; // Transform back to normal temperature scale
        }
    } // i

    // Update the global temperature
//#pragma omp parallel
//    {
//#pragma omp for reduction(+ : T_tot)
        for (int i = num_atoms_contact; i < N - num_atoms_contact; i++)
        {
            T_tot += site_temperature[i];
        }
//    }
    T_bg = T_tot / (N - 2*num_atoms_contact);
    result["Global temperature [K]"] = T_bg;
    free(T_vec);
    return result;
}


// update the local and global temperature in steady state
std::map<std::string, double> Device::updateLocalTemperatureSteadyState(double background_temp, double delta_t, double tau, double power_adjustment_term, double k_th_interface,
                                                                        double k_th_vacancies, double num_atoms_contact, std::vector<ELEMENT> metals)
{
    std::map<std::string, double> result;

    double T_tot = 0.0;           // [K] Background temperature
    double T_0 = background_temp; // [K] Temperature scale

    // Map the index to a new array
    int index_i, index_j;
    double T_transf;

    // Calculate constants
    double step_time = delta_t * tau;                                                                                                 // [a.u.]                                                               // [a.u.]
    const double p_transfer_vacancies = 1 / ((nn_dist * (1e-10) * k_th_interface) * (T_1 - background_temp));                         // [a.u.]
    const double p_transfer_non_vacancies = 1 / ((nn_dist * (1e-10) * k_th_vacancies) * (T_1 - background_temp));                     // [a.u.]

    // Iterate through all the sites
#pragma omp parallel for private(T_transf, index_i, index_j)
    for (int i = 0; i < N; i++)
    {

        T_transf = 0;
        index_i = index_mapping[i];

        if (index_i != -1)
        {

            for (int j = 0; j < N; j++)
            {

                index_j = index_mapping[j];

                if (index_j != -1)
                {

                    if (site_element[j] == VACANCY)
                    {

                        T_transf += laplacian_ss[index_i * N_interface + index_j] * (site_power[j]) * p_transfer_vacancies;
                    }
                    else
                    {

                        T_transf += laplacian_ss[index_i * N_interface + index_j] * (site_power[j]) * p_transfer_non_vacancies;
                    }
                }

            } // j

            // Update the temperature at the specific site
            site_temperature[i] = -T_transf * (T_1 - T_0) + T_0; // Transform back to normal temperature scale
        }
    } // i

// Update the global temperature
// #pragma omp parallel
//    {
// #pragma omp for reduction(+ : T_tot)
        for (int i = num_atoms_contact; i < N - num_atoms_contact; i++)
        {
            T_tot += site_temperature[i];
        }
//    }

    T_bg = T_tot / (N - 2*num_atoms_contact);
    result["Global temperature [K]"] = T_bg;
    return result;
}