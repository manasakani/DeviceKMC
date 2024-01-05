#include "Device.h"
#include "gpu_buffers.h"
#include "input_parser.h"
#include <cassert>
#include <boost/math/special_functions/gamma.hpp>

//remove:
#include <iomanip>

#ifdef USE_CUDA
    #include "cuda_wrapper.h"
#endif

void Graph::printAdjList()
{
    for (int i = 0; i < N; i++)
    {
        std::cout << "vertex " << i << "->";
        for (int j : l[i])
        {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }
}

// Construct the device from the structure (in the xyz files) and simulation parameters
Device::Device(std::vector<std::string> &xyz_files, KMCParameters &p)
{

    // initialize the random number generator
    random_generator.setSeed(p.rnd_seed);

    // parse xyz file(s)
    for (auto xyz_file : xyz_files)
    {
        this->N += read_xyz(xyz_file, site_element, site_x, site_y, site_z);
    }

    // set parameters
    this->lattice = p.lattice;
    this->pbc = p.pbc;
    this->nn_dist = p.nn_dist;
    this->sigma = p.sigma;
    this->k = k = 8.987552e9 / p.epsilon;
    this->T_bg = p.background_temp;

    // sort and prepare the raw coordinates
    // sort_by_x(site_x, site_y, site_z, site_element, lattice); // REMOVED
    sort_by_xyz(site_x, site_y, site_z, site_element, lattice); // REMOVED
    if (p.shift)
        translate_cell(site_x, site_y, site_z, N, lattice, p.shifts);

    // populate #interstitials, and #metals
    for (int i = 0; i < N; i++){
        if (site_element[i] == DEFECT)
        {
            this->N_int++;
        } else {
            this->N_atom++;
        }
        if (is_in_vector<ELEMENT>(p.metals, site_element[i]))
        {   
            site_is_metal.push_back(1);
            this->N_metals++;
        } else {
            site_is_metal.push_back(0);
        }
    }

    // initialize and construct the neighbor lists
    std::cout << "Building the neighbor list...\n";
    site_neighbors.initialize(N);
    constructSiteNeighborList();

    // check neighbors:
    for (int i = 0; i < N; i++)
    {
        if (site_neighbors.l[i].size() == 0){
            std::cout << "ERROR: Site with zero neighbors found at index: " << i << " at " << site_x[i] << " " << site_y[i] << " " << site_z[i] << "\n"; 
        }
    }

    // neighbor index for threads to use in the KMC step
    neigh_idx.resize(N * max_num_neighbors);
    #pragma omp parallel for
    for (auto i = 0; i < N; ++i) {
        int nb = 0;
        for (auto j: site_neighbors.l[i]) {
            neigh_idx[i * max_num_neighbors + nb] = j;
            nb++;
        }
        for (auto j = nb; j < max_num_neighbors; ++j) {
            neigh_idx[i * max_num_neighbors + j] = -1;
        }
    }

    // initialize the size of the field vectors
    site_charge.resize(N, 0);
    site_potential.resize(N, 0);
    site_power.resize(N, 0);
    site_temperature.resize(N, T_bg);

    std::cout << "Loaded " << N << " sites into device" << " : " << N_atom << " atoms and " << N_int << " interstitials " << "\n";

}

void Device::constructSiteNeighborList()
{
    
    //auto t0 = std::chrono::steady_clock::now();
    #pragma omp parallel
    {
    int local_max_nn = 0;

    // construct neighbor graph for all sites
    #pragma omp for 
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (is_neighbor(i, j)) 
            {
                site_neighbors.addEdge(i, j);
            }
        }

        if (site_neighbors.l[i].size() > local_max_nn) 
        {
            local_max_nn = site_neighbors.l[i].size();
        }
    }

    // set the maximum number of neighbors
    #pragma omp critical
    {
    if (local_max_nn > this->max_num_neighbors) 
    {
        this->max_num_neighbors = local_max_nn;
    }
    }

    }
    //auto t_f = std::chrono::steady_clock::now();
    //std::chrono::duration<double> diff_f = t_f - t0;
    //std::cout << "Time to build neighbor list: " << diff_f.count() << "\n";

    site_neighbors.is_constructed = 1;
    std::cout << "Maximum number of neighbors in device is: " << this->max_num_neighbors << "\n";
}

void Device::updateAtomLists()
{
    // reset the atoms arrays
    atom_x.clear();
    atom_y.clear();
    atom_z.clear();
    atom_ind.clear();
    atom_element.clear();
    atom_charge.clear();
    int atom_count = 0;

    // the elements of the atoms array are copies of the non-defects in the site array
    #pragma omp parallel for ordered schedule(static, 1)
    for (auto i = 0; i < N; i++)
    {
	    if (site_element[i] != DEFECT && site_element[i] != OXYGEN_DEFECT) // MODIFIED
        {
            #pragma omp ordered
            {
            atom_x.push_back(site_x[i]);	
            atom_y.push_back(site_y[i]);
            atom_z.push_back(site_z[i]);
            atom_element.push_back(site_element[i]);
            atom_charge.push_back(site_charge[i]);
            atom_ind.push_back(i);
            atom_count++;
            }
        }
    }

    this->N_atom = atom_count;
}

// construct laplacian and steady state laplacian
void Device::constructLaplacian(KMCParameters &p)
{
    print("Constructing graph Laplacian");

    // Get the number of interface atoms
    int N_left_tot = get_num_in_contacts(p.num_atoms_contact, "left");
    int N_right_tot = get_num_in_contacts(N_metals - p.num_atoms_contact, "right");
    N_interface = N - N_left_tot - N_right_tot;

    // Initialize the laplacian and inverse of the laplacian
    laplacian.resize(N_interface * N_interface);
    laplacian_ss.resize(N_interface * N_interface);
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

// returns number of sites of element
int Device::get_num_of_element(ELEMENT element_)
{

    int count = 0;
    #pragma omp parallel for reduction(+:count)
    for (int i = 0; i < N; i++)
    {
        if (site_element[i] == element_)
        {
            count++;
        }
    }
    return count;
}

// returns true if neighbor
bool Device::is_neighbor(int i, int j)
{

    if (site_neighbors.is_constructed){
        int count = std::count(site_neighbors.l[i].begin(), site_neighbors.l[i].end(), j);

        if (count > 0){
            return 1;
        }
        return 0;

    } else {

        std::vector<double> pos_i, pos_j;
        double dist, dist1, dist2;

        dist = site_dist(site_x[i], site_y[i], site_z[i], site_x[j], site_y[j], site_z[j], lattice, pbc);
        
        if (dist < nn_dist && i != j)
        {
            return 1;
        }
        return 0;
    }
}

// returns true if thing is present in the vector of things
template <typename T>
bool Device::is_in_vector(std::vector<T> things_, T thing_)
{
    for (auto t : things_)
    {
        if (t == thing_)
            return 1;
    }
    return 0;
}

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

// distributes some initial vacancies in the oxide
void Device::makeSubstoichiometric(double vacancy_concentration)
{

    int num_O, num_V_add, loc;
    double random_num;

    num_O = get_num_of_element(O_EL);
    num_V_add = vacancy_concentration * num_O;

    std::cout << num_V_add << " oxygen atoms will be converted to vacancies" << std::endl;
    while (num_V_add > 0)
    {
        random_num = random_generator.getRandomNumber();
        loc = random_num * N;
        if (site_element[loc] == O_EL)
        {
            site_element[loc] = VACANCY;
            num_V_add--;
        }
    }
}

// update the charge of each vacancy and ion
std::map<std::string, int> Device::updateCharge(GPUBuffers gpubuf, std::vector<ELEMENT> metals)
{
    std::map<std::string, int> result;

#ifdef USE_CUDA

    gpubuf.sync_HostToGPU(*this); // remove once full while loop is completed

    update_charge_gpu(gpubuf.site_element,
                      gpubuf.site_charge,
                      gpubuf.neigh_idx,
                      gpubuf.N_, gpubuf.nn_, gpubuf.metal_types, gpubuf.num_metal_types_);

    gpubuf.sync_GPUToHost(*this); // remove once full while loop is completed

#else
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

    result["Uncharged vacancies"] = uncharged_V_counter;
    result["Charged vacancies"] = V_counter - uncharged_V_counter;
    result["Uncharged oxygen ions"] = uncharged_Od_counter;
    result["Charged oxygen ions"] = Od_counter - uncharged_Od_counter;

#endif
    return result;
}

template <typename T>
void writeArrayToFile(const T* array, int numElements, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for(int i = 0; i < numElements; i++){
            file << array[i] << " "; 
        }
        file.close();
        std::cout << "Array data written to file: " << filename << std::endl;
    } else {
        std::cerr << "Unable to open the file for writing." << std::endl;
    }
}

// update the potential of each site
void Device::background_potential(cusolverDnHandle_t handle, int num_atoms_contact, double Vd, std::vector<double> lattice,
                                  double G_coeff, double high_G, double low_G, std::vector<ELEMENT> metals, int kmc_step_num)
{

    std::map<std::string, int> result;
    int N_left_tot = num_atoms_contact; // get_num_in_contacts(num_atoms_contact, "left");
    int N_right_tot = num_atoms_contact; //get_num_in_contacts(num_atoms_contact, "right"); 
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
                
    //@Alex Maeder: the matrix is D, which is indexed as above. To output D to a file:
    //int N_sq = N*N;
    //const std::string filename = "testK_wDiag.txt";
    //writeArrayToFile(K, N_sq, filename);
        

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

// assign potentials to sites:
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        if (i < N_left_tot)
        {
            site_potential[i] = VL[i];
        }
        else if ((i >= N_left_tot) && (i < (N - N_right_tot)))
        {
            site_potential[i] = -1 * Ksub[i - N_left_tot];
        }
        else if (i >= (N - N_right_tot))
        {
            site_potential[i] = VR[i - (N - N_right_tot)];
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
        site_potential[i] += V_temp;
    }
}

// update the potential of each site
void Device::updatePotential(cublasHandle_t handle_cublas, cusolverDnHandle_t handle_cusolver, 
                             GPUBuffers &gpubuf, KMCParameters &p, double Vd, int kmc_step_count)
{

    bool sparse_iterative_solver = 0;

#ifdef USE_CUDA

    int N_left_tot = p.num_atoms_contact; 
    int N_right_tot = p.num_atoms_contact; 

    gpubuf.sync_HostToGPU(*this); // comment out to avoid memory copy in GPU-only implementation
    
    if (sparse_iterative_solver) 
    {
        background_potential_gpu_sparse(handle_cublas, handle_cusolver, gpubuf, N, N_left_tot, N_right_tot,
                                        Vd, pbc, p.high_G, p.low_G, nn_dist, p.metals.size(), kmc_step_count);
    } else {
        background_potential_gpu(handle_cusolver, gpubuf, N, N_left_tot, N_right_tot,
                                 Vd, pbc, p.high_G, p.low_G, nn_dist, p.metals.size(), kmc_step_count);
    }

    // int mpi_rank, mpi_size, rows_per_process, row_idx_start, row_numbers;
    // MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    // MPI_Barrier(MPI_COMM_WORLD);

    // //1D distribution over sites 
    // rows_per_process = (N + mpi_size - 1) / mpi_size;
    // row_idx_start = mpi_rank * rows_per_process;
    // row_numbers = std::max(0, std::min(rows_per_process, N - row_idx_start));
    // std::cout << "poisson: rank " << mpi_rank << " handles rows from " << row_idx_start << " to " << (row_idx_start + row_numbers - 1) << std::endl;

    // poisson_gridless_mpi(p.num_atoms_contact, pbc, gpubuf.N_, gpubuf.lattice, gpubuf.sigma, gpubuf.k,
    //                      gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
    //                      gpubuf.site_charge, gpubuf.site_potential, row_idx_start, row_numbers);

    poisson_gridless_gpu(p.num_atoms_contact, pbc, gpubuf.N_, gpubuf.lattice, gpubuf.sigma, gpubuf.k,
                         gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
                         gpubuf.site_charge, gpubuf.site_potential);

    gpubuf.sync_GPUToHost(*this); // comment out to avoid memory copy in GPU-only implementation

#else
    // circuit-model-based potential solver
    background_potential(handle_cusolver, p.num_atoms_contact, Vd, lattice, p.G_coeff, p.high_G, p.low_G, p.metals, kmc_step_count);

    // gridless Poisson equation solver (using sum of gaussian charge distribution solutions)
    poisson_gridless(p.num_atoms_contact, p.lattice);
    
#endif
}

// update the power of each site
std::map<std::string, double> Device::updatePower(cublasHandle_t handle, cusolverDnHandle_t handle_cusolver, GPUBuffers &gpubuf, KMCParameters &p, double Vd)
{
    std::map<std::string, double> result;

#ifdef USE_CUDA

    int num_source_inj = p.num_atoms_first_layer;
    int num_ground_ext = num_source_inj;

    gpubuf.sync_HostToGPU(*this); // remove once full while loop is completed

    update_power_gpu(handle, handle_cusolver, gpubuf, N, num_source_inj, num_ground_ext,
                     Vd, pbc, p.high_G, p.low_G,
                     nn_dist, p.m_e, p.V0, p.metals.size(), p.t_ox, &imacro);

    gpubuf.sync_GPUToHost(*this); // remove once full while loop is completed

    // Postprocessing to get the total dissipated power
    double P_disp_tot = 0.0;
#pragma omp parallel for reduction(+ : P_disp_tot)
    for (int i = 0; i < N; i++)
    {
        P_disp_tot += site_power[i];
    }

    result["Total dissipated power"] = P_disp_tot;
    result["Current in uA"] = imacro * 1e6;
    
#else

    // Re-identify the atomic sites (differentiate from the vacancy sites and oxygen ions)
    updateAtomLists();

    double loop_G = p.high_G * 10000;                                                       // 'conductance' of the driver term for the NESS (converge to high value)
    double high_G = p.high_G * 100;                                                         // 'conductance' between metallic connections
    double low_G = p.low_G * 0.0001;                                                        // 'conductance' between insulating connections
    double tunnel_G = 0.01;                                                                 // coefficient for tunneling conductance terms

    int num_source_inj = p.num_atoms_first_layer;                                           // number of injection nodes (tied to source)
    int num_ground_ext = num_source_inj;                                                    // number of extraction nodes (tied to ground)
    int N_full = N_atom + 1 + 1;                                                            // total number of nodes, including injection and extraction
    int Nsub = N_full - 1;              
    double *M = (double *)calloc(N_full, sizeof(double));                                   // virtual potential vector
    double *X = (double *)calloc(N_full * N_full, sizeof(double));                          // conductance matrix
    double *D_T = (double *)malloc(Nsub * Nsub * sizeof(double));                           // copy buffer for conductance matrix

    M[0] = -loop_G * Vd;                                                                    // max Current extraction (ground)
    M[1] = loop_G * Vd;                                                                     // max Current injection (source)

#pragma omp parallel
{

// *** Build the X conductivity matrix ***

#pragma omp for 
        for (int i = 0; i < N_atom; i++)
        {
            for (int j = i; j < N_atom; j++)
            {

                double dist_angstrom = site_dist(atom_x[i], atom_y[i], atom_z[i],
                                                 atom_x[j], atom_y[j], atom_z[j], 
                                                 lattice, pbc);                              // [Angstrom] 3D distance between atoms i and j          

                bool neighbor = (dist_angstrom < nn_dist) && (i != j);

                // tunneling terms occur between not-neighbors
                if (i != j && !neighbor)
                {
                    bool any_vacancy1 =  atom_element[i] == VACANCY;
                    bool any_vacancy2 =  atom_element[j] == VACANCY;

                    // contacts, excluding the last layer
                    bool metal1p = is_in_vector<ELEMENT>(p.metals, atom_element[i]) 
                                                     && (atom_ind[i] > (5*num_source_inj))
                                                     && (atom_ind[i] < (N - 5*num_ground_ext - 540)); 

                    bool metal2p = is_in_vector<ELEMENT>(p.metals, atom_element[j])
                                                     && (atom_ind[j] > (5*num_source_inj))
                                                     && (atom_ind[j] < (N - 5*num_ground_ext - 540));   // REMOVE HARDCODE

                    // types of tunnelling conditions considered
                    bool trap_to_trap = (any_vacancy1 && any_vacancy2);
                    bool contact_to_trap = (any_vacancy1 && metal2p) || (any_vacancy2 && metal1p);
                    bool contact_to_contact = (metal1p && metal2p);

                    bool same_contact = ((atom_ind[i] < p.num_atoms_contact) && (atom_ind[j] < p.num_atoms_contact)) 
                                     || ((atom_ind[i] > N - p.num_atoms_contact) && atom_ind[j] > (N - p.num_atoms_contact));
                    
                    // compute the WKB tunneling coefficients for all the tunnelling conditions
                    if ((trap_to_trap || contact_to_trap || contact_to_contact) && !same_contact)       
                    {
                        // determine the distance over which the field drops
                        double xdist, x1, x2;
                        double start_x = atom_x[p.num_atoms_contact];
                        double end_x = atom_x[N_atom - p.num_atoms_contact - 540];          // REMOVE HARDCODE for Ti
                        x1 = atom_x[i]; x2 = atom_x[j];
                        if (atom_x[i] < start_x) { x1 = start_x;}                           // shift x-position to across the voltage drops
                        if (atom_x[i] > end_x) { x1 = end_x; } 
                        if (atom_x[j] < start_x) { x2 = start_x; } 
                        if (atom_x[j] > end_x) { x2 = end_x; } 
                        xdist = (1e-10) * (x2 - x1);                                        // [m] component of distance in direction of applied voltage

                        // tunnelling under forward bias
                        if (xdist > 0 && Vd > 0) 
                        {
                            
                            double local_V_drop = Vd * std::min(1.0, xdist / p.t_ox);       // [V] Applied electric field between the two sites 
                            double E1 = p.q * p.V0;                                         // [J] Energy distance to CB before tunnelling
                            double E2 = p.q * (p.V0 - local_V_drop);                        // [J] Energy distance to CB after tunnelling
                            double prefac = -(sqrt( 2 * p.m_e ) / p.h_bar) * (2.0 / 3.0);   // [s/(kg^1/2 * m^2)] coefficient inside the exponential
                            double dist = (1e-10)*dist_angstrom;                            // [m] 3D distance between atoms i and j

                            if (E2 > 0)                                                     // trapezoidal potential barrier (low field condition)
                            {            
                                // Conductance = - (q^2 / h_bar) * Tij                                                      
                                double T = exp(prefac * (dist / (E1 - E2)) * ( pow(E1, 1.5) - pow(E2, 1.5) ) );
                                X[N_full * (i + 2) + (j + 2)] = -2 * 3.8612e-5 * tunnel_G * T;
                                X[N_full * (j + 2) + (i + 2)] = -2 * 3.8612e-5 * tunnel_G * T;
                            }

                            if (E2 < 0)                                                     // triangular potential barrier (high field)
                            {
                                // Conductance = - (q^2 / h_bar) * Tij
                                double T = exp(prefac * (dist / (E1 - E2)) * ( pow(E1, 1.5) ));
                                X[N_full * (i + 2) + (j + 2)] = -2 * 3.8612e-5 * tunnel_G * T;
                                // T = exp(-1 * prefac * (dist / (E1 - E2)) * ( pow(E1, 1.5) ));
                                X[N_full * (j + 2) + (i + 2)] = -2 * 3.8612e-5 * tunnel_G * T;
                            }
                        }

                        // tunneling under reverse bias
                        if ((atom_x[j] < atom_x[i]) && Vd < 0)
                        {
                            std::cout << "Reverse bias tunneling terms not implemented yet!";
                            exit(1);
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

// *** Calculate the net current flowing into the device ***

    double I_macro = 0.0;                                                                   // net injected/extracted current

#pragma omp parallel
{
#pragma omp for reduction(+ : I_macro)
        for (int i = 0; i < N_atom; i++)
        {
            double I_pos_current = X[1 * N_full + (i + 2)] * (M[i + 2] - M[1]);             // injected  (M[1] = Vd)
            // I_pos_current = X[0 * N_full + (i + 2)] * (M[0] - M[i + 2]);                 // extracted (M[0] = 0)
            if ((I_pos_current > 0 && Vd > 0) || (I_pos_current < 0 && Vd < 0))
            {
                I_macro += I_pos_current;
            }
        }
}

    this->imacro = I_macro;
    double Geq = std::abs(I_macro / Vd);
    result["Current in uA"] = I_macro * 1e6;
    result["Conductance in uS"] = Geq * 1e6;
    // std::cout << I_macro * 1e6 << "\n";

// *** Calculate the dissipated power ***

if (p.solve_heating_local || p.solve_heating_global)
{

        double *I_neg = (double *)malloc(N_atom * N_atom * sizeof(double));
        double one_d = 1.0;
        double P_disp[N_atom];
        double min_V = *std::min_element(M + 2, M + N_full);
        double I_cal;                                                                           // single bond current
        int i, j;

    #pragma omp parallel private(I_cal, i, j)
    {
        // shifting the virtual potential by its minimum
        #pragma omp for
        for (i = 0; i < N_full; i++)
        {
            M[i] += std::abs(min_V);
        }

        // Collect the forward currents into I_neg
        #pragma omp for // collapse(2)
        for (i = 0; i < N_atom; i++)
        {
            int count = 0;
            for (j = 0; j < N_atom; j++)
            {

                double distNeighbor = site_dist(atom_x[i], atom_y[i], atom_z[i],
                                                atom_x[j], atom_y[j], atom_z[j], lattice, pbc);

                double neighbor = distNeighbor < nn_dist;
                double xdiff = (1e-10) * (atom_x[j] - atom_x[i]); // potential accross the x-direction => if x_j < x_i then Vdiff < 0

                I_neg[i * N_atom + j] = 0;
                I_cal = X[N_full * (i + 2) + (j + 2)] * (M[i + 2] - M[j + 2]); // current flows from j to i

                if (I_cal < 0 && Vd > 0 && i != j && neighbor)
                {
                    I_neg[i * N_atom + j] = -I_cal;
                }

                else if (I_cal < 0 && Vd > 0 && xdiff < p.t_ox * p.V0 / Vd && xdiff > nn_dist && !neighbor)
                { // excluding Fowler Nordheim tunneling
                    I_neg[i * N_atom + j] = -I_cal;
                }
            }
        }

    // Check whether there a negative entries in the X matrix

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

        // dissipated power at each atom
        char transa = 'T';
        char transb = 'N';
        double zero = 0.0;
        gemm(handle, &transa, &transb, &N_atom, &one, &N_atom, &one_d, I_neg, &N_atom, &M[2], &N_atom, &zero, P_disp, &N_atom);
        //char       *transa, *transb,      *m,   *n,      *k, *alpha,    *A,    *lda,    *B,    *ldb, *beta,     *C,    *ldc

    #pragma omp parallel for
        for (i = num_source_inj; i < N_atom - num_source_inj; i++)
        {

            double alpha;
            bool metal = is_in_vector<ELEMENT>(p.metals, atom_element[i]);
            bool vacancy = atom_element[i] == VACANCY;

            if (metal)
            {
                alpha = 0.0;
            }
            else if (vacancy)
            {
                alpha = 1;//0.1;
            }
            else
            {
                alpha = 1;//0.1;
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

    result["Total dissipated power"] = P_disp_tot;
    free(I_neg);

} // if (p.solve_heating_local || p.solve_heating_global)

    free(D_T);
    free(M);
    free(X);
    free(ipiv_T);

    // To do: put alpha in the parameter file
#endif

    return result;
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
//     else if (p.solve_heating_local)
//     {
//         std::cout << "This is not implemented on the GPU yet" << std::endl;
//     }

//     gpubuf.sync_GPUToHost(*this); // remove eventually

// #else

    result["Global temperature in K"] = p.background_temp;

    if (p.solve_heating_global)
    {
        return updateTemperatureGlobal(step_time, p.small_step, p.dissipation_constant,
                                       p.background_temp, p.t_ox, p.A, p.c_p);
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
// @param: step_time: time of the kmc time step
//         small_step: descretization time step
std::map<std::string, double> Device::updateTemperatureGlobal(double event_time, double small_step, double dissipation_constant,
                                                              double background_temp, double t_ox, double A, double c_p)
{
    // Map
    std::map<std::string, double> result;

    double P_tot = 0.0;
    double C_thermal = A * t_ox * c_p * (1e6); // [J/K]
    double T_global = T_bg;

// Calculate the total power dissipated in the lattice and at every site:
#pragma omp parallel for reduction(+ : P_tot)
    for (int i = 0; i < N; i++)
    {
        P_tot += site_power[i];
    }

    int num_steps = event_time / small_step;
    double T_upd = 0;

    for (int i = 0; i < num_steps; i++)
    {
        T_upd = (-1) * dissipation_constant * 1 / C_thermal * small_step * (T_global - background_temp) + 1 / C_thermal * P_tot * small_step + T_global;
        T_global = T_upd;
    }

    result["Total dissipated power in mW"] = P_tot * 1e3;
    result["Global temperature in K"] = T_global;
    T_bg = T_global;
    return result;
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
    result["Global temperature in K"] = T_bg;
    free(T_vec);
    return result;
}

// update the local and global temperature in steady state
std::map<std::string, double> Device::updateLocalTemperatureSteadyState(double background_temp, double delta_t, double tau, double power_adjustment_term, double k_th_interface,
                                                                        double k_th_vacancies, double num_atoms_contact, std::vector<ELEMENT> metals)
{
    // Map
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
    result["Global temperature in K"] = T_bg;
    return result;
}

void Device::writeSnapshot(std::string filename, std::string foldername)
{
    // cols of xyz file are "element x y z potential temperature"

    std::string path = "./" + foldername + "/" + filename;
    std::ofstream fout(path.c_str());
    fout << N << "\n";
    fout << "\n";

    for (int i = 0; i < N; i++)
    {
        fout << return_element(site_element[i]) << "   " << site_x[i] << "   " << site_y[i] << "   " << site_z[i] << "   " << site_potential[i] << "   " << site_temperature[i] << "\n";
    }
}
 // //check incomplete gamma function:
    // double ans = boost::math::tgamma(1.0, 1.0);
    // std::cout << ans << "\n";
    // exit(1);

   // int inde;
    // std::cout << "printing atom_el\n";
    // std::ofstream fout1("atom_el_with_int.txt");
    
    // for (inde = 0; inde < N_atom; ++inde) {
    //     fout1 << atom_element[inde] << ' ';
    // }
    // // Close the file
    // fout1.close();
    // exit(1);

    // int inde;
    // std::cout << "printing X\n";
    // std::cout << "N_full: " << N_full << "\n";
    // std::ofstream fout1("X_t20_1.txt");
    
    // for (inde = 0; inde < N_full * N_full; ++inde) {
    //     fout1 << X[inde] << ' ';
        
    //     // Add a newline character at the end of each row
    //     if ((inde + 1) % N_full == 0) {
    //         fout1 << '\n';
    //     }
    // }
    // // Close the file
    // fout1.close();
    // exit(1);

// old code for WKB tunneling:

// ** Rectangular barrier solution
                        // T = exp(-2 * sqrt((2 * p.m_e * p.V0 * eV_to_J) / (h_bar_sq)) * dist);
                        // G = 2 * 3.8612e-5 * T;
                        // std::cout << "G_rec: " << sqrt((2 * p.m_e ) / (h_bar_sq)) << "\t";
                        // X[N_full * (i + 2) + (j + 2)] = -G;
                        // X[N_full * (i + 2) + (j + 2)] = -G;


                        // double Vdiff = Vd;
                        // double xdiff = (1e-10) * (atom_x[j] - atom_x[i]); // potential accross the x-direction => if x_j < x_i then Vdiff < 0
                        // double b = Vdiff / p.t_ox;
                        // double a = 1e18; // high enough that it results in zero prob
                        // // compute the WKB tunneling coefficients (only xdiff > 0)
                        // if (abs(p.V0 / b - xdiff) < 1e-18 && xdiff > 0)
                        // { 
                        //     a = 2.0 / 3.0 * sqrt(p.V0) * xdiff;
                        // }
                        // else if (xdiff < p.V0 / b && xdiff > 0)// && xdiff > nn_dist)
                        // {                                                                         // if Vdiff < 0 then lower prob
                        //     a = -2.0 / 3.0 * (1 / b) * (pow(p.V0 - b * xdiff, 1.5) - pow(p.V0, 1.5)); // always +
                        // }
                        // else if (xdiff > p.V0 / b < 0 && xdiff > 0)
                        // {
                        //     a = -2.0/3.0*(1/b)*(-1)*pow(p.V0, 3/2); // always +
                        // }
                        // T = exp(-2 *sqrt( (2*p.m_e*eV_to_J)/(h_bar_sq) ) * a);
                        // G = 2 * 3.8612e-5 * T;
                        // X[N_full * (i + 2) + (j + 2)] = -G;
                        // X[N_full * (j + 2) + (i + 2)] = -G;