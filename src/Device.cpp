#include "Device.h"
#include "utils.h"
#include "random_num.h"
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <omp.h>
#include <map>

Site::Site() {}

void Site::init_site(int ind_, double x_, double y_, double z_, std::string element_)
{
    ind = ind_;
    pos.push_back(x_);
    pos.push_back(y_);
    pos.push_back(z_);
    element = element_;
}

void Site::disp_site()
{
    print("atom #" << ind << " with type " << element << ": "
                   << " at position " << pos[0] << " " << pos[1] << " " << pos[2] << " ");
}

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

// Construct the device
Device::Device(std::vector<std::string> &xyz_files, std::vector<double> lattice,
               bool shift, std::vector<double> shifts, bool pbc, double sigma, double epsilon,
               double nn_dist, double background_temp, unsigned int rnd_seed)
{

    // initialize the random number generator
    random_generator.setSeed(rnd_seed);

    // parse xyz file(s)
    std::vector<double> x, y, z;
    std::vector<std::string> elements;
    for (auto xyz_file : xyz_files)
    {
        this->N += read_xyz(xyz_file, elements, x, y, z);
    }

    // set parameters
    this->lattice = lattice;
    this->pbc = pbc;
    this->nn_dist = nn_dist;
    this->sigma = sigma;
    this->k = k = 8.987552e9 / epsilon;
    this->T_bg = background_temp;

    // sort and prepare the raw coordinates
    sort_by_x(x, y, z, elements, lattice);
    if (shift)
        translate_cell(x, y, z, N, lattice, shifts);

    // load the xyz info into the sites
    sites.resize(N);
    for (int i = 0; i < N; i++)
    {
        sites[i].init_site(i, x[i], y[i], z[i], elements[i]);

        if (sites[i].element == "d")
        {
            this->N_int++;
            sites[i].isdefect = true;
        }
        else
        {
            this->N_atom++;
        }
    }

    // initialize and construct the neighbor lists
    site_neighbors.initialize(N);
    constructSiteNeighborList();
    updateAtomNeighborList();

    // initialize the size of the field vectors
    site_charge.resize(N);
    site_potential.resize(N);
    site_power.resize(N);
    site_temperature.resize(N, T_bg);

    std::cout << "Loaded " << N << " sites into device"
              << "\n";
    std::cout << "Consisting of " << N_atom << " atoms and " << N_int << " interstitials "
              << "\n";
}

void Device::constructSiteNeighborList()
{

    double dist;

    // construct neighbor graph for all sites
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            dist = site_dist(sites[i].pos, sites[j].pos, lattice, pbc);
            if (dist < nn_dist && i != j)
            {
                site_neighbors.addEdge(sites[i].ind, sites[j].ind);
            }
        }
    }
}

void Device::updateAtomNeighborList()
{
    // updates (1) the atoms list and (2) the atom neighbor graph (excluding defects)

    // reset the atoms array and neighbor list
    atoms.clear();
    if (!atom_neighbors.l.empty())
    {
        atom_neighbors.erase();
    }
    int atom_count = 0;

    // !! check for race conditions or mem leak!!

    /*int threads_num = omp_get_max_threads();
    int local_iter_num = (int)std::ceil((double)N / threads_num);
    std::vector<std::vector<Site *>> atoms_local(threads_num);
    for (auto i = 0; i < threads_num; ++i)
    {
        atoms_local[i].reserve(local_iter_num);
    }

// locate the non-defect sites
#pragma omp parallel num_threads(1) // DEBUG
    {
        int thread_id = omp_get_thread_num();

#pragma omp for
        for (auto i = 0; i < N; i++)
        {
            if (sites[i].element != "d")
            {
                atoms_local[thread_id].push_back(&sites[i]);
            }
        }
    }

    // populate the atoms array with the non-defect sites
    for (auto i = 0; i < threads_num; ++i)
    {
        if (atoms_local[i].size() > 0)
        {
            atoms.insert(atoms.end(), atoms_local[i].begin(), atoms_local[i].end());
            atom_count += atoms_local[i].size();
            atoms_local[i].clear();
        }
    }

    // construct subset neighbor graph for atoms (exclude defects):
    atom_neighbors.initialize(atom_count);
    double dist;
#pragma omp parallel for private(dist) num_threads(1) // DEBUG
    for (auto i = 0; i < N_atom; i++)
    {
        for (auto j = 0; j < N_atom; j++)
        {
            dist = site_dist(atoms[i]->pos, atoms[j]->pos, lattice, pbc);
            if (dist < nn_dist && i != j)
            {
                atom_neighbors.addEdge(i, j);
            }
        }
    }*/

    // !! check for race conditions or mem leak!!

    // !! unoptimized !!

    // the elements of the atoms array are pointers to the atoms in the site array
    for (auto i = 0; i < N; i++)
    {
        if (sites[i].element != "d")
        {
            atoms.push_back(&sites[i]);
            atom_count++;
        }
    }

    // construct subset neighbor graph for atoms (exclude defects):
    double dist;
    atom_neighbors.initialize(atom_count);
    for (auto i = 0; i < atom_count; i++)
    {
        for (auto j = 0; j < atom_count; j++)
        {
            dist = site_dist(atoms[i]->pos, atoms[j]->pos, lattice, pbc);
            if (dist < nn_dist && i != j)
            {
                atom_neighbors.addEdge(i, j);
            }
        }
    }

    // !! unoptimized !!

    this->N_atom = atom_count;
}

// Computes the total number of atoms
int Device::get_num_metals(std::vector<std::string> metals)
{
    int count = 0;
    for (int i = 0; i < N; i++)
    {
        std::string element_ = sites[i].element;
        if (element_ == metals[0] || element_ == metals[1])
        {
            count++;
        }
    }
    return count;
}

// construct laplacian and steady state laplacian
void Device::constructLaplacian(double k_th_interface, double k_th_metal, double delta,
                                double delta_t, double tau, std::vector<std::string> metals, double background_temp,
                                double num_atoms_contact)
{
    print("constructing graph Laplacian");

    // Get the number of interface atoms
    int N_left_tot = get_num_in_contacts(num_atoms_contact, "left");
    int N_metals = get_num_metals(metals);
    int N_right_tot = get_num_in_contacts(N_metals - num_atoms_contact, "right");
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

    int info;
    // Map the index to a new array
    int new_index = 0;

    // Calculate constants
    double gamma = 1 / (delta * ((k_th_interface / k_th_metal) + 1)); // [a.u.]
    double step_time = delta_t * tau;                                 // [a.u.]

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
            index_mapping[i] = -1;
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
                metal_atom2 = is_present(metals, sites[j].element);

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

    // LU factorization of (I-L) (overwrite L_T with the factorization)
    dgetrf_(&N_interface, &N_interface, L_inv, &N_interface, ipiv_L_T, &info);

    // LU factorization of (L) (overwrite L_T with the factorization)
    dgetrf_(&N_interface, &N_interface, L_ss, &N_interface, ipiv_L_ss_T, &info);

    // Compute the inverse of the matrix L_T using the LU factorization (overwrite A with the factorization)
    dgetri_(&N_interface, L_inv, &N_interface, ipiv_L_T, work, &lwork, &info);

    // Compute the inverse of the matrix L_T using the LU factorization (overwrite A with the factorization)
    dgetri_(&N_interface, L_ss, &N_interface, ipiv_L_ss_T, work_ss, &lwork_ss, &info);

    // Update the inverse of the laplacian and steady state laplacian
#pragma omp parallel for collapse(2)
    for (int i = 0; i < N_interface; i++)
    {
        for (int j = 0; j < N_interface; j++)
        {
            laplacian[i * N_interface + j] = L_inv[i * N_interface + j];
        }
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < N_interface; i++)
    {
        for (int j = 0; j < N_interface; j++)
        {
            laplacian_ss[i * N_interface + j] = L_ss[i * N_interface + j];
        }
    }

    free(ipiv_L_T);
    free(L);
    free(ipiv_L_ss_T);
    free(work_ss);
    free(work);
    free(L_ss);
}

// returns number of sites of element
int Device::get_num_of_element(std::string element_)
{

    int count = 0;
    for (int i = 0; i < N; i++)
    {
        if (sites[i].element == element_)
        {
            count++;
        }
    }
    return count;
}

// returns true if neighbor
bool Device::is_neighbor(int i, int j)
{

    for (int a : site_neighbors.l[i])
    {
        if (a == j)
        {
            return 1;
        }
    }

    return 0;
}

// returns true if metal site
bool Device::is_present(std::vector<std::string> metals, std::string element_)
{
    for (int i = 0; i < metals.size(); i++)
    {
        if (metals[i] == element_)
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
            if (sites[count].element != "d")
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
            if (sites[count - 1].element != "d")
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

    num_O = get_num_of_element("O");
    num_V_add = vacancy_concentration * num_O;

    std::cout << num_V_add << " oxygen atoms will be converted to vacancies" << std::endl;
    while (num_V_add > 0)
    {
        random_num = random_generator.getRandomNumber();
        loc = random_num * N;
        if (sites[loc].element == "O")
        {
            sites[loc].element = "V";
            num_V_add--;
        }
    }
}

// update the charge of each vacancy and ion
std::map<std::string, int> Device::updateCharge(std::vector<std::string> metals)
{
    int Vnn;
    int uncharged_V_counter = 0;
    int V_counter = 0;
    int Od_counter = 0;
    int uncharged_Od_counter = 0;
    int nn_cond = 3;

#pragma omp parallel for private(Vnn) reduction(+ : uncharged_V_counter, uncharged_Od_counter, V_counter, Od_counter)
    for (int i = 0; i < N; i++)
    {

        if (sites[i].element == "V")
        {
            V_counter++;
            Vnn = 0;
            site_charge[i] = 2;
            for (int j : site_neighbors.l[i])
            {
                bool metal = is_present(metals, sites[j].element);
                if (sites[j].element == "V")
                {
                    Vnn++;
                };
                if (sites[j].element == metals[0] || sites[j].element == metals[1])
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

        if (sites[i].element == "Od")
        {
            Od_counter++;
            site_charge[i] = -2;
            for (int j : site_neighbors.l[i])
            {
                if (sites[j].element == metals[0] || sites[j].element == metals[1])
                {
                    site_charge[i] = 0;
                    uncharged_Od_counter++;
                    break;
                }
            }
        }
    }

    std::map<std::string, int> result;

    result["Uncharged vacancies"] = uncharged_V_counter;
    result["Charged vacancies"] = V_counter - uncharged_V_counter;
    result["Uncharged oxygen ions"] = uncharged_Od_counter;
    result["Charged oxygen ions"] = Od_counter - uncharged_Od_counter;

    return result;
}

// update the potential of each site
void Device::background_potential(int num_atoms_contact, double Vd, std::vector<double> lattice,
                                  double G_coeff, double high_G, double low_G, std::vector<std::string> metals)
{

    std::map<std::string, int> result;
    int N_left_tot = get_num_in_contacts(num_atoms_contact, "left");
    int N_right_tot = get_num_in_contacts(num_atoms_contact, "right");
    int N_interface = N - N_left_tot - N_right_tot;

    int one = 1;
    int info, cntr;

    double *K = (double *)calloc(N * N, sizeof(double));
    double *VL = (double *)malloc(N_left_tot * sizeof(double));
    double *VR = (double *)malloc(N_right_tot * sizeof(double));
    double *D = (double *)malloc(N_interface * N_interface * sizeof(double));

    double *Ksub = (double *)calloc(N_interface, sizeof(double));
    int *ipiv = (int *)calloc(N_interface, sizeof(int));

#pragma omp parallel
    {

#pragma omp for
        for (int i = 0; i < N_left_tot; i++)
        {
            VL[i] = 0;
        }

#pragma omp for
        for (int i = 0; i < N_right_tot; i++)
        {
            VR[i] = Vd;
        }

// construct parametrized conductivity matrix (populates the K array)
// neighbor conductances - off-diagonals of K
#pragma omp for
        for (int i = 0; i < N; i++)
        {
            bool metal1 = is_present(metals, sites[i].element);
            bool cvacancy1 = sites[i].element == "V" && site_charge[i] == 0;

            for (int j : site_neighbors.l[i])
            {
                bool metal2 = is_present(metals, sites[j].element);
                bool cvacancy2 = sites[j].element == "V" && site_charge[j] == 0;

                if ((metal1 && metal2) || (cvacancy1 && cvacancy2))
                {
                    K[N * sites[i].ind + sites[j].ind] = -high_G;
                    K[N * sites[j].ind + sites[i].ind] = -high_G;
                }
                else
                {
                    K[N * sites[i].ind + sites[j].ind] = -low_G;
                    K[N * sites[j].ind + sites[i].ind] = -low_G;
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

#pragma omp for collapse(2)
        for (int i = N_left_tot; i < N - N_right_tot; i++)
        {
            for (int j = N_left_tot; j < N - N_right_tot; j++)
            {
                D[(i - N_left_tot) * N_interface + (j - N_left_tot)] = K[i * N + j];
            }
        }

    } // thread meetup

    // do Ax = b -> VSW = -inv(D)*Ksub -> -D*VSW = Ksub
    dgesv_(&N_interface, &one, D, &N_interface, ipiv, Ksub, &N_interface, &info);

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

    if (info != 0)
    {
        print("Warning: error in linear system solver for background potential!");
    }

    // Solving the oxide potential considering only pairwise coulomb interactions.
    // solve poisson's equation with zero BC

    free(K);
    free(D);
    free(VL);
    free(VR);
    free(Ksub);
    free(ipiv);
}

void Device::poisson_gridless(int num_atoms_contact, std::vector<double> lattice)
{
    int N_left_tot = get_num_in_contacts(num_atoms_contact, "left");
    int N_right_tot = get_num_in_contacts(num_atoms_contact, "right");
    int N_interface = N - N_left_tot - N_right_tot;

#pragma omp parallel for
    for (int i = N_left_tot; i < N - N_right_tot; i++)
    {
        double V_temp = 0;
        double r_dist;

        for (int j = 0; j < N; j++)
        {

            if (i != j && site_charge[j] != 0)
            {
                r_dist = (1e-10) * site_dist(sites[i].pos, sites[j].pos, lattice, pbc);
                // solution for the potential r_dist away from a gaussian charge distribution with width sigma
                V_temp += site_charge[j] * erfc(r_dist / (sigma * sqrt(2))) * k * q / r_dist;
            }
        }
        site_potential[i] += V_temp;
    }
}

// update the potential of each site
void Device::updatePotential(int num_atoms_contact, double Vd, std::vector<double> lattice,
                             double G_coeff, double high_G, double low_G, std::vector<std::string> metals)
{
    // circuit-model-based potential solver
    background_potential(num_atoms_contact, Vd, lattice,
                         G_coeff, high_G, low_G, metals);

    // gridless Poisson equation solver (using sum of gaussian charge distribution solutions)
    poisson_gridless(num_atoms_contact, lattice);
}

// update the power of each site
std::map<std::string, double> Device::updatePower(cublasHandle_t handle, int num_atoms_first_layer, double Vd, double high_G, double low_G_1,
                                                  std::vector<std::string> metals, double m_e, double V0)
{
    // Map
    std::map<std::string, double> result;

    // Re-identify the atomic sites (differentiate from the vacancy sites)
    updateAtomNeighborList();

    // number of injection nodes
    int num_source_inj = num_atoms_first_layer;
    int num_ground_ext = num_source_inj;

    // total number of nodes, including injection and extraction
    int N_full = N_atom + 1 + 1;
    int Nsub = N_full - 1;
    double I_cal;
    double I_macro = 0.0;

    // needs to be init to zeros
    double *M = (double *)calloc(N_full, sizeof(double));
    double *X = (double *)calloc(N_full * N_full, sizeof(double));
    double *I_pos = (double *)calloc(N_atom * N_atom, sizeof(double));

    // does not need to be init to zeros
    double *D_T = (double *)malloc(Nsub * Nsub * sizeof(double));
    int *ipiv_T = (int *)malloc(Nsub * sizeof(int));

    int i, j, info;
    int one = 1;
    double zero = 0.0;
    double dist, G, T;
    char trans = 'N';

    M[0] = -high_G * Vd;
    M[1] = high_G * Vd;

#pragma omp parallel private(i, j)
    {

// Build the X conductivity matrix
#pragma omp for private(i, j, dist, T, G)
        for (i = 0; i < N_atom; i++)
        {

            for (j = i; j < N_atom; j++)
            {

                bool metal1, metal2, cvacancy1, cvacancy2, vacancy1, vacancy2, neighbor, V_V, V_contact;

                // contacts
                metal1 = is_present(metals, atoms[i]->element);
                metal2 = is_present(metals, atoms[j]->element);

                // conductive vacancies
                cvacancy1 = atoms[i]->element == "V" && site_charge[atoms[i]->ind] == 0;
                cvacancy2 = atoms[j]->element == "V" && site_charge[atoms[i]->ind] == 0;

                // charged vacancies
                vacancy1 = atoms[i]->element == "V" && site_charge[atoms[i]->ind] != 0;
                vacancy2 = atoms[j]->element == "V" && site_charge[atoms[i]->ind] != 0;

                neighbor = is_neighbor(i, j);

                // direct terms:
                if (i != j && neighbor)
                {
                    if ((metal1 && metal2) || (cvacancy1 && cvacancy2))
                    {
                        X[N_full * (i + 2) + (j + 2)] = -high_G;
                        X[N_full * (j + 2) + (i + 2)] = -high_G;
                    }
                    else
                    {
                        X[N_full * (i + 2) + (j + 2)] = -low_G_1;
                        X[N_full * (j + 2) + (i + 2)] = -low_G_1;
                    }
                }

                // tunneling terms
                if (i != j && !neighbor)
                {
                    V_V = (vacancy1 && vacancy2) || (vacancy2 && cvacancy1) || (vacancy1 && cvacancy2) || (cvacancy1 && cvacancy2);
                    V_contact = (vacancy1 && metal2) || (vacancy2 && metal1) || (cvacancy1 && metal2) || (cvacancy2 && metal1);

                    if (V_V || V_contact)
                    {
                        dist = (1e-10) * site_dist(atoms[i]->pos, atoms[j]->pos, lattice, pbc);
                        T = exp(-2 * sqrt((2 * m_e * V0 * eV_to_J) / (h_bar_sq)) * dist);
                        G = 2 * 3.8612e-5 * T;
                        X[N_full * (i + 2) + (j + 2)] = -G;
                        X[N_full * (j + 2) + (i + 2)] = -G;
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
            X[0 * N_full + 1] = -high_G;
            X[1 * N_full + 0] = -high_G;
        }

// diagonals of X
#pragma omp for
        for (i = 0; i < N_full; i++)
        {
            for (j = 0; j < N_full; j++)
            {
                if (i != j)
                {
                    X[i * N_full + i] += -1 * X[i * N_full + j];
                }
            }
        }

// Prepare D_T to solve for the virtual potentials
#pragma omp for collapse(2)
        for (i = 0; i < Nsub; i++)
        {
            for (j = 0; j < Nsub; j++)
            {
                D_T[i * Nsub + j] = X[i * N_full + j];
            }
        }
    }

    dgesv_(&Nsub, &one, D_T, &Nsub, ipiv_T, M, &Nsub, &info);
    // M now contains the virtual potentials

#pragma omp parallel private(I_cal, i, j)
    {

// bond-resolved currents
#pragma omp for
        for (i = 0; i < N_atom; i++)
        {
            for (j = i + 1; j < N_atom; j++)
            {
                I_cal = X[N_full * (i + 2) + (j + 2)] * (M[j + 2] - M[i + 2]);

                if (I_cal > 0 && Vd > 0)
                {
                    I_pos[i * N_atom + j] = I_cal;
                }
                else if (I_cal < 0 && Vd < 0)
                {
                    I_pos[i * N_atom + j] = I_cal;
                }
                else
                {
                    I_pos[j * N_atom + i] = -I_cal;
                }
            }
        }

// macroscopic current
#pragma omp for reduction(+ : I_macro)
        for (i = 2; i < N_atom; i++)
        {
            I_macro += I_pos[N_atom + i];
        }
    }

    double Geq = std::abs(I_macro / Vd);
    double *I_neg = (double *)malloc(N_atom * N_atom * sizeof(double));
    double one_d = 1.0;
    double P_disp[N_atom];
    double min_V = *std::min_element(M + 2, M + N_full);

#pragma omp parallel private(I_cal, i, j)
    {
// shifting the virtual potential by its minimum
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
                I_cal = X[N_full * (i + 2) + (j + 2)] * (M[j + 2] - M[i + 2]);

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
                    // I_neg[i*N_atom + i] += -1*I_neg[j*N_atom + i]; // sum col
                }
            }
        }
    }

    // dissipated power at each atom
    gemm(handle, &trans, &trans, &N_atom, &one, &N_atom, &one_d, I_neg, &N_atom, &M[2], &N_atom, &zero, P_disp, &N_atom);

#pragma omp parallel for
    for (i = num_source_inj; i < N_atom - num_source_inj; i++)
    {

        double alpha;
        bool metal = is_present(metals, atoms[i]->element);
        bool vacancy = atoms[i]->element == "V";

        if (metal)
        {
            alpha = 0.0;
        }
        else if (vacancy)
        {
            alpha = 0.10;
        }
        else
        {
            alpha = 0.20;
        }

        site_power[atoms[i]->ind] = -1 * alpha * P_disp[i];
    }

    free(D_T);
    free(M);
    free(X);
    free(I_pos);
    free(I_neg);
    free(ipiv_T);

    result["Current in uA"] = I_macro * 1e6;
    result["Conductance in uS"] = Geq * 1e6;
    // To do: put alpha in the parameter file
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
                                                             double k_th_vacancies, double num_atoms_contact, std::vector<std::string> metals)
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
    double step_time = t * tau;                                                                                                 // [a.u.]                                                               // [a.u.]
    const double p_transfer_vacancies = power_adjustment_term / ((nn_dist * (1e-10) * k_th_interface) * (T_1 - background_temp));     // [a.u.]
    const double p_transfer_non_vacancies = power_adjustment_term / ((nn_dist * (1e-10) * k_th_vacancies) * (T_1 - background_temp)); // [a.u.]

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

                    if (sites[j].element == "V")
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
#pragma omp parallel
    {
#pragma omp for reduction(+ : T_tot)
        for (int i = 0; i < N; i++)
        {
            T_tot += site_temperature[i];
        }
    }

    T_bg = T_tot / N;
    result["Global temperature in K"] = T_bg;
    free(T_vec);
    return result;
}

// update the local and global temperature in steady state
std::map<std::string, double> Device::updateLocalTemperatureSteadyState(double background_temp, double delta_t, double tau, double power_adjustment_term, double k_th_interface,
                                                                        double k_th_vacancies, double num_atoms_contact, std::vector<std::string> metals)
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
    const double p_transfer_vacancies = power_adjustment_term / ((nn_dist * (1e-10) * k_th_interface) * (T_1 - background_temp));     // [a.u.]
    const double p_transfer_non_vacancies = power_adjustment_term / ((nn_dist * (1e-10) * k_th_vacancies) * (T_1 - background_temp)); // [a.u.]

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

                    if (sites[j].element == "V")
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
#pragma omp parallel
    {
#pragma omp for reduction(+ : T_tot)
        for (int i = 0; i < N; i++)
        {
            T_tot += site_temperature[i];
        }
    }

    T_bg = T_tot / N;
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
        fout << sites[i].element << "   " << sites[i].pos[0] << "   " << sites[i].pos[1] << "   " << sites[i].pos[2] << "   " << site_potential[i] << "   " << site_temperature[i] << "\n";
    }
}
