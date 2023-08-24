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
               bool shift, std::vector<double> shifts, bool pbc, double nn_dist, double T_bg, unsigned int rnd_seed)
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

    // sort and prepare the raw coordinates
    this->lattice = lattice;
    this->pbc = pbc;
    this->nn_dist = nn_dist;
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

    int threads_num = omp_get_max_threads();
    int local_iter_num = (int)std::ceil((double)N / threads_num);
    std::vector<std::vector<Site *>> atoms_local(threads_num);
    for (auto i = 0; i < threads_num; ++i)
    {
        atoms_local[i].reserve(local_iter_num);
    }

// locate the non-defect sites
#pragma omp parallel
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
#pragma omp parallel for private(dist)
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
    }

    this->N_atom = atom_count;
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
        while (i > N - num_atoms_contact)
        {
            if (sites[count].element != "d")
            {
                i--;
            }
            count++;
        }
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

std::map<std::string, int> Device::updateCharge()
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
                if (sites[j].element == "V")
                {
                    Vnn++;
                };
                if (sites[j].element == "Ti" || sites[j].element == "N")
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
                if (sites[j].element == "Ti")
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

std::map<std::string, int> Device::updatePotential(int num_atoms_contact, double Vd, std::vector<double> lattice, bool pbc, double sigma, double k)
{
    std::map<std::string, int> result;
    int N_left_tot = get_num_in_contacts(num_atoms_contact, "left");
    int N_right_tot = get_num_in_contacts(num_atoms_contact, "right");
    int N_interface = N - N_left_tot - N_right_tot;

    // Ask about this
    double G_coeff = 1;
    double high_G = G_coeff * 1; //[S]
    double low_G = G_coeff * 0.00000001;
    std::vector<std::string> metals = {"Ti", "N"};

    int one = 1;
    int info, cntr;

    double *K = (double *)calloc(N * N, sizeof(double));

    double *VL = (double *)malloc(N_left_tot * sizeof(double));
    double *VR = (double *)malloc(N_right_tot * sizeof(double));
    double *D = (double *)malloc(N_interface * N_interface * sizeof(double));

    double *Ksub = (double *)calloc(N_interface, sizeof(double));
    int *ipiv = (int *)calloc(N_interface, sizeof(int));

#pragma omp parallel // num_threads(1)
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
                bool metal2 = is_present(metals, sites[i].element);
                bool cvacancy2 = sites[j].element == "V" && site_charge[i] == 0;

                if ((metal1 && metal2) || (cvacancy1 && cvacancy2))
                {
                    K[N * sites[i].ind + sites[j].ind] = -high_G;
                }
                else
                {
                    K[N * sites[i].ind + sites[j].ind] = -low_G;
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

    free(K);
    free(D);
    free(VL);
    free(VR);
    free(Ksub);
    free(ipiv);

    return result;
}

void Device::updatePower()
{
    // this function should populate the "site_power" attribute of this object
}

void Device::updateTemperature()
{
    // this function should populate the "site_temperature" attribute of this object
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
