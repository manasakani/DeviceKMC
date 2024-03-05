#include "Device.h"

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
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

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

    // sort and prepare the raw coordinates - DO NOT SORT
    // sort_by_xyz(site_x, site_y, site_z, site_element, lattice);              // uncomment to sort the sites
    
    // shift the coordinates across periodic boundaries
    if (p.shift)
        translate_cell(site_x, site_y, site_z, N, lattice, p.shifts);

    // populate #interstitials, and #metals
    for (int i = 0; i < N; i++){
        if (is_in_vector<ELEMENT>(p.metals, site_element[i]))
        {   
            site_is_metal.push_back(1);
            this->N_metals++;
        } else {
            site_is_metal.push_back(0);
        }
    }

    // initialize and construct the neighbor lists
    auto t0 = std::chrono::steady_clock::now();
    bool build_neighidx_on_gpu = 1;                         // build ONLY (gpu) neigh_idx and NOT (cpu) site_neighbors. 
                                                            // site_neighbors will remain uninitialized and cannot be used
    if (build_neighidx_on_gpu)
    {
        if (!mpi_rank)  std::cout << "Re-Building the neighbor list on GPU - max_num_neighbors is known!\n";
        max_num_neighbors = 52; // this should be known, if building the lists on the gpu!
    
        neigh_idx.resize(N * max_num_neighbors, -1);
        construct_site_neighbor_list_gpu(neigh_idx.data(), site_x.data(), site_y.data(), site_z.data(),
                                         lattice.data(), pbc, nn_dist, N, max_num_neighbors);

    } else {

        if (!mpi_rank)  std::cout << "Building the neighbor list...\n";
        site_neighbors.initialize(N);
        constructSiteNeighborList();
        if (!mpi_rank)  std::cout << "Maximum number of neighbors in device is: " << this->max_num_neighbors << "\n";

        // check neighbors:
        for (int i = 0; i < N; i++)
        {
            if (site_neighbors.l[i].size() == 0){
                std::cout << "ERROR: Site with zero neighbors found at index: " << i << " at " << site_x[i] << " " << site_y[i] << " " << site_z[i] << "\n"; 
            }
        }

        // neighbor index used in the gpu code
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
    }    
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "Time to build neighbor list: " << dt.count() << "\n";

    // initialize the size of the field vectors
    site_charge.resize(N, 0);
    site_CB_edge.resize(N, 0);
    site_potential_boundary.resize(N, 0);
    site_potential_charge.resize(N, 0);
    site_power.resize(N, 0);
    site_temperature.resize(N, T_bg);

    // Re-identify the atomic sites (differentiate from the vacancy sites and oxygen ions)
    updateAtomLists();
    // atom_CB_edge.resize(N_atom, 0);

    std::cout << "Loaded " << N << " sites into device" << " : " << N_atom << " atoms and " << N-N_atom << " interstitials " << "\n";

}

void Device::constructSiteNeighborList()
{
    
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
    atom_CB_edge.clear();
    int atom_count = 0;

    // the elements of the atoms array are copies of the non-defects in the site array
    #pragma omp parallel for ordered schedule(static, 1)
    for (auto i = 0; i < N; i++)
    {
	    if ((site_element[i] != DEFECT) && (site_element[i] != OXYGEN_DEFECT)) // defects and Oxygen ions are not included
        {
            #pragma omp ordered
            {
            atom_x.push_back(site_x[i]);	
            atom_y.push_back(site_y[i]);
            atom_z.push_back(site_z[i]);
            atom_element.push_back(site_element[i]);
            atom_charge.push_back(site_charge[i]);
            atom_CB_edge.push_back(site_CB_edge[i]);
            atom_ind.push_back(i);
            atom_count++;
            }
        }
    }
    this->N_atom = atom_count;

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

// distributes some initial vacancies in the oxide
void Device::makeSubstoichiometric(double vacancy_concentration)
{

    int num_O, num_V_add, loc;
    double random_num;

    // get number of oxygen atoms:
    num_O = 0;
    #pragma omp parallel for reduction(+:num_O)
    for (int i = 0; i < N; i++)
    {
        if (site_element[i] == O_EL)
        {
            num_O++;
        }
    }
    num_V_add = vacancy_concentration * num_O;

    std::cout << num_V_add << " oxygen atoms will be converted to vacancies" << std::endl;
    while (num_V_add > 0)
    {
        random_num = random_generator.getRandomNumber();
        loc = random_num * N_atom;                           //changed to N_atom

        if (atom_element[loc] == O_EL)
        {
            atom_element[loc] = VACANCY;
            site_element[atom_ind[loc]] = VACANCY;
            num_V_add--;
        }
    }
}


void Device::writeSnapshot(std::string filename, std::string foldername)
{
    // cols of xyz file are "element x y z potential temperature"

    std::string path = "./" + foldername + "/" + filename;
    std::ofstream fout(path.c_str());
    if (!fout.is_open()) {
        std::cout << "Error opening file: " << path << "\n";
    }
    fout << N << "\n";
    fout << "\n";

    for (int i = 0; i < N; i++)
    {
        //fout << return_element(site_element[i]) << "   " << site_x[i] << "   " << site_y[i] << "   " << site_z[i] << "   " << site_potential_boundary[i] + site_potential_charge[i] << "   " << site_power[i] << "\n";
        fout << return_element(site_element[i]) << "   " << site_x[i] << "   " << site_y[i] << "   " << site_z[i] << "   " << site_potential_charge[i] << "   " << site_power[i] << "\n";

    }
}