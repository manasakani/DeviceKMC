#include "Device.h"
#include "utils.h"
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <omp.h>

Site::Site(){}

void Site::init_site(int ind_, double x_, double y_, double z_, std::string element_){
  ind = ind_;
  pos.push_back(x_);
  pos.push_back(y_);
  pos.push_back(z_);
  element = element_;
}


void Site::disp_site(){
  print("atom #" << ind << " with type " << element << ": " << " at position " << pos[0] << " " << pos[1] << " " << pos[2] << " ");
}


// Construct the device
Device::Device(std::vector<std::string>& xyz_files, std::vector<double> lattice, 
			   bool shift, std::vector<double> shifts, bool pbc, double nn_dist, double T_bg){
	
	// parse xyz file(s)
	std::vector<double> x, y, z;
    std::vector<std::string> elements;
	for (auto xyz_file : xyz_files){
		this->N += read_xyz(xyz_file, elements, x, y, z);
    }
    
    // sort and prepare the raw coordinates
    this->lattice = lattice;
    this->pbc = pbc;
	this->nn_dist = nn_dist;
    sort_by_x(x, y, z, elements, lattice);
    if (shift) translate_cell(x, y, z, N, lattice, shifts);
    
    // load the xyz info into the sites
    sites.resize(N); 
    for (int i = 0; i < N; i++){
		sites[i].init_site(i, x[i], y[i], z[i], elements[i]);
		
		if (sites[i].element == "d"){
			this->N_int++;
			sites[i].isdefect = true;
		} else {
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
   
    std::cout << "Loaded " << N << " sites into device" << "\n";
    std::cout << "Consisting of " << N_atom << " atoms and " << N_int << " interstitials " << "\n";

}

void Device::constructSiteNeighborList(){
	
	double dist;
	
	// construct neighbor graph for all sites
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            dist = site_dist(sites[i].pos, sites[j].pos, lattice, pbc);
            if (dist < nn_dist && i != j){
                site_neighbors.addEdge(sites[i].ind, sites[j].ind);
            }
        }
    }

}

void Device::updateAtomNeighborList(){
    // updates (1) the atoms list and (2) the atom neighbor graph (excluding defects)
    atoms.clear();
    atom_neighbors.erase();
    
    int atom_count = 0;
    int threads_num = omp_get_max_threads();
    
    int local_iter_num = (int) std::ceil((double) N / threads_num);
    std::vector<std::vector<Site*>> atoms_local(threads_num);
    for (auto i = 0; i < threads_num; ++i) {
        atoms_local[i].reserve(local_iter_num);
    }
    
    // locate the non-defect sites
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
       
        #pragma omp for
        for (auto i = 0; i < N; i++) {
            if (sites[i].element != "d") {
                atoms_local[thread_id].push_back(&sites[i]);
            }
        }
    }
    
    // populate the atoms array with the non-defect sites
    for (auto i = 0; i < threads_num; ++i) {
        if (atoms_local[i].size() > 0) {			
            atoms.insert(atoms.end(), atoms_local[i].begin(), atoms_local[i].end());
            atom_count += atoms_local[i].size();
            atoms_local[i].clear();
        }
    }

    // construct subset neighbor graph for atoms (exclude defects):
    atom_neighbors.initialize(atom_count);
    double dist;
    #pragma omp parallel for private(dist)
    for(auto i = 0; i < N_atom; i++){
        for(auto j = 0; j < N_atom; j++){
            dist = site_dist(atoms[i]->pos, atoms[j]->pos, lattice, pbc);
            if (dist < nn_dist && i != j){
                atom_neighbors.addEdge(i, j);
            }
        }
    }
    
    //atom_neighbors.printAdjList();
       
}

//returns number of sites of element
int Device::get_num_of_element(std::string element_){
    
    int count = 0;
    for (int i = 0; i < N; i++) {
        if (sites[i].element == element_) {
            count++;
        }
    }
    return count;

}

// distributes some initial vacancies in the oxide
void Device::makeSubstoichiometric(double vacancy_concentration){

    //random number generator
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> rndm_double(0, 1);

    int num_O, num_V_add, loc;
    double random_num;

    num_O = get_num_of_element("O");
    num_V_add = vacancy_concentration*num_O;
    
    std::cout << num_V_add << " oxygen atoms will be converted to vacancies" << std::endl;
    while(num_V_add > 0){
        random_num = rndm_double(gen);
        loc = random_num*N;
        if (sites[loc].element == "O"){
            sites[loc].element = "V";
            num_V_add--;
        }
    }
}


void Device::updateCharge(){
    // this function should populate the "site_charge" attribute of this object 
}

void Device::updatePotential(){
    // this function should populate the "site_potential" attribute of this object 
}

void Device::updateTemperature(){
    // this function should populate the "site_temperature" attribute of this object 
}

void Device::writeSnapshot(std::string filename, std::string foldername){
	// cols of xyz file are "element x y z potential temperature"
    
    std::string path = "./" + foldername + "/" + filename;
    std::ofstream fout(path.c_str());
    fout << N << "\n";
    fout << "\n";

    for(int i = 0; i < N; i++){
        fout << sites[i].element <<"   "<< sites[i].pos[0] <<"   "<< sites[i].pos[1] <<"   "<< sites[i].pos[2] <<"   "<< site_potential[i] <<"   "<< site_temperature[i] << "\n";
    }
}
