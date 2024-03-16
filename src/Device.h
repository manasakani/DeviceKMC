// Device class: 
// Device.cpp, potential_solver.cpp, current_solver.cpp, heat_solver.cpp
#pragma once
#include "gpu_buffers.h"
#include "input_parser.h"
#include "random_num.h"
#include "utils.h"
#include <string>
#include <vector>
#include <random>
#include <omp.h>
#include <map>
#include <chrono>
#include <iomanip>

#include "gpu_solvers.h"

// #include <mpi.h>

// forward declaration of KMC Parameters class
class KMCParameters;

// Graph of neighbors (undirected graph)
struct Graph
{

    int N;                           // number of nodes in the graph
    std::vector<std::vector<int>> l; // list of arrays which define the neighbor connections
    bool is_constructed = 0;         // is the neighbor list populated?

    Graph() {}

    void initialize(int N)
    {
        this->N = N;
        l.resize(N);
    }

    void addEdge(int x, int y)
    {
        l[x].push_back(y);
        //l[y].push_back(x);
    }

    void erase()
    {
        for (int i = 0; i < N; i++)
        {
            l[i].clear();
        }
    }

    void printAdjList();

    ~Graph() {}
};

// A device is a collection of sites, a neighbor list, and their fields
class Device
{

public:

    // constants:
    int N = 0;                                          // number of sites in this device
    int N_atom = 0;                                     // number of atoms in this device
    int N_metals = 0;                                   // number of atoms identified as metals
    int max_num_neighbors = 0;                          // maximum number of neighbors per site
    int N_cutoff = 0;                                   // number of other potentially-charged sites within the cutoff radius
    double nn_dist;                                     // neighbor distance
    double sigma;                                       // gaussian width for potential solver
    double k;                                           //
    double T_bg;                                        // global background temperature
    bool pbc;                                           // is device periodic in the lateral directions?
    std::vector<double> lattice;                        // size of device box
    double imacro = 0.0;                                // macroscopic current

    // Neighbor lists:
    Graph site_neighbors;                               // list of neighbors of each site (including defects)
    std::vector<int> neigh_idx;                         // neighbors for the event list setup
    std::vector<int> cutoff_window;                     // Nx2 array of start and end indices for other sites within the cutoff radius
    std::vector<int> cutoff_idx;                        // Nxx array of all other potential point defects within the cutoff radius of i
    // std::vector<double> cutoff_dists;                   // Nxx array of distances to all other potential point defects within the cutoff radius of i

    // Site attributes:
    std::vector<double> site_x;
    std::vector<double> site_y;
    std::vector<double> site_z;
    std::vector<ELEMENT> site_element;
    std::vector<int> site_is_metal;                     // acts as a bool

    // Atom attributes:
    std::vector<double> atom_x;
    std::vector<double> atom_y;
    std::vector<double> atom_z;
    std::vector<double> atom_ind;
    std::vector<ELEMENT> atom_element;
    std::vector<int> atom_charge;
    std::vector<double> atom_CB_edge;                   // [eV] the conduction band edge of each atom

    // Fields:
    std::vector<int> site_charge;                       // [q] charge of each site
    std::vector<double> site_CB_edge;                   // [eV] the conduction band edge of each site
    std::vector<double> site_potential_boundary;        // [V] potential of each site (boundary solution, homogenous poisson)
    std::vector<double> site_potential_charge;          // [V] potential of each site (charge solution, inhomogenous poisson)
    std::vector<double> site_power;                     // [W] power dissipated at each site
    std::vector<double> site_temperature;               // [K] temperature of each site

    // re-usable matrices
    std::vector<double> laplacian;                      // laplacian matrix
    std::vector<double> laplacian_ss;                   // steady state laplacian
    std::vector<double> index_mapping;                  // index mappped

private:

    int N_interface = 0;
    RandomNumberGenerator random_generator;             // random number generator object for this device
    double kB = 8.617333262e-5;                         // [eV/K]
    double q = 1.60217663e-19;                          // [C]
    double h_bar_sq = 4.3957e-67;                       // [(Js)^2]
    double m_0 = 9.11e-31;                              // [kg]
    double eV_to_J = 1.6e-19;                           // [C]
    const double T_1 = 50;                              // [K] Normalization T_1 < background_temperature!!!


    //************************************
    // Main device functions / Device.cpp
    //************************************

    // Construct the device from the structure (in the xyz files) and simulation parameters
    public: Device(std::vector<std::string> &xyz_files, KMCParameters &p);

    // remove a specific percentage of oxygen from the lattice (convert to vacancies)
    public: void makeSubstoichiometric(double vacancy_concentration);

    // initialize site_neighbors depending on nn_dist
    private: void constructSiteNeighborList();

    // update the arrays of just the atoms (re-identify the defects)
    private: void updateAtomLists();

    // write an xyz file with [element, x, y, z, potential, temperature] data
    public: void writeSnapshot(std::string filename, std::string foldername);

    // returns true if neighbor
    private: bool is_neighbor(int i, int j);


    //**************************************************
    // Potential Solver functions / potential_solver.cpp
    //**************************************************

    // Solve the Laplace equation to get the CB edge along the device
    public: void setLaplacePotential(hipblasHandle_t handle_cublas, hipsolverHandle_t handle_cusolver, GPUBuffers gpubuf, 
                                     KMCParameters &p, double Vd);

    // update the charge of each vacancy and ion
    public: std::map<std::string, double> updateCharge(GPUBuffers gpubuf, std::vector<ELEMENT> metals);

    // // update the potential of each site with a cpu/gpu hybrid scheme
    // public: std::map<std::string, double> updatePotential_hybrid(hipblasHandle_t handle_cublas, hipsolverHandle_t handle_cusolver, 
    //                                                                      GPUBuffers &gpubuf, KMCParameters &p, double Vd, int kmc_step_count);

    // update the potential of each site
    public: std::map<std::string, double> updatePotential(hipblasHandle_t handle_cublas, hipsolverHandle_t handle_cusolver, 
                                                          GPUBuffers &gpubuf, KMCParameters &p, double Vd, int kmc_step_count);
    
    // resistive-network solver for the background potential
    private: void background_potential(hipsolverHandle_t handle, int num_atoms_contact, double Vd, std::vector<double> lattice,
                                       double G_coeff, double high_G, double low_G, std::vector<ELEMENT> metals, int kmc_step_num);

    // n-body poisson solver for the charged atoms - distance computations for cutoff_idx
    private: void poisson_gridless_indexed(int num_atoms_contact, std::vector<double> lattice);

    // n-body poisson solver for the charged atoms - all-to-all distance computation
    private: void poisson_gridless(int num_atoms_contact, std::vector<double> lattice);


    //*********************************************
    // Current Solver functions/ current_solver.cpp
    //*********************************************

    // update the power of each site
    public: std::map<std::string, double> updatePower(hipblasHandle_t handle, hipsolverHandle_t handle_cusolver, GPUBuffers &gpubuf, KMCParameters &p, double Vd);


    //****************************************
    // Heat Solver functions / heat_solver.cpp
    //****************************************

    // find the number of site objects located in the contacts
    private: int get_num_in_contacts(int num_atoms_contact, std::string contact_name_);

    // construct inverse of the laplacian and the steady state laplacian
    public: void constructLaplacian(KMCParameters &p);

    // update the temperature of each site
    public: std::map<std::string, double> updateTemperature(GPUBuffers &gpubuf, KMCParameters &p, double step_time);

    // update the global temperature
    private: void updateTemperatureGlobal(double event_time, double small_step, double dissipation_constant,
                                 double background_temp, double t_ox, double A, double c_p, std::map<std::string, double> result);

    // update the local and global temperatures
    private: std::map<std::string, double> updateLocalTemperature(double background_temp, double delta_t, double tau, double power_adjustment_term, double k_th_interface,
                                                         double k_th_vacancies, double num_atoms_contact, std::vector<ELEMENT> metals);

    // update the local and global temperatures in steady state
    private: std::map<std::string, double> updateLocalTemperatureSteadyState(double background_temp, double delta_t, double tau, double power_adjustment_term, double k_th_interface,
                                                                    double k_th_vacancies, double num_atoms_contact, std::vector<ELEMENT> metals);


    //*****************
    // Helper functions
    //*****************

    // returns true if thing is present in the vector of things
    template <typename T>
    bool is_in_vector(std::vector<T> things_, T thing_)
    {
        for (auto t : things_)
        {
            if (t == thing_)
                return 1;
        }
        return 0;
    }

    // write an array of numElements to filename
    template <typename T>
    void write_array_to_file(const T* array, int numElements, const std::string& filename) {
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

};
