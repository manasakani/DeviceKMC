// Device class
#include <string>
#include <vector>
#include <random>
#include <omp.h>
#include "random_num.h"
#include <map>
#include "utils.h"

#ifndef DEVICE_H
#define DEVICE_H

// Base struct for individual vacancy or atom:
struct Site
{

    int ind;                 // index in neighbor list
    std::vector<double> pos; // xyz position
    std::string element;     // atomic element ('d' for defect)
    bool isdefect;           // is this a interstitial position?

    Site();
    void init_site(int ind_, double x_, double y_, double z_, std::string element);
    void disp_site();
};

// generic graph
struct Graph
{

    int N;                           // number of nodes in the graph
    std::vector<std::vector<int>> l; // list of arrays which define the neighbor connections

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
    int N = 0;                 // number of sites in this device
    std::vector<Site> sites;   // list of sites in this device
    std::vector<Site *> atoms; // list of pointers to atoms in the sites array (exlcuding defects)
    Graph site_neighbors;      // list of neighbors of each site (including defects)
    Graph atom_neighbors;      // list of neighbors of each atom (excluding defects)
    double nn_dist;            // neighbor distance
    double sigma;              // gaussian width for potential solver
    double k;                  //
    double T_bg;               // global background temperature

    std::vector<int> site_charge;         // charge of each site
    std::vector<double> site_potential;   // potential of each site
    std::vector<double> site_power;       // power of each site
    std::vector<double> site_temperature; // temperature of each site

    std::vector<double> laplacian;     // laplacian matrix
    std::vector<double> laplacian_ss;  // steady state laplacian
    std::vector<double> index_mapping; // index mappped

    // constructor from input xyz file(s)
    Device(std::vector<std::string> &xyz_files, std::vector<double> lattice,
           bool shift, std::vector<double> shifts, bool pbc, double sigma, double epsilon,
           double nn_dist, double background_temp, unsigned int rnd_seed);

    // get number of sites with this element
    int get_num_of_element(std::string element_);

    // find the number of site objects located in the contacts
    int get_num_in_contacts(int num_atoms_contact, std::string contact_name_);

    // returns true if neighbor
    bool is_neighbor(int i, int j);

    // returns true if metal site
    bool is_present(std::vector<std::string> metals, std::string element_);

    // remove a specific percentage of oxygen from the lattice (convert to vacancies)
    void makeSubstoichiometric(double vacancy_concentration);

    // construct inverse of the laplacian and the steady state laplacian
    void constructLaplacian(cusolverDnHandle_t handle, double k_th_interface, double k_th_metal, double delta,
                            double delta_t, double tau, std::vector<std::string> metals, double background_temp,
                            double num_atoms_contact);

    // update the neighbor list of just the atoms (excluding the defects)
    void updateAtomNeighborList();

    // update the charge of each vacancy and ion
    std::map<std::string, int> updateCharge(std::vector<std::string> metals);

    int get_num_metals(std::vector<std::string> metals);

    void background_potential(cusolverDnHandle_t handle, int num_atoms_contact, double Vd, std::vector<double> lattice,
                              double G_coeff, double high_G, double low_G, std::vector<std::string> metals);

    void poisson_gridless(int num_atoms_contact, std::vector<double> lattice);

    // update the potential of each site
    void updatePotential(cusolverDnHandle_t handle, int num_atoms_contacts, double Vd, std::vector<double> lattice,
                         double G_coeff, double high_G, double low_G, std::vector<std::string> metals);

    // update the power of each site
    std::map<std::string, double> updatePower(cublasHandle_t handle, cusolverDnHandle_t handle_cusolver, int num_atoms_first_layer, double Vd, double high_G, double low_G_1,
                                              std::vector<std::string> metals, double m_e, double V0);

    // update the temperature of each site
    std::map<std::string, double> updateTemperatureGlobal(double event_time, double small_step, double dissipation_constant,
                                                          double background_temp, double t_ox, double A, double c_p);

    // update the local and global temperature
    std::map<std::string, double> updateLocalTemperature(double background_temp, double delta_t, double tau, double power_adjustment_term, double k_th_interface,
                                                         double k_th_vacancies, double num_atoms_contact, std::vector<std::string> metals);

    // update the local and global temperature in steady state
    std::map<std::string, double> updateLocalTemperatureSteadyState(double background_temp, double delta_t, double tau, double power_adjustment_term, double k_th_interface,
                                                                    double k_th_vacancies, double num_atoms_contact, std::vector<std::string> metals);

    // write an xyz file with [element, x, y, z, potential, temperature] data
    void writeSnapshot(std::string filename, std::string foldername);

private:
    int N_atom = 0; // number of atoms in this device
    int N_int = 0;  // number of available interstitial (defect) sites
    int N_interface = 0;
    std::vector<double> lattice;            // size of device box
    bool pbc;                               // is device periodic in the lateral directions?
    RandomNumberGenerator random_generator; // random number generator object for this device
    double kB = 8.617333262e-5;             // [eV/K]
    double q = 1.60217663e-19;              // [C]
    double h_bar_sq = 4.3957e-67;           // [(Js)^2]
    double m_0 = 9.11e-31;                  // [kg]
    double eV_to_J = 1.6e-19;               // [C]
    const double T_1 = 50;                  // [K] Normalization T_1 < background_temperature!!!
    // initialize site_neighbors depending on nn_dist
    void constructSiteNeighborList();

    // initialize laplacian for heat solver
    // void initLaplacian();
};
#endif
