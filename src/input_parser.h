#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "utils.h"


class KMCParameters
{

public:

	// random number seed
	unsigned int rnd_seed;

    // option for restart calculation
    bool restart;
    std::string restart_xyz_file;
    
    // I/O
    int log_freq;
    int output_freq;
    bool log_to_file;
    bool verbose;
    
    // device atomic structure
    std::string atom_xyz_file;
    std::string interstitial_xyz_file;
    bool pristine;
    bool shift;
    std::vector<double> shifts;
    double initial_vacancy_concentration;
    std::vector<double> lattice;
    double freq;
    double nn_dist;
    bool pbc;
    int num_atoms_first_layer;
    int num_layers_contact;
    int num_atoms_contact;
    int num_atoms_reservoir;
    std::vector<ELEMENT> metals;
    
    // Device constants
	double t_ox;                       	// thickness oxide in [m]
    double A;							// device area [m^2]
    
    // field solvers
    bool solve_potential;
    bool solve_current;
    bool solve_heating_global;
    bool solve_heating_local;
    bool perturb_structure;
    
    // Biasing scheme
    std::vector<double> V_switch;
    std::vector<double> t_switch;
    double Icc;
	double Rs; // [Ohm]

	// for potential solver
	double G_coeff = 1;
	double high_G; //[S]
	double low_G;
    double sigma; // [m]
    double epsilon;  //[1]
    
    // for current solver (tunneling parameters)
    double m_r; // [1]
    double V0;  // [eV]
    std::vector<double> alpha;
    
    // for temperature solver
    double k_therm;         // [W/mK]
    double background_temp; // [K]
    //global
    double dissipation_constant; // in [J/Ks] Tunable parameter
	double small_step;
	double event_time;
	//local
	double delta_t;                // step time [s] (rejection free event time == delta_t)
	double delta;                    // [a.u.]
	double power_adjustment_term ; // [a.u.]
	double L_char;               // characteristic length [m]
	double k_th_interface; // [W/mK]
    double tau;                                                    // Thermal rate constant [1/s]
    double k_th_metal;                                // [W/mK]
	double k_th_non_vacancy;                         // [W/mK]
	double k_th_vacancies;                             // [W/mK]
	double c_p;                               // in [J/Kcm^3]
    
    // Physical Constants
	double kB = 8.617333262e-5;        // [eV/K]
	double q = 1.60217663e-19;         // [C]
	double k;   					   // [N m^2 / C^2]
	double h_bar_sq = 4.3957e-67;      // [(Js)^2]
	double h_bar = 1.054571817e-34;    // [Js]
	double h_bar_eV = 6.582119569e-16; // [eV]
	double m_0 = 9.11e-31;             // [kg]
	double eV_to_J = 1.60217663e-19;          // [C]
	double m_e;			               // [kg]
    
	// constructor reads the parameters from the input file
	KMCParameters(std::string filename);
	
	// print all simulation parameters to file
	void print_to_file();

private:

	// get rid of inline comments
	std::string trimAfterDoubleSlash(std::string& input);
	
	// functions to read in different input types in the file
	bool read_bool(std::string line);
	unsigned int read_unsigned_int(std::string line);
	int read_int(std::string line);
	std::string read_string(std::string line);
	double read_double(std::string line);
    std::vector<double> read_vec_double(std::string line);
    std::vector<std::string> read_vec_string(std::string line);
        
    // initializes the parameters which are calculated based on other parameters
    void set_expression_parameters();
    
};

template <typename T>
void print_vec(std::vector<T>& vec){
	for (T& value : vec) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}