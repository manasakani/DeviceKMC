//******************************
//  Global Simulation Parameters
//******************************

#include <vector>
#include <string>

//restart calculation
bool restart = 0;
std::string restart_xyz_file = "snapshot_0_big.xyz";

//if not restart, draws from the atom coordinates file
std::string atom_xyz_file = "structure_files/atoms.xyz";
std::string interstitial_xyz_file = "structure_files/interstitials.xyz";
//std::vector<double> lattice = {93.72586, 53.82954, 53.27438}; // 2x2 device
std::vector<double> lattice = {93.72586, 26.914773122, 26.6371955996}; // 1x1 device

//add initial vacancy concentration if starting from pristine
bool pristine = !restart;
double initial_vacancy_concentration = 0.05;

//translate cell across the periodic boundary
bool shift = !restart;
std::vector<double> shifts = {0.0, 0.5, 0.5};

//neighbor distance:
double nn_dist = 3.5; // [Angstrom]

//field solvers
bool solve_potential = 1;
bool solve_current = 0;
bool solve_heating = 0;

//for potential solver:
double G_coeff = 1;
double high_G = G_coeff*1; //[S]
double low_G = G_coeff*0.00000001;
double sigma = 3.5e-10; // [m] 
double epsilon = 23.0; //[1]

//for current solver (tunneling parameters):
double m_r = 0.8; // [1]
double V0 = 3.0; // [eV]

//for temperature solver
//double alpha = 0.20; // [1]
double k_therm = 1.1; // [W/mK]
double background_temp = 400; // [K]

//toggle for periodic boundary conditions:
bool pbc = 0;

//applied bias to right contact:
double V_switch[] = {12.00}; // [V]

//timescale for switching:
double t_switch[] = {1}; // [s]

//enforced compliance current:
double Icc = 3e-3;//150e-6; // [A]

//frequency of device snapshot generation (per x kmc steps)
int log_freq = 100;

//log to output.log file? (default logs to console)
bool log_to_file = 1;

//print all the events that are selected in the output file?
bool verbose = 0;

//build the device layer by layer:
int numlayers = 5; 
int num_atoms_first_layer = 144;
int num_atoms_contact = 864;
std::vector<std::string> metals = { "Ti", "N" };

std::string layer_0_type = "contact";
double layer_0_E_gen_0 = 0.0;
double layer_0_E_rec_1 = 0.0;
double layer_0_E_diff_2 = 0.0;
double layer_0_E_diff_3 = 0.76;
double layer_0_start_x = -22.0;
double layer_0_end_x = 0.0;

std::string layer_1_type = "interface";
double layer_1_E_gen_0 = 3.93;
double layer_1_E_rec_1 = 0.0;
double layer_1_E_diff_2 = 1.09;
double layer_1_E_diff_3 = 0.76;
double layer_1_start_x = 0.0;
double layer_1_end_x = 3.0;

std::string layer_2_type = "oxide";
double layer_2_E_gen_0 = 3.93;
double layer_2_E_rec_1 = 0.0;
double layer_2_E_diff_2 = 1.09;
double layer_2_E_diff_3 = 0.76;
double layer_2_start_x = 3.0;
double layer_2_end_x = 49.3309;

std::string layer_3_type = "interface";
double layer_3_E_gen_0 = 1.66;
double layer_3_E_rec_1 = 0.0;
double layer_3_E_diff_2 = 1.09;
double layer_3_E_diff_3 = 0.76;
double layer_3_start_x = 49.3309;
double layer_3_end_x = 54.33894;

std::string layer_4_type = "contact";
double layer_4_E_gen_0 = 0.0;
double layer_4_E_rec_1 = 0.0;
double layer_4_E_diff_2 = 0.0;
double layer_4_E_diff_3 = 2.00; //CHANGED
double layer_4_start_x = 54.33894;
double layer_4_end_x = 90.0;


//Physical Constants
double kB = 8.617333262e-5; // [eV/K]
double q = 1.60217663e-19; // [C]
double k = 8.987552e9/epsilon; // [N m^2 / C^2]
double freq = 10e13; // [1/s]
double h_bar_sq = 4.3957e-67; // [(Js)^2]
double h_bar = 1.054571817e-34; // [Js]
double h_bar_eV = 6.582119569e-16; //[eV]
double m_0 = 9.11e-31; // [kg]
double eV_to_J = 1.6e-19; // [C]
double m_e = m_r*m_0; //[kg]
