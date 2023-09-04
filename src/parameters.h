//******************************
//  Global Simulation Parameters
//******************************

#include <vector>
#include <string>

// restart calculation
bool restart = 0;
std::string restart_xyz_file = "snapshot_0.xyz";

// if not restart, draws from the atom coordinates file
std::string atom_xyz_file = "structure_files/atoms.xyz";
std::string interstitial_xyz_file = "structure_files/interstitials.xyz";
// std::vector<double> lattice = {93.72586, 53.82954, 53.27438}; // 2x2 device
std::vector<double> lattice = {93.72586, 26.914773122, 26.6371955996}; // 1x1 device

// add initial vacancy concentration if starting from pristine
bool pristine = !restart;
double initial_vacancy_concentration = 0.05;
double freq = 10e13; // [1/s]
unsigned int rnd_seed = 1;

// translate cell across the periodic boundary
bool shift = !restart;
std::vector<double> shifts = {0.0, 0.5, 0.5};

// neighbor distance:
double nn_dist = 3.5; // [Angstrom]

// field solvers
bool solve_potential = 1;
bool solve_current = 0;
bool solve_heating = 0;

// for potential solver:
double G_coeff = 1;
double high_G = G_coeff * 1; //[S]
double low_G = G_coeff * 0.00000001;
double sigma = 3.5e-10; // [m]
double epsilon = 23.0;  //[1]

// for current solver (tunneling parameters):
double m_r = 0.8; // [1]
double V0 = 3.0;  // [eV]

// for temperature solver
// double alpha = 0.20; // [1]
double k_therm = 1.1;         // [W/mK]
double background_temp = 300; // [K]

// toggle for periodic boundary conditions:
bool pbc = 1;

// applied bias to right contact:
double V_switch[] = {13.00}; // [V]

// timescale for switching:
double t_switch[] = {1}; // [s]

// enforced compliance current:
double Icc = 3e-3; // 150e-6; // [A]

// frequency of device snapshot generation (per x kmc steps)
int log_freq = 10;

// frequency of printing the steps into the output file
int output_freq = 10;

// log to output.log file? (default logs to console)
bool log_to_file = 1;

// print all the events that are selected in the output file?
bool verbose = 0;

// build the device layer by layer:
int num_atoms_first_layer = 144;
int num_atoms_contact = 864;
std::vector<std::string> metals = {"Ti", "N"};

// Physical Constants
double kB = 8.617333262e-5;        // [eV/K]
double q = 1.60217663e-19;         // [C]
double k = 8.987552e9 / epsilon;   // [N m^2 / C^2]
double h_bar_sq = 4.3957e-67;      // [(Js)^2]
double h_bar = 1.054571817e-34;    // [Js]
double h_bar_eV = 6.582119569e-16; // [eV]
double m_0 = 9.11e-31;             // [kg]
double eV_to_J = 1.6e-19;          // [C]
double m_e = m_r * m_0;            // [kg]

// Device constants
const double t_ox = 52.6838e-10;                       // thickness oxide in [m]
const double A = 26.914773122e-10 * 26.6371955996e-10; // device area [m^2]
const double c_p = 1.92;                               // in [J/Kcm^3]
double k_th_metal = 29;                                // [W/mK]
double k_th_non_vacancy = 0.5;                         // [W/mK]
double k_th_vacancies = 5;                             // [W/mK]

// Power update
std::vector<double> alpha = {0.1, 0.2}; // power dissipation constant [vacancy site, non-vacancy site]

// Global thermal model
const double dissipation_constant = 70e-5; // in [J/Ks] Tunable parameter
const double small_step = 1e-16;
double event_time = 1e-13;

// Local thermal model
const double delta_t = 1e-14;                // step time [s] (rejection free event time == delta_t)
const double delta = 0.1;                    // [a.u.]
const double power_adjustment_term = 0.0001; // [a.u.]
const double L_char = 3.5e-10;               // characteristic length [m]

// Fixed parameters
const double k_th_interface = k_th_non_vacancy + (k_th_vacancies - k_th_non_vacancy) * initial_vacancy_concentration; // [W/mK]
const double tau = k_th_interface / (L_char * L_char * c_p * 1e6);                                                    // Thermal rate constant [1/s]
