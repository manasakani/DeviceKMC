#include <iostream>
#include <vector>
#include <cstdio>
#include <cstddef>
#include <stdlib.h>
#include <chrono>
#include "KMCProcess.h"
#include "parameters.h"
#include "utils.h"
#include "Device.h"
#include <map>

// main function for KMC simulation
int main()
{

    // set up logging
    std::ostringstream outputBuffer;
    std::ofstream outputFile("output.txt", std::ios_base::app);
    outputBuffer << "----------------------------\n";
    outputBuffer << "Starting Kinetic Monte Carlo\n";
    outputBuffer << "----------------------------\n";
    outputFile << outputBuffer.str();

    // Initialize device
    std::vector<std::string> xyz_files;
    if (restart)
    {
        const bool file_exists = location_exists(restart_xyz_file);
        if (!file_exists)
        {
            outputBuffer << "ERROR: Restart file " << restart_xyz_file << " not found!\n";
            outputFile << outputBuffer.str();
            return 0;
        }
        else
        {
            xyz_files.push_back(restart_xyz_file);
        }
    }
    else
    {
        xyz_files.push_back(atom_xyz_file);
        xyz_files.push_back(interstitial_xyz_file);
    }

    Device device(xyz_files, lattice, shift, shifts, pbc, sigma, epsilon, nn_dist, background_temp, rnd_seed);
    if (pristine)
        device.makeSubstoichiometric(initial_vacancy_concentration);
    outputBuffer << "Device was constructed\n";

    // ****** TESTING FIELD SOLVERS ********

    // Voltage
    /*double Vd;
    Vd = V_switch[0]; // Testing change this with the counter !!!

    // Initialize the fields
    std::map<std::string, int> chargeMap = device.updateCharge(metals);

    // Potential update
    device.updatePotential(num_atoms_contact, Vd, lattice,
                           G_coeff, high_G, low_G, metals);

    // Power update
    std::map<std::string, double> powerMap = device.updatePower(num_atoms_first_layer, Vd, high_G, low_G,
                                                                metals, m_e, V0);

    // Map
    std::map<std::string, double> temperatureMap = device.updateTemperatureGlobal(event_time, small_step, dissipation_constant,
                                                                                  background_temp, t_ox, A, c_p);

    for (const auto &pair : temperatureMap)
    {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    // Compute the laplacian
    device.constructLaplacian(k_th_interface, k_th_metal, delta,
                              delta_t, tau, metals, background_temp,
                              num_atoms_contact);

    int counter = 1;
    while (counter <= 0)
    {
        std::map<std::string, double> localTemperatureMap = device.updateLocalTemperature(background_temp, delta_t, tau, power_adjustment_term, k_th_interface,
                                                                                          k_th_vacancies, num_atoms_contact, metals);
        // Debug purposes
        if (counter == 0)
        {
            for (const auto &pair : localTemperatureMap)
            {
                std::cout << pair.first << ": " << pair.second << std::endl;
            }
        }

        counter++;
    }

    bool a = 1;
    if (a)
    {
        std::map<std::string, double> localTemperatureSSMap = device.updateLocalTemperatureSteadyState(background_temp, delta_t, tau, power_adjustment_term, k_th_interface,
                                                                                                       k_th_vacancies, num_atoms_contact, metals);
        // Debug purposes
        for (const auto &pair : localTemperatureSSMap)
        {
            std::cout << pair.first << ": " << pair.second << std::endl;
        }
    }

    const std::string file_name = "snapshot_init.xyz";
    device.writeSnapshot(file_name, "./");*/

    // ****** TESTING FIELD SOLVERS ********

    // Initialize KMC simulation

    KMCProcess sim(&device);
    outputBuffer << "initialized KMC simulation\n";
    outputBuffer.str(std::string());

    // loop over V_switch and t_switch
    double Vd, t, kmc_time, I_macro, T_kmc;
    int kmc_step_count;
    double *begin = std::begin(V_switch);
    double *end = std::end(V_switch);
    int vt_counter = 0;
    std::map<std::string, int> resultMap;

    std::chrono::duration<double> diff, diff_pot, diff_temp, diff_perturb;
    for (double *p = begin; p != end; ++p)
    {
        Vd = V_switch[vt_counter];
        t = t_switch[vt_counter];
        outputBuffer << "--------------------------------\n";
        outputBuffer << "Applied Voltage = " << Vd << " V\n";
        outputBuffer << "--------------------------------\n";

        const std::string folder_name = "Results_" + std::to_string(Vd);
        const bool folder_already_exists = location_exists(folder_name);
        if (!folder_already_exists)
        {
            const int error = mkdir(folder_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            outputBuffer << "Created folder: " << folder_name << '\n';
        }

        // KMC loop for atomic structure perturbations
        kmc_time = 0.0;
        kmc_step_count = 0;
        
        while (kmc_time < t)
        {
			outputBuffer << "--------------\n";
            outputBuffer << "KMC step count: " << kmc_step_count << "\n";
            
            // *** Solve fields ***
            
            // Charge and Potential
            auto t0 = std::chrono::steady_clock::now();
            if (solve_potential)
            {
                std::map<std::string, int> chargeMap = device.updateCharge(metals);
                resultMap.insert(chargeMap.begin(), chargeMap.end());
                
                device.updatePotential(num_atoms_contact, Vd, lattice,
									   G_coeff, high_G, low_G, metals);
            }
            auto t_pot = std::chrono::steady_clock::now();
            diff_pot = t_pot - t0;
            
            // Power and Temperature
            if (solve_current) {
				
				std::map<std::string, double> powerMap = device.updatePower(num_atoms_first_layer, Vd, high_G, low_G,
                                                                metals, m_e, V0);
                resultMap.insert(powerMap.begin(), powerMap.end());
				/*if (solve_heating){                                    
				}*/
			}
            auto t_temp = std::chrono::steady_clock::now();
            diff_temp = t_temp - t_pot;
            
            // generate xyz snapshot
            if (!(kmc_step_count % log_freq))
            {
                const std::string file_name = "snapshot_" + std::to_string(kmc_step_count) + ".xyz";
                device.writeSnapshot(file_name, folder_name);
            }

            // *** Execute events on structure ***
            
            kmc_time += sim.executeKMCStep(&device, freq, lattice, pbc);
            auto t_perturb = std::chrono::steady_clock::now();
            diff_perturb = t_perturb - t_temp;
            
            outputBuffer << "KMC time is: " << kmc_time << "\n";
            
            // load step results into print buffer
            for (const auto &pair : resultMap)
			{
				outputBuffer << pair.first << ": " << pair.second << std::endl;
			}

            // dump print buffer into the output file
            if (!(kmc_step_count % output_freq))
            {
                outputFile << outputBuffer.str();
                outputBuffer.str(std::string());
                resultMap.clear();
            }
            kmc_step_count++;

            // Timing info
            auto t1 = std::chrono::steady_clock::now();
            diff = t1 - t0;
            outputBuffer << "**Calculation times:**\n";
            outputBuffer << "Potential update: " << diff_pot.count() << "\n";
            outputBuffer << "Temperature update: " << diff_temp.count() << "\n";
            outputBuffer << "Structure perturbation: " << diff_perturb.count() << "\n";
            outputBuffer << "Total KMC Step: " << diff.count() << "\n";
            outputBuffer << "--------------------------------------";
        }
        const std::string file_name = "snapshot_" + std::to_string(kmc_step_count) + ".xyz";
        device.writeSnapshot(file_name, folder_name);
        vt_counter++;
    }
    
    // close logger
    outputFile << outputBuffer.str();
    outputFile.close();
    return 0;
}
