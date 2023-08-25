#include <iostream>
#include <vector>
#include <omp.h>
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
    double Vd;
    Vd = V_switch[0]; // Testing change this with the counter !!!

    // Initialize the fields
    std::map<std::string, int> chargeMap = device.updateCharge();

    // Potential update
    device.updatePotential(num_atoms_contact, Vd, lattice,
                           G_coeff, high_G, low_G, metals);

    // Power update
    std::map<std::string, double> powerMap = device.updatePower(num_atoms_first_layer, Vd, high_G, low_G,
                                                                metals, m_e, V0);

    // Map
    std::map<std::string, double> temperatureMap = device.updateTemperatureGlobal(event_time, small_step, dissipation_constant,
                                                                                  background_temp, t_ox, A, c_p);
    // Debug purposes
    for (const auto &pair : temperatureMap)
    {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    // device.updateTemperature();

    // Initial snapshot to test:
    const std::string file_name = "snapshot_init.xyz";
    device.writeSnapshot(file_name, "./");

    // ****** TESTING FIELD SOLVERS ********

    // Initialize KMC simulation
    KMCProcess sim(&device);
    std::cout << "initialized KMC simulation\n";

    // loop over V_switch and t_switch
    double t, kmc_time, I_macro, T_kmc;
    int kmc_step_count;
    double *begin = std::begin(V_switch);
    double *end = std::end(V_switch);
    int vt_counter = 0;

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
            std::cout << "Created folder: " << folder_name << '\n';
        }

        // KMC loop for atomic structure perturbations
        // I_macro = Icc;
        kmc_time = 0.0;
        kmc_step_count = 0;
        while (kmc_time < t)
        {
            outputBuffer << "KMC step count: " << kmc_step_count << "\n";

            // field solvers
            if (solve_potential)
            {
                std::map<std::string, int> resultMap = device.updateCharge();
                // device.updatePotential(num_atoms_contact, Vd, lattice, pbc);
            }

            /*if (solve_current) {
                if (solve_heating){
                    I_macro = update_current(sites, neighbors, N, Vd, 1, T_kmc);
                } else {
                    I_macro = update_current(sites, neighbors, N, Vd, 0, T_kmc);
                }
            } else {
                T_kmc = background_temp;
            } */

            // enforce compliance current
            /*if (I_macro > Icc){
                print("Compliance Current Icc reached");
                break;
            }*/

            // atomic structure perturbation step
            kmc_time += sim.executeKMCStep(&device, freq, lattice, pbc);
            outputBuffer << "KMC time is: " << kmc_time << "\n";

            // generate xyz snapshot
            if (!(kmc_step_count % log_freq))
            {
                const std::string file_name = "snapshot_" + std::to_string(kmc_step_count) + ".xyz";
                device.writeSnapshot(file_name, "./");
            }

            // dump output buffer into file
            if (!(kmc_step_count % output_freq))
            {
                outputFile << outputBuffer.str();
            }
            kmc_step_count++;

            // print some timing info here
        }
        const std::string file_name = "snapshot_" + std::to_string(kmc_step_count) + ".xyz";
        device.writeSnapshot(file_name, "./");
        vt_counter++;
    }

    // outer simulation loop over Vd, t:
    //    prepare output folder and update the simulation
    //    inner simulation loop over KMC steps:
    //        update all the fields
    // std::map<std::string, int> resultMap = device.updateCharge();
    // std::map<std::string, int> potentialMap = device.updatePotential(num_atoms_contact, Vd, lattice, pbc, sigma, k);
    // device.updateTemperature();
    //        execute a KMC step on the device
    // sim.executeKMCStep(&device);
    // 		  logging:
    //		  if output_buf_step % 10
    //		  (dump the output buffer into the output file)

    // close logger
    outputFile << outputBuffer.str();
    outputFile.close();
    return 0;
}
