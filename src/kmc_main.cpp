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

    Device device(xyz_files, lattice, shift, shifts, pbc, nn_dist, background_temp, rnd_seed);
    if (pristine)
        device.makeSubstoichiometric(initial_vacancy_concentration);
    outputBuffer << "Device was constructed\n";

    // Voltage
    double Vd;
    Vd = V_switch[0]; // Testing change this with the counter !!!

    // Initialize the fields - implement these member functions inside Device.cpp!
    std::map<std::string, int> resultMap = device.updateCharge();

    // Debug purposes
    for (const auto &pair : resultMap)
    {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    std::map<std::string, int> potentialMap = device.updatePotential(num_atoms_contact, Vd, lattice, pbc, sigma, k);
    device.updateTemperature();

    // Initial snapshot to test:
    const std::string file_name = "snapshot_init.xyz";
    device.writeSnapshot(file_name, "./");

    // Initialize KMC simulation
    // instantiate a KMCprocess object by sending a pointer to this device and a dictionary of input settings
    KMCProcess sim(&device);

    // outer simulation loop over Vd, t:
    //    prepare output folder and update the simulation
    //    inner simulation loop over KMC steps:
    //        update all the fields
    //        execute a KMC step on the device

    // close logger
    outputFile << outputBuffer.str();
    outputFile.close();
    return 0;
}
