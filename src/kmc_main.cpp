/*
@author: Manasa Kaniselvan 
@date: 2023-08
Copyright 2023 ETH Zurich and the Computational Nanoelectronics Group. All rights reserved.
*/

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstddef>
#include <stdlib.h>
#include <chrono>
#include <map>
// #include <mpi.h>

#include "KMCProcess.h"
#include "utils.h"
#include "Device.h"
#include "gpu_buffers.h"
#include "input_parser.h"

#ifdef USE_CUDA
#include "cuda_wrapper.h"
#endif

int main(int argc, char **argv)
{

    //***********************************
    // Setup accelerators (GPU/Multicore)
    //***********************************

    // int mpi_rank, mpi_size;
    // MPI_Init(&argc, &argv);
    // MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    // if (!mpi_rank)
    // {
#ifdef USE_CUDA
    char gpu_string[1000];
    get_gpu_info(gpu_string, 0);
    printf("Will use this GPU: %s\n", gpu_string);
    // printf("Using %i MPI process(es) with %s GPU(s)\n", mpi_size, gpu_string);
    set_gpu(0);
#else
    std::cout << "Simulation will not use any accelerators.\n";
#endif

#pragma omp parallel
{
    #pragma omp single
    {
        int num_threads = omp_get_num_threads();
        std::cout << "Number of OMP threads: " << num_threads << std::endl;
    }
}

    // }
    // MPI_Barrier(MPI_COMM_WORLD);

    //***************************************
    // Parse inputs and setup output logging
    //***************************************
    
    KMCParameters p(argv[1]);                                                       // stores simulation parameters
    std::ostringstream outputBuffer;                                                // holds output data to dump into a txt file
    std::remove("output.txt");
    std::ofstream outputFile("output.txt", std::ios_base::app);

    //*******************************************************
    // Initialize the Atomistic simulation domain (the Device)
    //*******************************************************

    std::vector<std::string> xyz_files;
    if (p.restart)
    {
        const bool file_exists = location_exists(p.restart_xyz_file);
        if (!file_exists)
        {
            outputBuffer << "ERROR: Restart file " << p.restart_xyz_file << " not found!\n";
            outputFile << outputBuffer.str();
            outputBuffer.str(std::string());
            return 0;
        }
        else
        {
            outputBuffer << "Restarting from " << p.restart_xyz_file << "\n";
            xyz_files.push_back(p.restart_xyz_file);
        }
    }
    else
    {   
        xyz_files.push_back(p.atom_xyz_file);
        xyz_files.push_back(p.interstitial_xyz_file);
    }

    std::cout << "Constructing device...\n"; 
    Device device(xyz_files, p);                                                    // the simulation domain and field solver functions

    std::chrono::duration<double> diff_laplacian;
    auto t_lap0 = std::chrono::steady_clock::now();
    if (p.solve_heating_local)
    {
        device.constructLaplacian(p);
    }
    auto t_lap1 = std::chrono::steady_clock::now();
    diff_laplacian = t_lap1 - t_lap0;
    outputBuffer << "**Calculation time for the laplacian:**\n";
    outputBuffer << "Laplacian update: " << diff_laplacian.count() << "\n";
    outputBuffer.str(std::string());

    if (p.pristine)
        device.makeSubstoichiometric(p.initial_vacancy_concentration);

    //******************************
    // Initialize the KMC Simulation
    //******************************

    KMCProcess sim(device, p.freq);                                                // stores the division of the device into KMC regions
                                                                                   // update the site-site combinations where events are possible
    //*****************************
    // Setup and handle GPU buffers
    //*****************************

#ifdef USE_CUDA
    GPUBuffers gpubuf(sim.layers, sim.site_layer, sim.freq,                         // handles GPU memory management of the device attributes
                      device.N, device.N_atom, device.site_x, device.site_y, device.site_z,
                      device.max_num_neighbors, device.sigma, device.k, 
                      device.lattice, device.neigh_idx, p.metals, p.metals.size());
    initialize_sparsity(gpubuf, p.pbc, p.nn_dist, p.num_atoms_contact);

#else
    GPUBuffers gpubuf;
#endif

    // CUDA library handles
    cublasHandle_t handle = CreateCublasHandle(0);
    cusolverDnHandle_t handle_cusolver = CreateCusolverDnHandle(0);                                   

    // loop over V_switch and t_switch
    double Vd, t, kmc_time, step_time, I_macro, T_kmc, V_vcm;
    int kmc_step_count;
    std::map<std::string, double> resultMap;
    std::string file_name;
    std::chrono::duration<double> diff, diff_pot, diff_power, diff_temp, diff_perturb;

    for (int vt_counter = 0; vt_counter < p.V_switch.size(); vt_counter++)
    {

        // MPI_Barrier(MPI_COMM_WORLD);                                                // ensure that all MPI ranks have up-to-date gpubufs and loop vars  

        Vd = p.V_switch[vt_counter];
        t = p.t_switch[vt_counter];
        V_vcm = Vd;
        I_macro = 0.0;
        outputBuffer << "--------------------------------\n";
        outputBuffer << "Applied Voltage = " << Vd << " V\n";
        outputBuffer << "--------------------------------\n";

        // solve the Laplace Equation to get the CB edge energy at this voltage
        device.setLaplacePotential(p, Vd);

        const std::string folder_name = "Results_" + std::to_string(Vd);
        const bool folder_already_exists = location_exists(folder_name);
        if (folder_already_exists)
        {
            std::remove(folder_name.c_str());
        }
        const int error = mkdir(folder_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        outputBuffer << "Created folder: " << folder_name << '\n';

        kmc_time = 0.0;
        kmc_step_count = 0;

        // ********************************************************
        // ***************** MAIN KMC LOOP ************************
        // **** Update fields and execute events on structure *****
        // ********************************************************
// #ifdef USE_CUDA
//         gpubuf.sync_HostToGPU(device);
// #endif
        while (kmc_time < t)
        {
            outputBuffer << "--------------\n";
            outputBuffer << "KMC step count: " << kmc_step_count << "\n";

            // handle any input IR drop:
            V_vcm = Vd - I_macro * p.Rs;
            outputBuffer << "V_vcm: " << V_vcm << "\n";

            // Update potential
            auto t0 = std::chrono::steady_clock::now();
            if (p.solve_potential)
            {
                std::map<std::string, int> chargeMap = device.updateCharge(gpubuf, p.metals);           // update site-resolved charge
                device.updatePotential(handle, handle_cusolver, gpubuf, p, Vd, kmc_step_count);         // update site-resolved potential
                resultMap.insert(chargeMap.begin(), chargeMap.end());                                   // update output file with collected metrics
            }
            auto t_pot = std::chrono::steady_clock::now();
            diff_pot = t_pot - t0;

            // int charge_sum = 0;
            // for(auto c : device.site_charge)
            // {
            //     charge_sum += c;
            // }
            // std::cout << "sum of charges: " << charge_sum << "\n";
            // exit(21);

            // generate xyz snapshot
            if (!(kmc_step_count % p.log_freq))
            {
                std::string file_name = "snapshot_" + std::to_string(kmc_step_count) + ".xyz";
                device.writeSnapshot(file_name, folder_name);
            }

            // Execute events and update time
            if (p.perturb_structure){                                  
                step_time = sim.executeKMCStep(gpubuf, device);                                         // execute events on the structure
                kmc_time += step_time;                                                                  // internal simulation timescale
            } else {                                                                                    // run in IV or field-solver testing
                kmc_time = t;
            }
            auto t_perturb = std::chrono::steady_clock::now();
            diff_perturb = t_perturb - t_pot;

            // Update current and joule heating
            if (p.solve_current)
            {
                std::map<std::string, double> powerMap = device.updatePower(handle, handle_cusolver,    // update site-resolved dissipated power
                                                                            gpubuf, p, Vd);
                resultMap.insert(powerMap.begin(), powerMap.end());

                auto t_power = std::chrono::steady_clock::now();
                diff_power = t_power - t_perturb;
                I_macro = device.imacro;

                // Temperature
                if (p.solve_heating_global || p.solve_heating_local)                                     // update site-resolved heat
                {
                    std::map<std::string, double> temperatureMap = device.updateTemperature(gpubuf, p, step_time);
                    resultMap.insert(temperatureMap.begin(), temperatureMap.end());
                }

                auto t_temp = std::chrono::steady_clock::now();
                diff_temp = t_temp - t_power;
            }

            // ********************************************************
            // ******************** Log results ***********************
            // ********************************************************

            outputBuffer << "KMC time is: " << kmc_time << "\n";

            // load step results into print buffer
            for (const auto &pair : resultMap)
            {
                outputBuffer << pair.first << ": " << pair.second << std::endl;
            }
            resultMap.clear();

            // dump print buffer into the output file
            if (!(kmc_step_count % p.output_freq))
            {
                outputFile << outputBuffer.str();
                outputBuffer.str(std::string());
            }
            kmc_step_count++;

            if (I_macro > p.Icc)
            {
                outputBuffer << "I_macro > Icc, compliance current reached.\n";
                break;
            }

            // Log timing info
            auto t1 = std::chrono::steady_clock::now();
            diff = t1 - t0;
            outputBuffer << "**Calculation times:**\n";
            outputBuffer << "Potential update: " << diff_pot.count() << "\n";
            outputBuffer << "Power update: " << diff_power.count() << "\n";
            outputBuffer << "Temperature update: " << diff_temp.count() << "\n";
            outputBuffer << "Structure perturbation: " << diff_perturb.count() << "\n";
            outputBuffer << "Total KMC Step: " << diff.count() << "\n";
            outputBuffer << "--------------------------------------";
        } // while (kmc_time < t)

// Get device attributes from GPU memory
// #ifdef USE_CUDA
//         gpubuf.sync_GPUToHost(device);
// #endif

        const std::string file_name = "snapshot_" + std::to_string(kmc_step_count) + ".xyz";
        device.writeSnapshot(file_name, folder_name);
    } // for (int vt_counter = 0; vt_counter < p.V_switch.size(); vt_counter++)


#ifdef USE_CUDA
    gpubuf.freeGPUmemory();
    CheckCublasError(cublasDestroy(handle));
#endif

    // close logger
    outputFile << outputBuffer.str();
    outputFile.close();
    return 0;

    // Finalize MPI
    // MPI_Finalize();
}
