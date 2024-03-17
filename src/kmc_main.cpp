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
#include <iomanip>
#include <mpi.h>

#include "KMCProcess.h"
#include "utils.h"
#include "Device.h"
#include "gpu_buffers.h"
#include "input_parser.h"

#include "rocm_smi/rocm_smi.h"

#ifdef USE_CUDA
#include "gpu_solvers.h"
#endif

std::string getHipErrorString(hipError_t error) {
    switch (error) {
        case hipSuccess:
            return "hipSuccess";
        case hipErrorInvalidValue:
            return "hipErrorInvalidValue";
        case hipErrorOutOfMemory:
            return "hipErrorOutOfMemory";
        // Add more cases as needed
        default:
            return "Unknown HIP error";
    }
}

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}


int main(int argc, char **argv)
{

    //***********************************
    // Initialize MPI
    //***********************************
    MPI_Init(&argc, &argv);

    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    //***********************************
    // Setup accelerators (GPU)
    //***********************************

    char* slurm_localid = getenv("SLURM_LOCALID");
    int localid = -1;
    if (slurm_localid != nullptr) {
        localid = atoi(slurm_localid);
        std::cout << "Rank " << mpi_rank << " has SLURM_LOCALID " << localid << std::endl;
    } else {
        std::cerr << "Rank " << mpi_rank << " cannot access SLURM_LOCALID" << std::endl;
        exit(1);
    }

    char* rocr_visible_devices = getenv("ROCR_VISIBLE_DEVICES");
    if (rocr_visible_devices != nullptr) {
        std::cout << "Rank " << mpi_rank << " ROCR_VISIBLE_DEVICES: " << rocr_visible_devices << std::endl;
    } else {
        std::cerr << "Rank " << mpi_rank << " ROCR_VISIBLE_DEVICES not set" << std::endl;
        exit(1);
    }

    hipError_t hipStatus;
    int device_id = localid; 
    hipStatus = hipSetDevice(device_id);

    hipDeviceProp_t dprop;
    hipGetDeviceProperties(&dprop, device_id);
    if (hipStatus == hipSuccess)
    {
        std::cout << "Rank " << mpi_rank << " successfully set device " << device_id << std::endl;
    } else {
        std::cerr << "Rank " << mpi_rank << " failed to set device " << device_id << std::endl;
        std::cerr << "Error: " << getHipErrorString(hipStatus) << std::endl;
    }

    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

#pragma omp parallel
{
    #pragma omp single
    {
        int num_threads = omp_get_num_threads();
        if (!mpi_rank)
        std::cout << "Number of OMP threads: " << num_threads << std::endl;
    }
}

    //***************************************
    // Parse inputs and setup output logging
    //***************************************
    
    KMCParameters p(argv[1]);                                                       // stores simulation parameters
    std::ostringstream outputBuffer;                                                // holds output data to dump into a txt file
    std::string output_filename = "output" + std::to_string(mpi_size) + "_" + std::to_string(mpi_rank) + ".txt";
    std::remove(output_filename.c_str());
    std::ofstream outputFile(output_filename, std::ios_base::app);

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

    if (!mpi_rank)
    std::cout << "Constructing device...\n"; 
    Device device(xyz_files, p);                                                    // contains the simulation domain and field solver functions
    
    if (p.solve_heating_local)                                                      // build the Laplacian to solve for the local temperature distribution
    {
        std::chrono::duration<double> diff_laplacian;
        auto t_lap0 = std::chrono::steady_clock::now();
        device.constructLaplacian(p);
        auto t_lap1 = std::chrono::steady_clock::now();
        diff_laplacian = t_lap1 - t_lap0;
        outputBuffer << "**Calculation time for the laplacian:**\n";
        outputBuffer << "Laplacian update: " << diff_laplacian.count() << "\n";
        outputBuffer.str(std::string());
    }

    if (p.pristine)                                                                // convert an initial percentage of oxygen atoms to vacancies
        device.makeSubstoichiometric(p.initial_vacancy_concentration);
    
    //******************************
    // Initialize the KMC Simulation
    //******************************
    
    KMCProcess sim(device, p.freq);                                                // stores the division of the device into KMC 'layers' with different EA

    //*****************************
    // Setup GPU memory management
    //*****************************

#ifdef USE_CUDA
    GPUBuffers gpubuf(sim.layers, sim.site_layer, sim.freq,                         
                      device.N, device.N_atom, device.site_x, device.site_y, device.site_z,
                      device.max_num_neighbors, device.sigma, device.k, 
                      device.lattice, device.neigh_idx, device.cutoff_window, device.cutoff_idx, p.metals, p.metals.size(),
                      MPI_COMM_WORLD, p.num_atoms_first_layer);
    gpubuf.sync_HostToGPU(device);                                                                  // initialize the device attributes in gpu memory


    if (p.solve_potential || p.solve_current)
    {
        initialize_sparsity_K(gpubuf, p.pbc, p.nn_dist, p.num_atoms_first_layer); // for K
        // initialize_sparsity_T(gpubuf, p.pbc, p.nn_dist, p.num_atoms_first_layer, p.num_atoms_first_layer, p.num_layers_contact);
    }

    // make layer arrays and copy them to const memory
    std::vector<double> E_gen_host, E_rec_host, E_Vdiff_host, E_Odiff_host;
    for (auto l : sim.layers){
        E_gen_host.push_back(l.E_gen_0);
        E_rec_host.push_back(l.E_rec_1);
        E_Vdiff_host.push_back(l.E_diff_2); 
        E_Odiff_host.push_back(l.E_diff_3);
    }
    int num_layers = sim.layers.size();
    copytoConstMemory(E_gen_host, E_rec_host, E_Vdiff_host, E_Odiff_host); 
#else
    GPUBuffers gpubuf;
#endif

    // Create hip library handles to pass into the gpu_Device functions
    hipblasHandle_t handle;
    hipblasCreate(&handle);
    hipsolverHandle_t handle_cusolver;
    hipsolverCreate(&handle_cusolver);                           

    // loop over V_switch and t_switch
    double Vd, t, kmc_time, step_time, I_macro, T_kmc, V_vcm;                                       // KMC loop variables
    int kmc_step_count;                                                                             // tracks the number of KMC steps per bias point
    std::map<std::string, double> resultMap;                                                        // dictionary of output quantities which are dumped to output.log
    std::chrono::duration<double> diff, diff_pot, diff_power, diff_temp, diff_perturb;              // track computation time of the different modules

    for (int vt_counter = 0; vt_counter < p.V_switch.size(); vt_counter++)
    {

        Vd = p.V_switch[vt_counter];                                                                // [V] applied voltage at this bias point
        t = p.t_switch[vt_counter];                                                                 // [s] physical duration of the applied voltage
        V_vcm = Vd;                                                                                 // [V] voltage dropped across rram (different only if there is a series resistor)
        I_macro = 0.0;                                                                              // [uA] net current leaving the device
        
        outputBuffer << "--------------------------------\n";
        outputBuffer << "Applied Voltage = " << Vd << " V\n";
        outputBuffer << "--------------------------------\n";

        // solve the Laplace Equation to get the CB edge energy at this voltage
        if (p.solve_current)
        {
            device.setLaplacePotential(handle, handle_cusolver, gpubuf, p, Vd);                     // homogenous poisson equation with contact BC
        }

        // setup output folder
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

        std::string file_name = "snapshot_init.xyz";
        device.writeSnapshot(file_name, folder_name);

#ifdef USE_CUDA
        gpubuf.sync_HostToGPU(device);                                                                  // initialize the device attributes in gpu memory
#endif

        gpuErrchk( hipDeviceSynchronize() ); //debug

        while (kmc_time < t)
        {
            outputBuffer << "--------------\n";
            outputBuffer << "KMC step count: " << kmc_step_count << "\n";
            auto t0 = std::chrono::steady_clock::now();

            // Update potential
            if (p.solve_potential)
            {
                // update site-resolved charge
                std::cout << "mpi rank" << mpi_rank << "  updating charge " << std::endl;
                std::map<std::string, double> chargeMap = device.updateCharge(gpubuf, p.metals);           
                resultMap.insert(chargeMap.begin(), chargeMap.end());               
                
                std::cout << "mpi rank" << mpi_rank << " starting updating boundary potential " << std::endl;
                // update site-resolved potential
                std::map<std::string, double> potentialMap = device.updatePotential(handle, handle_cusolver, gpubuf, p, Vd, kmc_step_count);
                resultMap.insert(potentialMap.begin(), potentialMap.end());      
                std::cout << "mpi rank" << mpi_rank << " done updating boundary potential " << std::endl;                        
            }

            // generate xyz snapshot
            if (!(kmc_step_count % p.log_freq))
            {
#ifdef USE_CUDA
        gpubuf.sync_GPUToHost(device);
#endif
                if (!mpi_rank){
                    std::cout << "making snapshot" << std::endl;
                    std::string file_name = "snapshot_" + std::to_string(kmc_step_count) + ".xyz";
                    device.writeSnapshot(file_name, folder_name);
                }
            }

            // Execute events and update kmc_time
            std::cout << "mpi rank" << mpi_rank << " is starting kmc events " << std::endl;
            if (p.perturb_structure){                                  
                std::map<std::string, double> kmcMap = sim.executeKMCStep(gpubuf, device, &step_time);   // execute events on the structure
                kmc_time += step_time;
                resultMap.insert(kmcMap.begin(), kmcMap.end());                                   
            } else {           
                // debug:                                                                         // run in IV or field-solver testing
                if (kmc_step_count > 10)
                {
                    kmc_time = t;
                }
            }
            std::cout << "mpi rank" << mpi_rank << " is done kmc events " << std::endl;
           
            // Update current and joule heating
            if (p.solve_current)
            {
                std::map<std::string, double> powerMap = device.updatePower(handle, handle_cusolver,    // update site-resolved dissipated power
                                                                            gpubuf, p, Vd);
                resultMap.insert(powerMap.begin(), powerMap.end());
                I_macro = device.imacro;

                // Temperature
                if (p.solve_heating_global || p.solve_heating_local)                                     // update site-resolved heat
                {
                    std::map<std::string, double> temperatureMap = device.updateTemperature(gpubuf, p, step_time);
                    resultMap.insert(temperatureMap.begin(), temperatureMap.end());
                }
            }

            auto tfield = std::chrono::steady_clock::now();
            std::chrono::duration<double> dt_field = tfield - t0;

            if (!mpi_rank) 
            {
                std::cout << "**********************************\n";
                std::string rocm_smi_output = exec("rocm-smi --showmeminfo vram");
                std::cout << rocm_smi_output;
                std::cout << "**********************************\n";
            }
            
            // ********************************************************
            // ******************** Log results ***********************
            // ********************************************************

            outputBuffer << "KMC time is: " << kmc_time << "\n";
            std::cout << "mpi rank" << mpi_rank << " logging results " << std::endl;
            
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

            if (I_macro > p.Icc*(1e-6))
            {
                outputBuffer << "I_macro > Icc, compliance current reached.\n";
                break;
            }

            auto tlog = std::chrono::steady_clock::now();
            std::chrono::duration<double> dt_log = tlog - tfield;

            // Log timing info
            auto t1 = std::chrono::steady_clock::now();
            std::chrono::duration<double> dt = t1 - t0;
            outputBuffer << "Z - calculation time - all fields [s]: " << dt_field.count() << "\n";
            outputBuffer << "Z - calculation time - logging results [s]: " << dt_log.count() << "\n";
            outputBuffer << "Z - calculation time - KMC superstep [s]: " << dt.count() << "\n";
            outputBuffer << "--------------------------------------";
            
            gpuErrchk( hipDeviceSynchronize() ); //debug

        } // while (kmc_time < t)

// Get device attributes from GPU memory
#ifdef USE_CUDA
        gpubuf.sync_GPUToHost(device);
#endif
        if (!mpi_rank)
        {
            const std::string file_name = "snapshot_" + std::to_string(kmc_step_count) + ".xyz";
            std::cout << "logging the last snapshot\n";
            //print kmc step count
            std::cout << "KMC step count: " << kmc_step_count << "\n";
            device.writeSnapshot(file_name, folder_name);
        }

    } // for (int vt_counter = 0; vt_counter < p.V_switch.size(); vt_counter++)

#ifdef USE_CUDA
    gpubuf.freeGPUmemory();
    CheckCublasError(hipblasDestroy(handle));
#endif

    // close logger
    outputFile << outputBuffer.str();
    outputFile.close();

    //***********************************
    // Finalize MPI
    //***********************************
    MPI_Finalize();


    return 0;
}
