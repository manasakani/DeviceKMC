#include <iostream>
#include <vector>
#include <cstdio>
#include <cstddef>
#include <stdlib.h>
#include <chrono>
#include <map>

#include "KMCProcess.h"
#include "utils.h"
#include "Device.h"
#include "gpu_buffers.h"
#include "input_parser.h"
#include "cuda_wrapper.h"

// main function for KMC simulation
int main(int argc, char **argv)
{

    // parse inputs
    KMCParameters p(argv[1]);

    // set up logging
    std::ostringstream outputBuffer;
    std::ofstream outputFile("output.txt", std::ios_base::app);
    outputBuffer << "----------------------------\n";
    outputBuffer << "Starting Kinetic Monte Carlo\n";
    outputBuffer << "----------------------------\n";
    outputFile << outputBuffer.str();

    // check for accelerators
    std::cout << "checking for an accelerator...\n";
    char gpu_string[1000];
    get_gpu_info(gpu_string, 0);
#ifdef USE_CUDA
    printf("Will use this GPU: %s\n", gpu_string);
    set_gpu(0);
#else
    std::cout << "Simulation will not use the GPU.\n";
#endif
    cublasHandle_t handle = CreateCublasHandle(0);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    cusolverDnHandle_t handle_cusolver = CreateCusolverDnHandle(0);

    // Initialize device
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
            outputFile << outputBuffer.str();
            xyz_files.push_back(p.restart_xyz_file);
        }
    }
    else
    {
        xyz_files.push_back(p.atom_xyz_file);
        xyz_files.push_back(p.interstitial_xyz_file);
    }

    std::cout << "Constructing device...\n";
    Device device(xyz_files, p.lattice, p.metals, p.shift, p.shifts, p.pbc, p.sigma, p.epsilon, p.nn_dist, p.background_temp, p.rnd_seed);

    std::chrono::duration<double> diff_laplacian;
    auto t_lap0 = std::chrono::steady_clock::now();
    if (p.solve_heating_local)
    {
        device.constructLaplacian(handle_cusolver, p.k_th_interface, p.k_th_metal, p.delta,
                                  p.delta_t, p.tau, p.metals, p.background_temp,
                                  p.num_atoms_contact);
    }
    auto t_lap1 = std::chrono::steady_clock::now();
    diff_laplacian = t_lap1 - t_lap0;
    outputBuffer << "**Calculation time for the laplacian:**\n";
    outputBuffer << "Laplacian update: " << diff_laplacian.count() << "\n";
    outputFile << outputBuffer.str();

    if (p.pristine)
        device.makeSubstoichiometric(p.initial_vacancy_concentration);

    // Initialize KMC simulation
    KMCProcess sim(&device, p.freq);
    outputBuffer.str(std::string());

    // Initialize device attributes on GPU
#ifdef USE_CUDA
        GPUBuffers gpubuf(device.N, device.max_num_neighbors);
        gpubuf.upload_HostToGPU(device); 
#endif

    // loop over V_switch and t_switch
    double Vd, t, kmc_time, step_time, I_macro, T_kmc;
    int kmc_step_count;
    std::map<std::string, double> resultMap;
    std::string file_name;
    std::chrono::duration<double> diff, diff_pot, diff_power, diff_temp, diff_perturb;

    for (size_t vt_counter = 0; vt_counter < p.V_switch.size(); vt_counter++)
    {
        Vd = p.V_switch[vt_counter];
        t = p.t_switch[vt_counter];
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

        kmc_time = 0.0;
        kmc_step_count = 0;
        file_name = "snapshot_init.xyz";
        device.writeSnapshot(file_name, folder_name);

        // KMC loop for atomic structure perturbations with these conditions
        while (kmc_time < t)
        {
            outputBuffer << "--------------\n";
            outputBuffer << "KMC step count: " << kmc_step_count << "\n";

            // *** Update fields and execute events on structure ***

            // Charge and Potential
            auto t0 = std::chrono::steady_clock::now();
            if (p.solve_potential)
            {
#ifdef USE_CUDA
                int esum = 0;
                for (auto e : device.site_charge){
                    esum += e;
                }
                std::cout << "charge total before: " <<  esum << "\n";

                device.updateCharge_gpu(gpubuf);
                gpubuf.download_GPUToHost(device); // keep moving this download down

                esum = 0;
                for (auto e : device.site_charge){
                    esum += e;
                }
                std::cout << "charge total after: " <<  esum << "\n";
                exit(1);
#else
                std::cout << "updating charge on CPU\n";
                int esum = 0;
                for (auto e : device.site_charge){
                    esum += e;
                }
                std::cout << "charge total before: " <<  esum << "\n";

                std::map<std::string, int> chargeMap = device.updateCharge(p.metals);
                resultMap.insert(chargeMap.begin(), chargeMap.end());

                esum = 0;
                for (auto e : device.site_charge){
                    esum += e;
                }
                std::cout << "charge total after: " <<  esum << "\n";
                exit(1);
#endif
                device.updatePotential(handle_cusolver, p.num_atoms_contact, Vd, p.lattice,
                                       p.G_coeff, p.high_G, p.low_G, p.metals);
            }
            auto t_pot = std::chrono::steady_clock::now();
            diff_pot = t_pot - t0;

            // KMC update step 
            step_time = sim.executeKMCStep(device);
            double temperature_time = kmc_time;
            kmc_time += step_time;
            auto t_perturb = std::chrono::steady_clock::now();
            diff_perturb = t_perturb - t_pot;

            // Power and Temperature
            /*if (p.solve_current)
            {

                std::map<std::string, double> powerMap = device.updatePower(handle, handle_cusolver, p.num_atoms_first_layer, Vd, p.high_G, p.low_G,
                                                                            p.metals, p.m_e, p.V0);
                resultMap.insert(powerMap.begin(), powerMap.end());
                auto t_power = std::chrono::steady_clock::now();
                diff_power = t_power - t_perturb;

                if (p.solve_heating_global)
                {
                    std::map<std::string, double> temperatureMap = device.updateTemperatureGlobal(step_time, p.small_step, p.dissipation_constant,
                                                                                                  p.background_temp, p.t_ox, p.A, p.c_p);
                    resultMap.insert(temperatureMap.begin(), temperatureMap.end());
                }
                if (p.solve_heating_local)
                { 
                    // use this to modify the rates
                    if (step_time > 1e3 * p.delta_t)
                    { 
                        // use steady state solution
                        std::map<std::string, double> localTemperatureMap = device.updateLocalTemperatureSteadyState(p.background_temp, step_time, p.tau, p.power_adjustment_term, p.k_th_interface,
                                                                                                                     p.k_th_vacancies, p.num_atoms_contact, p.metals);
                        resultMap.insert(localTemperatureMap.begin(), localTemperatureMap.end());
                    }
                    else
                    {
                        for (int i = 0; i < int(step_time / p.delta_t); ++i)
                        {
                            std::map<std::string, double> localTemperatureMap = device.updateLocalTemperature(p.background_temp, step_time, p.tau, p.power_adjustment_term, p.k_th_interface,
                                                                                                              p.k_th_vacancies, p.num_atoms_contact, p.metals);

                            resultMap.insert(localTemperatureMap.begin(), localTemperatureMap.end());
                        }
                    }
                }

                auto t_temp = std::chrono::steady_clock::now();
                diff_temp = t_temp - t_power;
            }*/

            // *** Log results ***

            outputBuffer << "KMC time is: " << kmc_time << "\n";

            // load step results into print buffer
            for (const auto &pair : resultMap) { outputBuffer << pair.first << ": " << pair.second << std::endl; }
            resultMap.clear();

            // dump print buffer into the output file
            if (!(kmc_step_count % p.output_freq))
            {
                outputFile << outputBuffer.str();
                outputBuffer.str(std::string());
            }
            kmc_step_count++;

            // generate xyz snapshot
            if (!(kmc_step_count % p.log_freq))
            {
                std::string file_name = "snapshot_" + std::to_string(kmc_step_count) + ".xyz";
                device.writeSnapshot(file_name, folder_name);
            }

            // *** Log timing info ***  

            auto t1 = std::chrono::steady_clock::now();
            diff = t1 - t0;
            outputBuffer << "**Calculation times:**\n";
            outputBuffer << "Potential update: " << diff_pot.count() << "\n";
            outputBuffer << "Power update: " << diff_power.count() << "\n";
            outputBuffer << "Temperature update: " << diff_temp.count() << "\n";
            outputBuffer << "Structure perturbation: " << diff_perturb.count() << "\n";
            outputBuffer << "Total KMC Step: " << diff.count() << "\n";
            outputBuffer << "--------------------------------------";
        }
        const std::string file_name = "snapshot_" + std::to_string(kmc_step_count) + ".xyz";
        device.writeSnapshot(file_name, folder_name);
        vt_counter++;
    }

#ifdef USE_CUDA
        gpubuf.freeGPUmemory();
#endif

    CheckCublasError(cublasDestroy(handle));

    // close logger
    outputFile << outputBuffer.str();
    outputFile.close();
    return 0;
}