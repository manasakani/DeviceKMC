// KMC Process class
#include "random_num.h"
#include "KMCProcess.h"
#include "structure_input.h"
#include "Device.h"
#include "utils.h"
#include "cuda_wrapper.h"
#include <iostream>
#include <list>
#include <algorithm>
#include <numeric>

void Layer::init_layer(std::string type_, double E_gen_0_, double E_rec_1_, double E_diff_2_, double E_diff_3_, double start_x_, double end_x_)
{
    type = type_;
    E_gen_0 = E_gen_0_;
    E_rec_1 = E_rec_1_;
    E_diff_2 = E_diff_2_;
    E_diff_3 = E_diff_3_;
    start_x = start_x_;
    end_x = end_x_;
    init_vac_percentage = 0.0;
}

void Layer::disp_layer()
{
    print("Layer of type " << type << " from " << start_x << " to " << end_x);
}

KMCProcess::KMCProcess(Device *device, double _freq)
{

    // initialize random number generator
    random_generator.setSeed(rnd_seed_kmc);
    this->freq = _freq;

    // intialize device layers
    int layerID;
    this->layers.resize(numlayers);
    layers[0].init_layer(layer_0_type, layer_0_E_gen_0, layer_0_E_rec_1, layer_0_E_diff_2, layer_0_E_diff_3, layer_0_start_x, layer_0_end_x);
    layers[1].init_layer(layer_1_type, layer_1_E_gen_0, layer_1_E_rec_1, layer_1_E_diff_2, layer_1_E_diff_3, layer_1_start_x, layer_1_end_x);
    layers[2].init_layer(layer_2_type, layer_2_E_gen_0, layer_2_E_rec_1, layer_2_E_diff_2, layer_2_E_diff_3, layer_2_start_x, layer_2_end_x);
    layers[3].init_layer(layer_3_type, layer_3_E_gen_0, layer_3_E_rec_1, layer_3_E_diff_2, layer_3_E_diff_3, layer_3_start_x, layer_3_end_x);
    layers[4].init_layer(layer_4_type, layer_4_E_gen_0, layer_4_E_rec_1, layer_4_E_diff_2, layer_4_E_diff_3, layer_4_start_x, layer_4_end_x);

    // assign layer IDs for all the sites
    for (int i = 0; i < device->N; i++)
    {
        layerID = 1000;
        for (int j = 0; j < numlayers; j++)
        {
            if (layers[j].start_x <= device->site_x[i] && device->site_x[i] <= layers[j].end_x)
            {
                layerID = j;
            }
        }
        if (layerID == 1000)
        {
            print("Site #" << i << " is not inside the device!");
            abort();
        }
        site_layer.push_back(layerID);
    }
}

double KMCProcess::executeKMCStep(Device &device)
{
    // ** Build event list **

    int num_sites = device.N;
    int num_neigh = device.max_num_neighbors;
    EVENTTYPE *event_type = new EVENTTYPE[num_sites * num_neigh];
    double    *event_prob = new    double[num_sites * num_neigh];

    // iterates through all possible site-neighbor pairs
    #pragma omp parallel for
    for (auto idx = 0; idx < num_sites * num_neigh; ++idx) {

        EVENTTYPE event_type_ = NULL_EVENT;
        double P = 0;

        // site-neighbor pair indicies for this element of neigh_idx
        int i = std::floor(idx / num_neigh);
        auto j = device.neigh_idx[idx]; 

        // j is -1 if there is no neighbor at this position
        if (j >= 0 && j < num_sites) { 

            double r_dist = (1e-10) * site_dist(device.site_x[i], device.site_y[i], device.site_z[i], 
                                                device.site_x[j], device.site_y[j], device.site_z[j], device.lattice, device.pbc);

            // Generation
            if (device.site_element[i] == DEFECT && device.site_element[j] == O)
            {

                double E = 2 * (device.site_potential[i] - device.site_potential[j]);
                double zero_field_energy = layers[site_layer[j]].E_gen_0;
                event_type_ = VACANCY_GENERATION;
                double Ekin = kB * (device.site_temperature[j] - device.site_temperature[i]);
                double EA = zero_field_energy - E - Ekin;
                P = exp(-1 * EA / (kB * device.T_bg)) * freq;
            }

            // Recombination
            else if (device.site_element[i] == OXYGEN_DEFECT && device.site_element[j] == VACANCY)
            {
                int charge_abs = 2;
                double self_int_V = v_solve(r_dist, charge_abs, device.sigma, device.k, q);

                int charge_state = device.site_charge[i] - device.site_charge[j];
                double E = charge_state * (device.site_potential[i] - device.site_potential[j] + (charge_state / 2) * self_int_V);
                double zero_field_energy = layers[site_layer[j]].E_rec_1;

                event_type_ = VACANCY_RECOMBINATION;
                double Ekin = kB * (device.site_temperature[i] - device.site_temperature[j]);
                double EA = zero_field_energy - E - Ekin;
                P = exp(-1 * EA / (kB * device.T_bg)) * freq;
            }

            // Vacancy diffusion
            if (device.site_element[i] == VACANCY && device.site_element[j] == O)
            {

                double self_int_V = 0.0;
                if (device.site_charge[i] != 0)
                {
                    double self_int_V = v_solve(r_dist, device.site_charge[i], device.sigma, device.k, q);
                }

                event_type_ = VACANCY_DIFFUSION;
                double E = (device.site_charge[i] - device.site_charge[j]) * (device.site_potential[i] - device.site_potential[j] + self_int_V);
                double zero_field_energy = layers[site_layer[i]].E_diff_2;
                double Ekin = kB * (device.site_temperature[j] - device.site_temperature[i]);
                double EA = zero_field_energy - E - Ekin;
                P = exp(-1 * EA / (kB * device.T_bg)) * freq;
            }

            // Ion diffusion
            else if (device.site_element[i] == OXYGEN_DEFECT && device.site_element[j] == DEFECT)
            {
                int charge_abs = 2;
                double self_int_V = 0.0;
                if (device.site_charge[i] != 0)
                {
                    self_int_V = v_solve(r_dist, charge_abs, device.sigma, device.k, q);
                }
                double E = (device.site_charge[i] - device.site_charge[j]) * (device.site_potential[i] - device.site_potential[j] - self_int_V);
                double zero_field_energy = layers[site_layer[j]].E_diff_3;

                event_type_ = ION_DIFFUSION;

                double Ekin = kB * (device.site_temperature[i] - device.site_temperature[j]);
                double EA = zero_field_energy - E - Ekin;
                P = exp(-1 * EA / (kB * device.T_bg)) * freq;
            }
        }
        event_type[idx] = event_type_;
        event_prob[idx] = P;
    }

    // ** Event execution **

    double *event_prob_cum = new double[num_sites * num_neigh];
    double event_time = 0.0;
    while (event_time < 1 / freq) {

        // NOTE: We can optimize this by only updating the required values
        // get the cumulative sum of the probabilities
        inclusive_prefix_sum<double>(event_prob, event_prob_cum, num_sites * num_neigh);
        // std::inclusive_scan(event_prob, event_prob + num_sites * num_neigh, event_prob_cum);

        // Select an event
        double Psum = event_prob_cum[num_sites * num_neigh - 1];
        double number = random_generator.getRandomNumber() * Psum;
        int event_idx = std::upper_bound(event_prob_cum, event_prob_cum + num_sites * num_neigh, number) - event_prob_cum;
        double sel_event_prob = event_prob_cum[event_idx];

        //std::cout << "Searching for " << number << " in [" << event_prob_cum[0] << ", " << Psum << "]" << std::endl;

        EVENTTYPE sel_event_type = event_type[event_idx];

        // find the site-neighbor pair corresponding to this event
        int i = std::floor(event_idx / num_neigh);
        auto j = device.neigh_idx[event_idx];

        // std::cout << "Selected event index: " << event_idx << " with type "
        //          << event_type[event_idx] << " and probability " << event_prob[event_idx]
        //          << " (" << sel_event_prob << ")" << " to happen between " << device.site_element[i]
        //          << " and " << device.site_element[j] << std::endl;

        // Event execution
        switch (sel_event_type)
        {
        case VACANCY_GENERATION:
        {
            if (device.site_element[i] != DEFECT || device.site_element[j] != O)
            {
                print("Wrong event type - VACANCY_GENERATION!");
                print(return_element(device.site_element[i]) << " and " << return_element(device.site_element[j]));
            }

            // turn the defect (site_1) into an oxygen ion:
            device.site_element[i] = OXYGEN_DEFECT;
            device.site_charge[i] = -2;

            // turn the oxygen (site_2) into a charged vacancy:
            device.site_element[j] = VACANCY;
            device.site_charge[j] = 2;

            break;
        }
        case VACANCY_RECOMBINATION:
        {
            if (device.site_element[i] != OXYGEN_DEFECT || device.site_element[j] != VACANCY)
            {
                print("Wrong event type - VACANCY_RECOMBINATION!");
                print(return_element(device.site_element[i]) << " and " << return_element(device.site_element[j]));
            }

            // turn the oxygen (site_1) into a defect
            device.site_element[i] = DEFECT;
            device.site_charge[i] = 0;

            // turn the vacancy (site_2) into an oxygen atom:
            device.site_element[j] = O;
            device.site_charge[j] = 0;

            break;
        }
        case VACANCY_DIFFUSION:
        {
            if (device.site_element[i] != VACANCY || device.site_element[j] != O)
            {
                print("Wrong event type - VACANCY_DIFFUSION!");
                print(return_element(device.site_element[i]) << " and " << return_element(device.site_element[j]));
            }

            swap_values(&device.site_element[i], &device.site_element[j]);
            swap_values(&device.site_charge[i], &device.site_charge[j]);

            break;
        }
        case ION_DIFFUSION:
        {
            if (device.site_element[i] != OXYGEN_DEFECT || device.site_element[j] != DEFECT)
            {
                print("Wrong event type - ION_DIFFUSION!");
                print(return_element(device.site_element[i]) << " and " << return_element(device.site_element[j]));
            }

            swap_values(&device.site_element[i], &device.site_element[j]);
            swap_values(&device.site_charge[i], &device.site_charge[j]);

            break;
        }
        default:
            print("error: unidentified event key found");
        }

        // Deactivate conflicting events
        int i_, j_;
#pragma omp parallel private(i_, j_)
{
        // other site's events with i or j
        #pragma omp for
        for (auto idx = 0; idx < num_sites * num_neigh; ++idx){
            i_ = std::floor(idx / num_neigh);
            j_ = device.neigh_idx[idx];

            if (i == i_ || j == j_ || i == j_ || j == i_){
                event_type[idx] = NULL_EVENT;
                event_prob[idx] = 0.0;
            }
        }
        // i's events with its neighbors
        #pragma omp for
        for (auto neigh_idx = i * num_neigh; neigh_idx < (i + 1)*num_neigh; ++neigh_idx){
            event_type[neigh_idx] = NULL_EVENT;
            event_prob[neigh_idx] = 0.0;
        }
        // j's events with its neighbors
        #pragma omp for
        for (auto neigh_idx = j * num_neigh; neigh_idx < (j + 1)*num_neigh; ++neigh_idx){
            event_type[neigh_idx] = NULL_EVENT;
            event_prob[neigh_idx] = 0.0;
        }
}
        event_time = -log(random_generator.getRandomNumber()) / Psum;
    }

    delete[] event_type;
    delete[] event_prob;
    delete[] event_prob_cum;
    return event_time;
}

double KMCProcess::executeKMCStep_gpu(GPUBuffers gpubuf){

    execute_kmc_step_gpu(gpubuf.N_, gpubuf.nn_, 
                         gpubuf.site_x, gpubuf.site_y, gpubuf.site_z, 
                         gpubuf.site_potential, 
                         gpubuf.site_temperature,
                         gpubuf.site_element, gpubuf.site_charge);

    std::cout << "got here"; exit(1);
}
