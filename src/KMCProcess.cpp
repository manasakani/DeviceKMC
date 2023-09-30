// KMC Process class
#include "random_num.h"
#include "KMCProcess.h"
#include "structure_input.h"
#include "Device.h"
#include "utils.h"
#include <iostream>
#include <list>

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

// template <typename T, size_t N>
// void inclusive_prefix_sum(const T input[N], T output[N]) {
//     if (N == 0) return;

//     output[0] = input[0];
//     for (size_t i = 1; i < N; ++i) {
//         output[i] = output[i - 1] + input[i];
//     }
// }

void inclusive_prefix_sum(double* input, double* output, size_t N) {
    if (N == 0) return;

    output[0] = input[0];
    for (size_t i = 1; i < N; ++i) {
        output[i] = output[i - 1] + input[i];
    }
}

double KMCProcess::executeKMCStep(Device &device)
{
    std::cout << "starting KMC step\n";

    // build event list
    int num_sites = device.N;
    int num_neigh = device.max_num_neighbors;

    EVENTTYPE *event_type = new EVENTTYPE[num_sites * num_neigh];
    double    *event_prob = new double[num_sites * num_neigh];

    // iterates through all possible site-neighbor pairs
    #pragma omp parallel for
    for (auto idx = 0; idx < num_sites * num_neigh; ++idx) {

        EVENTTYPE event_type_ = NULL_EVENT;
        double P = 0;

        auto i = idx / num_neigh;
        auto j = device.neigh_idx[idx]; 

        // j is -1 if no more neighbors exist at this position of neigh_idx
        if (j >= 0 && j < num_sites) { 

            double r_dist = (1e-10) * site_dist(device.site_x[i], device.site_y[i], device.site_z[i], 
                                                device.site_x[j], device.site_y[j], device.site_z[j], device.lattice, device.pbc);

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
                double Ekin = kB * (device.site_temperature[i] - device.site_temperature[j]);
                double EA = zero_field_energy - E - Ekin;
                P = exp(-1 * EA / (kB * device.T_bg)) * freq;
            }
        }
        event_type[idx] = event_type_;
        event_prob[idx] = P;
    }

    std::cout << "Starting the event execution loop ..." << std::endl;
    double *event_prob_cum = new double[num_sites * num_neigh];

    double event_time = 0.0;
    while (event_time < 1 / freq) {
        // NOTE: We can optimize this by only updating the required values
        //std::inclusive_scan(event_prob, event_prob + num_sites * num_neigh, event_prob_cum);
        inclusive_prefix_sum(event_prob, event_prob_cum, num_sites * num_neigh);

        double Psum = event_prob_cum[num_sites * num_neigh - 1];
        double number = this->random_generator.getRandomNumber() * Psum;

        std::cout << "Searching for " << number << " in [" << event_prob_cum[0] << ", " << Psum << "]" << std::endl;
        int event_idx = std::upper_bound(event_prob_cum, event_prob_cum + num_sites * num_neigh, number) - event_prob_cum;
        double sel_event_prob = event_prob_cum[event_idx];

        std::cout << "Selected event index: " << event_idx << " with type "
                  << event_type[event_idx] << " and probability " << event_prob[event_idx]
                  << " (" << sel_event_prob << ")" << std::endl;

        EVENTTYPE sel_event_type = event_type[event_idx];
        auto i = event_idx / num_neigh;
        auto j = device.neigh_idx[event_idx];
        ELEMENT site_1 = device.site_element[i];
        ELEMENT site_2 = device.site_element[j];

        int charge_1 = device.site_charge[i];
        int charge_2 = device.site_charge[j];

        switch (sel_event_type)
        {
        case VACANCY_DIFFUSION:
        {
            print(site_1);
            print(site_2);
            if (site_1 != VACANCY || site_2 != O)
            {
                print("Wrong event type!");
            }

            int vacancy_charge = charge_1;
            int oxygen_charge = charge_2;

            // turn the vacancy (site_1) into an oxygen
            site_1 = O;
            charge_1 = oxygen_charge;

            // turn the oxygen (site_2) into vacancy
            site_2 = VACANCY;
            charge_2 = vacancy_charge;

            break;
        }
        default:
            print("error: unidentified event key found");
        }

        // update the attributes of these sites
        device.site_element[i] = site_1;
        device.site_element[j] = site_2;
        device.site_charge[i] = charge_1;
        device.site_charge[j] = charge_2;

        // std::cout << "Executed event!" << std::endl;
        // std::cout << device.sites[i].element << " " << device.sites[j].element << std::endl;
        // std::cout << device.site_charge[i] << " " << device.site_charge[j] << std::endl;

        for (auto neigh_idx = i * num_neigh; neigh_idx < (i + 1) * num_neigh; ++neigh_idx) {
            event_type[neigh_idx] = NULL_EVENT;
            event_prob[neigh_idx] = 0.0;
        }
        for (auto neigh_idx = j * num_neigh; neigh_idx < (j + 1) * num_neigh; ++neigh_idx) {
            event_type[neigh_idx] = NULL_EVENT;
            event_prob[neigh_idx] = 0.0;
        }

        event_time = -log(random_generator.getRandomNumber()) / sel_event_prob;
    }

    delete[] event_type;
    delete[] event_prob;
    delete[] event_prob_cum;

    //std::cout << "Event time: " << event_time << std::endl;
    //return event_time;
    return 1.0;

}
