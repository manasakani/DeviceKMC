// KMC Process class
#include "random_num.h"
#include "KMCProcess.h"
#include "structure_input.h"
#include "Device.h"
#include <iostream>
#include <list>
#include <algorithm>
#include <numeric>

#ifdef USE_CUDA
    #include "cuda_wrapper.h"
#endif


KMCProcess::KMCProcess(Device &device, double _freq)
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
    for (int i = 0; i < device.N; i++)
    {
        layerID = 1000;
        for (int j = 0; j < numlayers; j++)
        {
            if (layers[j].start_x <= device.site_x[i] && device.site_x[i] <= layers[j].end_x)
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

    // initialize the affected neighborhood (at the start, this just includes all the sites)
    affected_neighborhood.resize(device.N * device.max_num_neighbors, 0);

    #pragma omp parallel for
    for (int i = 0; i < device.N * device.max_num_neighbors; i++)
    {
        affected_neighborhood[i] = i;
    }

    // initialize the size of the event list:
    // event_types = new EVENTTYPE[device.N * device.max_num_neighbors];
    // update_event_types(device);
}


void KMCProcess::update_events_and_rates(Device &device, EVENTTYPE *event_type, double *event_prob)
{
    
    int num_sites = device.N;
    int num_neigh = device.max_num_neighbors;

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
                                                device.site_x[j], device.site_y[j], device.site_z[j], 
                                                device.lattice, device.pbc);                                    // [m]

            // Generation
            if (device.site_element[i] == DEFECT && device.site_element[j] == O_EL)
            {

                double E = 2 * (device.site_potential[i] - device.site_potential[j]);
                double zero_field_energy = layers[site_layer[j]].E_gen_0;
                event_type_ = VACANCY_GENERATION;
                double Ekin = 0; //kB * (device.site_temperature[j] - device.T_bg); //double Ekin = kB * (device.site_temperature[j] - device.site_temperature[i]);
                double EA = zero_field_energy - E - Ekin;
                P = exp(-1 * EA / (kB * device.T_bg)) * freq;
            }

            // Recombination
            if (device.site_element[i] == OXYGEN_DEFECT && device.site_element[j] == VACANCY) // ELSE IF
            {
                int charge_abs = 2;
                double self_int_V = v_solve(r_dist, charge_abs, device.sigma, device.k, q);

                int charge_state = device.site_charge[i] - device.site_charge[j];
                double E = charge_state * (device.site_potential[i] - device.site_potential[j] + (charge_state / 2) * self_int_V);
                double zero_field_energy = layers[site_layer[j]].E_rec_1;

                event_type_ = VACANCY_RECOMBINATION;
                double Ekin = 0; //kB * (device.site_temperature[i] - device.T_bg); //double Ekin = kB * (device.site_temperature[i] - device.site_temperature[j]);
                double EA = zero_field_energy - E - Ekin;
                P = exp(-1 * EA / (kB * device.T_bg)) * freq;
            }

            // Vacancy diffusion
            if (device.site_element[i] == VACANCY && device.site_element[j] == O_EL)
            {

                double self_int_V = 0.0;
                if (device.site_charge[i] != 0)
                {
                    self_int_V = v_solve(r_dist, device.site_charge[i], device.sigma, device.k, q);
                }

                event_type_ = VACANCY_DIFFUSION;
                double E = (device.site_charge[i] - device.site_charge[j]) * (device.site_potential[i] - device.site_potential[j] + self_int_V);
                double zero_field_energy = layers[site_layer[i]].E_diff_2;
                double Ekin = 0; //kB * (device.site_temperature[i] - device.T_bg); //kB * (device.site_temperature[j] - device.site_temperature[i]);
                double EA = zero_field_energy - E - Ekin;
                // std::cout << "energy for moving this vacancy: " << EA << " : " << zero_field_energy << "\n";
                P = exp(-1 * EA / (kB * device.T_bg)) * freq;
            }

            // Ion diffusion
            if (device.site_element[i] == OXYGEN_DEFECT && device.site_element[j] == DEFECT) 
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

                double Ekin = 0; //kB * (device.site_temperature[i] - device.T_bg); //kB * (device.site_temperature[i] - device.site_temperature[j]);
                double EA = zero_field_energy - E - Ekin;
                P = exp(-1 * EA / (kB * device.T_bg)) * freq;
            }
        }
        event_type[idx] = event_type_;
        event_prob[idx] = P;
    }
}

void KMCProcess::update_affected_neighborhood(int event_idx, Device &device)
{

    int i = std::floor(event_idx / device.max_num_neighbors);
    int j = device.neigh_idx[event_idx];

    for (auto nn = 0; nn < device.max_num_neighbors; ++nn){
        this->affected_neighborhood.push_back(i * device.max_num_neighbors + nn);
        this->affected_neighborhood.push_back(j * device.max_num_neighbors + nn);
    }

    for (auto idx = 0; idx < device.N * device.max_num_neighbors; ++idx){
        int i_ = std::floor(idx / device.max_num_neighbors);
        int j_ = device.neigh_idx[idx];

        if (i == i_ || j == j_ || i == j_ || j == i_){
            this->affected_neighborhood.push_back(idx);
        }
    }
}

void KMCProcess::execute_event(Device &device, EVENTTYPE sel_event_type, int i, int j)
{
    switch (sel_event_type)
        {
        case VACANCY_GENERATION:
        {
            if (device.site_element[i] != DEFECT || device.site_element[j] != O_EL)
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
            device.site_element[j] = O_EL;
            device.site_charge[j] = 0;

            break;
        }
        case VACANCY_DIFFUSION:
        {
            if (device.site_element[i] != VACANCY || device.site_element[j] != O_EL)
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
}


double KMCProcess::executeKMCStep(GPUBuffers gpubuf, Device &device)
{

#ifdef USE_CUDA

    gpubuf.sync_HostToGPU(device); // remove once full while loop is completed

    double event_time = execute_kmc_step_gpu(device.N, device.max_num_neighbors, gpubuf.neigh_idx, gpubuf.site_layer,
                        gpubuf.lattice, device.pbc, gpubuf.T_bg, 
                        gpubuf.freq, gpubuf.sigma, gpubuf.k,
                        gpubuf.site_x, gpubuf.site_y, gpubuf.site_z, 
                        gpubuf.site_potential, gpubuf.site_temperature,
                        gpubuf.site_element, gpubuf.site_charge, random_generator, device.neigh_idx.data());

    gpubuf.sync_GPUToHost(device); // remove once full while loop is completed

#else

    int num_sites = device.N;
    int num_neigh = device.max_num_neighbors;

    EVENTTYPE *event_type = new EVENTTYPE[num_sites * num_neigh];
    double    *event_prob = new    double[num_sites * num_neigh];
    double    *event_prob_cum = new double[num_sites * num_neigh];

    // ** Build the event and rate lists **

    // update_event_types(device);
    // update_rates(device, event_prob);
    update_events_and_rates(device, event_type, event_prob);

    // reset the affected neighborhood for the next events
    this->affected_neighborhood.clear();
   
    // ** Event execution loop **

    double event_time = 0.0;
    while (event_time < 1 / freq) {

        // get the cumulative sum of the probabilities
        inclusive_prefix_sum<double>(event_prob, event_prob_cum, num_sites * num_neigh);

        // Select an event
        double Psum = event_prob_cum[num_sites * num_neigh - 1];
        double number = random_generator.getRandomNumber() * Psum;
        int event_idx = std::upper_bound(event_prob_cum, event_prob_cum + num_sites * num_neigh, number) - event_prob_cum;
        double sel_event_prob = event_prob_cum[event_idx];
        EVENTTYPE sel_event_type = event_type[event_idx];

        // find the site-neighbor pair corresponding to this event
        int i = std::floor(event_idx / num_neigh);
        int j = device.neigh_idx[event_idx];

        // std::cout << "selected " << number << " in [" << event_prob_cum[0] << ", " << Psum << "]" << std::endl;
        // std::cout << "Selected event index: " << event_idx << " with type "
        //          << event_type[event_idx] << " and probability " << event_prob[event_idx]
        //          << " (" << sel_event_prob << ")" << " to happen between " << return_element(device.site_element[i])
        //          << " and " << return_element(device.site_element[j]) << std::endl;

        execute_event(device, sel_event_type, i, j);

        // ** Deactivate conflicting events **

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

        // i's  and j's events with their neighbors
        #pragma omp for
        for (auto nn = 0; nn < num_neigh; ++nn){
            event_type[i * num_neigh + nn] = NULL_EVENT;
            event_type[j * num_neigh + nn] = NULL_EVENT;
            event_prob[i * num_neigh + nn] = 0.0;
            event_prob[j * num_neigh + nn] = 0.0;
        }
}
        // update event time based on the sum of all rates
        event_time = -log(random_generator.getRandomNumber()) / Psum;

        // collect the affected neighborhood
        update_affected_neighborhood(event_idx, device);
    }

    delete[] event_prob;
    delete[] event_prob_cum;
    delete[] event_type;

#endif

    return event_time;
}


// // builds the list of event types
// void KMCProcess::update_event_types(Device &device)
// {
//     #pragma omp parallel for
//     for (int k = 0; k < affected_neighborhood.size(); ++k) {

//         // site-neighbor pair indicies for this element of neigh_idx
//         int idx = affected_neighborhood[k];
//         int i = std::floor(idx / device.max_num_neighbors);
//         auto j = device.neigh_idx[idx]; 

//         // all pairs where no process can occur are represented by NULL_EVENT in the event_types list
//         EVENTTYPE event_type_ = NULL_EVENT;

//         // j is -1 if there is no neighbor at this position
//         if (j >= 0 && j < device.N) { 

//             if (device.site_element[i] == DEFECT && device.site_element[j] == O_EL)
//             {
//                 event_type_ = VACANCY_GENERATION;
//             }
//             else if (device.site_element[i] == OXYGEN_DEFECT && device.site_element[j] == VACANCY)
//             {
//                 event_type_ = VACANCY_RECOMBINATION;
//             }
//             if (device.site_element[i] == VACANCY && device.site_element[j] == O_EL)
//             {
//                 event_type_ = VACANCY_DIFFUSION;
//             }
//             else if (device.site_element[i] == OXYGEN_DEFECT && device.site_element[j] == DEFECT)
//             {
//                 event_type_ = ION_DIFFUSION;
//             }
//         }
//         event_types[idx] = event_type_;
//     }
//     std::cout << "updated event list for affected neighborhood.\n";
// }

// // updates all the rates for possible events
// void KMCProcess::update_rates(Device &device, double *event_prob)
// {
//     int num_sites = device.N;
//     int num_neigh = device.max_num_neighbors;
    
//     #pragma omp parallel for
//     for (auto idx = 0; idx < num_sites * num_neigh; ++idx) {

//         double P = 0;

//         // site_neighbor pairs for which events exist
//         if (event_types[idx] != NULL_EVENT)
//         {

//             // site-neighbor pair indicies for this element of neigh_idx
//             int i = std::floor(idx / num_neigh);
//             auto j = device.neigh_idx[idx]; 

//             // j is -1 if there is no neighbor at this position
//             if (j >= 0 && j < num_sites) { 

//                 double r_dist = (1e-10) * site_dist(device.site_x[i], device.site_y[i], device.site_z[i], 
//                                                     device.site_x[j], device.site_y[j], device.site_z[j], device.lattice, device.pbc);

//                 // Generation
//                 if (event_types[idx] == VACANCY_GENERATION)
//                 {

//                     double E = 2 * (device.site_potential[i] - device.site_potential[j]);
//                     double zero_field_energy = layers[site_layer[j]].E_gen_0;
//                     double Ekin = kB * (device.site_temperature[j] - device.T_bg); //double Ekin = kB * (device.site_temperature[j] - device.site_temperature[i]);
//                     double EA = zero_field_energy - E - Ekin;
//                     P = exp(-1 * EA / (kB * device.T_bg)) * freq;
//                 }

//                 // Recombination
//                 else if (event_types[idx] == VACANCY_RECOMBINATION)
//                 {
//                     int charge_abs = 2;
//                     double self_int_V = v_solve(r_dist, charge_abs, device.sigma, device.k, q);

//                     int charge_state = device.site_charge[i] - device.site_charge[j];
//                     double E = charge_state * (device.site_potential[i] - device.site_potential[j] + (charge_state / 2) * self_int_V);
//                     double zero_field_energy = layers[site_layer[j]].E_rec_1;

//                     double Ekin = kB * (device.site_temperature[i] - device.T_bg); //double Ekin = kB * (device.site_temperature[i] - device.site_temperature[j]);
//                     double EA = zero_field_energy - E - Ekin;
//                     P = exp(-1 * EA / (kB * device.T_bg)) * freq;
//                 }

//                 // Vacancy diffusion
//                 if (event_types[idx] == VACANCY_DIFFUSION)
//                 {

//                     double self_int_V = 0.0;
//                     if (device.site_charge[i] != 0)
//                     {
//                         double self_int_V = v_solve(r_dist, device.site_charge[i], device.sigma, device.k, q);
//                     }
//                     double E = (device.site_charge[i] - device.site_charge[j]) * (device.site_potential[i] - device.site_potential[j] + self_int_V);
//                     double zero_field_energy = layers[site_layer[i]].E_diff_2;
//                     double Ekin = kB * (device.site_temperature[i] - device.T_bg); //kB * (device.site_temperature[j] - device.site_temperature[i]);
//                     double EA = zero_field_energy - E - Ekin;
//                     P = exp(-1 * EA / (kB * device.T_bg)) * freq;
//                 }

//                 // Ion diffusion
//                 else if (event_types[idx] == ION_DIFFUSION)
//                 {
//                     int charge_abs = 2;
//                     double self_int_V = 0.0;
//                     if (device.site_charge[i] != 0)
//                     {
//                         self_int_V = v_solve(r_dist, charge_abs, device.sigma, device.k, q);
//                     }
//                     double E = (device.site_charge[i] - device.site_charge[j]) * (device.site_potential[i] - device.site_potential[j] - self_int_V);
//                     double zero_field_energy = layers[site_layer[j]].E_diff_3;

//                     double Ekin = kB * (device.site_temperature[i] - device.T_bg); //kB * (device.site_temperature[i] - device.site_temperature[j]);
//                     double EA = zero_field_energy - E - Ekin;
//                     P = exp(-1 * EA / (kB * device.T_bg)) * freq;
//                 }
//             }
//             event_prob[idx] = P;
//         } 

//         // site_neighbor pairs for which events are not possible
//         else 
//         {
//             event_prob[idx] = 0.0;
//         }
//     }
// }

// // old version which updates the entire rate list.
// double KMCProcess::executeKMCStep(GPUBuffers gpubuf, Device &device)
// {

// #ifdef USE_CUDA

//     gpubuf.sync_HostToGPU(device); // remove once full while loop is completed

//     double event_time = execute_kmc_step_gpu(device.N, device.max_num_neighbors, gpubuf.neigh_idx, gpubuf.site_layer,
//                         gpubuf.lattice, device.pbc, gpubuf.T_bg, 
//                         gpubuf.freq, gpubuf.sigma, gpubuf.k,
//                         gpubuf.site_x, gpubuf.site_y, gpubuf.site_z, 
//                         gpubuf.site_potential, gpubuf.site_temperature,
//                         gpubuf.site_element, gpubuf.site_charge, random_generator, device.neigh_idx.data());

//     gpubuf.sync_GPUToHost(device); // remove once full while loop is completed

// #else

//     // ** Build event list **

//     int num_sites = device.N;
//     int num_neigh = device.max_num_neighbors;
//     EVENTTYPE *event_type = new EVENTTYPE[num_sites * num_neigh];
//     double    *event_prob = new    double[num_sites * num_neigh];

//     // iterates through all possible site-neighbor pairs
//     #pragma omp parallel for
//     for (auto idx = 0; idx < num_sites * num_neigh; ++idx) {

//         EVENTTYPE event_type_ = NULL_EVENT;
//         double P = 0;

//         // site-neighbor pair indicies for this element of neigh_idx
//         int i = std::floor(idx / num_neigh);
//         auto j = device.neigh_idx[idx]; 

//         // j is -1 if there is no neighbor at this position
//         if (j >= 0 && j < num_sites) { 

//             double r_dist = (1e-10) * site_dist(device.site_x[i], device.site_y[i], device.site_z[i], 
//                                                 device.site_x[j], device.site_y[j], device.site_z[j], device.lattice, device.pbc);

//             // Generation
//             if (device.site_element[i] == DEFECT && device.site_element[j] == O_EL)
//             {

//                 double E = 2 * (device.site_potential[i] - device.site_potential[j]);
//                 double zero_field_energy = layers[site_layer[j]].E_gen_0;
//                 event_type_ = VACANCY_GENERATION;
//                 double Ekin = kB * (device.site_temperature[j] - device.T_bg); //double Ekin = kB * (device.site_temperature[j] - device.site_temperature[i]);
//                 double EA = zero_field_energy - E - Ekin;
//                 P = exp(-1 * EA / (kB * device.T_bg)) * freq;
//             }

//             // Recombination
//             else if (device.site_element[i] == OXYGEN_DEFECT && device.site_element[j] == VACANCY)
//             {
//                 int charge_abs = 2;
//                 double self_int_V = v_solve(r_dist, charge_abs, device.sigma, device.k, q);

//                 int charge_state = device.site_charge[i] - device.site_charge[j];
//                 double E = charge_state * (device.site_potential[i] - device.site_potential[j] + (charge_state / 2) * self_int_V);
//                 double zero_field_energy = layers[site_layer[j]].E_rec_1;

//                 event_type_ = VACANCY_RECOMBINATION;
//                 double Ekin = kB * (device.site_temperature[i] - device.T_bg); //double Ekin = kB * (device.site_temperature[i] - device.site_temperature[j]);
//                 double EA = zero_field_energy - E - Ekin;
//                 P = exp(-1 * EA / (kB * device.T_bg)) * freq;
//             }

//             // Vacancy diffusion
//             if (device.site_element[i] == VACANCY && device.site_element[j] == O_EL)
//             {

//                 double self_int_V = 0.0;
//                 if (device.site_charge[i] != 0)
//                 {
//                     double self_int_V = v_solve(r_dist, device.site_charge[i], device.sigma, device.k, q);
//                 }

//                 event_type_ = VACANCY_DIFFUSION;
//                 double E = (device.site_charge[i] - device.site_charge[j]) * (device.site_potential[i] - device.site_potential[j] + self_int_V);
//                 double zero_field_energy = layers[site_layer[i]].E_diff_2;
//                 double Ekin = kB * (device.site_temperature[i] - device.T_bg); //kB * (device.site_temperature[j] - device.site_temperature[i]);
//                 double EA = zero_field_energy - E - Ekin;
//                 P = exp(-1 * EA / (kB * device.T_bg)) * freq;
//             }

//             // Ion diffusion
//             else if (device.site_element[i] == OXYGEN_DEFECT && device.site_element[j] == DEFECT)
//             {
//                 int charge_abs = 2;
//                 double self_int_V = 0.0;
//                 if (device.site_charge[i] != 0)
//                 {
//                     self_int_V = v_solve(r_dist, charge_abs, device.sigma, device.k, q);
//                 }
//                 double E = (device.site_charge[i] - device.site_charge[j]) * (device.site_potential[i] - device.site_potential[j] - self_int_V);
//                 double zero_field_energy = layers[site_layer[j]].E_diff_3;

//                 event_type_ = ION_DIFFUSION;

//                 double Ekin = kB * (device.site_temperature[i] - device.T_bg); //kB * (device.site_temperature[i] - device.site_temperature[j]);
//                 double EA = zero_field_energy - E - Ekin;
//                 P = exp(-1 * EA / (kB * device.T_bg)) * freq;
//             }
//         }
//         event_type[idx] = event_type_;
//         event_prob[idx] = P;
//     }

//     // ** Event execution **

//     double *event_prob_cum = new double[num_sites * num_neigh];
//     double event_time = 0.0;
//     while (event_time < 1 / freq) {

//         // NOTE: We can optimize this by only updating the required values
//         // get the cumulative sum of the probabilities
//         inclusive_prefix_sum<double>(event_prob, event_prob_cum, num_sites * num_neigh);

//         // Select an event
//         double Psum = event_prob_cum[num_sites * num_neigh - 1];
//         double number = random_generator.getRandomNumber() * Psum;
//         int event_idx = std::upper_bound(event_prob_cum, event_prob_cum + num_sites * num_neigh, number) - event_prob_cum;
//         double sel_event_prob = event_prob_cum[event_idx];

//         //std::cout << "Searching for " << number << " in [" << event_prob_cum[0] << ", " << Psum << "]" << std::endl;

//         EVENTTYPE sel_event_type = event_type[event_idx];

//         // find the site-neighbor pair corresponding to this event
//         int i = std::floor(event_idx / num_neigh);
//         auto j = device.neigh_idx[event_idx];

//         // std::cout << "Selected event index: " << event_idx << " with type "
//         //          << event_type[event_idx] << " and probability " << event_prob[event_idx]
//         //          << " (" << sel_event_prob << ")" << " to happen between " << device.site_element[i]
//         //          << " and " << device.site_element[j] << std::endl;

//         // Event execution
//         switch (sel_event_type)
//         {
//         case VACANCY_GENERATION:
//         {
//             if (device.site_element[i] != DEFECT || device.site_element[j] != O_EL)
//             {
//                 print("Wrong event type - VACANCY_GENERATION!");
//                 print(return_element(device.site_element[i]) << " and " << return_element(device.site_element[j]));
//             }

//             // turn the defect (site_1) into an oxygen ion:
//             device.site_element[i] = OXYGEN_DEFECT;
//             device.site_charge[i] = -2;

//             // turn the oxygen (site_2) into a charged vacancy:
//             device.site_element[j] = VACANCY;
//             device.site_charge[j] = 2;

//             break;
//         }
//         case VACANCY_RECOMBINATION:
//         {
//             if (device.site_element[i] != OXYGEN_DEFECT || device.site_element[j] != VACANCY)
//             {
//                 print("Wrong event type - VACANCY_RECOMBINATION!");
//                 print(return_element(device.site_element[i]) << " and " << return_element(device.site_element[j]));
//             }

//             // turn the oxygen (site_1) into a defect
//             device.site_element[i] = DEFECT;
//             device.site_charge[i] = 0;

//             // turn the vacancy (site_2) into an oxygen atom:
//             device.site_element[j] = O_EL;
//             device.site_charge[j] = 0;

//             break;
//         }
//         case VACANCY_DIFFUSION:
//         {
//             if (device.site_element[i] != VACANCY || device.site_element[j] != O_EL)
//             {
//                 print("Wrong event type - VACANCY_DIFFUSION!");
//                 print(return_element(device.site_element[i]) << " and " << return_element(device.site_element[j]));
//             }

//             swap_values(&device.site_element[i], &device.site_element[j]);
//             swap_values(&device.site_charge[i], &device.site_charge[j]);

//             break;
//         }
//         case ION_DIFFUSION:
//         {
//             if (device.site_element[i] != OXYGEN_DEFECT || device.site_element[j] != DEFECT)
//             {
//                 print("Wrong event type - ION_DIFFUSION!");
//                 print(return_element(device.site_element[i]) << " and " << return_element(device.site_element[j]));
//             }

//             swap_values(&device.site_element[i], &device.site_element[j]);
//             swap_values(&device.site_charge[i], &device.site_charge[j]);

//             break;
//         }
//         default:
//             print("error: unidentified event key found");
//         }

//         // Deactivate conflicting events
//         int i_, j_;
// #pragma omp parallel private(i_, j_)
// {
//         // other site's events with i or j
//         #pragma omp for
//         for (auto idx = 0; idx < num_sites * num_neigh; ++idx){
//             i_ = std::floor(idx / num_neigh);
//             j_ = device.neigh_idx[idx];

//             if (i == i_ || j == j_ || i == j_ || j == i_){
//                 event_type[idx] = NULL_EVENT;
//                 event_prob[idx] = 0.0;
//             }
//         }
//         // i's events with its neighbors
//         #pragma omp for
//         for (auto neigh_idx = i * num_neigh; neigh_idx < (i + 1)*num_neigh; ++neigh_idx){
//             event_type[neigh_idx] = NULL_EVENT;
//             event_prob[neigh_idx] = 0.0;
//         }
//         // j's events with its neighbors
//         #pragma omp for
//         for (auto neigh_idx = j * num_neigh; neigh_idx < (j + 1)*num_neigh; ++neigh_idx){
//             event_type[neigh_idx] = NULL_EVENT;
//             event_prob[neigh_idx] = 0.0;
//         }
// }
//         event_time = -log(random_generator.getRandomNumber()) / Psum;
//     }

//     delete[] event_type;
//     delete[] event_prob;
//     delete[] event_prob_cum;

// #endif

//     return event_time;
// }