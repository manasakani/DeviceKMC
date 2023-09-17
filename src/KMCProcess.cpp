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

KMCProcess::KMCProcess(Device *device)
{

    // initialize random number generator
    random_generator.setSeed(rnd_seed_kmc);

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
            if (layers[j].start_x <= device->sites[i].pos[0] && device->sites[i].pos[0] <= layers[j].end_x)
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

Event *KMCProcess::pick_and_get_event(std::list<Event> &event_list, int event_list_size, double Psum)
{

    int i;
    double random_num = random_generator.getRandomNumber();
    double Ppointer = random_num * Psum;
    double p_sum = 0;

    for (auto it = event_list.begin(); it != event_list.end(); ++it)
    {
        p_sum += it->prob;

        if (p_sum > Ppointer)
        {
            return &(*it);
        }
    }

    return &(event_list.back());
}

void KMCProcess::execute_event(Site *site_1, Site *site_2, int &event_type, int &charge_1, int &charge_2)
{
    /*Key for event_type:
     * 0 - Vacancy/Ion Pair Generation
     * 1 - Vacancy/Ion Pair Recombination
     * 2 - Vacancy Diffusion
     * 3 - Ion Diffusion
     */
    std::string event_name;

    switch (event_type)
    {
    case 0:
    {
        if (site_1->element != "d" || site_2->element != "O")
        {
            print("Wrong event type!");
        }
        event_name = "vacancy/ion pair generation";

        // turn the defect (site_1) into an oxygen ion:
        site_1->element = "Od";
        charge_1 = -2;

        // turn the oxygen (site_2) into a charged vacancy:
        site_2->element = "V";
        charge_2 = 2;

        break;
    }
    case 1:
    {
        if (site_1->element != "Od" || site_2->element != "V")
        {
            print("Wrong event type!");
        }
        event_name = "vacancy/ion pair recombination";

        // turn the oxygen (site_1) into a defect
        site_1->element = "d";
        charge_1 = 0;

        // turn the vacancy (site_2) into an oxygen atom:
        site_2->element = "O";
        charge_2 = 0;

        break;
    }
    case 2:
    {
        event_name = "vacancy diffusion";
        if (site_1->element != "V" || site_2->element != "O")
        {
            print("Wrong event type!");
        }

        int vacancy_charge = charge_1;
        int oxygen_charge = charge_2;

        // turn the vacancy (site_1) into an oxygen
        site_1->element = "O";
        charge_1 = oxygen_charge;

        // turn the oxygen (site_2) into vacancy
        site_2->element = "V";
        charge_2 = vacancy_charge;

        break;
    }
    case 3:
    {
        if (site_1->element != "Od" || site_2->element != "d")
        {
            print("Wrong event type!");
        }
        event_name = "ion diffusion";
        int oxygen_charge = charge_1;

        // turn the oxygen (site_1) into a defect
        site_1->element = "d";
        charge_1 = 0;

        // turn the defect (site_2) into an oxygen
        site_2->element = "Od";
        charge_2 = oxygen_charge;

        break;
    }
    default:
        print("error: unidentified event key found");
    }
    // print("executed an " << event_name << " event between " << site_1->element << "-" << site_1->ind << " and " << site_2->element << "-" << site_2->ind << " with charges " << charge_1 << " and " << charge_2;
}

double KMCProcess::executeKMCStep(Device *device, double freq, std::vector<double> lattice, bool pbc)
{

    // build event list
    std::list<Event> event_list;
    long double Psum = 0.0;
    Event *t_temp;

#pragma omp parallel
    {
        // this thread gets its own event list and adds events to it
        std::list<Event> t_event_list;

        std::string site1_type, site2_type;
        int charge_abs, charge_state, event_type_;
        double zero_field_energy, E, EA, r_dist, self_int_V, event_temp, delta_temp, Ekin;
        long double P;
        double T_kmc = device->T_bg; // CHANGE WITH REAL TEMPERATURE

#pragma omp for schedule(dynamic) reduction(+ : Psum)
        for (int i = 0; i < device->N; i++)
        {

            for (int j : device->site_neighbors.l[i])
            {

                r_dist = (1e-10) * site_dist(device->sites[i].pos, device->sites[j].pos, lattice, pbc);
                site1_type = device->sites[i].element;
                site2_type = device->sites[j].element;

                // Generation
                if (site1_type == "d" && site2_type == "O")
                {

                    E = 2 * (device->site_potential[i] - device->site_potential[j]);
                    zero_field_energy = layers[site_layer[j]].E_gen_0;
                    event_type_ = 0;
                    Ekin = kB * (device->site_temperature[i] - device->site_temperature[j]);
                    EA = zero_field_energy - E - Ekin;
                    P = exp(-1 * EA / (kB * T_kmc)) * freq;
                    t_event_list.emplace_back(i, j, event_type_, P);

                    Psum += P;
                }

                // Recombination
                else if (site1_type == "Od" && site2_type == "V")
                {

                    charge_abs = 2;
                    self_int_V = v_solve(r_dist, charge_abs, device->sigma, device->k, q);

                    charge_state = device->site_charge[i] - device->site_charge[j];
                    E = charge_state * (device->site_potential[i] - device->site_potential[j] + (charge_state / 2) * self_int_V);
                    zero_field_energy = layers[site_layer[j]].E_rec_1;

                    event_type_ = 1;
                    Ekin = kB * (device->site_temperature[i] - device->site_temperature[j]);
                    EA = zero_field_energy - E - Ekin;
                    P = exp(-1 * EA / (kB * T_kmc)) * freq;
                    t_event_list.emplace_back(i, j, event_type_, P);

                    Psum += P;
                }

                // Vacancy diffusion
                else if (site1_type == "V" && site2_type == "O")
                {

                    if (device->site_charge[i] != 0)
                    {
                        self_int_V = v_solve(r_dist, device->site_charge[i], device->sigma, device->k, q);
                    }
                    else
                    {
                        self_int_V = 0;
                    }

                    E = (device->site_charge[i] - device->site_charge[j]) * (device->site_potential[i] - device->site_potential[j] + self_int_V);
                    zero_field_energy = layers[site_layer[i]].E_diff_2;

                    event_type_ = 2;
                    Ekin = kB * (device->site_temperature[i] - device->site_temperature[j]);
                    EA = zero_field_energy - E - Ekin;
                    P = exp(-1 * EA / (kB * T_kmc)) * freq;
                    t_event_list.emplace_back(i, j, event_type_, P);

                    Psum += P;
                }

                // Ion diffusion
                else if (site1_type == "Od" && site2_type == "d")
                {

                    charge_abs = 2;

                    if (device->site_charge[i] != 0)
                    {
                        self_int_V = v_solve(r_dist, charge_abs, device->sigma, device->k, q);
                    }
                    else
                    {
                        self_int_V = 0;
                    }
                    E = (device->site_charge[i] - device->site_charge[j]) * (device->site_potential[i] - device->site_potential[j] - self_int_V);
                    zero_field_energy = layers[site_layer[j]].E_diff_3;

                    event_type_ = 3;

                    Ekin = kB * (device->site_temperature[i] - device->site_temperature[j]);
                    EA = zero_field_energy - E - Ekin;
                    P = exp(-1 * EA / (kB * T_kmc)) * freq;
                    t_event_list.emplace_back(i, j, event_type_, P);

                    Psum += P;
                }
            }
        }

// link all the thread-local event lists
#pragma omp critical
        {
            event_list.splice(event_list.end(), t_event_list);
        }
    }

    // print_event_list(event_list, "events.txt", 1e-100);
    // print("size of event list: " << event_list.size());
    // print("total Psum: " << Psum);

    // Event execution loop:
    double event_time = 0.0;
    int event_cntr, track_ind, event_ind, site_1_ind, site_2_ind;
    bool conflict_1, conflict_2, conflict_3, conflict_4;
    Event *selected_event;
    Event *temp;

    while (event_time < 1 / freq)
    {
        event_cntr = event_list.size();

        if (event_cntr == 0)
        {
            print("entire event list completed - this should not happen.");
            break;
        }

        // select and execute an event:
        selected_event = pick_and_get_event(event_list, event_cntr, Psum);

        /*print("-------");
        print("before event, ind1 was: " << device->sites[selected_event->ind1].element << " with charge " << device->site_charge[selected_event->ind1]);
        print("before event, ind2 was: " << device->sites[selected_event->ind2].element << " with charge " << device->site_charge[selected_event->ind2]);*/

        execute_event(&(device->sites[selected_event->ind1]),
                      &(device->sites[selected_event->ind2]),
                      selected_event->event_type,
                      device->site_charge[selected_event->ind1],
                      device->site_charge[selected_event->ind2]);

        /*print("after event, ind1 was: " << device->sites[selected_event->ind1].element << " with charge " << device->site_charge[selected_event->ind1]);
        print("after event, ind2 was: " << device->sites[selected_event->ind2].element << " with charge " << device->site_charge[selected_event->ind2]);*/

        // remove all events containing the indices of sites[i] and sites[j] from the event list
        track_ind = 0;
        Psum = 0.0;
        auto temp = event_list.begin();
        site_1_ind = selected_event->ind1;
        site_2_ind = selected_event->ind2;

        while (temp != event_list.end())
        {
            conflict_1 = (temp->ind1 == site_1_ind);
            conflict_2 = (temp->ind1 == site_2_ind);
            conflict_3 = (temp->ind2 == site_1_ind);
            conflict_4 = (temp->ind2 == site_2_ind);

            if (conflict_1 || conflict_2 || conflict_3 || conflict_4)
            {
                temp = event_list.erase(temp);
            }
            else
            {
                Psum += temp->prob;
                temp++;
                track_ind++;
            }
        }

        event_time = -log(random_generator.getRandomNumber()) / Psum;
        // print("kmc event time: " << event_time << ". Psum: " << Psum);
    }

    // free memory from remaining events
    event_list.clear();
    return event_time;
}
