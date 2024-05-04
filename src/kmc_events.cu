#include "hip/hip_runtime.h"
#include "gpu_solvers.h"
#include <omp.h>
// Constants needed:
constexpr double kB = 8.617333262e-5;           // [eV/K]

#define NUM_THREADS 512
#define MAX_NUM_LAYERS 5

// in GPU cache
__constant__ double E_gen_const[MAX_NUM_LAYERS];
__constant__ double E_rec_const[MAX_NUM_LAYERS];
__constant__ double E_Vdiff_const[MAX_NUM_LAYERS];
__constant__ double E_Odiff_const[MAX_NUM_LAYERS];

// void get_gpu_info(char *gpu_string, int dev){
//     struct hipDeviceProp_t dprop;
//     hipError_t cudaStatus;

//     cudaStatus = hipSetDevice(dev);
//     if (cudaStatus != hipSuccess) {
//         fprintf(stderr, "hipSetDevice failed! Error: %s\n", hipGetErrorString(cudaStatus));
//         // Handle the error or exit the program
//         exit(EXIT_FAILURE);
//     }

//     hipGetDeviceProperties(&dprop, dev);
//     strcpy(gpu_string,dprop.name);
// }

void get_gpu_info(char *gpu_string, int dev) {
    hipDeviceProp_t dprop;
    // hipError_t hipStatus;

    // // hipStatus = hipSetDevice(dev);
    // if (hipStatus != hipSuccess) {
    //     fprintf(stderr, "hipSetDevice failed! Error: %s\n", hipGetErrorString(hipStatus));
    //     exit(EXIT_FAILURE);
    // }

    hipGetDeviceProperties(&dprop, dev);
    sprintf(gpu_string, "GPU %d", dev);
}

// void set_gpu(int dev){
//  hipSetDevice(dev);
// }

__global__ void build_event_list(const int N, const int nn, const int *neigh_idx, 
                                 const int *layer, const double *lattice, const int pbc, 
                                 const double *T_bg, const double *freq, const double *sigma, const double *k, 
                                 const double *posx, const double *posy, const double *posz,
                                 const double *potential_charge, const double *temperature,
                                 const ELEMENT *element, const int *charge, EVENTTYPE *event_type, double *event_prob)
{
    int total_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int idx = total_tid; idx < N * nn; idx += total_threads) {

        EVENTTYPE event_type_ = NULL_EVENT;
        double P = 0.0;

        int i = static_cast<int>(floorf(idx / nn));
        int j = neigh_idx[idx];

        // condition for neighbor existing
        if (j >= 0 && j < N) {
            double dist = 1e-10 * site_dist_gpu(posx[i], posy[i], posz[i], 
                                                posx[j], posy[j], posz[j], 
                                                lattice[0], lattice[1], lattice[2], pbc);

            // potential_charge now contains the sum of the potential

            // Generation
            if (element[i] == DEFECT && element[j] == O_EL)
            {

                double E = 2 * ((potential_charge[i]) - (potential_charge[j]));
                double zero_field_energy = E_gen_const[layer[j]]; 
                event_type_ = VACANCY_GENERATION;
                double Ekin = 0; // kB * (temperature[j] - (*T_bg)); //kB * (temperature[j] - temperature[i]);
                double EA = zero_field_energy - E - Ekin;
                P = exp(-1 * EA / (kB * (*T_bg))) * (*freq);
            }

            // Recombination
            if (element[i] == OXYGEN_DEFECT && element[j] == VACANCY) 
            {
                int charge_abs = 2;
                double self_int_V = v_solve_gpu(dist, charge_abs, sigma, k);

                int charge_state = charge[i] - charge[j];
                double E = charge_state * ((potential_charge[i]) - (potential_charge[j]) + (charge_state / 2) * self_int_V);
                double zero_field_energy = E_rec_const[layer[j]];

                event_type_ = VACANCY_RECOMBINATION;
                double Ekin = 0; //kB * (temperature[i] - (*T_bg)); //kB * (temperature[i] - temperature[j]);
                double EA = zero_field_energy - E - Ekin;
                P = exp(-1 * EA / (kB * (*T_bg))) * (*freq);
            }

            // Vacancy diffusion
            if (element[i] == VACANCY && element[j] == O_EL)
            {

                double self_int_V = 0.0;
                if (charge[i] != 0)
                {
                    self_int_V = v_solve_gpu(dist, charge[i], sigma, k);
                }

                event_type_ = VACANCY_DIFFUSION;
                double E = (charge[i] - charge[j]) * ((potential_charge[i]) - (potential_charge[j]) + self_int_V);
                double zero_field_energy = E_Vdiff_const[layer[j]];  
                double Ekin = 0;//kB * (temperature[i] - (*T_bg)); //kB * (temperature[j] - temperature[i]);
                double EA = zero_field_energy - E - Ekin;
                P = exp(-1 * EA / (kB * (*T_bg))) * (*freq);
            }

            // Ion diffusion
            if (element[i] == OXYGEN_DEFECT && element[j] == DEFECT)
            {
                int charge_abs = 2;
                double self_int_V = 0.0;
                if (charge[i] != 0)
                {                    
                    self_int_V = v_solve_gpu(dist, charge_abs, sigma, k);
                }

                double E = (charge[i] - charge[j]) * ((potential_charge[i]) - (potential_charge[j]) - self_int_V);
                double zero_field_energy = E_Odiff_const[layer[j]];

                event_type_ = ION_DIFFUSION;
                double Ekin = 0; //kB * (temperature[i] - (*T_bg)); //kB * (temperature[i] - temperature[j]);
                double EA = zero_field_energy - E - Ekin;
                P = exp(-1 * EA / (kB * (*T_bg))) * (*freq);
            }
        }
        event_type[idx] = event_type_;
        event_prob[idx] = P;
    }
}

// builds only part of the event list
// from start_i to start_i + size_i
__global__ void build_event_list_split(const int N, const int size_i, const int start_i,
                                 const int nn, const int *neigh_idx, 
                                 const int *layer, const double *lattice, const int pbc, 
                                 const double *T_bg, const double *freq, const double *sigma, const double *k, 
                                 const double *posx, const double *posy, const double *posz,
                                 const double *potential_charge, const double *temperature,
                                 const ELEMENT *element, const int *charge, EVENTTYPE *event_type, double *event_prob)
{
    int total_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int id = total_tid; id < size_i * nn; id += total_threads) {
        EVENTTYPE event_type_ = NULL_EVENT;
        double P = 0.0;

        // access neigh_idx with id and not idx
        int idx = id + start_i * nn;
        int i = id / nn + start_i;
        int j = neigh_idx[id];

        double epsilon = 1e-200; // for exponential overflow

        // condition for neighbor existing
        if (j >= 0 && j < N) {
            double dist = 1e-10 * site_dist_gpu(posx[i], posy[i], posz[i], 
                                                posx[j], posy[j], posz[j]);

            // Generation
            if (element[i] == DEFECT && element[j] == O_EL)
            {

                double E = 2 * ((potential_charge[i]) - (potential_charge[j]));
                double zero_field_energy = E_gen_const[layer[j]]; 
                event_type_ = VACANCY_GENERATION;
                double Ekin = 0; // kB * (temperature[j] - (*T_bg)); //kB * (temperature[j] - temperature[i]);
                double EA = zero_field_energy - E - Ekin;
                // P = exp(-1 * EA / (kB * (*T_bg))) * (*freq);
                P = (*freq) * (1 / (exp(EA / (kB * (*T_bg))) + epsilon) );
            }

            // Recombination
            if (element[i] == OXYGEN_DEFECT && element[j] == VACANCY) 
            {
                int charge_abs = 2;
                double self_int_V = v_solve_gpu(dist, charge_abs, sigma, k);

                int charge_state = charge[i] - charge[j];
                double E = charge_state * ((potential_charge[i]) - (potential_charge[j]) + (charge_state / 2) * self_int_V);
                double zero_field_energy = E_rec_const[layer[j]];

                event_type_ = VACANCY_RECOMBINATION;
                double Ekin = 0; //kB * (temperature[i] - (*T_bg)); //kB * (temperature[i] - temperature[j]);
                double EA = zero_field_energy - E - Ekin;
                // P = exp(-1 * EA / (kB * (*T_bg))) * (*freq);
                P = (*freq) * (1 / (exp(EA / (kB * (*T_bg))) + epsilon) );
            }

            // Vacancy diffusion
            if (element[i] == VACANCY && element[j] == O_EL)
            {

                double self_int_V = 0.0;
                if (charge[i] != 0)
                {
                    self_int_V = v_solve_gpu(dist, charge[i], sigma, k);
                }

                event_type_ = VACANCY_DIFFUSION;
                double E = (charge[i] - charge[j]) * ((potential_charge[i]) - (potential_charge[j]) + self_int_V);
                double zero_field_energy = E_Vdiff_const[layer[j]];  
                double Ekin = 0;//kB * (temperature[i] - (*T_bg)); //kB * (temperature[j] - temperature[i]);
                double EA = zero_field_energy - E - Ekin;
                // P = exp(-1 * EA / (kB * (*T_bg))) * (*freq);
                P = (*freq) * (1 / (exp(EA / (kB * (*T_bg))) + epsilon) );
            }

            // Ion diffusion
            if (element[i] == OXYGEN_DEFECT && element[j] == DEFECT)
            {
                int charge_abs = 2;
                double self_int_V = 0.0;
                if (charge[i] != 0)
                {                    
                    self_int_V = v_solve_gpu(dist, charge_abs, sigma, k);
                }

                double E = (charge[i] - charge[j]) * ((potential_charge[i]) - (potential_charge[j]) - self_int_V);
                double zero_field_energy = E_Odiff_const[layer[j]];

                event_type_ = ION_DIFFUSION;
                double Ekin = 0; //kB * (temperature[i] - (*T_bg)); //kB * (temperature[i] - temperature[j]);
                double EA = zero_field_energy - E - Ekin;
                // P = exp(-1 * EA / (kB * (*T_bg))) * (*freq);
                P = (*freq) * (1 / (exp(EA / (kB * (*T_bg))) + epsilon) );
            }
        }
        event_type[id] = event_type_;
        event_prob[id] = P;
    }
}


__global__ void zero_out_events(EVENTTYPE *event_type, double *event_prob, const int *neigh_idx, int N, int nn, int i_to_delete, int j_to_delete){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    int j;
    for (int id = idx; id < N * nn; id += blockDim.x * gridDim.x){
        i = id / nn;
        j = neigh_idx[id];

        if (i == i_to_delete || j == j_to_delete || i == j_to_delete || j == i_to_delete){
            event_type[id] = NULL_EVENT;
            event_prob[id] = 0.0;
        }
    }
}

__global__ void zero_out_events_split(EVENTTYPE *event_type, double *event_prob, const int *neigh_idx,
        const int size_i, const int start_i,
        int nn, int i_to_delete, int j_to_delete)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long i;
    long j;

    for (int id = idx; id < (size_t)size_i * (size_t)nn; id += blockDim.x * gridDim.x){
        i = id / nn + start_i;
        // j = neigh_idx[id+start_i*nn];
        j = neigh_idx[id];

        if ( j >=0 && (i == i_to_delete || j == j_to_delete || i == j_to_delete || j == i_to_delete)){
            event_type[id] = NULL_EVENT;
            event_prob[id] = 0.0;
        }
    }

}

__global__ void read_out_event(
    int *ijevent_to_delete,
    const int i,
    const int *j,
    const EVENTTYPE *type
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx == 0){
        ijevent_to_delete[0] = i;
        ijevent_to_delete[1] = *j;
        ijevent_to_delete[2] = int(*type);
    }
}

template <typename T>
__device__ void swap(
    T *a, T *b
){
    T temp = *a;
    *a = *b;
    *b = temp;
}

__global__ void execute_event(
    ELEMENT *site_element,
    int *site_charge,
    int *ijevent_to_delete
){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx == 0){
        int i_host = ijevent_to_delete[0];
        int j_host = ijevent_to_delete[1];
        EVENTTYPE sel_event_type = EVENTTYPE(ijevent_to_delete[2]);

        if(sel_event_type == VACANCY_GENERATION){
            site_element[i_host] = OXYGEN_DEFECT;
            site_element[j_host] = VACANCY;
            site_charge[i_host] = -2;
            site_charge[j_host] = 2;
        }
        //TODO generalizable
        else if(sel_event_type == VACANCY_RECOMBINATION){
            site_element[i_host] = DEFECT;
            site_element[j_host] = O_EL;
            site_charge[i_host] = 0;
            site_charge[j_host] = 0;

        }
        else if(sel_event_type == VACANCY_DIFFUSION){
            // swap
            swap<ELEMENT>(site_element + i_host, site_element + j_host);
            swap<int>(site_charge + i_host, site_charge + j_host);
        }
        else if(sel_event_type == ION_DIFFUSION){
            // swap
            swap<ELEMENT>(site_element + i_host, site_element + j_host);
            swap<int>(site_charge + i_host, site_charge + j_host);
        }

    }
} 

double execute_kmc_step_mpi(
        MPI_Comm comm,
        const int N,
        const int *count,
        const int *displs,
        const int nn, const int *neigh_idx, const int *site_layer,
        const double *lattice, const int pbc, const double *T_bg, 
        const double *freq, const double *sigma, const double *k,
        const double *posx, const double *posy, const double *posz, 
        const double *site_potential_charge, const double *site_temperature,
        ELEMENT *site_element, int *site_charge, RandomNumberGenerator &rng)
{


    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int measurements = 1;
    // double time_list[measurements];
    // double time_events[measurements];

    // the KMC event list arrays only exist in gpu memory
    EVENTTYPE *event_type_local_d; 
    double    *event_prob_local_d; 
    gpuErrchk( hipMalloc((void**)&event_type_local_d, (size_t)count[rank] * (size_t)nn * sizeof(EVENTTYPE)) );
    gpuErrchk( hipMalloc((void**)&event_prob_local_d, (size_t)count[rank] * (size_t)nn * sizeof(double)) );
    double *event_prob_cum_local_d;
    gpuErrchk( hipMalloc((void**)&event_prob_cum_local_d, (size_t)count[rank] * (size_t)nn * sizeof(double)) );
    double *event_prob_cum_global_h;
    gpuErrchk( hipMallocHost((void**)&event_prob_cum_global_h, size * sizeof(double)));

    int ijevent_to_delete[3];
    int *ijevent_to_delete_d;
    gpuErrchk( hipMalloc((void**)&ijevent_to_delete_d, 3 * sizeof(int)) );

    double event_time = 0.0;
    int event_counter = 0;

    double *Psum_host;
    gpuErrchk(hipMallocHost((void**)&Psum_host, 1 * sizeof(double)));

    // **************************
    // **** Build Event List ****
    // **************************


    for(int i = 0; i < measurements; i++){

        // gpuErrchk( hipMemset(event_prob_local_d, 0, (size_t)count[rank] * (size_t)nn * sizeof(double)) );
        // gpuErrchk( hipMemset(event_type_local_d, 0, (size_t)count[rank] * (size_t)nn * sizeof(EVENTTYPE)) );


        // hipDeviceSynchronize();
        // MPI_Barrier(comm);

        // auto start = std::chrono::high_resolution_clock::now();


        int num_threads = 1024;
        int num_blocks = ((size_t)count[rank] * (size_t)nn - 1) / num_threads + 1;

        // populate the event_type and event_prob arrays:
        // only your part of the event list
        // TODO use COO
        build_event_list_split<<<num_blocks, num_threads>>>(N,
                                                    count[rank], displs[rank],
                                                    nn, neigh_idx, 
                                                    site_layer, lattice, pbc,
                                                    T_bg, freq, sigma, k,
                                                    posx, posy, posz, 
                                                    site_potential_charge, site_temperature, 
                                                    site_element, site_charge,
                                                    event_type_local_d, event_prob_local_d);

        // hipDeviceSynchronize();
        // MPI_Barrier(comm);
        
        // auto end = std::chrono::high_resolution_clock::now();
        // time_list[i] = std::chrono::duration<double>(end - start).count();
        // if(rank == 0){
        //     std::cout << "Time to build event list: " << time_list[i] << std::endl;
        // }

    }
    // EVENTTYPE *event_type_local_d_copy; 
    // double    *event_prob_local_d_copy; 
    // gpuErrchk( hipMalloc((void**)&event_type_local_d_copy, (size_t)count[rank] * (size_t)nn * sizeof(EVENTTYPE)) );
    // gpuErrchk( hipMalloc((void**)&event_prob_local_d_copy, (size_t)count[rank] * (size_t)nn * sizeof(double)) );

    // gpuErrchk( hipMemcpy(event_type_local_d_copy, event_type_local_d, (size_t)count[rank] * (size_t)nn * sizeof(EVENTTYPE), hipMemcpyDeviceToDevice) );
    // gpuErrchk( hipMemcpy(event_prob_local_d_copy, event_prob_local_d, (size_t)count[rank] * (size_t)nn * sizeof(double), hipMemcpyDeviceToDevice) );


    // **************************
    // ** Event Execution Loop **
    // **************************
    for(int j = 0; j < measurements; j++){

        event_time = 0.0;
        event_counter = 0;

        // gpuErrchk( hipMemcpy(event_type_local_d, event_type_local_d_copy, (size_t)count[rank] * (size_t)nn * sizeof(EVENTTYPE), hipMemcpyDeviceToDevice) );
        // gpuErrchk( hipMemcpy(event_prob_local_d, event_prob_local_d_copy, (size_t)count[rank] * (size_t)nn * sizeof(double), hipMemcpyDeviceToDevice) );
        // gpuErrchk( hipMemset(event_prob_cum_local_d, 0, (size_t)count[rank] * (size_t)nn * sizeof(double)) );
        // hipDeviceSynchronize();
        // MPI_Barrier(comm);

        // auto start = std::chrono::high_resolution_clock::now();

        int threads_single_block = 64;


        double freq_h;
        gpuErrchk( hipMemcpy(&freq_h, freq, 1 * sizeof(double), hipMemcpyDeviceToHost) );
        while (event_time < 1 / freq_h) {
        // while (event_counter < 1000) {
            event_counter++;  

            // get the cumulative sum of the probabilities
            thrust::inclusive_scan(thrust::device, event_prob_local_d, event_prob_local_d + (size_t)count[rank] * (size_t)nn, event_prob_cum_local_d);
            
            // select an event

            // allgather with host pointers
            gpuErrchk( hipMemcpy(Psum_host, event_prob_cum_local_d + (size_t)count[rank] * (size_t)nn - 1, sizeof(double), hipMemcpyDeviceToHost) );
            MPI_Allgather(Psum_host, 1, MPI_DOUBLE, event_prob_cum_global_h, 1, MPI_DOUBLE, comm);

            // allgather with device pointers:
            // MPI_Allgather(event_prob_cum_local_d + count[rank] * nn - 1, 1, MPI_DOUBLE, event_prob_cum_global_h, 1, MPI_DOUBLE, comm);

            for (int i = 1; i < size; i++){
                event_prob_cum_global_h[i] += event_prob_cum_global_h[i-1];
            }

            //TODO: cuda random number
            double number = rng.getRandomNumber() * event_prob_cum_global_h[size-1];
            // figure out which rank has the number
            int source_rank;
            for (int i = 0; i < size; i++){
                if (number < event_prob_cum_global_h[i]){
                    source_rank = i;
                    break;
                }
            }

            if(rank == source_rank){
                // shift random number to the correct range
                if(rank > 0){
                    number -= event_prob_cum_global_h[rank-1];
                }
            
                int event_idx = thrust::upper_bound(thrust::device, event_prob_cum_local_d, event_prob_cum_local_d + (size_t)count[rank] * (size_t)nn, number) - event_prob_cum_local_d;
                // std::cout << "selected event: " << event_idx << "\n";
                read_out_event<<<1,threads_single_block>>>(
                        ijevent_to_delete_d,
                        event_idx / nn + displs[rank],
                        neigh_idx + event_idx,
                        event_type_local_d + event_idx);
                gpuErrchk( hipMemcpy(ijevent_to_delete,
                    ijevent_to_delete_d, 3 * sizeof(int), hipMemcpyDeviceToHost) );
                MPI_Bcast(ijevent_to_delete, 3, MPI_INT, source_rank, comm);
            }
            else{
                MPI_Bcast(ijevent_to_delete, 3, MPI_INT, source_rank, comm);
                // memcpy to device
                gpuErrchk( hipMemcpy(ijevent_to_delete_d, ijevent_to_delete, 3 * sizeof(int), hipMemcpyHostToDevice) );
            }

            // execute the event on the SoA

            execute_event<<<1, threads_single_block>>>(site_element, site_charge, ijevent_to_delete_d);
            
            // TODO ijevent_to_delete only on device
            int i_host = ijevent_to_delete[0];
            int j_host = ijevent_to_delete[1];

            int threads = 1024;
            int blocks = ((size_t)count[rank] * (size_t)nn + threads - 1) / threads;
            zero_out_events_split<<<blocks, threads>>>(event_type_local_d, event_prob_local_d, neigh_idx,
                count[rank], displs[rank],
                nn, i_host, j_host);
            event_time = -log(rng.getRandomNumber()) / event_prob_cum_global_h[size-1];
        }
        // hipDeviceSynchronize();
        // MPI_Barrier(comm);
        
        // auto end = std::chrono::high_resolution_clock::now();
        // time_events[j] = std::chrono::duration<double>(end - start).count();

        // if(rank == 0){
        //     std::cout << "Time to execute events: " << time_events[j] << std::endl;
        //     std::cout << "Number of KMC events: " << event_counter << "\n";
        //     std::cout << "Event time: " << event_time << "\n";
        // }
    }
    // if(rank == 0){
    //     std::string base_path =  "final_reordered/";
    //     std::ofstream time_file;
    //     std::string time_file_name = base_path + "time_event_list_" + std::to_string(size) +".txt";
    //     time_file.open(time_file_name);
    //     for(int i = 0; i < measurements; i++){
    //         time_file << time_list[i] << "\n";
    //     }
    //     time_file.close();


    //     std::ofstream time_file_events;
    //     std::string time_file_events_name = base_path + "time_event_selection_" + std::to_string(size) +".txt";
    //     time_file_events.open(time_file_events_name);
    //     for(int i = 0; i < measurements; i++){
    //         time_file_events << time_events[i] << "\n";
    //     }
    //     time_file_events.close();
    // }

    if(rank == 0){
        std::cout << "Number of KMC events: " << event_counter << "\n";
        std::cout << "Event time: " << event_time << "\n";
    }

    // MPI_Barrier(comm);
    // exit(0);


    gpuErrchk( hipFree(event_prob_cum_local_d) );
    gpuErrchk( hipFree(event_type_local_d) );
    gpuErrchk( hipFree(event_prob_local_d) );
    gpuErrchk( hipFreeHost(event_prob_cum_global_h));
    return event_time;    
}


void copytoConstMemory(std::vector<double> E_gen, std::vector<double> E_rec, std::vector<double> E_Vdiff, std::vector<double> E_Odiff)
{
    gpuErrchk( hipMemcpyToSymbol(HIP_SYMBOL(E_gen_const), E_gen.data(), E_gen.size() * sizeof(double)) );
    gpuErrchk( hipMemcpyToSymbol(HIP_SYMBOL(E_rec_const), E_rec.data(), E_rec.size() * sizeof(double)) );
    gpuErrchk( hipMemcpyToSymbol(HIP_SYMBOL(E_Vdiff_const), E_Vdiff.data(), E_Vdiff.size() * sizeof(double)) );
    gpuErrchk( hipMemcpyToSymbol(HIP_SYMBOL(E_Odiff_const), E_Odiff.data(), E_Odiff.size() * sizeof(double)) );
}