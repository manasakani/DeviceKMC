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

void get_gpu_info(char *gpu_string, int dev){
    struct cudaDeviceProp dprop;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(dev);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        // Handle the error or exit the program
        exit(EXIT_FAILURE);
    }

    cudaGetDeviceProperties(&dprop, dev);
    strcpy(gpu_string,dprop.name);
}

void set_gpu(int dev){
 cudaSetDevice(dev);
}

__global__ void build_event_list(const int N, const int nn, const int *neigh_idx, 
                                 const int *layer, const double *lattice, const int pbc, 
                                 const double *T_bg, const double *freq, const double *sigma, const double *k, 
                                 const double *posx, const double *posy, const double *posz,
                                 const double *potential_boundary, const double *potential_charge, const double *temperature,
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

            // Generation
            if (element[i] == DEFECT && element[j] == O_EL)
            {

                double E = 2 * ((potential_boundary[i] + potential_charge[i]) - (potential_boundary[j] + potential_charge[j]));
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
                double E = charge_state * ((potential_boundary[i] + potential_charge[i]) - (potential_boundary[j] + potential_charge[j]) + (charge_state / 2) * self_int_V);
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
                double E = (charge[i] - charge[j]) * ((potential_boundary[i] + potential_charge[i]) - (potential_boundary[j] + potential_charge[j]) + self_int_V);
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

                double E = (charge[i] - charge[j]) * ((potential_boundary[i] + potential_charge[i]) - (potential_boundary[j] + potential_charge[j]) - self_int_V);
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


double execute_kmc_step_gpu(const int N, const int nn, const int *neigh_idx, const int *site_layer,
                            const double *lattice, const int pbc, const double *T_bg, 
                            const double *freq, const double *sigma, const double *k,
                            const double *posx, const double *posy, const double *posz, 
                            const double *site_potential_boundary, const double *site_potential_charge, const double *site_temperature,
                            ELEMENT *site_element, int *site_charge, RandomNumberGenerator &rng, const int *neigh_idx_host){

    // **************************
    // **** Build Event List ****
    // **************************

    double time_event_list = 0.0;


    // the KMC event list arrays only exist in gpu memory
    EVENTTYPE *event_type; 
    double    *event_prob; 
    gpuErrchk( cudaMalloc((void**)&event_type, N * nn * sizeof(EVENTTYPE)) );
    gpuErrchk( cudaMalloc((void**)&event_prob, N * nn * sizeof(double)) );
                
    int num_threads = 512;
    int num_blocks = (N * nn - 1) / num_threads + 1;

    time_event_list = -omp_get_wtime();
    // populate the event_type and event_prob arrays:
    build_event_list<<<num_blocks, num_threads>>>(N, nn, neigh_idx, 
                                                  site_layer, lattice, pbc,
                                                  T_bg, freq, sigma, k,
                                                  posx, posy, posz, 
                                                  site_potential_boundary, site_potential_charge, site_temperature, 
                                                  site_element, site_charge, event_type, event_prob);

    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );
    time_event_list += omp_get_wtime();
    // **************************
    // ** Event Execution Loop **
    // **************************

    // helper variables:
    // NOTE: INITIALIZE THESE ON GPU AND USE MEMCPY DEVICETODEVICE INSTEAD
    int two_host = 2;
    int two_neg_host = -2;
    int zero_host = 0;
    ELEMENT defect_element_host = DEFECT;
    ELEMENT O_defect_element_host = OXYGEN_DEFECT;
    ELEMENT vacancy_element_host = VACANCY;
    ELEMENT O_element_host = O_EL;

    double *event_prob_cum;
    gpuErrchk( cudaMalloc((void**)&event_prob_cum, N * nn * sizeof(double)) );
 
    double freq_host;
    gpuErrchk( cudaMemcpy(&freq_host, freq, 1 * sizeof(double), cudaMemcpyDeviceToHost) );


    double time_incl_sum = 0.0;
    double time_upper_bound = 0.0;
    double time_which_event = 0.0;
    double time_zero_prob = 0.0;

    double event_time = 0.0;
    int counter = 0;
    while (event_time < 1 / freq_host) {
        counter++;  
        // get the cumulative sum of the probabilities
        time_incl_sum -= omp_get_wtime();
        thrust::inclusive_scan(thrust::device, event_prob, event_prob + N * nn, event_prob_cum);
        time_incl_sum += omp_get_wtime();

        time_upper_bound -= omp_get_wtime();
        // select an event
        double Psum_host;
        gpuErrchk( cudaMemcpy(&Psum_host, event_prob_cum + N * nn - 1, sizeof(double), cudaMemcpyDeviceToHost) );
        double number = rng.getRandomNumber() * Psum_host;
        int event_idx = thrust::upper_bound(thrust::device, event_prob_cum, event_prob_cum + N * nn, number) - event_prob_cum;
        // std::cout << "selected event: " << event_idx << "\n";
        time_upper_bound += omp_get_wtime();

        time_which_event -= omp_get_wtime();
        EVENTTYPE sel_event_type = NULL_EVENT;
        gpuErrchk( cudaMemcpy(&sel_event_type, event_type + event_idx, sizeof(EVENTTYPE), cudaMemcpyDeviceToHost) );

        // test output:
        // double sel_event_prob;
        // gpuErrchk( cudaMemcpy(&sel_event_prob, event_prob + event_idx, sizeof(double), cudaMemcpyDeviceToHost) );
        // std::cout << "Selected event index: " << event_idx << " with type "
        //           << sel_event_type << " and probability " << sel_event_prob << std::endl;

        // get attributes of the sites involved:
        int i_host = static_cast<int>(floorf(event_idx / nn));
        int j_host;
        ELEMENT element_i_host, element_j_host;
        int charge_i_host, charge_j_host;

        gpuErrchk( cudaMemcpy(&j_host, neigh_idx + event_idx, sizeof(int), cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(&element_i_host, site_element + i_host, sizeof(ELEMENT), cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(&element_j_host, site_element + j_host, sizeof(ELEMENT), cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(&charge_i_host, site_charge + i_host, sizeof(int), cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaMemcpy(&charge_j_host, site_charge + j_host, sizeof(int), cudaMemcpyDeviceToHost) );

        // Event execution loop
        switch (sel_event_type)
        {
        case VACANCY_GENERATION:
        {
            if (element_i_host != DEFECT || element_j_host != O_EL)
            {
                print("Wrong event type - VACANCY_GENERATION!");
                print(return_element(element_i_host) << " and " << return_element(element_j_host));
            }

            // turn the defect (i) into an oxygen ion
            // turn the oxygen (j) into a charged vacancy
            gpuErrchk( cudaMemcpy(site_element + i_host, &O_defect_element_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_element + j_host, &vacancy_element_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + i_host, &two_neg_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + j_host, &two_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );

            break;
        }
        case VACANCY_RECOMBINATION:
        {
            if (element_i_host != OXYGEN_DEFECT || element_j_host != VACANCY)
            {
                print("Wrong event type - VACANCY_RECOMBINATION!");
                print(return_element(element_i_host) << " and " << return_element(element_j_host));
            }

            // turn the oxygen (i) into a defect
            // turn the vacancy (j) into an oxygen atom
            gpuErrchk( cudaMemcpy(site_element + i_host, &defect_element_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_element + j_host, &O_element_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + i_host, &zero_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + j_host, &zero_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );

            break;
        }
        case VACANCY_DIFFUSION:
        {
            if (element_i_host != VACANCY || element_j_host != O_EL)
            {
                print("Wrong event type - VACANCY_DIFFUSION!");
                print(return_element(element_i_host) << " and " << return_element(element_j_host));
            }

            // swap element and charge
            gpuErrchk( cudaMemcpy(site_element + i_host, &element_j_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_element + j_host, &element_i_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + i_host, &charge_j_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + j_host, &charge_i_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );

            break;
        }
        case ION_DIFFUSION:
        {
            if (element_i_host != OXYGEN_DEFECT || element_j_host != DEFECT)
            {
                print("Wrong event type - ION_DIFFUSION!");
                print(return_element(element_i_host) << " and " << return_element(element_j_host));
            }

            // swap element and charge
            gpuErrchk( cudaMemcpy(site_element + i_host, &element_j_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_element + j_host, &element_i_host, 1 * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + i_host, &charge_j_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );
            gpuErrchk( cudaMemcpy(site_charge + j_host, &charge_i_host, 1 * sizeof(int), cudaMemcpyHostToDevice) );

            break;
        }
        default:
            print("error: unidentified event key found: ");
            print(sel_event_type);
        }
        time_which_event += omp_get_wtime();
        time_zero_prob -= omp_get_wtime();
        int threads = 1024;
        int blocks = (N * nn + threads - 1) / threads;
        zero_out_events<<<blocks, threads>>>(event_type, event_prob,
            neigh_idx, N, nn, i_host, j_host);


        // // other site's events with i or j
        // int i_, j_;
        // for (auto idx = 0; idx < N * nn; ++idx){
        //     i_ = std::floor(idx / nn);
        //     j_ = neigh_idx_host[idx];

        //     if (i_host == i_ || j_host == j_ || i_host == j_ || j_host == i_){
        //         gpuErrchk( cudaMemset(event_type + idx, NULL_EVENT, 1 * sizeof(EVENTTYPE)) );
        //         gpuErrchk( cudaMemset(event_prob + idx, 0.0, 1 * sizeof(double)) );
        //     }
        // }
    
        // remove the conflicting events i an j and their probabilities
        gpuErrchk( cudaMemset(event_type + i_host * nn, NULL_EVENT, (nn - 1) * sizeof(EVENTTYPE)) );
        gpuErrchk( cudaMemset(event_type + j_host * nn, NULL_EVENT, (nn - 1) * sizeof(EVENTTYPE)) );
        gpuErrchk( cudaMemset(event_prob + i_host * nn, 0, (nn - 1) * sizeof(double)) );
        gpuErrchk( cudaMemset(event_prob + j_host * nn, 0, (nn - 1) * sizeof(double)) );
        gpuErrchk( cudaDeviceSynchronize() );
        time_zero_prob += omp_get_wtime();
        event_time = -log(rng.getRandomNumber()) / Psum_host;
    }
    // std::cout << "Time to build event list: " << time_event_list << "\n";
    // std::cout << "Time to calculate inclusive sum: " << time_incl_sum << "\n";
    // std::cout << "Time to calculate upper bound: " << time_upper_bound << "\n";
    // std::cout << "Time to select event: " << time_which_event << "\n";
    // std::cout << "Time to zero out conflicting events: " << time_zero_prob << "\n";


    std::cout << "Number of KMC events: " << counter << "\n";


    gpuErrchk( cudaFree(event_prob_cum) );
    gpuErrchk( cudaFree(event_type) );
    gpuErrchk( cudaFree(event_prob) );

    return event_time;    
}


#ifndef COMPILE_WITH_TESTS
void copytoConstMemory(std::vector<double> E_gen, std::vector<double> E_rec, std::vector<double> E_Vdiff, std::vector<double> E_Odiff)
{   
    gpuErrchk( cudaMemcpyToSymbol(E_gen_const, E_gen.data(), E_gen.size() * sizeof(double)) );
    gpuErrchk( cudaMemcpyToSymbol(E_rec_const, E_rec.data(), E_rec.size() * sizeof(double)) );
    gpuErrchk( cudaMemcpyToSymbol(E_Vdiff_const, E_Vdiff.data(), E_Vdiff.size() * sizeof(double)) );
    gpuErrchk( cudaMemcpyToSymbol(E_Odiff_const, E_Odiff.data(), E_Odiff.size() * sizeof(double)) );
}
#endif

// *** The Graveyard of Code ***

    // dump A to file:
    // dump_csr_matrix_txt(N_interface, A_nnz, A_row_ptr_d, A_col_indices_d, A_data_d, kmc_step_count);

    // # if __CUDA_ARCH__>=200
    // printf("%i \n", tid);
    // #endif  

    // // debug - use floor() for i
    // std::vector<double> new_k(N * N);
    // gpuErrchk( cudaMemcpy(new_k.data(), gpu_k , N * N * sizeof(double), cudaMemcpyDeviceToHost) );
    // std::cout << "copied";
    // std::ofstream fout("new_k.txt");
    // for(int i = 0; i< N*N; i++){
    //     if (new_k[i] != 0){
    //         fout << new_k[i]; 
    //         fout << ' ';
    //     }
    // }
    // exit(1);

    // // debug
    // double *M = (double *)calloc(N, sizeof(double));
    // gpuErrchk( cudaMemcpy(M, gpubuf.site_potential, N * sizeof(double), cudaMemcpyDeviceToHost) );
    // std::cout << "copied\n";
    // std::ofstream fout2("gpu_M.txt");
    // for(int i = 0; i< N; i++){
    //     if (M[i] != 0){
    //         fout2 << M[i]; 
    //         fout2 << ' ';
    //     }
    // }


    // // ************************************************************
    // // 1. Convert D matrix to sparse COO format using the COOElement struct
    // int numNonZero = 0;

    // COOElement* d_cooData;
    // cudaMalloc((void**)&d_cooData, N_interface * N_interface * sizeof(COOElement));

    // int* d_numNonZero;
    // cudaMalloc((void**)&d_numNonZero, sizeof(int));
    // cudaMemcpy(d_numNonZero, &numNonZero, sizeof(int), cudaMemcpyHostToDevice);

    // int numThreadsPerBlock = 256;
    // int numBlocks = (N_interface + numThreadsPerBlock - 1) / numThreadsPerBlock;
    // extractCOOData<<<numBlocks, numThreadsPerBlock>>>(gpu_D, N_interface, d_cooData, d_numNonZero);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    // cudaMemcpy(&numNonZero, d_numNonZero, sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << "number of nonzeros in COO rep: " << numNonZero << "\n";

    // // ************************************************************
    // // 2. Sort and filter the coordinates
    // thrust::device_vector<COOElement> d_cooDataVec(d_cooData, d_cooData + numNonZero);
    
    // // Sort the COO data based on row-major order 
    // thrust::sort(thrust::device, d_cooDataVec.begin(), d_cooDataVec.end(), [] __device__ (const COOElement& a, const COOElement& b) {
    //     return a.row < b.row || (a.row == b.row && a.col < b.col);
    // });

    // // Remove duplicate entries 
    // auto newEnd = thrust::unique(thrust::device, d_cooDataVec.begin(), d_cooDataVec.end(), [] __device__ (const COOElement& a, const COOElement& b) {
    //     return a.row == b.row && a.col == b.col;
    // });
    // numNonZero = newEnd - d_cooDataVec.begin();

    // // ************************************************************
    // // 3. Extract CSR data from COO representation and nnz
    // int* d_csrRowPtr;
    // int* d_csrColIndices;
    // double* d_csrValues;

    // cudaMalloc((void**)&d_csrRowPtr, (N_interface + 1) * sizeof(int));   // +1 for the row pointers
    // cudaMalloc((void**)&d_csrColIndices, numNonZero * sizeof(int));      // Allocate space for the maximum number of non-zero elements
    // cudaMalloc((void**)&d_csrValues, numNonZero * sizeof(double));       // Allocate space for the maximum number of non-zero elements
    // cudaMemset(d_csrRowPtr, 0, (N_interface + 1) * sizeof(int));
    // gpuErrchk( cudaDeviceSynchronize() );

    // extractCOOToCSR<<<numBlocks, numThreadsPerBlock>>>(thrust::raw_pointer_cast(d_cooDataVec.data()), numNonZero, N_interface, d_csrRowPtr, d_csrColIndices, d_csrValues);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    // std::cout << "number of nonzeros in CSR rep: " << numNonZero << "\n";




// //debug
    // double *cpu_x = new double[(N_atom + 2) * (N_atom + 2)];
    // cudaMemcpy(cpu_x, gpu_x, sizeof(double) * (N_atom + 2) * (N_atom + 2), cudaMemcpyDeviceToHost);
    // std::cout << "printing X_gpu\n";
    // std::ofstream fout2("X_gpu_noT.txt");
    // int row, col;
    // for (row = 0; row < (N_atom + 2); row++) {
    // for (col = 0; col < (N_atom + 2); col++) {
    //     fout2 << cpu_x[row * (N_atom + 2) + col] << ' ';
    // }
    // fout2 << '\n';
    // }
    // fout2.close(); 
    // exit(1);
    // //debug end


    // // dense to sparse conversion:
    // cusparseHandle_t cusparseHandle;
    // cusparseCreate(&cusparseHandle);
    // cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_DEVICE);
    
    // gpuErrchk( cudaMemcpy2D(gpu_A, (N_atom + 1) * sizeof(double), gpu_x, (N_atom + 2) * sizeof(double), (N_atom + 1) * sizeof(double), (N_atom + 1), cudaMemcpyDeviceToDevice) );
    // int numRows = N_atom + 1;
    // int numCols = N_atom + 1;
    // double *csrValues;
    // int *csrRowPtr;
    // int *csrColInd;
    // int nnz = 0;
    // cudaMalloc((void**) &csrRowPtr, (N_atom + 1 + 1) * sizeof(int));
    // denseToCSR(cusparseHandle, gpu_A, numRows, numCols, &csrValues, &csrRowPtr, &csrColInd, &nnz);
    // std::cout << "dense nnz: " << nnz << "\n";
