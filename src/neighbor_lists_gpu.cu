#include "gpu_solvers.h"

//**************************************************************************
// Initializes and populates the neighbor index lists used in the simulation
//**************************************************************************
// NOTE: THE CUTOFF_DISTS IS NOT BEING POPULATED DUE TO OOM AT LARGER DEVICE SIZES

__global__ void populate_cutoff_window(int *cutoff_window, const double *posx, const double *posy, const double *posz,
                                       const double *lattice, const bool pbc, const double cutoff_radius, const int N)
{
    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;

    // each thread finds its window
    for (auto i = tid_total; i < N; i += num_threads_total)
    {
        bool lower_window_found = false;
        bool upper_window_found = false;

        for (auto j = 0; j < N; j++)
        {
            if (!lower_window_found)
            {
                double dist = site_dist_gpu(posx[i], posy[i], posz[i], posx[j], posy[j], posz[j], lattice[0], lattice[1], lattice[2], pbc);
                bool in_window = (dist < cutoff_radius);

                if (in_window)
                {
                    cutoff_window[i*2 + 0] = j; // start index of window
                    lower_window_found = true;
                }
            }
        }

        for (auto j = N-1; j >= 0; j--)
        {
            // int j = N - j_idx;
            if (!upper_window_found)
            {
                double dist = site_dist_gpu(posx[i], posy[i], posz[i], posx[j], posy[j], posz[j], lattice[0], lattice[1], lattice[2], pbc);
                bool in_window = (dist < cutoff_radius);

                if (in_window)
                {
                    cutoff_window[i*2 + 1] = j; // end index of window
                    upper_window_found = true;
                }
            }
        }

    }
}

__global__ void populate_neighbor_list(int *neigh_idx, const double *posx, const double *posy, const double *posz,
                                       const double *lattice, const bool pbc, const double nn_dist, const int N, const int nn)
{
    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;

    // each thread works on a site and writes the indices of its neighbors to its row in neigh_idx
    for (auto i = tid_total; i < N; i += num_threads_total)
    {
        int counter = 0;
        for (auto j = 0; j < N; j++)
        {
            double dist = site_dist_gpu(posx[i], posy[i], posz[i], posx[j], posy[j], posz[j], lattice[0], lattice[1], lattice[2], pbc);
            bool neighbor = (dist < nn_dist && i != j);
            if (neighbor && counter < nn)
            {
                neigh_idx[i*nn + counter] = j;
                counter++;
            }
        }
    }
}

__global__ void getsize_cutoff_idx(int *cutoff_size, const ELEMENT *element, const double *posx, const double *posy, const double *posz,
                                   const double *lattice, const bool pbc, const double cutoff_radius, const int N)
{
    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;

    // each thread works on a site and writes the indices of its neighbors to its row in neigh_idx
    for (auto i = tid_total; i < N; i += num_threads_total)
    {
        cutoff_size[i] = 0;
        for (auto j = 0; j < N; j++)
        {
            double dist = site_dist_gpu(posx[i], posy[i], posz[i], posx[j], posy[j], posz[j], lattice[0], lattice[1], lattice[2], pbc);
            bool in_cutoff = (dist < cutoff_radius && i != j);
            bool possibly_charged = (element[j] == OXYGEN_DEFECT) || (element[j] == O_EL) || (element[j] == VACANCY) || (element[j] == DEFECT);

            if (in_cutoff && possibly_charged)
            {
                cutoff_size[i]++;
            }
        }
    }
}

// memory access fault caused by using this
__global__ void populate_cutoff_idx(int *cutoff_idx, const ELEMENT *element, const double *posx, const double *posy, const double *posz,
                                    const double cutoff_radius, const int N, const int max_num_cutoff)
{
    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;

    // each thread works on a site and writes the indices of its neighbors to its row in neigh_idx
    int counter = 0;
    for (int i = tid_total; i < N; i += num_threads_total)
    {
        for (int j = 0; j < N; j++)
        {
            double dist = site_dist_gpu(posx[i], posy[i], posz[i], posx[j], posy[j], posz[j]);
            bool in_cutoff = (dist < cutoff_radius);
            bool possibly_charged = (element[j] == OXYGEN_DEFECT) || (element[j] == O_EL) || (element[j] == VACANCY) || (element[j] == DEFECT);

            // if (in_cutoff && possibly_charged && (counter < max_num_cutoff) && (i != j))
            if (in_cutoff && possibly_charged && (counter < max_num_cutoff) && (i != j))
            {
                long int idx_next = (size_t)i*(size_t)max_num_cutoff + (size_t)counter;
                cutoff_idx[idx_next] = j; // 10, 100, 1000 works with if ( counter == 0 )
                // cutoff_idx[i*max_num_cutoff + counter] = j; // 10, 100, 1000 works with if ( counter == 0 )
                // cutoff_idx[i*(max_num_cutoff-1000) + counter] = j; // 10, 100, 1000 works with if ( counter == 0 )
                counter++;
            }
        }
        counter = 0;
    }
}

__global__ void populate_cutoff_dists(double *cutoff_dists, const ELEMENT *element, const double *posx, const double *posy, const double *posz,
                                      const double *lattice, const bool pbc, const double cutoff_radius, const int N, const int max_num_cutoff)
{
    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;

    // each thread works on a site
    for (auto i = tid_total; i < N; i += num_threads_total)
    {
        int counter = 0;
        for (auto j = 0; j < N; j++)
        {
            double dist = site_dist_gpu(posx[i], posy[i], posz[i], posx[j], posy[j], posz[j], lattice[0], lattice[1], lattice[2], pbc);
            bool in_cutoff = (dist < cutoff_radius && i != j);
            bool possibly_charged = (element[j] == OXYGEN_DEFECT) || (element[j] == O_EL) || (element[j] == VACANCY) || (element[j] == DEFECT);

            if (in_cutoff && possibly_charged && (counter < max_num_cutoff))
            {
                cutoff_dists[i*max_num_cutoff + counter] = dist;
                counter++;
            }
        }
    }
}


// void construct_site_neighbor_list_gpu(int *neigh_idx, int *cutoff_window, std::vector<int> &cutoff_idx, std::vector<double> &cutoff_dists,
//                                       const ELEMENT *site_element, const double *posx, const double *posy, const double *posz, 
//                                       const double *lattice, const bool pbc, double nn_dist, double cutoff_radius, int N, int max_num_neighbors)
// {

//     double *d_posx, *d_posy, *d_posz, *d_lattice;
//     int *d_neigh_idx, *d_cutoff_window;
//     ELEMENT *d_element;

//     // Allocate and copy input arrays 
//     gpuErrchk( hipMalloc((void**)&d_element, N * sizeof(ELEMENT)) );
//     gpuErrchk( hipMalloc((void**)&d_posx, N * sizeof(double)) );
//     gpuErrchk( hipMalloc((void**)&d_posy, N * sizeof(double)) );
//     gpuErrchk( hipMalloc((void**)&d_posz, N * sizeof(double)) );
//     gpuErrchk( hipMalloc((void**)&d_lattice, 3 * sizeof(double)) ); 
//     gpuErrchk( hipMalloc((void**)&d_neigh_idx, N * max_num_neighbors * sizeof(int)) );
//     gpuErrchk( hipMalloc((void**)&d_cutoff_window, N * 2 * sizeof(int)) );
//     gpuErrchk( hipMemcpy(d_element, site_element, N * sizeof(ELEMENT), hipMemcpyHostToDevice) );
//     gpuErrchk( hipMemcpy(d_neigh_idx, neigh_idx, N * max_num_neighbors * sizeof(int), hipMemcpyHostToDevice) );
//     gpuErrchk( hipMemcpy(d_cutoff_window, cutoff_window, N * 2 * sizeof(int), hipMemcpyHostToDevice) );
//     gpuErrchk( hipMemcpy(d_posx, posx, N * sizeof(double), hipMemcpyHostToDevice) );
//     gpuErrchk( hipMemcpy(d_posy, posy, N * sizeof(double), hipMemcpyHostToDevice) );
//     gpuErrchk( hipMemcpy(d_posz, posz, N * sizeof(double), hipMemcpyHostToDevice) );
//     gpuErrchk( hipMemcpy(d_lattice, lattice, 3 * sizeof(double), hipMemcpyHostToDevice) ); 

//     int num_threads = 512;

//     // *** construct site neighbor list: list of indices of the neighbors of each site
//     int num_blocks = (N * max_num_neighbors - 1) / num_threads + 1;
//     hipLaunchKernelGGL(populate_neighbor_list, num_blocks, num_threads, 0, 0, d_neigh_idx, d_posx, d_posy, d_posz, d_lattice, pbc, nn_dist, N, max_num_neighbors);
//     gpuErrchk( hipPeekAtLastError() );
//     gpuErrchk( hipDeviceSynchronize() );

//     // *** construct cutoff window: start-index and end-idx of other sites within the cutoff radius
//     num_blocks = (N * 2 - 1) / num_threads + 1;
//     hipLaunchKernelGGL(populate_cutoff_window, num_blocks, num_threads, 0, 0, d_cutoff_window, d_posx, d_posy, d_posz, d_lattice, pbc, cutoff_radius, N);
//     gpuErrchk( hipPeekAtLastError() );
//     gpuErrchk( hipDeviceSynchronize() );

//     // *** construct cutoff indices: list of indices of other sites within the cutoff radius
//     int *d_num_cutoff_idx;
//     hipMalloc((void**)&d_num_cutoff_idx, N * sizeof(int));

//     num_blocks = (N - 1) / num_threads + 1;
//     hipLaunchKernelGGL(getsize_cutoff_idx, num_blocks, num_threads, 0, 0, d_num_cutoff_idx, d_element, d_posx, d_posy, d_posz, d_lattice, pbc, cutoff_radius, N);
//     gpuErrchk( hipPeekAtLastError() );
//     gpuErrchk( hipDeviceSynchronize() );

//     //
//     int* h_num_cutoff_idx = new int[N];
//     hipMemcpy(h_num_cutoff_idx, d_num_cutoff_idx, N * sizeof(int), hipMemcpyDeviceToHost);
//     for (int i = 0; i < 100; ++i)
//     {
//         std::cout << "Element " << i << ": " << h_num_cutoff_idx[i] << std::endl;
//     }
//     delete[] h_num_cutoff_idx;
//     //

//     // thrust::device_vector<int> d_vec_num_cutoff_idx(d_num_cutoff_idx, d_num_cutoff_idx + N);
//     // int max_num_cutoff = thrust::reduce(thrust::device, d_vec_num_cutoff_idx.begin(), d_vec_num_cutoff_idx.end(), 0, thrust::maximum<int>());

//     int max_num_cutoff = thrust::reduce(d_num_cutoff_idx, d_num_cutoff_idx + N, 0, thrust::maximum<int>());
//     // std::cout << "Max number cutoff: " << max_num_cutoff << std::endl;

//     int *d_cutoff_idx;
//     gpuErrchk( hipMalloc((void**)&d_cutoff_idx, N * max_num_cutoff * sizeof(int)) );
//     gpuErrchk( hipMemset(d_cutoff_idx, -1, N * max_num_cutoff * sizeof(int)) );     // unused neighbor elements are set to -1
//     gpuErrchk( hipDeviceSynchronize() );

//     num_blocks = (N * max_num_cutoff - 1) / num_threads + 1;
//     hipLaunchKernelGGL(populate_cutoff_idx, num_blocks, num_threads, 0, 0, d_cutoff_idx, d_element, d_posx, d_posy, d_posz, d_lattice, pbc, cutoff_radius, N, max_num_cutoff);
//     gpuErrchk( hipPeekAtLastError() );
//     gpuErrchk( hipDeviceSynchronize() );

//     std::cout << "max_num_cutoff: " << max_num_cutoff << "\n";

//     // CUTOFF DISTANCES ARE NOT POPULATED
//     // *** construct cutoff distances: list of distances of other sites within the cutoff radius
//     // double *d_cutoff_dists;
//     // gpuErrchk( hipMalloc((void**)&d_cutoff_dists, N * max_num_cutoff * sizeof(double)) );
//     // gpuErrchk( hipMemset(d_cutoff_dists, 0, N * max_num_cutoff * sizeof(double)) );     // unused neighbor elements are set to 0
//     // gpuErrchk( hipDeviceSynchronize() );

//     // num_blocks = (N * max_num_cutoff - 1) / num_threads + 1;
//     // hipLaunchKernelGGL(populate_cutoff_dists, num_blocks, num_threads, 0, 0, d_cutoff_dists, d_element, d_posx, d_posy, d_posz, d_lattice, pbc, cutoff_radius, N, max_num_cutoff);
//     // gpuErrchk( hipPeekAtLastError() );
//     // gpuErrchk( hipDeviceSynchronize() );

//     // get the neighbor lists back to host
//     hipMemcpy(neigh_idx, d_neigh_idx, N * max_num_neighbors * sizeof(int), hipMemcpyDeviceToHost);
//     hipMemcpy(cutoff_window, d_cutoff_window, N * 2 * sizeof(int), hipMemcpyDeviceToHost);
//     cutoff_idx.resize(N * max_num_cutoff, 0);
//     hipMemcpy(cutoff_idx.data(), d_cutoff_idx, N * max_num_cutoff * sizeof(int), hipMemcpyDeviceToHost);
//     cutoff_dists.resize(N * max_num_cutoff, 0);
//     // hipMemcpy(cutoff_dists.data(), d_cutoff_dists, N * max_num_cutoff * sizeof(double), hipMemcpyDeviceToHost);

//     hipFree(d_posx);
//     hipFree(d_posy);
//     hipFree(d_posz);
//     hipFree(d_lattice);
//     hipFree(d_neigh_idx);
//     hipFree(d_cutoff_window);
//     hipFree(d_num_cutoff_idx);
//     hipFree(d_cutoff_idx);
//     // hipFree(d_cutoff_dists);
// }


std::string exec1(const char* cmd) {
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

void construct_site_neighbor_list_gpu(int *neigh_idx, int *cutoff_window, std::vector<int> &cutoff_idx,
                                      const ELEMENT *site_element, const double *posx, const double *posy, const double *posz, 
                                      const double *lattice, const bool pbc, double nn_dist, double cutoff_radius, int N, int max_num_neighbors)
{

    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (!mpi_rank) 
    {
        std::cout << "**************start of construct_site_neighbor_list_gpu********************\n";
        std::string rocm_smi_output = exec1("rocm-smi --showmeminfo vram");
        std::cout << rocm_smi_output;
        std::cout << "**********************************\n";
    }

    double *d_posx, *d_posy, *d_posz, *d_lattice;
    int *d_neigh_idx, *d_cutoff_window;
    ELEMENT *d_element;

    // Allocate and copy input arrays 
    gpuErrchk( hipMalloc((void**)&d_element, N * sizeof(ELEMENT)) );
    gpuErrchk( hipMalloc((void**)&d_posx, N * sizeof(double)) );
    gpuErrchk( hipMalloc((void**)&d_posy, N * sizeof(double)) );
    gpuErrchk( hipMalloc((void**)&d_posz, N * sizeof(double)) );
    gpuErrchk( hipMalloc((void**)&d_lattice, 3 * sizeof(double)) ); 
    gpuErrchk( hipMalloc((void**)&d_neigh_idx, (size_t)N * (size_t)max_num_neighbors * sizeof(int)) );
    gpuErrchk( hipMalloc((void**)&d_cutoff_window, N * 2 * sizeof(int)) );
    gpuErrchk( hipMemcpy(d_element, site_element, N * sizeof(ELEMENT), hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(d_neigh_idx, neigh_idx, (size_t)N * (size_t)max_num_neighbors * sizeof(int), hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(d_cutoff_window, cutoff_window, N * 2 * sizeof(int), hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(d_posx, posx, N * sizeof(double), hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(d_posy, posy, N * sizeof(double), hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(d_posz, posz, N * sizeof(double), hipMemcpyHostToDevice) );
    gpuErrchk( hipMemcpy(d_lattice, lattice, 3 * sizeof(double), hipMemcpyHostToDevice) ); 

    int num_threads = 1024; //change to 2014
    int num_blocks = (N - 1) / num_threads + 1;

    // *** construct site neighbor list: list of indices of the neighbors of each site
    // int num_blocks = (N * max_num_neighbors - 1) / num_threads + 1;
    populate_neighbor_list<<<num_blocks, num_threads>>>(d_neigh_idx, d_posx, d_posy, d_posz, d_lattice, pbc, nn_dist, N, max_num_neighbors);
    gpuErrchk( hipPeekAtLastError() );
    gpuErrchk( hipDeviceSynchronize() );

    // *** construct cutoff window: start-index and end-idx of other sites within the cutoff radius
    // num_blocks = (N * 2 - 1) / num_threads + 1;
    populate_cutoff_window<<<num_blocks, num_threads>>>(d_cutoff_window, d_posx, d_posy, d_posz, d_lattice, pbc, cutoff_radius, N);
    gpuErrchk( hipPeekAtLastError() );
    gpuErrchk( hipDeviceSynchronize() );

    // *** construct cutoff indices: list of indices of other sites within the cutoff radius
    int *d_num_cutoff_idx;
    gpuErrchk( hipMalloc((void**)&d_num_cutoff_idx, N * sizeof(int)) );
    gpuErrchk( hipMemset(d_num_cutoff_idx, 0, N * sizeof(int)) ); // set to zero

    // num_blocks = (N - 1) / num_threads + 1;
    getsize_cutoff_idx<<<num_blocks, num_threads>>>(d_num_cutoff_idx, d_element, d_posx, d_posy, d_posz, d_lattice, pbc, cutoff_radius, N);
    gpuErrchk( hipPeekAtLastError() );
    gpuErrchk( hipDeviceSynchronize() );
    int max_num_cutoff = thrust::reduce(d_num_cutoff_idx, d_num_cutoff_idx + N, 0, thrust::maximum<int>());
    std::cout << "Max number cutoff: " << max_num_cutoff << std::endl;

    if (!mpi_rank) 
    {
        std::cout << "**************before ********************\n";
        std::string rocm_smi_output = exec1("rocm-smi --showmeminfo vram");
        std::cout << rocm_smi_output;
        std::cout << "**********************************\n";
    }

    // print max_num_cutoff * N in gigabytes
    std::cout << "memcon for cutoff_idx: " << (size_t)max_num_cutoff * (size_t)N * sizeof(int) / 1e9 << " GB" << std::endl;
    fflush(stdout);

    // print N and max_num_cutoff:
    std::cout << "N: " << N << " max_num_cutoff: " << max_num_cutoff << std::endl;
    std::cout << (size_t)N * (size_t)max_num_cutoff << "\n";

    int *d_cutoff_idx;
    gpuErrchk( hipMalloc((void**)&d_cutoff_idx, (size_t)N * (size_t)max_num_cutoff * sizeof(int)) );
    gpuErrchk( hipMemset(d_cutoff_idx, -1, (size_t)N * (size_t)max_num_cutoff * sizeof(int)) );     // unused neighbor elements are set to -1
    gpuErrchk( hipDeviceSynchronize() );

    // num_blocks = (N + num_threads - 1) / num_threads;
    populate_cutoff_idx<<<num_blocks, num_threads>>>(d_cutoff_idx, d_element, d_posx, d_posy, d_posz, cutoff_radius, N, max_num_cutoff);
    gpuErrchk( hipPeekAtLastError() );
    gpuErrchk( hipDeviceSynchronize() );

    if (!mpi_rank) 
    {
        std::cout << "*************after********************\n";
        std::string rocm_smi_output = exec1("rocm-smi --showmeminfo vram");
        std::cout << rocm_smi_output;
        std::cout << "**********************************\n";
    }

    // *** construct cutoff distances: list of distances of other sites within the cutoff radius
    // double *d_cutoff_dists;
    // gpuErrchk( hipMalloc((void**)&d_cutoff_dists, N * max_num_cutoff * sizeof(double)) );
    // gpuErrchk( hipMemset(d_cutoff_dists, 0, N * max_num_cutoff * sizeof(double)) );     // unused neighbor elements are set to 0
    // gpuErrchk( hipDeviceSynchronize() );

    // num_blocks = (N * max_num_cutoff - 1) / num_threads + 1;
    // populate_cutoff_dists<<<num_blocks, num_threads>>>(d_cutoff_dists, d_element, d_posx, d_posy, d_posz, d_lattice, pbc, cutoff_radius, N, max_num_cutoff);
    // gpuErrchk( hipPeekAtLastError() );
    // gpuErrchk( hipDeviceSynchronize() );

    // get the neighbor lists back to host
    hipMemcpy(neigh_idx, d_neigh_idx, (size_t)N * (size_t)max_num_neighbors * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(cutoff_window, d_cutoff_window, N * 2 * sizeof(int), hipMemcpyDeviceToHost);
    cutoff_idx.resize((size_t)N * (size_t)max_num_cutoff, 0);
    hipMemcpy(cutoff_idx.data(), d_cutoff_idx, (size_t)N * (size_t)max_num_cutoff * sizeof(int), hipMemcpyDeviceToHost);

    // mpi rank 0 pauses for a second so that smi will update
    // if (!mpi_rank) 
    // {
    //     std::this_thread::sleep_for(std::chrono::seconds(1));
    // }

    // if (!mpi_rank) 
    // {
    //     std::cout << "**************before cuda free********************\n";
    //     std::string rocm_smi_output = exec1("rocm-smi --showmeminfo vram");
    //     std::cout << rocm_smi_output;
    //     std::cout << "**********************************\n";
    // }

    hipFree(d_element);
    hipFree(d_posx);
    hipFree(d_posy);
    hipFree(d_posz);
    hipFree(d_lattice);
    hipFree(d_neigh_idx);
    hipFree(d_cutoff_window);
    hipFree(d_num_cutoff_idx);
    hipFree(d_cutoff_idx);
    
    gpuErrchk( hipPeekAtLastError() );
    //synchronize gpu
    gpuErrchk( hipDeviceSynchronize() );
    // hipFree(d_cutoff_dists);

    // if (!mpi_rank) 
    // {
    //     std::cout << "**************after cuda free********************\n";
    //     std::string rocm_smi_output = exec1("rocm-smi --showmeminfo vram");
    //     std::cout << rocm_smi_output;
    //     std::cout << "**********************************\n";
    // }

    MPI_Barrier(MPI_COMM_WORLD);
    // exit(1);
}

        // std::cout << "--------------------------\n";
        // std::string rocm_smi_output = exec1("rocm-smi ---gpureset");
        // std::cout << rocm_smi_output;
        // std::cout << "--------------------------\n";
        // MPI_Barrier(MPI_COMM_WORLD);