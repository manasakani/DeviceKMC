#include "gpu_solvers.h"

//**************************************************************************
// Initializes and populates the neighbor index lists used in the simulation
//**************************************************************************
// NOTE: THE CUTOFF_DISTS IS NOT BEING POPULATED DUE TO OOM AT LARGER DEVICE SIZES

__global__ void populate_cutoff_window(int *cutoff_window, const double *posx, const double *posy, const double *posz,
                                       const double cutoff_radius, const int N, const int counts, const int displ)
{
    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;

    // each thread finds its window
    for (int idx = tid_total; idx < counts; idx += num_threads_total)
    {
        int i = idx + displ;
        bool lower_window_found = false;
        bool upper_window_found = false;

        for (auto j = 0; j < N; j++)
        {
            if (!lower_window_found)
            {
                double dist = site_dist_gpu(posx[i], posy[i], posz[i], posx[j], posy[j], posz[j]);
                bool in_window = (dist < cutoff_radius);

                if (in_window)
                {
                    cutoff_window[idx*2 + 0] = j; // start index of window
                    lower_window_found = true;
                }
            }
        }

        for (auto j = N-1; j >= 0; j--)
        {
            // int j = N - j_idx;
            if (!upper_window_found)
            {
                double dist = site_dist_gpu(posx[i], posy[i], posz[i], posx[j], posy[j], posz[j]);
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
                                       const double nn_dist, const int N, const int nn, const int counts, const int displ)
{
    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;

    // each thread works on a site and writes the indices of its neighbors to its row in neigh_idx
    for (int idx = tid_total; idx < counts; idx += num_threads_total)
    {
        int counter = 0;
        int i = idx + displ;

        for (int j = 0; j < N; j++)
        {
            double dist = site_dist_gpu(posx[i], posy[i], posz[i], posx[j], posy[j], posz[j]);
            bool neighbor = (dist < nn_dist && i != j);
            if (neighbor && counter < nn)
            {
                neigh_idx[(size_t)idx*(size_t)nn + (size_t)counter] = j;
                counter++;
            }
        }
    }
}

__global__ void getsize_cutoff_idx(int *cutoff_size, const ELEMENT *element, const double *posx, const double *posy, const double *posz,
                                   const double cutoff_radius, const int N, const int counts, const int displ)
{
    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;

    // each thread works on a site and writes the indices of its neighbors to its row in neigh_idx
    for (int idx = tid_total; idx < counts; idx += num_threads_total)
    {
        int i = idx + displ;
        cutoff_size[idx] = 0;

        for (auto j = 0; j < N; j++)
        {
            double dist = site_dist_gpu(posx[i], posy[i], posz[i], posx[j], posy[j], posz[j]);
            bool in_cutoff = (dist < cutoff_radius && i != j);
            bool possibly_charged = (element[j] == OXYGEN_DEFECT) || (element[j] == O_EL) || (element[j] == VACANCY) || (element[j] == DEFECT);

            if (in_cutoff && possibly_charged)
            {
                cutoff_size[idx]++;
            }
        }
    }
}

// memory access fault caused by using this
__global__ void populate_cutoff_idx(int *cutoff_idx, const ELEMENT *element, const double *posx, const double *posy, const double *posz,
                                    const double cutoff_radius, const int N, const int max_num_cutoff, const int counts, const int displ)
{
    int tid_total = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads_total = blockDim.x * gridDim.x;

    // each thread works on a site and writes the indices of its neighbors to its row in neigh_idx
    for (int idx = tid_total; idx < counts; idx += num_threads_total)
    {
        int counter = 0;
        int i = idx + displ;

        for (int j = 0; j < N; j++)
        {
            double dist = site_dist_gpu(posx[i], posy[i], posz[i], posx[j], posy[j], posz[j]);
            bool in_cutoff = (dist < cutoff_radius);
            bool possibly_charged = (element[j] == OXYGEN_DEFECT) || (element[j] == O_EL) || (element[j] == VACANCY) || (element[j] == DEFECT);

            // if (in_cutoff && possibly_charged && (counter < max_num_cutoff) && (i != j))
            if (in_cutoff && possibly_charged && (counter < max_num_cutoff) && (i != j))
            {
                long int idx_next = (size_t)idx*(size_t)max_num_cutoff + (size_t)counter;

                cutoff_idx[idx_next] = j; 
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


// void compute_neighbor_lists(Device &device, GPUBuffers &gpubuf, KMCParameters &p)
// {
//     int mpi_size, mpi_rank;
//     MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
//     MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
//     double cutoff_radius = 20;                               // [A] interaction cutoff radius for charge contribution to potential

//     int N = gpubuf.N_;
//     int max_num_neighbors = 52;
//     double nn_dist = 3.5;
//     int rank = gpubuf.rank;
//     int size = gpubuf.size;
//     int counts_this_rank = gpubuf.count_sites[rank];
//     int displs_this_rank = gpubuf.displ_sites[rank];

//     int num_threads = 1024; 
//     int num_blocks = (counts_this_rank - 1) / num_threads + 1;

//     hipMalloc((void**)&gpubuf.neigh_idx, (size_t)counts_this_rank * (size_t)max_num_neighbors *sizeof(ELEMENT));

//     // *** construct site neighbor list: list of indices of the neighbors of each site
//     gpuErrchk( hipMemset(gpubuf.neigh_idx, -1, (size_t)counts_this_rank * (size_t)max_num_neighbors * sizeof(int)) );     // unused neighbor elements are set to -1
//     populate_neighbor_list<<<num_blocks, num_threads>>>(gpubuf.neigh_idx, gpubuf.site_x, gpubuf.site_y, gpubuf.site_z, 
//                                                         nn_dist, N, max_num_neighbors, counts_this_rank, displs_this_rank);
//     gpuErrchk( hipPeekAtLastError() );
//     gpuErrchk( hipDeviceSynchronize() );

//     // // *** construct cutoff window: start-index and end-idx of other sites within the cutoff radius - BUG
//     // populate_cutoff_window<<<num_blocks, num_threads>>>(gpubuf.cutoff_idx, gpubuf.site_x, gpubuf.site_y, gpubuf.site_z, 
//     //                                                     cutoff_radius, N, counts_this_rank, displs_this_rank);
//     // gpuErrchk( hipPeekAtLastError() );
//     // gpuErrchk( hipDeviceSynchronize() );

//     // *** construct cutoff indices: list of indices of other sites within the cutoff radius
//     int *d_num_cutoff_idx;
//     gpuErrchk( hipMalloc((void**)&d_num_cutoff_idx, counts_this_rank * sizeof(int)) );
//     gpuErrchk( hipMemset(d_num_cutoff_idx, 0, counts_this_rank * sizeof(int)) ); // set to zero

//     getsize_cutoff_idx<<<num_blocks, num_threads>>>(d_num_cutoff_idx, gpubuf.site_element, gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
//                                                     cutoff_radius, N, counts_this_rank, displs_this_rank);
//     gpuErrchk( hipPeekAtLastError() );
//     gpuErrchk( hipDeviceSynchronize() );
    
//     int max_num_cutoff = thrust::reduce(d_num_cutoff_idx, d_num_cutoff_idx + counts_this_rank, 0, thrust::maximum<int>());
//     gpubuf.N_cutoff_ = max_num_cutoff;

//     // print max_num_cutoff * N in gigabytes
//     std::cout << "rank : " << rank << " memcon for cutoff_idx: " << (size_t)max_num_cutoff * (size_t)counts_this_rank * sizeof(int) / 1e9 << " GB" << std::endl;
//     fflush(stdout);

//     // print N and max_num_cutoff:
//     std::cout << "rank : " << rank << " N: " << N << " max_num_cutoff: " << max_num_cutoff << std::endl;
//     std::cout << (size_t)N * (size_t)max_num_cutoff << "\n";

//     // int *d_cutoff_idx;
//     gpuErrchk( hipMalloc((void**)&gpubuf.cutoff_idx, (size_t)counts_this_rank * (size_t)max_num_cutoff * sizeof(int)) );
//     gpuErrchk( hipMemset(gpubuf.cutoff_idx, -1, (size_t)counts_this_rank * (size_t)max_num_cutoff * sizeof(int)) );     // unused neighbor elements are set to -1
//     gpuErrchk( hipDeviceSynchronize() );

//     // num_blocks = (N + num_threads - 1) / num_threads;
//     populate_cutoff_idx<<<num_blocks, num_threads>>>(gpubuf.cutoff_idx, gpubuf.site_element, gpubuf.site_x, gpubuf.site_y, gpubuf.site_z, cutoff_radius, N, 
//                                                      max_num_cutoff, counts_this_rank, displs_this_rank);
//     gpuErrchk( hipPeekAtLastError() );
//     gpuErrchk( hipDeviceSynchronize() );

//     hipFree(d_num_cutoff_idx);

//     if (!rank) 
//     {
//         std::cout << "*********************************\n";
//         std::cout << "MPI Rank: " << rank << std::endl;
//         std::string rocm_smi_output = exec1("rocm-smi --showmeminfo vram");
//         std::cout << rocm_smi_output;
//         std::cout << "**********************************\n";
//         fflush(stdout);
//     }
//     MPI_Barrier(MPI_COMM_WORLD);
// }

void compute_neighbor_list(MPI_Comm &event_comm, int *counts, int *displ, Device &device, GPUBuffers &gpubuf, KMCParameters &p)
{
    int size, rank;
    MPI_Comm_size(event_comm, &size);
    MPI_Comm_rank(event_comm, &rank);
    double cutoff_radius = 20;                               // [A] interaction cutoff radius for charge contribution to potential

    int N = gpubuf.N_;
    int max_num_neighbors = 52;
    double nn_dist = 3.5;
    int counts_this_rank = counts[rank];
    int displs_this_rank = displ[rank];

    hipMalloc((void**)&gpubuf.neigh_idx, (size_t)counts_this_rank * (size_t)max_num_neighbors *sizeof(ELEMENT));

    // *** construct site neighbor list: list of indices of the neighbors of each site
    int num_threads = 1024; 
    int num_blocks = (counts_this_rank - 1) / num_threads + 1;
    gpuErrchk( hipMemset(gpubuf.neigh_idx, -1, (size_t)counts_this_rank * (size_t)max_num_neighbors * sizeof(int)) );     // unused neighbor elements are set to -1
    populate_neighbor_list<<<num_blocks, num_threads>>>(gpubuf.neigh_idx, gpubuf.site_x, gpubuf.site_y, gpubuf.site_z, 
                                                        nn_dist, N, max_num_neighbors, counts_this_rank, displs_this_rank);
    gpuErrchk( hipPeekAtLastError() );
    gpuErrchk( hipDeviceSynchronize() );

    if (!rank) 
    {
        std::cout << "*********************************\n";
        std::cout << "MPI Rank: " << rank << std::endl;
        std::string rocm_smi_output = exec1("rocm-smi --showmeminfo vram");
        std::cout << rocm_smi_output;
        std::cout << "**********************************\n";
        fflush(stdout);
    }
}


void compute_cutoff_list(MPI_Comm &pairwise_comm, int *counts, int *displ, Device &device, GPUBuffers &gpubuf, KMCParameters &p)
{
    int size, rank;
    MPI_Comm_size(pairwise_comm, &size);
    MPI_Comm_rank(pairwise_comm, &rank);
    double cutoff_radius = 20;                               // [A] interaction cutoff radius for charge contribution to potential

    int N = gpubuf.N_;
    int max_num_neighbors = 52;
    double nn_dist = 3.5;
    int counts_this_rank = counts[rank];
    int displs_this_rank = displ[rank];

    int num_threads = 1024; 
    int num_blocks = (counts_this_rank - 1) / num_threads + 1;

    // print counts_this_rank and displs_this_rank
    std::cout << "rank : " << rank << " counts_this_rank: " << counts_this_rank << " displs_this_rank: " << displs_this_rank << std::endl;

    // *** construct cutoff indices: list of indices of other sites within the cutoff radius
    int *d_num_cutoff_idx;
    gpuErrchk( hipMalloc((void**)&d_num_cutoff_idx, counts_this_rank * sizeof(int)) );
    gpuErrchk( hipMemset(d_num_cutoff_idx, 0, counts_this_rank * sizeof(int)) ); // set to zero

    getsize_cutoff_idx<<<num_blocks, num_threads>>>(d_num_cutoff_idx, gpubuf.site_element, gpubuf.site_x, gpubuf.site_y, gpubuf.site_z,
                                                    cutoff_radius, N, counts_this_rank, displs_this_rank);
    gpuErrchk( hipPeekAtLastError() );
    gpuErrchk( hipDeviceSynchronize() );

    // std::cout << "cutoff rad: " << cutoff_radius << std::endl;
    // std::cout << "N: " << N << std::endl;

    // // copy back and print gpubuf.site_element
    // ELEMENT *site_element = new ELEMENT[N];
    // hipMemcpy(site_element, gpubuf.site_element, N * sizeof(ELEMENT), hipMemcpyDeviceToHost);
    // for (int i = 0; i < N; i++)
    // {
    //     std::cout << "site_element[" << i << "] = " << site_element[i] << std::endl;
    // }

    // double *site_x = new double[N];
    // hipMemcpy(site_x, gpubuf.site_x, N * sizeof(double), hipMemcpyDeviceToHost);
    // for (int i = 0; i < N; i++)
    // {
    //     std::cout << "site_x[" << i << "] = " << site_x[i] << std::endl;
    // }

    int max_num_cutoff = thrust::reduce(d_num_cutoff_idx, d_num_cutoff_idx + counts_this_rank, 0, thrust::maximum<int>());
    MPI_Allreduce(MPI_IN_PLACE, &max_num_cutoff, 1, MPI_INT, MPI_MAX, pairwise_comm);
    gpubuf.N_cutoff_ = max_num_cutoff;
    std::cout << "max num cutoff " << max_num_cutoff << std::endl; 

    // print max_num_cutoff * N in gigabytes
    std::cout << "rank : " << rank << " memcon for cutoff_idx: " << (size_t)max_num_cutoff * (size_t)counts_this_rank * sizeof(int) / 1e9 << " GB" << std::endl;
    fflush(stdout);

    // print N and max_num_cutoff:
    std::cout << "rank : " << rank << " N: " << N << " max_num_cutoff: " << max_num_cutoff << std::endl;

    // int *d_cutoff_idx;
    gpuErrchk( hipMalloc((void**)&gpubuf.cutoff_idx, (size_t)counts_this_rank * (size_t)max_num_cutoff * sizeof(int)) );
    gpuErrchk( hipMemset(gpubuf.cutoff_idx, -1, (size_t)counts_this_rank * (size_t)max_num_cutoff * sizeof(int)) );     // unused neighbor elements are set to -1
    gpuErrchk( hipDeviceSynchronize() );

    populate_cutoff_idx<<<num_blocks, num_threads>>>(gpubuf.cutoff_idx, gpubuf.site_element, gpubuf.site_x, gpubuf.site_y, gpubuf.site_z, cutoff_radius, N, 
                                                     max_num_cutoff, counts_this_rank, displs_this_rank);
    gpuErrchk( hipPeekAtLastError() );
    gpuErrchk( hipDeviceSynchronize() );

    hipFree(d_num_cutoff_idx);

    if (!rank) 
    {
        std::cout << "*********************************\n";
        std::cout << "Memory consumption after constructing device and computing neighbor lists" << std::endl;
        std::string rocm_smi_output = exec1("rocm-smi --showmeminfo vram");
        std::cout << rocm_smi_output;
        std::cout << "**********************************\n";
        fflush(stdout);
    }
}