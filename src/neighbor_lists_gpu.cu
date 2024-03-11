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

__global__ void populate_cutoff_idx(int *cutoff_idx, const ELEMENT *element, const double *posx, const double *posy, const double *posz,
                                    const double *lattice, const bool pbc, const double cutoff_radius, const int N, const int max_num_cutoff)
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
            bool in_cutoff = (dist < cutoff_radius && i != j);
            bool possibly_charged = (element[j] == OXYGEN_DEFECT) || (element[j] == O_EL) || (element[j] == VACANCY) || (element[j] == DEFECT);

            if (in_cutoff && possibly_charged && (counter < max_num_cutoff))
            {
                cutoff_idx[i*max_num_cutoff + counter] = j;
                counter++;
            }
        }
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


void construct_site_neighbor_list_gpu(int *neigh_idx, int *cutoff_window, std::vector<int> &cutoff_idx, std::vector<double> &cutoff_dists,
                                      const ELEMENT *site_element, const double *posx, const double *posy, const double *posz, 
                                      const double *lattice, const bool pbc, double nn_dist, double cutoff_radius, int N, int max_num_neighbors)
{

    double *d_posx, *d_posy, *d_posz, *d_lattice;
    int *d_neigh_idx, *d_cutoff_window;
    ELEMENT *d_element;

    // Allocate and copy input arrays 
    gpuErrchk( cudaMalloc((void**)&d_element, N * sizeof(ELEMENT)) );
    gpuErrchk( cudaMalloc((void**)&d_posx, N * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_posy, N * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_posz, N * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_lattice, 3 * sizeof(double)) ); 
    gpuErrchk( cudaMalloc((void**)&d_neigh_idx, N * max_num_neighbors * sizeof(int)) );
    gpuErrchk( cudaMalloc((void**)&d_cutoff_window, N * 2 * sizeof(int)) );
    gpuErrchk( cudaMemcpy(d_element, site_element, N * sizeof(ELEMENT), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_neigh_idx, neigh_idx, N * max_num_neighbors * sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_cutoff_window, cutoff_window, N * 2 * sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_posx, posx, N * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_posy, posy, N * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_posz, posz, N * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_lattice, lattice, 3 * sizeof(double), cudaMemcpyHostToDevice) ); 

    int num_threads = 512;

    // *** construct site neighbor list: list of indices of the neighbors of each site
    int num_blocks = (N * max_num_neighbors - 1) / num_threads + 1;
    populate_neighbor_list<<<num_blocks, num_threads>>>(d_neigh_idx, d_posx, d_posy, d_posz, d_lattice, pbc, nn_dist, N, max_num_neighbors);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // *** construct cutoff window: start-index and end-idx of other sites within the cutoff radius
    num_blocks = (N * 2 - 1) / num_threads + 1;
    populate_cutoff_window<<<num_blocks, num_threads>>>(d_cutoff_window, d_posx, d_posy, d_posz, d_lattice, pbc, cutoff_radius, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // *** construct cutoff indices: list of indices of other sites within the cutoff radius
    int *d_num_cutoff_idx;
    cudaMalloc((void**)&d_num_cutoff_idx, N * sizeof(int));

    num_blocks = (N - 1) / num_threads + 1;
    getsize_cutoff_idx<<<num_blocks, num_threads>>>(d_num_cutoff_idx, d_element, d_posx, d_posy, d_posz, d_lattice, pbc, cutoff_radius, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    thrust::device_vector<int> d_vec_num_cutoff_idx(d_num_cutoff_idx, d_num_cutoff_idx + N);
    int max_num_cutoff = thrust::reduce(thrust::device, d_vec_num_cutoff_idx.begin(), d_vec_num_cutoff_idx.end(), 0, thrust::maximum<int>());

    int *d_cutoff_idx;
    gpuErrchk( cudaMalloc((void**)&d_cutoff_idx, N * max_num_cutoff * sizeof(int)) );
    gpuErrchk( cudaMemset(d_cutoff_idx, -1, N * max_num_cutoff * sizeof(int)) );     // unused neighbor elements are set to -1
    gpuErrchk( cudaDeviceSynchronize() );

    num_blocks = (N * max_num_cutoff - 1) / num_threads + 1;
    populate_cutoff_idx<<<num_blocks, num_threads>>>(d_cutoff_idx, d_element, d_posx, d_posy, d_posz, d_lattice, pbc, cutoff_radius, N, max_num_cutoff);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // CUTOFF DISTANCES ARE NOT POPULATED
    // *** construct cutoff distances: list of distances of other sites within the cutoff radius
    // double *d_cutoff_dists;
    // gpuErrchk( cudaMalloc((void**)&d_cutoff_dists, N * max_num_cutoff * sizeof(double)) );
    // gpuErrchk( cudaMemset(d_cutoff_dists, 0, N * max_num_cutoff * sizeof(double)) );     // unused neighbor elements are set to 0
    // gpuErrchk( cudaDeviceSynchronize() );

    // num_blocks = (N * max_num_cutoff - 1) / num_threads + 1;
    // populate_cutoff_dists<<<num_blocks, num_threads>>>(d_cutoff_dists, d_element, d_posx, d_posy, d_posz, d_lattice, pbc, cutoff_radius, N, max_num_cutoff);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    // get the neighbor lists back to host
    cudaMemcpy(neigh_idx, d_neigh_idx, N * max_num_neighbors * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cutoff_window, d_cutoff_window, N * 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cutoff_idx.resize(N * max_num_cutoff, 0);
    cudaMemcpy(cutoff_idx.data(), d_cutoff_idx, N * max_num_cutoff * sizeof(int), cudaMemcpyDeviceToHost);
    cutoff_dists.resize(N * max_num_cutoff, 0);
    // cudaMemcpy(cutoff_dists.data(), d_cutoff_dists, N * max_num_cutoff * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_posx);
    cudaFree(d_posy);
    cudaFree(d_posz);
    cudaFree(d_lattice);
    cudaFree(d_neigh_idx);
    cudaFree(d_cutoff_window);
    cudaFree(d_num_cutoff_idx);
    cudaFree(d_cutoff_idx);
    // cudaFree(d_cutoff_dists);
}
