#include "dist_objects.h"

Distributed_vector::Distributed_vector(
    int matrix_size,
    int *counts,
    int *displacements,
    int number_of_neighbours,
    int *neighbours,
    MPI_Comm comm
){
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    this->matrix_size = matrix_size;
    this->comm = comm;
    this->counts = new int[size];
    this->displacements = new int[size];
    for(int i = 0; i < size; i++){
        this->counts[i] = counts[i];
        this->displacements[i] = displacements[i];
    }
    rows_this_rank = counts[rank];
    this->number_of_neighbours = number_of_neighbours;
    this->neighbours = new int[number_of_neighbours];
    for(int k = 0; k < number_of_neighbours; k++){
        this->neighbours[k] = neighbours[k];
    }
    vec_h = new double*[number_of_neighbours];
    vec_d = new double*[number_of_neighbours];
    descriptors = new hipsparseDnVecDescr_t[number_of_neighbours];
    for(int k = 0; k < number_of_neighbours; k++){
        int neighbour_idx = neighbours[k];
        cudaErrchk(hipHostMalloc(&vec_h[k], counts[neighbour_idx]*sizeof(double)));
        for(int i = 0; i < counts[neighbour_idx]; i++){
            vec_h[k][i] = 0.0;
        }
        cudaErrchk(hipMalloc(&vec_d[k], counts[neighbour_idx]*sizeof(double)));
        cudaErrchk(hipMemset(vec_d[k], 0, counts[neighbour_idx]*sizeof(double)));
        cusparseErrchk(hipsparseCreateDnVec(&descriptors[k], counts[neighbour_idx], vec_d[k], HIP_R_64F));

    }
}

Distributed_vector::~Distributed_vector(){
    delete[] counts;
    delete[] displacements;
    delete[] neighbours;
    for(int k = 0; k < number_of_neighbours; k++){
        cudaErrchk(hipHostFree(vec_h[k]));
        cudaErrchk(hipFree(vec_d[k]));
        cusparseErrchk(hipsparseDestroyDnVec(descriptors[k]));
    }
    delete[] vec_h;
    delete[] vec_d;
    delete[] descriptors;
}
