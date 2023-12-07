#include <iostream>
// #include <petscksp.h>
// #include <petscsys.h>
#include <string>
#include "utils.h"
#include <mpi.h>


int main(int argc, char **argv) {
    // older version of petsc on daint
    // replace by PetscCall()
    // CHKERRQ(PetscInitialize(&argc, &argv, NULL, NULL));
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string base_path = "../test_data";
    std::string matrix_parameters_path = base_path + "/matrix_parameters.txt";

    int matrix_size;
    load_matrix_parameters(matrix_parameters_path, &matrix_size);


    std::cout << "Hello World from rank " << rank << std::endl;
    std::cout << "Matrix size: " << matrix_size << " from rank " << rank << std::endl;

    // CHKERRQ(PetscFinalize());
    return 0;
}
