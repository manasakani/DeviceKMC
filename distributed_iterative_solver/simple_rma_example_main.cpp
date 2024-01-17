#include <mpi.h>
#include <iostream>
#include <unistd.h>  

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "Hello from rank " << rank << " of " << size << std::endl;

    int matrix_size = 11111;

    int recvcounts[size];
    int displs[size];
    int rows_per_rank = matrix_size / size;    
    for (int i = 0; i < size; ++i) {
        if(i < matrix_size % size){
            recvcounts[i] = rows_per_rank+1;
        }
        else{
            recvcounts[i] = rows_per_rank;
        }
    }
    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
        displs[i] = displs[i-1] + recvcounts[i-1];
    }
    int row_start_index = displs[rank];
    rows_per_rank = recvcounts[rank];

    int measurements = 10000;


    for(int i = 0; i < measurements; i++){
        double *vec_global = new double[matrix_size];
        double *vec_local = new double[rows_per_rank];




        MPI_Win win;
        // initialize memory for RMA
        MPI_Win_create(vec_local, rows_per_rank*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        for(int j = 0; j < rows_per_rank; j++){
            vec_local[j] = 1.0;
            (vec_global+displs[rank])[j] = 1.0;
        }

        // for(int i = 0; i < size-1; i++){
        //     // loop over neighbors
        //     MPI_Win_lock(MPI_LOCK_SHARED, (rank+i+1) % size, 0, win);
        //     int get_idx = (rank+i+1) % size;
        //     MPI_Get(vec_global+displs[get_idx], recvcounts[get_idx], MPI_DOUBLE, get_idx, 0, recvcounts[get_idx], MPI_DOUBLE, win);
        //     MPI_Win_flush(get_idx, win);
        //     MPI_Win_unlock(get_idx, win);
        // }
        // MPI_Win_fence(0, win);
        // MPI_Win_fence(0, win);

        MPI_Win_lock_all(0, win);
        for(int i = 0; i < size-1; i++){
            // loop over neighbors
            int get_idx = (rank+i+1) % size;
            MPI_Get(vec_global+displs[get_idx], recvcounts[get_idx], MPI_DOUBLE, get_idx, 0, recvcounts[get_idx], MPI_DOUBLE, win);
        }
        MPI_Win_flush_all(win);
        MPI_Win_unlock_all(win);
        // MPI_Win_lock_all(0, win);
        // for(int i = 0; i < size-1; i++){
        //     // loop over neighbors
        //     int get_idx = (rank+i+1) % size;
        //     MPI_Get(vec_global+displs[get_idx], recvcounts[get_idx], MPI_DOUBLE, get_idx, 0, recvcounts[get_idx], MPI_DOUBLE, win);
        //     MPI_Win_flush(get_idx, win);
        // }
        // MPI_Win_unlock_all(win);


        MPI_Win_fence(0, win);
        MPI_Win_fence(0, win);



        for(int j = 0; j < matrix_size; j++){
            if(std::abs(vec_global[j] - 1.0) > 1e-16){
                std::cout << rank << " Error in global vector at index " << j << " with value " << vec_global[j] << std::endl;
            }
        }

        delete[] vec_global;
        delete[] vec_local;
        MPI_Win_free(&win);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for(int i = 0; i < size; i++){
        if(rank == i){
            std::cout << "Rank " << rank << " displs[rank] " << displs[rank] << " displs[rank]+recvcounts[get_idx] " << displs[rank]+recvcounts[rank]<< std::endl;
        }
        sleep(0.1);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    

    MPI_Finalize();
    return 0;
}
