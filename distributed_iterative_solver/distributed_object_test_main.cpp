#include "distributed_objects.h"
#include "utils.h"
#include <unistd.h>  

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::cout << "rank: " << rank << std::endl;

    int matsize = 7;
    std::string data_path = "/scratch/snx3000/amaeder/"+std::to_string(matsize)+"k_piz_daint_data";
    data_path = "/usr/scratch/mont-fort17/almaeder/kmc_"+std::to_string(matsize)+"k/system_K";
    std::string save_path = "/usr/scratch/mont-fort17/almaeder/kmc_"+std::to_string(matsize)+"k/split_matrix";

    int matrix_size;
    int nnz;     
    if(matsize == 7){
        matrix_size = 7302;
        nnz = 186684;        
    }
    else if(matsize == 80){
        matrix_size = 70630;
        nnz = 1719652;        
    }
    else{
        matrix_size = 262144;
        nnz = 16481266;
    }

    int *counts = new int[size];
    int *displacements = new int[size];

    split_matrix(matrix_size, size, counts, displacements);

    std::cout << rank << " " << counts[rank] << " " << displacements[rank] << std::endl;


    std::string data_filename = data_path + "/A_data"+std::to_string(0)+".bin";
    std::string row_ptr_filename = data_path + "/A_row_ptr"+std::to_string(0)+".bin";
    std::string col_indices_filename = data_path + "/A_col_indices"+std::to_string(0)+".bin";
    std::string rhs_filename = data_path + "/A_rhs"+std::to_string(0)+".bin";
    std::string solution_filename = data_path + "/solution"+std::to_string(0)+".bin";

    double *data = new double[nnz];
    int *row_ptr = new int[matrix_size+1];
    int *col_indices = new int[nnz];

    load_binary_array<double>(data_filename, data, nnz);
    load_binary_array<int>(row_ptr_filename, row_ptr, matrix_size+1);
    load_binary_array<int>(col_indices_filename, col_indices, nnz);

    int *row_ptr_local = new int[counts[rank]+1];
    for (int i = 0; i < counts[rank]+1; ++i) {
        row_ptr_local[i] = row_ptr[i+displacements[rank]] - row_ptr[displacements[rank]];
    }
    int nnz_local = row_ptr_local[counts[rank]];
    int *col_indices_local = new int[nnz_local];
    double *data_local = new double[nnz_local];
    for (int i = 0; i < nnz_local; ++i) {
        col_indices_local[i] = col_indices[i+row_ptr[displacements[rank]]];
        data_local[i] = data[i+row_ptr[displacements[rank]]];
    }

    distributed_matrix A_distributed(
        matrix_size,
        nnz_local,
        counts,
        displacements,
        col_indices_local,
        row_ptr_local,
        data_local,
        MPI_COMM_WORLD
    );

    // // assert both matrices are equal
    // for(int i = 0; i < nnz_local; i++){
    //     if(std::abs(data_local[i] - A_distributed.data_h[0][i]) > 1e-10){
    //         std::cout << "data not equal" << std::endl;
    //         std::cout << data_local[i] << " " << A_distributed.data_h[0][i] << std::endl;
    //     }
    //     if(col_indices_local[i] != A_distributed.col_indices_h[0][i]){
    //         std::cout << "col_indices not equal" << std::endl;
    //         std::cout << col_indices_local[i] << " " << A_distributed.col_indices_h[0][i] << std::endl;
    //     }
    // }
    // for(int i = 0; i < counts[rank]+1; i++){
    //     if(row_ptr_local[i] != A_distributed.row_ptr_h[0][i]){
    //         std::cout << "row_ptr not equal" << std::endl;
    //         std::cout << row_ptr_local[i] << " " << A_distributed.row_ptr_h[0][i] << std::endl;
    //     }
    // }
    for(int k = 0; k < A_distributed.number_of_neighbours; k++){
        save_bin_array<int>(A_distributed.row_ptr_h[k], A_distributed.counts[rank]+1,
            save_path+"/A_row_ptr" +
            std::to_string(size)+ "_"+std::to_string(rank)+"_"+std::to_string(k)+".bin");
        save_bin_array<int>(A_distributed.col_indices_h[k], A_distributed.nnz_per_neighbour[k],
            save_path+"/A_col_indices" +
            std::to_string(size)+ "_"+std::to_string(rank)+"_"+std::to_string(k)+".bin");        
    }

    for(int i = 0; i < size; i++){
        if(i == rank){
            std::cout << "rank, NNZ per neighbour: " << rank;
            for(int k = 0; k < A_distributed.number_of_neighbours; k++){
                std::cout << " " << A_distributed.nnz_per_neighbour[k];
            }        
            std::cout << std::endl;   
        }
        sleep(1);
        MPI_Barrier(MPI_COMM_WORLD);
    }


    delete[] counts;
    delete[] displacements;
    delete[] data;
    delete[] row_ptr;
    delete[] col_indices;
    delete[] row_ptr_local;
    delete[] col_indices_local;
    delete[] data_local;

    MPI_Finalize();
    return 0;
}