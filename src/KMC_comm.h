#pragma once
#include <mpi.h>

class KMC_comm {
public:
    int rank_global;
    int rank_K;
    int rank_T;
    int rank_pairwise;
    int rank_events;

    int size_global;
    int size_K;
    int size_T;
    int size_pairwise;
    int size_events;

    int root_K;
    int root_T;
    int root_pairwise;

    MPI_Group group_global;
    MPI_Group group_K;
    MPI_Group group_T;
    MPI_Group group_pairwise;
    
    MPI_Comm comm_K;
    MPI_Comm comm_T;
    MPI_Comm comm_pairwise;
    MPI_Comm comm_events;

    int *counts_K;
    int *counts_T;
    int *counts_pairwise;
    int *counts_events;

    int *displs_K;
    int *displs_T;
    int *displs_pairwise;
    int *displs_events;

    KMC_comm(
        MPI_Comm comm_global,
        int nrows_K,
        int nrows_T,
        int nrows_pairwise,
        int nrows_events
    ){
        comm_events = comm_global;
        MPI_Comm_rank(comm_global, &rank_global);
        MPI_Comm_size(comm_global, &size_global);
        //TODO determine better fractions
        if(size_global < 3){
            std::cerr << "Need at least 3 ranks to split the communicator" << std::endl;
            exit(1);
        } 
        else {
            int frac = size_global / 3;
            // size_K = 4; //frac;
            // size_T = 1; //frac + size_global % 3;
            // size_pairwise = 3; // frac;
            size_K = frac;
            size_T = frac + size_global % 3;
            size_pairwise = frac;

        }
        if(size_K + size_T + size_pairwise != size_global){
            std::cerr << "Error in splitting the communicator" << std::endl;
            exit(1);
        }

        int ranks_K[size_K], ranks_T[size_T], ranks_pairwise[size_pairwise];
        for (int i = 0; i < size_T; i++){
            ranks_T[i] = i;
        }
        for (int i = 0; i < size_K; i++){
            ranks_K[i] = i + size_T;
        }
        for (int i = 0; i < size_pairwise; i++){
            ranks_pairwise[i] = i + size_T + size_K;
        }
        if(ranks_pairwise[size_pairwise-1] != size_global - 1){
            std::cerr << "Error in splitting the communicator" << std::endl;
            exit(1);
        }
        root_T = 0;
        root_K = size_T;
        root_pairwise = size_T + size_K;


        MPI_Comm_group(comm_global, &group_global);
        MPI_Group_incl(group_global, size_K, ranks_K, &group_K);
        MPI_Group_incl(group_global, size_T, ranks_T, &group_T);
        MPI_Group_incl(group_global, size_pairwise, ranks_pairwise, &group_pairwise);
        MPI_Comm_create(MPI_COMM_WORLD, group_K, &comm_K);
        MPI_Comm_create(MPI_COMM_WORLD, group_T, &comm_T);
        MPI_Comm_create(MPI_COMM_WORLD, group_pairwise, &comm_pairwise);

        if (comm_K != MPI_COMM_NULL) {
            std::cout << "Rank " << rank_global << " is in comm_K" << std::endl;
        }
        if (comm_T != MPI_COMM_NULL) {
            std::cout << "Rank " << rank_global << " is in comm_T" << std::endl;
        }
        if (comm_pairwise != MPI_COMM_NULL) {
            std::cout << "Rank " << rank_global << " is in comm_pairwise" << std::endl;
        }

        MPI_Group_size(group_K, &size_K);
        MPI_Group_size(group_T, &size_T);
        MPI_Group_size(group_pairwise, &size_pairwise);
        MPI_Group_size(group_global, &size_events);
        MPI_Group_rank(group_K, &rank_K);
        MPI_Group_rank(group_T, &rank_T);
        MPI_Group_rank(group_pairwise, &rank_pairwise);
        MPI_Group_rank(group_global, &rank_events);

        root_T = 0;
        root_K = 0;
        root_pairwise = 0;
        rank_K = rank_global;
        rank_T = rank_global;
        rank_pairwise = rank_global;
        rank_events = rank_global;
        size_K = size_global;
        size_T = size_global;
        size_pairwise = size_global;
        size_events = size_global;
        comm_K = comm_global;
        comm_T = comm_global;
        comm_pairwise = comm_global;
        comm_events = comm_global;

        if (comm_K != MPI_COMM_NULL) {
            counts_K = new int[size_K];
            displs_K = new int[size_K];
            int rows_per_rank_K = nrows_K / size_K;
            for (int i = 0; i < size_K; ++i) {
                if(i < nrows_K % size_K){
                    counts_K[i] = rows_per_rank_K+1;
                }
                else{
                    counts_K[i] = rows_per_rank_K;
                }
            }
            displs_K[0] = 0;
            for (int i = 1; i < size_K; ++i) {
                displs_K[i] = displs_K[i-1] + counts_K[i-1];
            }
        }

        if (comm_T != MPI_COMM_NULL) {
            counts_T = new int[size_T];
            displs_T = new int[size_T];
            int rows_per_rank_T = nrows_T / size_T;
            for (int i = 0; i < size_T; ++i) {
                if(i < nrows_T % size_T){
                    counts_T[i] = rows_per_rank_T+1;
                }
                else{
                    counts_T[i] = rows_per_rank_T;
                }
            }
            displs_T[0] = 0;
            for (int i = 1; i < size_T; ++i) {
                displs_T[i] = displs_T[i-1] + counts_T[i-1];
            }
        }

        if (comm_pairwise != MPI_COMM_NULL) {
            counts_pairwise = new int[size_pairwise];
            displs_pairwise = new int[size_pairwise];
            int rows_per_rank_pairwise = nrows_pairwise / size_pairwise;
            for (int i = 0; i < size_pairwise; ++i) {
                if(i < nrows_pairwise % size_pairwise){
                    counts_pairwise[i] = rows_per_rank_pairwise+1;
                }
                else{
                    counts_pairwise[i] = rows_per_rank_pairwise;
                }
            }
            displs_pairwise[0] = 0;
            for (int i = 1; i < size_pairwise; ++i) {
                displs_pairwise[i] = displs_pairwise[i-1] + counts_pairwise[i-1];
            }
        }
        counts_events = new int[size_global];
        displs_events = new int[size_global];
        int rows_per_rank_events = nrows_events / size_global;
        for (int i = 0; i < size_global; ++i) {
            if(i < nrows_events % size_global){
                counts_events[i] = rows_per_rank_events+1;
            }
            else{
                counts_events[i] = rows_per_rank_events;
            }
        }
        displs_events[0] = 0;
        for (int i = 1; i < size_global; ++i) {
            displs_events[i] = displs_events[i-1] + counts_events[i-1];
        }


    }

    // ~KMC_comm(){
    //     MPI_Group_free(&group_global);
    //     MPI_Group_free(&group_K);
    //     MPI_Group_free(&group_T);
    //     MPI_Group_free(&group_pairwise);        
    //     if (comm_K != MPI_COMM_NULL) {
    //         delete[] counts_K;
    //         delete[] displs_K;
    //         MPI_Comm_free(&comm_K);
    //     }
    //     if (comm_T != MPI_COMM_NULL) {
    //         delete[] counts_T;
    //         delete[] displs_T;
    //         MPI_Comm_free(&comm_T);
    //     }
    //     if (comm_pairwise != MPI_COMM_NULL) {
    //         delete[] counts_pairwise;
    //         delete[] displs_pairwise;
    //         MPI_Comm_free(&comm_pairwise);
    //     }
    //     delete[] counts_events;
    //     delete[] displs_events;
    // };

};
