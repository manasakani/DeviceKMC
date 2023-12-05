/*
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-09

Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
*/

#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <fstream>
#include <string>
#include <iostream>
#include <mkl.h>


#include <Eigen/Dense>

#include "system_solve_benchmark.h"
#include "utils.h"


char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

int main(int argc, char *argv[]){

    if(argc != 13){
        std::cout << "Usage: ./main" << std::endl <<
        "-nmeas <number of measurements>" << std::endl <<
        "-sstep <kmc to start from>" << std::endl <<
        "-estep <kmc to end excluded>" << std::endl <<
        "-abstol <absolute tolerance to compare reference>" << std::endl << 
        "-reltol <relative tolerance to compare reference>" << std::endl <<
        "-restol <CG limit for residual>" << std::endl;
        return 0;
    }

    int nmeas = std::stoi(getCmdOption(argv, argv + argc, "-nmeas"));
    if(nmeas < 0){
        std::printf("Number of measurements must be positive\n");
        return 0;
    }

    
    int sstep = std::stoi(getCmdOption(argv, argv + argc, "-sstep"));
    int estep = std::stoi(getCmdOption(argv, argv + argc, "-estep"));
    int steps_to_measure = estep - sstep;
    std::printf("Number of measurements: %d\n", nmeas);
    std::printf("KMC steps to measure: %d\n", steps_to_measure);

    double abstol = std::stod(getCmdOption(argv, argv + argc, "-abstol"));
    double reltol = std::stod(getCmdOption(argv, argv + argc, "-reltol"));
    double restol = std::stod(getCmdOption(argv, argv + argc, "-restol"));
    if(abstol < 0){
        std::printf("Absolute tolerance must be positive\n");
        return 0;
    }
    else{
        std::printf("Absolute tolerance: %f\n", abstol);
    }
    if(reltol < 0){
        std::printf("Relative tolerance must be positive\n");
        return 0;
    }
    else{
        std::printf("Relative tolerance: %f\n", reltol);
    }
    if(restol < 0){
        std::printf("Residual tolerance must be positive\n");
        return 0;
    }
    else{
        std::printf("Residual tolerance: %f\n", restol);
    }


    int matrix_size = 7302;
    int number_of_nonzero = 186684;


    matrix_size = 26396;
    number_of_nonzero = 628526;
    matrix_size = 70630;
    number_of_nonzero = 1719652;

    bool flag_verbose = false;
    bool flag_failed = false;


    bool flag_dense = false;
    bool flag_calc_bandwidth = false;
    bool flag_save_times = true;
    bool flag_band = false;
    bool flag_save_relative_error = true;
    bool flag_save_steps = true;

    //print the matrix parameters
    std::printf("Matrix parameters:\n");
    std::printf("Matrix size: %d\n", matrix_size);
    std::printf("Number of nonzero: %d\n", number_of_nonzero);



    double *dense_matrix = NULL;
    double *dense_matrix_copy = NULL;
    if(flag_dense){
        dense_matrix = (double*)malloc(matrix_size*matrix_size*sizeof(double));
        dense_matrix_copy = (double*)malloc(matrix_size*matrix_size*sizeof(double));  
    }

    double *data = (double*)malloc(number_of_nonzero*sizeof(double));
    int *indices = (int*)malloc(number_of_nonzero*sizeof(int));
    int *indptr = (int*)malloc((matrix_size+1)*sizeof(int));
    double *rhs = (double*)malloc(matrix_size*sizeof(double));
    
    double *rhs_copy = (double*)malloc(matrix_size*sizeof(double));
    double *reference_solution = (double*)malloc(matrix_size*sizeof(double));
    double *previous_solution = (double*)malloc(matrix_size*sizeof(double));


    int kku = 483;
    int kkl = 483;
    int kkd = 483;

    kku = 1753;
    kkl = 1753;
    kkd = 1753;


    double *matrix_band_LU;
    double *matrix_band_LU_copy;
    double *matrix_band_CHOL;
    
    double *matrix_band_CHOL_copy;
    if(flag_band){
        matrix_band_LU = (double*)malloc((2*kku+kkl+1)*matrix_size*sizeof(double));
        matrix_band_LU_copy = (double*)malloc((2*kku+kkl+1)*matrix_size*sizeof(double));
        matrix_band_CHOL = (double*)malloc((kkd+1)*matrix_size*sizeof(double));
        matrix_band_CHOL_copy = (double*)malloc((kkd+1)*matrix_size*sizeof(double));
    }

    bool measurements_correct = true;


    int steps_CG[steps_to_measure];
    int steps_CG_guess[steps_to_measure];
    int steps_CG_jacobi[steps_to_measure];
    int steps_CG_jacobi_guess[steps_to_measure];
    int steps_CG_ILU[steps_to_measure];

    double CG_relative_error[steps_to_measure];
    double CG_guess_relative_error[steps_to_measure];
    double CG_jacobi_relative_error[steps_to_measure];
    double CG_jacobi_guess_relative_error[steps_to_measure];
    double CG_ILU_relative_error[steps_to_measure];

    //std::string base_path = "/usr/scratch/mont-fort17/almaeder/kmc_7k/system_K/";
    //std::string base_path = "/usr/scratch/mont-fort17/almaeder/kmc_28k/system_K/";
    std::string base_path = "/usr/scratch/mont-fort17/almaeder/kmc_80k/system_K/";
    for(int step = sstep; step < estep; step++){

        int step_to_measure_previous = step - 1;
        if(step == 0){
            std::printf("No previous step exists\n");
            std::printf("Use solution of step itself\n");
            step_to_measure_previous = 0;
        }


        // Get matrix parameters

        std::string path_data = base_path + "A_data";
        std::string path_indices = base_path + "A_col_indices";
        std::string path_indptr = base_path + "A_row_ptr";
        std::string path_rhs = base_path + "A_rhs";
        std::string path_reference_solution = base_path + "solution";
        std::string path_reference_solution_previous_step = base_path + "solution";

        path_data += std::to_string(step) + ".bin";
        path_indices += std::to_string(step) + ".bin";
        path_indptr += std::to_string(step) + ".bin";
        path_rhs += std::to_string(step) + ".bin";
        path_reference_solution += std::to_string(step) + ".bin";
        path_reference_solution_previous_step += std::to_string(step_to_measure_previous) + ".bin";

        std::cout << path_data << std::endl;
        std::cout << path_indices << std::endl;
        std::cout << path_indptr << std::endl;
        std::cout << path_rhs << std::endl;
        std::cout << path_reference_solution << std::endl;
        std::cout << path_reference_solution_previous_step << std::endl;

        if(!load_binary_array<double>(path_data, data, number_of_nonzero)){
            if(flag_verbose){
                std::printf("Error loading data\n");
            }
            flag_failed = true;
        }
        if(!load_binary_array<int>(path_indices, indices, number_of_nonzero)){
            if(flag_verbose){
                std::printf("Error loading indices\n");
            }
            flag_failed = true;
        }
        

        if(!load_binary_array<int>(path_indptr, indptr, matrix_size+1)){
            if(flag_verbose){
                std::printf("Error loading indptr\n");
            }
            flag_failed = true;
        }


        if(!load_binary_array<double>(path_rhs, rhs, matrix_size)){
            if(flag_verbose){
                std::printf("Error loading rhs\n");
            }
            flag_failed = true;
        }
        if(!load_binary_array<double>(path_reference_solution, reference_solution, matrix_size)){
            if(flag_verbose){
                std::printf("Error loading reference solution\n");
            }
            flag_failed = true;
        }
        if(!load_binary_array<double>(path_reference_solution_previous_step, previous_solution, matrix_size)){
            if(flag_verbose){
                std::printf("Error loading reference solution previous step\n");
            }
            flag_failed = true;
        }


        mkl_set_num_threads(14);

        std::printf("Step: %d\n", step);

        if(!flag_failed){
            bool reference_correct = true;

            double times_gesv[nmeas];
            double times_posv[nmeas];
            double times_gbsv[nmeas];
            double times_pbsv[nmeas];
            double times_CG[nmeas];
            double times_CG_ILU[nmeas];
            double times_CG_guess[nmeas];
            double times_CG_jacobi[nmeas];
            double times_CG_jacobi_guess[nmeas];
            double times_cusolver_dense_LU[nmeas];
            double times_cusolver_dense_CHOL[nmeas];
            double times_cusolver_sparse_CHOL[nmeas];
            
            if(flag_dense){
                sparse_to_dense<double>(
                    dense_matrix,
                    data,
                    indices,
                    indptr,
                    matrix_size);


                for(int i = 0; i < nmeas; i++){
                    copy_array<double>(dense_matrix, dense_matrix_copy, matrix_size*matrix_size);
                    copy_array<double>(rhs, rhs_copy, matrix_size);

                    times_gesv[i] = solve_mkl_dgesv(
                        dense_matrix_copy,
                        rhs_copy,
                        reference_solution,
                        matrix_size,
                        abstol,
                        reltol,
                        flag_verbose);

                    std::printf("Time MKL dgesv: %f\n", times_gesv[i]);
                    
                }


                for(int i = 0; i < nmeas; i++){
                    copy_array<double>(dense_matrix, dense_matrix_copy, matrix_size*matrix_size);
                    copy_array<double>(rhs, rhs_copy, matrix_size);

                    times_cusolver_dense_LU[i] = solve_cusolver_dense_LU(
                        dense_matrix_copy,
                        rhs_copy,
                        reference_solution,
                        matrix_size,
                        abstol,
                        reltol,
                        flag_verbose);

                    std::printf("Time cusolver dense LU: %f\n", times_cusolver_dense_LU[i]);
                    

                }

                for(int i = 0; i < nmeas; i++){
                    copy_array<double>(dense_matrix, dense_matrix_copy, matrix_size*matrix_size);
                    copy_array<double>(rhs, rhs_copy, matrix_size);

                    times_cusolver_dense_CHOL[i] = solve_cusolver_dense_CHOL(
                        dense_matrix_copy,
                        rhs_copy,
                        reference_solution,
                        matrix_size,
                        abstol,
                        reltol,
                        flag_verbose);

                    std::printf("Time cusolver dense CHOL: %f\n", times_cusolver_dense_CHOL[i]);

                }


                for(int i = 0; i < nmeas; i++){
                    copy_array<double>(rhs, rhs_copy, matrix_size);

                    times_cusolver_sparse_CHOL[i] = solve_cusolver_sparse_CHOL(
                        data,
                        indices,
                        indptr,
                        rhs_copy,
                        reference_solution,
                        number_of_nonzero,
                        matrix_size,
                        abstol,
                        reltol,
                        flag_verbose);

                    std::printf("Time cusolver sparse CHOL: %f\n", times_cusolver_sparse_CHOL[i]);

                }

            }

            if(flag_band){
                for(int i = 0; i < nmeas; i++){
                    copy_array<double>(dense_matrix, dense_matrix_copy, matrix_size*matrix_size);
                    copy_array<double>(rhs, rhs_copy, matrix_size);

                    times_posv[i] = solve_mkl_dposv(
                        dense_matrix_copy,
                        rhs_copy,
                        reference_solution,
                        matrix_size,
                        abstol,
                        reltol,
                        flag_verbose);

                    std::printf("Time MKL dposv: %f\n", times_posv[i]);
                }

                sparse_to_band_for_LU<double>(
                    data,
                    indices,
                    indptr,
                    matrix_band_LU,
                    matrix_size,
                    kku,
                    kkl);


                for(int i = 0; i < nmeas; i++){
                    copy_array<double>(matrix_band_LU, matrix_band_LU_copy, matrix_size*(2*kku+kkl+1));
                    copy_array<double>(rhs, rhs_copy, matrix_size);
                    times_gbsv[i] = solve_mkl_dgbsv(
                        matrix_band_LU_copy,
                        rhs_copy,
                        reference_solution,
                        matrix_size,
                        kku,
                        kkl,
                        abstol,
                        reltol,
                        flag_verbose);

                    std::printf("Time MKL gbsv: %f\n", times_gbsv[i]);
                }

                sparse_to_band_for_U_CHOL<double>(
                    data,
                    indices,
                    indptr,
                    matrix_band_CHOL,
                    matrix_size,
                    kkd);


                for(int i = 0; i < nmeas; i++){
                    copy_array<double>(matrix_band_CHOL, matrix_band_CHOL_copy, matrix_size*(kkd+1));
                    copy_array<double>(rhs, rhs_copy, matrix_size);
                    times_pbsv[i] = solve_mkl_dpbsv(
                        matrix_band_CHOL_copy,
                        rhs_copy,
                        reference_solution,
                        matrix_size,
                        kkd,
                        abstol,
                        reltol,
                        flag_verbose);

                    std::printf("Time MKL pbsv: %f\n", times_pbsv[i]);

                }

            }



            if(flag_calc_bandwidth){
                int ku = 0;
                int kl = 0;
                calc_bandwidth_sparse(
                    indices,
                    indptr,
                    matrix_size,
                    &ku,
                    &kl);    
                std::printf("Upper Bandwidth: %d\n", ku);
                std::printf("Lower Bandwidth: %d\n", kl);                                
            }

            

            for(int i = 0; i < nmeas; i++){
                copy_array<double>(rhs, rhs_copy, matrix_size);
                double starting_guess[matrix_size];
                // bit overkill to set zero in every measurement
                for(int j = 0; j < matrix_size; j++){
                    starting_guess[j] = 0.0;
                }
                times_CG[i] = solve_cusparse_CG(
                    data,
                    indices,
                    indptr,
                    rhs_copy,
                    reference_solution,
                    starting_guess,
                    number_of_nonzero,
                    matrix_size,
                    abstol,
                    reltol,
                    restol,
                    flag_verbose,
                    &steps_CG[step],
                    &CG_relative_error[step]);

                std::printf("Time cusparse CG: %f\n", times_CG[i]);


            }


            for(int i = 0; i < nmeas; i++){
                copy_array<double>(rhs, rhs_copy, matrix_size);
                double starting_guess[matrix_size];
                for(int j = 0; j < matrix_size; j++){
                    starting_guess[j] = previous_solution[j];
                }
                times_CG_guess[i] = solve_cusparse_CG(
                    data,
                    indices,
                    indptr,
                    rhs_copy,
                    reference_solution,
                    starting_guess,
                    number_of_nonzero,
                    matrix_size,
                    abstol,
                    reltol,
                    restol,
                    flag_verbose,
                    &steps_CG_guess[step],
                    &CG_guess_relative_error[step]);

                std::printf("Time cusparse CG with guess: %f\n", times_CG_guess[i]);

            }



            for(int i = 0; i < nmeas; i++){
                copy_array<double>(rhs, rhs_copy, matrix_size);

                times_CG_ILU[i] = solve_cusparse_ILU_CG(
                    data,
                    indices,
                    indptr,
                    rhs_copy,
                    reference_solution,
                    number_of_nonzero,
                    matrix_size,
                    abstol,
                    reltol,
                    restol,
                    flag_verbose,
                    &steps_CG_ILU[step],
                    &CG_ILU_relative_error[step]);

                std::printf("Time cusparse ILU CG: %f\n", times_CG_ILU[i]);

            }


            for(int i = 0; i < nmeas; i++){
                copy_array<double>(rhs, rhs_copy, matrix_size);
                double starting_guess[matrix_size];
                // bit overkill to set zero in every measurement
                for(int j = 0; j < matrix_size; j++){
                    starting_guess[j] = 0.0;
                }
                times_CG_jacobi[i] = solve_cusparse_CG_jacobi(
                    data,
                    indices,
                    indptr,
                    rhs_copy,
                    reference_solution,
                    starting_guess,
                    number_of_nonzero,
                    matrix_size,
                    abstol,
                    reltol,
                    restol,
                    flag_verbose,
                    &steps_CG_jacobi[step],
                    &CG_jacobi_relative_error[step]);

                std::printf("Time cusparse jacobi CG: %f\n", times_CG_jacobi[i]);

            }

            for(int i = 0; i < nmeas; i++){
                copy_array<double>(rhs, rhs_copy, matrix_size);
                double starting_guess[matrix_size];
                // bit overkill to set zero in every measurement
                for(int j = 0; j < matrix_size; j++){
                    starting_guess[j] = previous_solution[j];
                }
                times_CG_jacobi_guess[i] = solve_cusparse_CG_jacobi(
                    data,
                    indices,
                    indptr,
                    rhs_copy,
                    reference_solution,
                    starting_guess,
                    number_of_nonzero,
                    matrix_size,
                    abstol,
                    reltol,
                    restol,
                    flag_verbose,
                    &steps_CG_jacobi_guess[step],
                    &CG_jacobi_guess_relative_error[step]);

                std::printf("Time cusparse jacobi CG with guess: %f\n", times_CG_jacobi_guess[i]);

            }
            std::cout << "Relative error CG: " << CG_relative_error[step] << std::endl;
            std::cout << "Relative error CG guess: " << CG_guess_relative_error[step] << std::endl;
            std::cout << "Relative error CG jacobi: " << CG_jacobi_relative_error[step] << std::endl;
            std::cout << "Relative error CG jacobi guess: " << CG_jacobi_guess_relative_error[step] << std::endl;
            std::cout << "Relative error CG ILU: " << CG_ILU_relative_error[step] << std::endl;


            if(flag_dense && !reference_correct){
                std::printf("Error in reference\n");
            }
            else{
                std::printf("Reference correct\n");
            }
            if(flag_save_times){
                if(flag_dense){
                    std::ofstream outputFile_times_gesv;
                    std::string file_gesv = base_path + "results/" + "times_gesv"
                     + "_" + std::to_string(int(std::log10(restol))) +"_"+ std::to_string(sstep) + "_" +std::to_string(estep) + "_";
                    file_gesv += std::to_string(step);
                    file_gesv +=  ".txt";
                    outputFile_times_gesv.open(file_gesv);
                    if(outputFile_times_gesv.is_open()){
                        for(int i = 0; i < nmeas; i++){
                            outputFile_times_gesv << times_gesv[i] << " ";
                        }
                        outputFile_times_gesv << '\n';
                    }
                    else{
                        std::printf("Error opening file\n");
                    }
                    outputFile_times_gesv.close();

                    std::ofstream outputFile_times_posv;
                    std::string file_posv = base_path + "results/" + "times_posv"
                     + "_" + std::to_string(int(std::log10(restol))) +"_"+ std::to_string(sstep) + "_" +std::to_string(estep) + "_";
                    file_posv += std::to_string(step);
                    file_posv +=  ".txt";
                    outputFile_times_posv.open(file_posv);
                    if(outputFile_times_posv.is_open()){
                        for(int i = 0; i < nmeas; i++){
                            outputFile_times_posv << times_posv[i] << " ";
                        }
                        outputFile_times_posv << '\n';
                    }
                    else{
                        std::printf("Error opening file\n");
                    }
                    outputFile_times_posv.close();

                    std::ofstream outputFile_times_cusolver_dense_LU;
                    std::string file_cusolver_dense_LU = base_path + "results/" + "times_cusolver_dense_LU"
                     + "_" + std::to_string(int(std::log10(restol))) +"_"+ std::to_string(sstep) + "_" +std::to_string(estep) + "_";
                    file_cusolver_dense_LU += std::to_string(step);
                    file_cusolver_dense_LU +=  ".txt";
                    outputFile_times_cusolver_dense_LU.open(file_cusolver_dense_LU);
                    if(outputFile_times_cusolver_dense_LU.is_open()){
                        for(int i = 0; i < nmeas; i++){
                            outputFile_times_cusolver_dense_LU << times_cusolver_dense_LU[i] << " ";
                        }
                        outputFile_times_cusolver_dense_LU << '\n';
                    }
                    else{
                        std::printf("Error opening file\n");
                    }
                    outputFile_times_cusolver_dense_LU.close();

                    std::ofstream outputFile_times_cusolver_dense_CHOL;
                    std::string file_cusolver_dense_CHOL = base_path + "results/" + 
                    "times_cusolver_dense_CHOL"
                     + "_" + std::to_string(int(std::log10(restol))) +"_"+ std::to_string(sstep) + "_" +std::to_string(estep) + "_";
                    file_cusolver_dense_CHOL += std::to_string(step);
                    file_cusolver_dense_CHOL +=  ".txt";
                    outputFile_times_cusolver_dense_CHOL.open(file_cusolver_dense_CHOL);
                    if(outputFile_times_cusolver_dense_CHOL.is_open()){
                        for(int i = 0; i < nmeas; i++){
                            outputFile_times_cusolver_dense_CHOL << times_cusolver_dense_CHOL[i] << " ";
                        }
                        outputFile_times_cusolver_dense_CHOL << '\n';
                    }
                    else{
                        std::printf("Error opening file\n");
                    }
                    outputFile_times_cusolver_dense_CHOL.close();

                    std::ofstream outputFile_times_gbsv;
                    std::string file_gbsv = base_path + "results/" + "times_gbsv";
                    file_gbsv += std::to_string(step);
                    file_gbsv +=  ".txt";
                    outputFile_times_gbsv.open(file_gbsv);
                    if(outputFile_times_gbsv.is_open()){
                        for(int i = 0; i < nmeas; i++){
                            outputFile_times_gbsv << times_gbsv[i] << " ";
                        }
                        outputFile_times_gbsv << '\n';
                    }
                    else{
                        std::printf("Error opening file\n");
                    }
                    outputFile_times_gbsv.close();

                    std::ofstream outputFile_times_cusolver_sparse_CHOL;
                    std::string file_cusolver_sparse_CHOL = base_path + "results/" + "times_cusolver_sparse_CHOL"
                     + "_" + std::to_string(int(std::log10(restol))) +"_"+ std::to_string(sstep) + "_" +std::to_string(estep) + "_";
                    file_cusolver_sparse_CHOL += std::to_string(step);
                    file_cusolver_sparse_CHOL +=  ".txt";
                    outputFile_times_cusolver_sparse_CHOL.open(file_cusolver_sparse_CHOL);
                    if(outputFile_times_cusolver_sparse_CHOL.is_open()){
                        for(int i = 0; i < nmeas; i++){
                            outputFile_times_cusolver_sparse_CHOL << times_cusolver_sparse_CHOL[i] << " ";
                        }
                        outputFile_times_cusolver_sparse_CHOL << '\n';
                    }
                    else{
                        std::printf("Error opening file\n");
                    }
                    outputFile_times_cusolver_sparse_CHOL.close();

                    std::ofstream outputFile_times_pbsv;
                    std::string file_pbsv = base_path + "results/" + "times_pbsv"
                     + "_" + std::to_string(int(std::log10(restol))) +"_"+ std::to_string(sstep) + "_" +std::to_string(estep) + "_";
                    file_pbsv += std::to_string(step);
                    file_pbsv +=  ".txt";
                    outputFile_times_pbsv.open(file_pbsv);
                    if(outputFile_times_pbsv.is_open()){
                        for(int i = 0; i < nmeas; i++){
                            outputFile_times_pbsv << times_pbsv[i] << " ";
                        }
                        outputFile_times_pbsv << '\n';
                    }
                    else{
                        std::printf("Error opening file\n");
                    }
                    outputFile_times_pbsv.close();

                }


                std::ofstream outputFile_times_CG;
                std::string file_CG = base_path + "results/" + "times_CG"
                     + "_" + std::to_string(int(std::log10(restol))) +"_"+ std::to_string(sstep) + "_" +std::to_string(estep) + "_";
                file_CG += std::to_string(step);
                file_CG +=  ".txt";
                outputFile_times_CG.open(file_CG);
                if(outputFile_times_CG.is_open()){
                    for(int i = 0; i < nmeas; i++){
                        outputFile_times_CG << times_CG[i] << " ";
                    }
                    outputFile_times_CG << '\n';
                }
                else{
                    std::printf("Error opening file\n");
                }
                outputFile_times_CG.close();

                std::ofstream outputFile_times_CG_guess;
                std::string file_CG_guess = base_path + "results/" + "times_CG_guess"
                     + "_" + std::to_string(int(std::log10(restol))) +"_"+ std::to_string(sstep) + "_" +std::to_string(estep) + "_";
                file_CG_guess += std::to_string(step);
                file_CG_guess +=  ".txt";
                outputFile_times_CG_guess.open(file_CG_guess);
                if(outputFile_times_CG_guess.is_open()){
                    for(int i = 0; i < nmeas; i++){
                        outputFile_times_CG_guess << times_CG_guess[i] << " ";
                    }
                    outputFile_times_CG_guess << '\n';
                }
                else{
                    std::printf("Error opening file\n");
                }
                outputFile_times_CG_guess.close();

                std::ofstream outputFile_times_CG_jacobi;
                std::string file_CG_jacobi = base_path + "results/" + "times_CG_jacobi"
                     + "_" + std::to_string(int(std::log10(restol))) +"_"+ std::to_string(sstep) + "_" +std::to_string(estep) + "_";
                file_CG_jacobi += std::to_string(step);
                file_CG_jacobi +=  ".txt";
                outputFile_times_CG_jacobi.open(file_CG_jacobi);
                if(outputFile_times_CG_jacobi.is_open()){
                    for(int i = 0; i < nmeas; i++){
                        outputFile_times_CG_jacobi << times_CG_jacobi[i] << " ";
                    }
                    outputFile_times_CG_jacobi << '\n';
                }
                else{
                    std::printf("Error opening file\n");
                }
                outputFile_times_CG_jacobi.close();

                std::ofstream outputFile_times_CG_jacobi_guess;
                std::string file_CG_jacobi_guess = base_path + "results/" + "times_CG_jacobi_guess"
                     + "_" + std::to_string(int(std::log10(restol))) +"_"+ std::to_string(sstep) + "_" +std::to_string(estep) + "_";
                file_CG_jacobi_guess += std::to_string(step);
                file_CG_jacobi_guess +=  ".txt";
                outputFile_times_CG_jacobi_guess.open(file_CG_jacobi_guess);
                if(outputFile_times_CG_jacobi_guess.is_open()){
                    for(int i = 0; i < nmeas; i++){
                        outputFile_times_CG_jacobi_guess << times_CG_jacobi_guess[i] << " ";
                    }
                    outputFile_times_CG_jacobi_guess << '\n';
                }
                else{
                    std::printf("Error opening file\n");
                }
                outputFile_times_CG_jacobi_guess.close();

                std::ofstream outputFile_times_CG_ILU;
                std::string file_CG_ILU = base_path + "results/" + "times_CG_ILU";
                file_CG_ILU += std::to_string(step);
                file_CG_ILU +=  ".txt";
                outputFile_times_CG_ILU.open(file_CG_ILU);
                if(outputFile_times_CG_ILU.is_open()){
                    for(int i = 0; i < nmeas; i++){
                        outputFile_times_CG_ILU << times_CG_ILU[i] << " ";
                    }
                    outputFile_times_CG_ILU << '\n';
                }
                else{
                    std::printf("Error opening file\n");
                }
                outputFile_times_CG_ILU.close();
                std::cout << "saved times" << std::endl;
            }

            
        }
    }
    if(flag_save_steps){
        std::ofstream outputFile_steps_CG;
        std::string path_steps_CG = base_path + "results/" + "steps_CG_" 
        + std::to_string(int(std::log10(restol))) +"_"+ std::to_string(sstep) + "_" +std::to_string(estep) + ".txt";
        outputFile_steps_CG.open(path_steps_CG);
        if(outputFile_steps_CG.is_open()){
            for(int i = 0; i < steps_to_measure; i++){
                outputFile_steps_CG << steps_CG[i] << " ";
            }
            outputFile_steps_CG << '\n';
        }
        else{
            std::printf("Error opening file\n");
        }
        outputFile_steps_CG.close();

        std::ofstream outputFile_steps_CG_guess;
        std::string path_steps_CG_guess = base_path + "results/" + "steps_CG_guess_" 
        + std::to_string(int(std::log10(restol))) +"_"+ std::to_string(sstep) + "_" +std::to_string(estep) + ".txt";
        outputFile_steps_CG_guess.open(path_steps_CG_guess);
        if(outputFile_steps_CG_guess.is_open()){
            for(int i = 0; i < steps_to_measure; i++){
                outputFile_steps_CG_guess << steps_CG_guess[i] << " ";
            }
            outputFile_steps_CG_guess << '\n';
        }
        else{
            std::printf("Error opening file\n");
        }
        outputFile_steps_CG_guess.close();

        std::ofstream outputFile_steps_CG_jacobi;
        std::string path_steps_CG_jacobi = base_path + "results/" + "steps_CG_jacobi_" 
        + std::to_string(int(std::log10(restol))) +"_"+ std::to_string(sstep) + "_" +std::to_string(estep) + ".txt";
        outputFile_steps_CG_jacobi.open(path_steps_CG_jacobi);
        if(outputFile_steps_CG_jacobi.is_open()){
            for(int i = 0; i < steps_to_measure; i++){
                outputFile_steps_CG_jacobi << steps_CG_jacobi[i] << " ";
            }
            outputFile_steps_CG_jacobi << '\n';
        }
        else{
            std::printf("Error opening file\n");
        }
        outputFile_steps_CG_jacobi.close();

        std::ofstream outputFile_steps_CG_jacobi_guess;
        std::string path_steps_CG_jacobi_guess = base_path + "results/" + "steps_CG_jacobi_guess_" 
        + std::to_string(int(std::log10(restol))) +"_"+ std::to_string(sstep) + "_" +std::to_string(estep) + ".txt";
        outputFile_steps_CG_jacobi_guess.open(path_steps_CG_jacobi_guess);
        if(outputFile_steps_CG_jacobi_guess.is_open()){
            for(int i = 0; i < steps_to_measure; i++){
                outputFile_steps_CG_jacobi_guess << steps_CG_jacobi_guess[i] << " ";
            }
            outputFile_steps_CG_jacobi_guess << '\n';
        }
        else{
            std::printf("Error opening file\n");
        }
        outputFile_steps_CG_jacobi_guess.close();

        std::ofstream outputFile_steps_CG_ILU;
        std::string path_steps_CG_ILU = base_path + "results/" + "steps_CG_ILU_" 
        + std::to_string(int(std::log10(restol))) +"_"+ std::to_string(sstep) + "_" +std::to_string(estep) + ".txt";
        outputFile_steps_CG_ILU.open(path_steps_CG_ILU);
        if(outputFile_steps_CG_ILU.is_open()){
            for(int i = 0; i < steps_to_measure; i++){
                outputFile_steps_CG_ILU << steps_CG_ILU[i] << " ";
            }
            outputFile_steps_CG_ILU << '\n';
        }
        else{
            std::printf("Error opening file\n");
        }
        outputFile_steps_CG_ILU.close();
    }
    if(flag_save_relative_error){
        std::ofstream outputFile_CG_relative_error;
        std::string path_CG_relative_error = base_path + "results/" + "CG_relative_error_"
            + std::to_string(int(std::log10(restol))) +"_" + std::to_string(sstep) + "_" +std::to_string(estep) + ".txt";
        outputFile_CG_relative_error.open(path_CG_relative_error);
        if(outputFile_CG_relative_error.is_open()){
            for(int i = 0; i < steps_to_measure; i++){
                outputFile_CG_relative_error << CG_relative_error[i] << " ";
            }
            outputFile_CG_relative_error << '\n';
        }
        else{
            std::printf("Error opening file\n");
        }
        outputFile_CG_relative_error.close();

        std::ofstream outputFile_CG_guess_relative_error;
        std::string path_CG_guess_relative_error = base_path + "results/" + "CG_guess_relative_error_"
            + std::to_string(int(std::log10(restol))) +"_" + std::to_string(sstep) + "_" +std::to_string(estep) + ".txt";
        outputFile_CG_guess_relative_error.open(path_CG_guess_relative_error);
        if(outputFile_CG_guess_relative_error.is_open()){
            for(int i = 0; i < steps_to_measure; i++){
                outputFile_CG_guess_relative_error << CG_guess_relative_error[i] << " ";
            }
            outputFile_CG_guess_relative_error << '\n';
        }
        else{
            std::printf("Error opening file\n");
        }
        outputFile_CG_guess_relative_error.close();

        std::ofstream outputFile_CG_jacobi_relative_error;
        std::string path_CG_jacobi_relative_error = base_path + "results/" + "CG_jacobi_relative_error_"
            + std::to_string(int(std::log10(restol))) +"_" + std::to_string(sstep) + "_" +std::to_string(estep) + ".txt";
        outputFile_CG_jacobi_relative_error.open(path_CG_jacobi_relative_error);
        if(outputFile_CG_jacobi_relative_error.is_open()){
            for(int i = 0; i < steps_to_measure; i++){
                outputFile_CG_jacobi_relative_error << CG_jacobi_relative_error[i] << " ";
            }
            outputFile_CG_jacobi_relative_error << '\n';
        }
        else{
            std::printf("Error opening file\n");
        }
        outputFile_CG_jacobi_relative_error.close();

        std::ofstream outputFile_CG_jacobi_guess_relative_error;
        std::string path_CG_jacobi_guess_relative_error = base_path + "results/" + "CG_jacobi_guess_relative_error_"
            + std::to_string(int(std::log10(restol))) +"_" + std::to_string(sstep) + "_" +std::to_string(estep) + ".txt";
        outputFile_CG_jacobi_guess_relative_error.open(path_CG_jacobi_guess_relative_error);
        if(outputFile_CG_jacobi_guess_relative_error.is_open()){
            for(int i = 0; i < steps_to_measure; i++){
                outputFile_CG_jacobi_guess_relative_error << CG_jacobi_guess_relative_error[i] << " ";
            }
            outputFile_CG_jacobi_guess_relative_error << '\n';
        }
        else{
            std::printf("Error opening file\n");
        }
        outputFile_CG_jacobi_guess_relative_error.close();

        std::ofstream outputFile_CG_ILU_relative_error;
        std::string path_CG_ILU_relative_error = base_path + "results/" + "CG_ILU_relative_error_"
            + std::to_string(int(std::log10(restol))) +"_" + std::to_string(sstep) + "_" +std::to_string(estep) + ".txt";
        outputFile_CG_ILU_relative_error.open(path_CG_ILU_relative_error);
        if(outputFile_CG_ILU_relative_error.is_open()){
            for(int i = 0; i < steps_to_measure; i++){
                outputFile_CG_ILU_relative_error << CG_ILU_relative_error[i] << " ";
            }
            outputFile_CG_ILU_relative_error << '\n';
        }
        else{
            std::printf("Error opening file\n");
        }
        outputFile_CG_ILU_relative_error.close();
    }


    if(!measurements_correct){
        std::printf("Error in one of the measurements\n");
    }
    else{
        std::printf("All measurements correct\n");
    }

    if(flag_dense){
        free(dense_matrix);
        free(dense_matrix_copy);
    }
    if(flag_band){
        free(matrix_band_LU);
        free(matrix_band_LU_copy);
        free(matrix_band_CHOL);
        free(matrix_band_CHOL_copy);        
    }

    free(rhs_copy);
    free(data);
    free(indices);
    free(indptr);
    free(rhs);

    return 0;
}








