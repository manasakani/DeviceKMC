/*
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-09

Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
*/

#include <cstdio>
#include <cstdlib>
#include <complex>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <iomanip>

#include "utils.h"


template<typename T>
bool load_binary_array(
    std::string filename, 
    T *array, 
    int size)
{
    std::FILE *fp;

    fp = std::fopen(filename.c_str(), "rb");
    if (fp == nullptr) {
        std::printf("Error opening file\n");
        return false;
    }

    std::fread(array, sizeof(T), size, fp);

    std::fclose(fp);
    return true;
}
template bool load_binary_array<double>(std::string filename, double *array, int size);
template bool load_binary_array<int>(std::string filename, int *array, int size);



template<typename T>
bool load_text_array(
    std::string filename, 
    T *array,
    int size)
{

    std::ifstream ifile(filename, std::ios::in);
    if (!ifile.is_open()) {
        std::cout << filename << std::endl;
        std::printf("Error opening file\n");
        return false;
    }

    // problem on how the text file is saved
    // i.e. savetxt from numpy does 1.123e01 instead of 11.23
    // fix, but not possible for complex numbers
    // i.e. templating brings no benefit:)
    double num = 0.0;
    //keep storing values from the text file so long as data exists:
    for (int i = 0; i < size; i++) {
        ifile >> num;
        array[i] = (T)num;
    }

    ifile.close();

    return true;
}
// Explicit instantiation of the template
// else not found in compilation
// other option would be to put the implementation in the header file
template bool load_text_array<double>(std::string filename, double* matrix, int size);
template bool load_text_array<int>(std::string filename, int* matrix, int size);

template <typename T>
bool save_text_array(
    std::string filename,
    const T* array,
    int size)
{

    std::ofstream file(filename);
    if (file.is_open()) {
        for(int i = 0; i < size; i++){
            file << array[i] << " "; 
        }
        file.close();
        std::printf("Array data written to file.\n");
        return true;
    } else {
        std::printf("Unable to open the file for writing.\n");
        return false;
    }
}
template bool save_text_array<double>(std::string filename, const double* array, int size);

template <typename T>
void save_bin_array(T* array, int numElements, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<char*>(array), numElements*sizeof(T));
        file.close();
        std::cout << "Array data written to file: " << filename << std::endl;
    } else {
        std::cerr << "Unable to open the file for writing." << std::endl;
    }
}
template void save_bin_array<double>(double* array, int numElements, const std::string& filename);

template<typename T>
void sparse_to_dense(
    T *dense_matrix,
    T *data,
    int *indices,
    int *indptr,
    int matrix_band)
{
    for (int i = 0; i < matrix_band; i++) {
        for (int j = 0; j < matrix_band; j++) {
            // could not work for complex data type
            dense_matrix[i*matrix_band + j] = T(0);
        }
    }
    for(int i = 0; i < matrix_band; i++){
        for(int j = indptr[i]; j < indptr[i+1]; j++){
            T tmp = data[j];
            dense_matrix[i*matrix_band + indices[j]] = tmp;
        }
    }
}

template void sparse_to_dense<double>(double *dense_matrix,
    double *data,
    int *indices,
    int *indptr,
    int matrix_band);

template<typename T>
void copy_array(
    T *array,
    T *copy,
    int size)
{
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        copy[i] = array[i];
    }
}
template void copy_array<double>(double *array, double *copy, int size);

template<typename T>
bool assert_array_elementwise(
    T *array_test,
    T *array_ref,
    double abstol,
    double reltol,
    int size)
{
    for (int i = 0; i < size; i++) {
        if (std::abs(array_test[i] - array_ref[i]) > reltol * std::abs(array_ref[i]) + abstol) {
            std::cout << std::fixed << std::setprecision(12) << array_test[i] << " " << array_ref[i] << std::endl;
            std::printf("Arrays are elementwise not the same\n");
            return false;
        }
    }
    return true;
}
template bool assert_array_elementwise<double>(
    double *array_test,
    double *array_ref,
    double abstol,
    double reltol,
    int size);


template<typename T>
bool assert_array_magnitude(
    T *array_test,
    T *array_ref,
    double abstol,
    double reltol,
    int size,
    double *relative_error)
{
    double sum_difference = 0.0;
    double sum_ref = 0.0;
    for (int i = 0; i < size; i++) {
        sum_difference += std::abs(array_test[i] - array_ref[i]) *std::abs(array_test[i] - array_ref[i]);
        sum_ref += std::abs(array_ref[i])*std::abs(array_ref[i]);

    }
    sum_difference = std::sqrt(sum_difference);
    sum_ref = std::sqrt(sum_ref);

    relative_error[0] = sum_difference / sum_ref;

    if (sum_difference > reltol * sum_ref + abstol) {
        std::printf("Arrays are in magnitude not the same\n");
        std::cout << "Difference " << sum_difference << std::endl;
        std::cout << "Mixed tolerance " << reltol * sum_ref + abstol << std::endl;
        return false;
    }


    return true;
}
template bool assert_array_magnitude<double>(
    double *array_test,
    double *array_ref,
    double abstol,
    double reltol,
    int size,
    double *relative_error);



bool are_equals(
    std::complex<double> *A,
    std::complex<double> *B,
    unsigned int matrix_band, 
    unsigned int blocksize)
{
    // Check that the two parsed matrices are equals
    for (unsigned int i = 0; i < matrix_band; i++) {
        for (unsigned int j = 0; j < blocksize; j++) {
            if (std::abs(A[i * matrix_band + j] - B[i * matrix_band + j]) > 1e-10) {
                return false;
            }
        }
    }
    return true;
}

template<typename T>
void calc_bandwidth_dense(
    T * matrix,
    int matrix_size,
    int * ku,
    int * kl)
{
    int ku_tmp = 0;
    int kl_tmp = 0;
    for(int i = 0; i < matrix_size; i++){
        for(int j = 0; j < matrix_size; j++){
            if(matrix[i*matrix_size + j] != T(0)){
                if(i > j){
                    if(kl_tmp < i - j){
                        kl_tmp = i - j;
                    }
                }
                else{
                    if(ku_tmp < j - i){
                        ku_tmp = j - i;
                    }
                }
            }
        }
    }
    *ku = ku_tmp;
    *kl = kl_tmp;
}
template void calc_bandwidth_dense<double>(double * matrix, int matrix_size, int * ku, int * kl);

void calc_bandwidth_sparse(
    int * indices,
    int * indptr,
    int matrix_size,
    int * ku,
    int * kl)
{
    int ku_tmp = 0;
    int kl_tmp = 0;
    for(int i = 0; i < matrix_size; i++){
        for(int j = indptr[i]; j < indptr[i+1]; j++){
            if(i > indices[j]){
                if(kl_tmp < i - indices[j]){
                    kl_tmp = i - indices[j];
                }
            }
            else{
                if(ku_tmp < indices[j] - i){
                    ku_tmp = indices[j] - i;
                }
            }
        }
    }
    *ku = ku_tmp;
    *kl = kl_tmp;
}


template<typename T>
void dense_to_band_for_LU(
    T *dense_matrix,
    T *matrix_band,
    int matrix_size,
    int ku,
    int kl)
{
    for(int i = 0; i<2*ku+1+kl;i++){
        for(int j = 0; j < matrix_size;j++){
            matrix_band[i*matrix_size + j] = T(0);
        }
    }

    for(int i = 0; i < matrix_size; i++){
        for(int j = 0; j < matrix_size; j++){
            if(dense_matrix[matrix_size*i + j] != T(0)){
                matrix_band[2*ku + i - j + j*(2*ku+kl+1)] = dense_matrix[matrix_size*i + j];
            }
        }
    }
}
template void dense_to_band_for_LU<double>(double *dense_matrix, double *matrix_band, int matrix_size, int ku, int kl);


template<typename T>
void sparse_to_band_for_LU(
    T *data,
    int *indices,
    int *indptr,
    T *matrix_band,
    int matrix_size,
    int ku,
    int kl)
{
    for(int i = 0; i<2*ku+1+kl;i++){
        for(int j = 0; j < matrix_size;j++){
            matrix_band[i*matrix_size + j] = T(0);
        }
    }

    for(int i = 0; i < matrix_size; i++){
        for(int j = indptr[i]; j < indptr[i+1]; j++){
            matrix_band[2*ku + i - indices[j] + indices[j]*(2*ku+kl+1)] = data[j];
        }
    }
}
template void sparse_to_band_for_LU<double>(double *data, int *indices, int *indptr, double *matrix_band, int matrix_size, int ku, int kl);



template<typename T>
void dense_to_band_for_U_CHOL(
    T *dense_matrix,
    T *matrix_band,
    int matrix_size,
    int kd)
{
    for(int i = 0; i<1+kd;i++){
        for(int j = 0; j < matrix_size;j++){
            matrix_band[i*matrix_size + j] = T(0);
        }
    }

    for(int i = 0; i < matrix_size; i++){
        for(int j = 0; j < matrix_size; j++){
            if(dense_matrix[matrix_size*i + j] != T(0) && j-i >= 0){
                matrix_band[kd + i - j + j*(kd+1)] = dense_matrix[matrix_size*i + j];
            }
        }
    }
}
template void dense_to_band_for_U_CHOL<double>(double *dense_matrix, double *matrix_band, int matrix_size, int kd);

template<typename T>
void sparse_to_band_for_U_CHOL(
    T *data,
    int *indices,
    int *indptr,
    T *matrix_band,
    int matrix_size,
    int kd)
{
    for(int i = 0; i<1+kd;i++){
        for(int j = 0; j < matrix_size;j++){
            matrix_band[i*matrix_size + j] = T(0);
        }
    }

    for(int i = 0; i < matrix_size; i++){
        for(int j = indptr[i]; j < indptr[i+1]; j++){
            if(indices[j] - i >= 0){
                matrix_band[kd + i - indices[j] + indices[j]*(kd+1)] = data[j];
            }
        }
    }
}
template void sparse_to_band_for_U_CHOL<double>(double *data, int *indices, int *indptr, double *matrix_band, int matrix_size, int kd); 


template<typename T>
bool assert_symmetric(
    T *dense_matrix,
    int matrix_size,
    double abstol,
    double reltol)
{
    bool symmetric = true;
    for(int i = 0; i< matrix_size;i++){
        for(int j = 0; j < matrix_size;j++){
            if( std::abs(dense_matrix[i*matrix_size + j]-dense_matrix[j*matrix_size + i]) 
                > std::abs(dense_matrix[j*matrix_size + i]) * reltol + abstol){
                symmetric = false;
            }
        }
    }
    return symmetric;
}
template bool assert_symmetric<double>(
    double *dense_matrix,
    int matrix_size,
    double abstol,
    double reltol);