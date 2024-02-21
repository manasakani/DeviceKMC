#include <string>
#include <iostream>
#include <fstream>
#include <cmath>

void split_matrix(
    int matrix_size,
    int size,
    int *counts,
    int *displacements);

template <typename T>
void save_bin_array(T* array, int numElements, const std::string& filename);

template<typename T>
void load_binary_array(
    std::string filename, 
    T *array, 
    int size);

void load_matrix_parameters(
    std::string filename, 
    int *matrix_size);

void extract_diagonal(
    double *data,
    int *row_ptr,
    int *col_indices,
    double *diagonal,
    int matrix_size);

void symmetric_precondition_matrix(
    double *data,
    int *row_ptr,
    int *col_indices,
    double *diagonal,
    int matrix_sizes);

void precondition_vector(
    double *vector,
    double *diagonal,
    int matrix_size);

void unpreecondition_vector(
    double *vector,
    double *diagonal,
    int matrix_size);

void save_measurements(
    std::string path,
    double *times,
    int measurements,
    bool append);

std::string get_filename(
    std::string path,
    std::string filename,
    int number_of_kmc_steps,
    int size,
    int rank);