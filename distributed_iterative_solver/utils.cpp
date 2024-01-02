#include "utils.h"

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
template void save_bin_array<int>(int* array, int numElements, const std::string& filename);

template<typename T>
void load_binary_array(
    std::string filename, 
    T *array, 
    int size)
{
    std::FILE *fp;

    fp = std::fopen(filename.c_str(), "rb");
    if (fp == nullptr) {
        std::printf("Error opening file\n");
    }

    std::fread(array, sizeof(T), size, fp);

    std::fclose(fp);
}
template void load_binary_array<double>(std::string filename, double *array, int size);
template void load_binary_array<int>(std::string filename, int *array, int size);

void load_matrix_parameters(
    std::string filename, 
    int *matrix_size)
{
    FILE *fp;

    fp = std::fopen(filename.c_str(), "r");
    if (fp == NULL) {
        std::printf("Error opening file\n");
    }

    std::fscanf(fp, "%u", matrix_size);

    std::fclose(fp);

}

void extract_diagonal(
    double *data,
    int *row_ptr,
    int *col_indices,
    double *diagonal,
    int matrix_size)
{
    #pragma omp parallel for
    for(int i = 0; i < matrix_size; i++){
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            if(col_indices[j] == i){
                diagonal[i] = data[j];
            }
        }
    }
}

void symmetric_precondition_matrix(
    double *data,
    int *row_ptr,
    int *col_indices,
    double *diagonal,
    int matrix_size)
{
    #pragma omp parallel for
    for(int i = 0; i < matrix_size; i++){
        for(int j = row_ptr[i]; j < row_ptr[i+1]; j++){
            data[j] = data[j] / sqrt(diagonal[i]) / sqrt(diagonal[col_indices[j]]);
        }
    }
}

void precondition_vector(
    double *vector,
    double *diagonal,
    int matrix_size)
{
    #pragma omp parallel for
    for(int i = 0; i < matrix_size; i++){
        vector[i] = vector[i] / sqrt(diagonal[i]);
    }
}

void unpreecondition_vector(
    double *vector,
    double *diagonal,
    int matrix_size)
{
    #pragma omp parallel for
    for(int i = 0; i < matrix_size; i++){
        vector[i] = vector[i] * sqrt(diagonal[i]);
    }
}