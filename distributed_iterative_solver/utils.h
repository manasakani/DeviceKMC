#include <string>
#include <iostream>
#include <fstream>

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