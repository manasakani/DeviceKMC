//*****************
// Utility functions
//*****************
#pragma once
#include <sys/stat.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <numeric>
#include <algorithm>
#include <math.h>
#include <omp.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

// #ifndef UTILS_H
// #define UTILS_H

#define print(x) std::cout << x << std::endl

// lapack and blas functions:
extern "C"
{
    extern void dgesv_(int *, int *, double *, int *, int *, double *, int *, int *);
    extern void dgesvd_(char *, char *, int *, int *, double *, int *, double *, double *, int *, double *, int *, double *, int *, int *);
    extern void dgemm_(char *, char *, int *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
    extern void dgetri_(int *, double *, int *, int *, double *, int *, int *);
    extern void dgetrf_(int *, int *, double *, int *, int *, int *);
}

// Elements of the periodic table, converted from the input file
enum ELEMENT : int
{
    DEFECT,         // d - lattice interstitial site
    OXYGEN_DEFECT,  // Od - Oxygen defect/interstitial
    VACANCY,        // V - Lattice vacancy
    O, Hf, Ni, Ti, Pt, N,
};

// convert string to element type
ELEMENT update_element(std::string element_);

// convert element type to string
std::string return_element(ELEMENT element_);

// check if file/folder exists here
inline bool location_exists(const std::string &fname)
{
    struct stat buffer;
    return (stat(fname.c_str(), &buffer) == 0);
}

// swap two elements in an array
template <typename T>
void swap_values(T* a, T* b){
    T temp_a = *a;
    *a = *b;
    *b = temp_a;
}

// cumulative sum of array elements
template <typename T>
void inclusive_prefix_sum(T* input, T* output, size_t N) {
    if (N == 0) return;

    output[0] = input[0];
    for (size_t i = 1; i < N; ++i) {
        output[i] = output[i - 1] + input[i];
    }
}

// potential solution for gaussian charge distribution
inline double v_solve(double &r_dist, int &charge, double &sigma, double &k, double &q) { return static_cast<double>(charge) * erfc(r_dist / (sigma * sqrt(2))) * k * q / r_dist; }

// read xyz and populate the xyz coordinate array and lattice array
int read_xyz(std::string filename, std::vector<ELEMENT> &elements,
             std::vector<double> &x, std::vector<double> &y, std::vector<double> &z);

// returns the 3D distance which can include pbc. Note that only y-z periodicity exists!
double site_dist(std::vector<double> pos1, 
                 std::vector<double> pos2, std::vector<double> lattice, bool pbc);
double site_dist(double pos1x, double pos1y, double pos1z, 
                 double pos2x, double pos2y, double pos2z, std::vector<double> lattice, bool pbc);

// sort coordinates by x-axis value
void sort_by_x(std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, std::vector<ELEMENT> &elements, std::vector<double> lattice);

// shifts the first atom position to 0 in the selected dims. Expects the coordinates to be sorted!
void center_coords(std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, int N, bool dim[]);

// translates the periodic boundaries of the cell
void translate_cell(std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, int N, std::vector<double> lattice, std::vector<double> shifts);

// CUDA/cuBLAS/CuSolver
void CheckCublasError(cublasStatus_t const& status);
void CheckCusolverDnError(cusolverStatus_t const &status);
cublasHandle_t CreateCublasHandle(int device);
cusolverDnHandle_t CreateCusolverDnHandle(int device);

// GEMM
void gemm(cublasHandle_t handle, char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);

// GESV (by LU decomposition)
void gesv(cusolverDnHandle_t handle, int *N, int *nrhs, double *A, int *lda, int *ipiv, double *B, int *ldb, int *info);

// error checking for CUDA calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    //   if (abort) exit(code);
   }
}

// #endif