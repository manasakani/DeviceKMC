//*****************
// Utility functions
//*****************
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

#ifndef UTILS_H
#define UTILS_H

#define print(x) std::cout << x << std::endl

// lapack and blas functions:
extern "C"
{
    extern void dgesv_(int *, int *, double *, int *, int *, double *, int *, int *);
    extern void dgesvd_(char *, char *, int *, int *, double *, int *, double *, double *, int *, double *, int *, double *, int *, int *);
    extern void dgemm_(char *, char *, int *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
}

// check if file/folder exists here
inline bool location_exists(const std::string &fname)
{
    struct stat buffer;
    return (stat(fname.c_str(), &buffer) == 0);
}

inline double v_solve(double &r_dist, int &charge, double &sigma, double &k, double &q){return charge * erfc( r_dist/(sigma*sqrt(2)) )*k*q/r_dist;}

// read xyz and populate the xyz coordinate array and lattice array
int read_xyz(std::string filename, std::vector<std::string> &elements,
             std::vector<double> &x, std::vector<double> &y, std::vector<double> &z);

// returns the 3D distance between two Site objects. Note that only y-z periodicity exists!
double site_dist(std::vector<double> pos1, std::vector<double> pos2, std::vector<double> lattice, bool pbc);

// sort coordinates by x-axis value
void sort_by_x(std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, std::vector<std::string> &elements, std::vector<double> lattice);

// shifts the first atom position to 0 in the selected dims. Expects the coordinates to be sorted!
void center_coords(std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, int N, bool dim[]);

// translates the periodic boundaries of the cell
void translate_cell(std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, int N, std::vector<double> lattice, std::vector<double> shifts);

#endif
