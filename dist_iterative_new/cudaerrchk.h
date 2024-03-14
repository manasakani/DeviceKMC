// Copyright 2023 under ETH Zurich DPHPC project course. All rights reserved.
#pragma once
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <hipsolver.h>
#include <hipblas.h>
#include <hipsparse.h>
#include <cstdio>

#define cudaErrchk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(hipError_t code, const char *file, int line, bool abort=true)
{
   if (code != hipSuccess) 
   {
      std::printf("CUDAassert: %s %s %d\n", hipGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#define cusolverErrchk(ans) { cusolverAssert((ans), __FILE__, __LINE__); }
inline void cusolverAssert(hipsolverStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != HIPSOLVER_STATUS_SUCCESS) 
   {
        //Did not find a counter part to hipGetErrorString in cusolver
        std::printf("CUSOLVERassert: %s %s %d\n", hipGetErrorString((hipError_t)code), file, line);
        if (abort) exit(code);
   }
}


#define cublasErrchk(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(hipblasStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != HIPBLAS_STATUS_SUCCESS) 
   {
        //Did not find a counter part to hipGetErrorString in cublas
        std::printf("CUBLASassert: %s %s %d\n", hipGetErrorString((hipError_t)code), file, line);
        if (abort) exit(code);
   }
}

#define cusparseErrchk(ans) { cusparseAssert((ans), __FILE__, __LINE__); }
inline void cusparseAssert(hipsparseStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != HIPSPARSE_STATUS_SUCCESS) 
   {
        //Did not find a counter part to hipGetErrorString in cusolver
        fprintf(stderr,"CUSPARSEassert: %s %s %d\n", hipGetErrorString((hipError_t)code), file, line);
        if (abort) exit(code);
   }
}
