
#pragma once
#include <cuda_runtime.h>
#include <cusparse.h>
#include <mpi.h>
#include "cudaerrchk.h"
#include "dist_objects.h"
#include "utils_cg.h"

namespace dspmv{


void gpu_packing(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    cusparseDnVecDescr_t &vecAp_local,
    cudaStream_t &default_stream,
    cusparseHandle_t &default_cusparseHandle);

void gpu_packing_cam(
    Distributed_matrix &A_distributed,
    Distributed_vector &p_distributed,
    cusparseDnVecDescr_t &vecAp_local,
    cudaStream_t &default_stream,
    cusparseHandle_t &default_cusparseHandle);

} // namespace dspmv