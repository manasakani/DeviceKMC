//*****************
// Utility functions
//*****************

#include "utils.h"
// #include <cuda_runtime.h>
// #include <cublas_v2.h>

int read_xyz(std::string filename, std::vector<std::string> &elements,
             std::vector<double> &x, std::vector<double> &y, std::vector<double> &z)
{
    int N;
    std::string line, temp;
    std::ifstream xyz(filename);
    std::getline(xyz, line);
    std::istringstream iss1(line);
    iss1 >> N;
    std::getline(xyz, line);

    double x_, y_, z_;
    std::string element_;
    for (int i = 0; i < N; i++)
    {
        getline(xyz, line);
        std::istringstream iss(line);
        iss >> element_ >> x_ >> y_ >> z_;
        elements.push_back(element_);
        x.push_back(x_);
        y.push_back(y_);
        z.push_back(z_);
    }
    xyz.close();
    return N;
}

double site_dist(std::vector<double> pos1, std::vector<double> pos2, std::vector<double> lattice, bool pbc)
{

    double dist = 0;
    double dist_xyz[3];

    if (pbc == 1)
    {
        double dist_frac[3] = {pos1[0] / lattice[0] - pos2[0] / lattice[0],
                               pos1[1] / lattice[1] - pos2[1] / lattice[1],
                               pos1[2] / lattice[2] - pos2[2] / lattice[2]};

        dist_frac[1] -= int(dist_frac[1] + 0.5);
        dist_frac[2] -= int(dist_frac[2] + 0.5);

        dist_xyz[1] = dist_frac[1] * lattice[1];
        dist_xyz[2] = dist_frac[2] * lattice[2];

        dist = sqrt(pow(pos2[0] - pos1[0], 2) + pow(dist_xyz[1], 2) + pow(dist_xyz[2], 2));
    }
    else
    {
        dist = sqrt(pow(pos2[0] - pos1[0], 2) + pow(pos2[1] - pos1[1], 2) + pow(pos2[2] - pos1[2], 2));
    }

    return dist;
}

void sort_by_x(std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, std::vector<std::string> &elements, std::vector<double> lattice)
{

    const std::size_t size = x.size();
    std::vector<std::size_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0);

    auto cmp = [&x](std::size_t i, std::size_t j)
    {
        return x[i] < x[j];
    };

    std::sort(indices.begin(), indices.end(), cmp);
    std::vector<double> x_sorted(size);
    std::vector<double> y_sorted(size);
    std::vector<double> z_sorted(size);
    std::vector<std::string> elements_sorted(size);

    for (std::size_t i = 0; i < size; ++i)
    {
        const std::size_t index = indices[i];
        x_sorted[i] = x[index];
        y_sorted[i] = y[index];
        z_sorted[i] = z[index];
        elements_sorted[i] = elements[index];
    }

    x = std::move(x_sorted);
    y = std::move(y_sorted);
    z = std::move(z_sorted);
    elements = std::move(elements_sorted);
}

void center_coords(std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, int N, bool dim[])
{
    double min_x = *min_element(x.begin(), x.end()); // x[0];
    double min_y = *min_element(y.begin(), y.end()); // y[0];
    double min_z = *min_element(z.begin(), z.end()); // z[0];

    for (int i = 0; i < N; i++)
    {
        if (dim[0])
            x[i] -= min_x;
        if (dim[1])
            y[i] -= min_y;
        if (dim[2])
            z[i] -= min_z;
    }
}

void translate_cell(std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, int N, std::vector<double> lattice, std::vector<double> shifts)
{

    bool dims[3];
    dims[0] = (shifts[0] == 0.0) ? 0 : 1;
    dims[1] = (shifts[1] == 0.0) ? 0 : 1;
    dims[2] = (shifts[2] == 0.0) ? 0 : 1;

    print("Shifting unit cell by: " << dims[0] * shifts[0] << "x, " << dims[1] * shifts[1] << "y, " << dims[2] * shifts[2] << "z");
    center_coords(x, y, z, N, dims);

    double cut_x = lattice[0] * shifts[0];
    double cut_y = lattice[1] * shifts[1];
    double cut_z = lattice[2] * shifts[2];

    for (int i = 0; i < N; i++)
    {
        if (dims[0] && x[i] < cut_x)
        {
            x[i] += lattice[0];
        }
        if (dims[1] && y[i] < cut_y)
        {
            y[i] += lattice[1];
        }
        if (dims[2] && z[i] < cut_z)
        {
            z[i] += lattice[2];
        }
    }

    center_coords(x, y, z, N, dims);
}

// *****************************************************************
// *** CuBLAS calls ***
// *****************************************************************

void CheckCublasError(cublasStatus_t const& status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cuBLAS failed with error code: " +
                             std::to_string(status));
  }
}

cublasHandle_t CreateCublasHandle(int device) {
  if (device >= 0) {
    if (cudaSetDevice(device) != cudaSuccess) {
      throw std::runtime_error("Failed to set CUDA device.");
    }
  }
  cublasHandle_t handle;
  CheckCublasError(cublasCreate(&handle));
  return handle;
}

void gemm(cublasHandle_t handle, char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc) {

    // printf("Executing GEMM ...\n");
// {
//         double * gpu_A;
//         DACE_GPU_CHECK(cudaMalloc((void**)&gpu_A, (M * N) * sizeof(double)));
//         double * gpu_x;
//         DACE_GPU_CHECK(cudaMalloc((void**)&gpu_x, (N * N) * sizeof(double)));
//         double * gpu___return;
//         DACE_GPU_CHECK(cudaMalloc((void**)&gpu___return, M * sizeof(double)));

//         DACE_GPU_CHECK(cudaMemcpyAsync(gpu_A, A, (M * N) * sizeof(double), cudaMemcpyHostToDevice, __state->gpu_context->streams[0]));
//         DACE_GPU_CHECK(cudaMemcpyAsync(gpu_x, x, (N * N) * sizeof(double), cudaMemcpyHostToDevice, __state->gpu_context->streams[1]));

//         DACE_GPU_CHECK(cudaEventRecord(__state->gpu_context->events[0], __state->gpu_context->streams[1]));
//         DACE_GPU_CHECK(cudaStreamWaitEvent(__state->gpu_context->streams[0], __state->gpu_context->events[0], 0));

//         {
//             double * _A = &gpu_A[0];
//             double * _x = &gpu_x[1];
//             double* _y = gpu___return;

//             ///////////////////
//             int __dace_current_stream_id = 0;
//             cudaStream_t __dace_current_stream = __state->gpu_context->streams[__dace_current_stream_id];
//             const int __dace_cuda_device = 0;
//             cublasHandle_t &__dace_cublas_handle = __state->cublas_handle.Get(__dace_cuda_device);
//             cublasSetStream(__dace_cublas_handle, __dace_current_stream);

//             cublasDgemv(__dace_cublas_handle, CUBLAS_OP_T, N, M, __state->cublas_handle.Constants(__dace_cuda_device).DoublePone(), _A, N,
//             _x, N, __state->cublas_handle.Constants(__dace_cuda_device).DoubleZero(), _y, 1);

//             ///////////////////

//         }
//         DACE_GPU_CHECK(cudaMemcpyAsync(__return, gpu___return, M * sizeof(double), cudaMemcpyDeviceToHost, __state->gpu_context->streams[0]));
//         DACE_GPU_CHECK(cudaStreamSynchronize(__state->gpu_context->streams[0]));

//         DACE_GPU_CHECK(cudaFree(gpu_A));
//         DACE_GPU_CHECK(cudaFree(gpu_x));
//         DACE_GPU_CHECK(cudaFree(gpu___return));

//     }

#ifdef USE_CUDA

    printf("Executing GEMM on GPU ...\n");

    double *gpu_A, *gpu_B, *gpu_C, *gpu_alpha, *gpu_beta;
    cudaMalloc((void**)&gpu_A, ((*m) * (*k)) * sizeof(double));
    cudaMalloc((void**)&gpu_B, ((*k) * (*n)) * sizeof(double));
    cudaMalloc((void**)&gpu_C, ((*m) * (*n)) * sizeof(double));
    cudaMalloc((void**)&gpu_alpha, sizeof(double));
    cudaMalloc((void**)&gpu_beta, sizeof(double));

    cudaMemcpy(gpu_A, A, ((*m) * (*k)) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, B, ((*k) * (*n)) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_C, C, ((*m) * (*n)) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_alpha, alpha, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_beta, beta, sizeof(double), cudaMemcpyHostToDevice);

    // auto handle = CreateCublasHandle(0);
    // cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    //printf("M, N, K, alpha, beta = %d, %d, %d, %lf, %lf\n", *m, *n, *k, *alpha, *beta);
    //printf("lda, ldb, ldc = %d, %d, %d\n", *lda, *ldb, *ldc);

    // CheckCublasError(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, *m, *n, *k, gpu_alpha, A, *lda, B, *ldb, gpu_beta, C, *ldc));
    CheckCublasError(cublasDgemv(handle, CUBLAS_OP_N, *m, *k, gpu_alpha, gpu_A, *lda, gpu_B, 1, gpu_beta, gpu_C, 1));
    // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, *m, *n, *k, gpu_alpha, A, *k, B, *n, gpu_beta, C, *n);
    cudaDeviceSynchronize();

    printf("Done\n");

    // CheckCublasError(cublasDestroy(handle));

    cudaMemcpy(C, gpu_C, ((*m) * (*n)) * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
    cudaFree(gpu_alpha);
    cudaFree(gpu_beta);

#elif
    printf("Executing GEMM on CPU ...\n");
    dgemm_(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif

}

// *****************************************************************
// *** CuSolver calls ***
// *****************************************************************

cusolverDnHandle_t CreateCusolverDnHandle(int device) {
  if (cudaSetDevice(device) != cudaSuccess) {
    throw std::runtime_error("Failed to set CUDA device.");
  }
  cusolverDnHandle_t handle;
  CheckCusolverDnError(cusolverDnCreate(&handle));
  return handle;
}

static void CheckCusolverDnError(cusolverStatus_t const& status) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error("cuSOLVER failed with error code: " +
                             std::to_string(status));
  }
}

void gesv(cusolverDnHandle_t handle, int *N, int *nrhs, double *A, int *lda, int *ipiv, double *B, int *ldb, int *info) {

#ifdef USE_CUDA
    // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/getrf/cusolver_getrf_example.cu
    printf("Executing linear system solver on GPU ...\n");

    int lwork = 0;                /* size of workspace */
    double *gpu_work = nullptr;   /* device workspace for getrf */
    int *gpu_info = nullptr;      /* error info */
    double *gpu_A, *gpu_B;
    int *gpu_ipiv;

    cudaMalloc((void**)&gpu_A, ((*N) * (*N)) * sizeof(double));
    cudaMalloc((void**)&gpu_B, ((*N) * (*nrhs)) * sizeof(double));
    cudaMalloc((void**)&gpu_ipiv, (*N) * sizeof(int));
    cudaMalloc((void **)(&gpu_info), sizeof(int));
    //cudaMalloc(reinterpret_cast<void **>(&gpu_info), sizeof(int));

    cudaMemcpy(gpu_A, A, ((*N) * (*N)) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, B, ((*N) * (*nrhs)) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_ipiv, ipiv, (*N) * sizeof(int), cudaMemcpyHostToDevice);

    CheckCusolverDnError(cusolverDnDgetrf_bufferSize(handle, *N, *N, gpu_A, *lda, &lwork));
    cudaMalloc((void **)(&gpu_work), sizeof(double) * lwork);
    //cudaMalloc(reinterpret_cast<void **>(&gpu_work), sizeof(double) * lwork);

    // Solve Ax=B through LU factorization
    CheckCusolverDnError(cusolverDnDgetrf(handle, *N, *N, gpu_A, *lda, gpu_work, gpu_ipiv, gpu_info));
    //cudaMemcpy(&info, gpu_info, sizeof(int), cudaMemcpyDeviceToHost);
    //printf("info for cusolverDnDgetrf: %i \n", info);
    cudaDeviceSynchronize();

    CheckCusolverDnError(cusolverDnDgetrs(handle, CUBLAS_OP_N, *N, *nrhs, gpu_A, *lda, gpu_ipiv, gpu_B, *ldb, gpu_info));
    //cudaMemcpy(&info, gpu_info, sizeof(int), cudaMemcpyDeviceToHost);
    //printf("info for cusolverDnDgetrs: %i \n", info);
    cudaDeviceSynchronize();
    // Result is in B
    cudaMemcpy(B, gpu_B, ((*N) * (*nrhs)) * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_ipiv);
    cudaFree(gpu_work);
    cudaFree(gpu_info);

#else

    dgesv_(N, nrhs, A, lda, ipiv, B, ldb, info);
    print(*info);

#endif

}

