#include <iostream>
#include <string>
#include "utils.h"
#include <mpi.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_runtime.h>
#include "utils_gpu.h"
#include <cublas_v2.h>

#include <ginkgo/ginkgo.hpp>

using ValueType = double;
using RealValueType = gko::remove_complex<ValueType>;
using IndexType = int;
using vec = gko::matrix::Dense<ValueType>;
using real_vec = gko::matrix::Dense<RealValueType>;
using mcsr = gko::matrix::Csr<ValueType, IndexType>;
using cg = gko::solver::Cg<ValueType>;
using mtx = gko::matrix::Csr<ValueType, IndexType>;

int main(int argc, char **argv) {
    // older version of petsc on daint
    // replace by PetscCall()
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    std::cout << "provided " << provided << std::endl;

    std::cout << gko::version_info::get() << std::endl;

    auto exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());

    // IndexType indices[] = {0, 1, 2, 3};
    // IndexType indptr[] = {0, 1, 2, 3, 4};
    // ValueType values[] = {1.0, 2.0, 3.0, 4.0};
    // ValueType rhs[] = {1.0, 2.0, 3.0, 4.0};
    // ValueType x[] = {0.0, 0.0, 0.0, 0.0};

    // IndexType *indices_d;
    // IndexType *indptr_d;
    // ValueType *values_d;
    // ValueType *rhs_d;
    // ValueType *x_d;

    // cudaMalloc((void **)&indices_d, 4 * sizeof(IndexType));
    // cudaMalloc((void **)&indptr_d, 5 * sizeof(IndexType));
    // cudaMalloc((void **)&values_d, 4 * sizeof(ValueType));
    // cudaMalloc((void **)&rhs_d, 4 * sizeof(ValueType));
    // cudaMalloc((void **)&x_d, 4 * sizeof(ValueType));

    // cudaMemcpy(indices_d, indices, 4 * sizeof(IndexType), cudaMemcpyHostToDevice);
    // cudaMemcpy(indptr_d, indptr, 5 * sizeof(IndexType), cudaMemcpyHostToDevice);
    // cudaMemcpy(values_d, values, 4 * sizeof(ValueType), cudaMemcpyHostToDevice);
    // cudaMemcpy(rhs_d, rhs, 4 * sizeof(ValueType), cudaMemcpyHostToDevice);
    // cudaMemcpy(x_d, x, 4 * sizeof(ValueType), cudaMemcpyHostToDevice);

    // create sparse csr matrix
    // const gko::dim<2> dimension = gko::dim<2>(4, 4);
    // std::unique_ptr<const gko::matrix::Csr<ValueType, IndexType>> matrix =  mcsr::create_const(exec, dimension,
    //                                 gko::detail::const_array_view<ValueType>(exec, 4, values),
    //                                 gko::detail::const_array_view<IndexType>(exec, 4, indices),
    //                                 gko::detail::const_array_view<IndexType>(exec, 5, indptr));

    // // create rhs vector
    // std::unique_ptr<const gko::matrix::Dense<ValueType>> b_vec = vec::create_const(exec, gko::dim<2>(4, 1),
    //                                 gko::detail::const_array_view<ValueType>(exec, 4, rhs), 1);
    // std::unique_ptr<gko::matrix::Dense<ValueType>> x_vec = vec::create(exec, gko::dim<2>(4, 1),
    //                                 gko::array<>(exec, 4, x, std::default_delete<double>), 1);

    // auto x_vec2 = gko::matrix::Dense<ValueType>(exec, gko::dim<2>(4, 1), gko::array<>(exec, 4, x, std::default_delete<double>), 1);
    // auto x_vec = vec::create(exec, gko::dim<2>(4, 1));


    auto A = gko::share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    auto b = gko::read<vec>(std::ifstream("data/b.mtx"), exec);
    auto x = gko::read<vec>(std::ifstream("data/x0.mtx"), exec);

    auto solver_gen =
        cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(20u),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_reduction_factor(1e-15))
            .on(exec);
    // Solve system
    auto solver = solver_gen->generate(A);
    solver->apply(b, x);


    gko::write(std::cout, x);
    // cudaFree(indices_d);
    // cudaFree(indptr_d);
    // cudaFree(values_d);
    // cudaFree(rhs_d);
    // cudaFree(x_d);

    MPI_Finalize();


    return 0;
}
