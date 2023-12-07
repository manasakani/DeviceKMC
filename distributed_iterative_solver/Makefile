CXX = g++
NVCC = nvcc
CUDA_ROOT = /usr/local/cuda
MPI_ROOT = /usr/ela/local/linux-local/mpich-3.4.2/gcc/
MKL_ROOT = /usr/pack/intel_compiler-2020-af/x64/compilers_and_libraries_2019.0.117/linux/mkl
PETSC_ROOT = /usr/scratch/mont-fort17/almaeder/libraries/petsc
PETSC_ARCH_ROOT = /usr/scratch/mont-fort17/almaeder/libraries/petsc/arch-linux-c-debug

CXXFLAGS = -Wall -Wextra -fopenmp -std=c++14 -I"${MKL_ROOT}/include" -I"${MPI_ROOT}/include" -I"${PETSC_ROOT}/include" -I"${PETSC_ARCH_ROOT}/include"
NVCCFLAGS = -arch=sm_60 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -std=c++14 -Xcompiler -fopenmp -I${CUDA_ROOT}/include -I"${MKL_ROOT}/include" -I"${PETSC_ROOT}/include"
LDFLAGS = -L"${CUDA_ROOT}/lib64" -L"${MPI_ROOT}/lib" -L"${MKL_ROOT}/lib/intel64" -L"${PETSC_ROOT}/lib" -L"${PETSC_ARCH_ROOT}/lib" -lpetsc -lcuda -lcudart -lcublas -lcusolver -lcusparse -lmkl_intel_lp64 -lmkl_gnu_thread -lm -lmkl_core -lmpi

SOURCES= main.cpp utils.cpp test.cu

CPP_SOURCES = $(filter %.cpp, $(SOURCES))
CU_SOURCES = $(filter %.cu, $(SOURCES))

CPP_OBJ_FILES=$(CPP_SOURCES:.cpp=.o)
CU_OBJ_FILES =$(CU_SOURCES:.cu=.o)
BINARY = main

.PHONY: all
all: $(BINARY)

$(BINARY): $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	$(CXX) $(CXXFLAGS) $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $@ $(LDFLAGS)


# Rule for compiling C++ source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for compiling CUDA source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@


.PHONY: clean
clean:
	rm -f $(BINARY) *.o
