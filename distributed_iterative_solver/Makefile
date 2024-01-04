# CC = g++
# NVCC = nvcc
# CUDA_ROOT = /usr/local/cuda
# MPI_ROOT = /usr/ela/local/linux-local/mpich-3.4.2/gcc/
# MKL_ROOT = /usr/pack/intel_compiler-2020-af/x64/compilers_and_libraries_2019.0.117/linux/mkl
# PETSC_ROOT = /usr/scratch/mont-fort17/almaeder/libraries/petsc
# PETSC_ARCH_ROOT = /usr/scratch/mont-fort17/almaeder/libraries/petsc/arch-linux-c-debug

# CCFLAGS = -Wall -Wextra -fopenmp -std=c++14 -I"${MKL_ROOT}/include" -I"${MPI_ROOT}/include" -I"${PETSC_ROOT}/include" -I"${PETSC_ARCH_ROOT}/include"
# NVCCFLAGS = -arch=sm_60 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -std=c++14 -Xcompiler -fopenmp -I${CUDA_ROOT}/include -I"${MKL_ROOT}/include" -I"${PETSC_ROOT}/include"
# LDFLAGS = -L"${CUDA_ROOT}/lib64" -L"${MPI_ROOT}/lib" -L"${MKL_ROOT}/lib/intel64" -L"${PETSC_ROOT}/lib" -L"${PETSC_ARCH_ROOT}/lib" -lpetsc -lcuda -lcudart -lcublas -lcusolver -lcusparse -lmkl_intel_lp64 -lmkl_gnu_thread -lm -lmkl_core -lmpi

# /opt/cray/pe/tpsl/20.03.2/GNU/82/sandybridge/lib

NVCC = nvcc

# madman installed petsc with cuda, but i do not know with which mpi version it was compiled
# PETSC_DIR=/apps/daint/UES/6.0.UP02/sandbox-ws/petsc-maint
# PETSC_ARCH=arch-gnu-xc30-daint-cuda

CCFLAGS = -Wall -Wextra -fopenmp -std=c++14 -I"${MPICH_DIR}/include" -I"${PETSC_DIR}/${PETSC_ARCH}/include" -I"${PETSC_DIR}/include"
NVCCFLAGS = -arch=sm_60 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -std=c++14 -Xcompiler -fopenmp -I"${PETSC_DIR}/${PETSC_ARCH}/include" -I"${PETSC_DIR}/include"
LDFLAGS = -L"${MPICH_DIR}/lib" -L"${PETSC_DIR}/${PETSC_ARCH}/lib" -L"${PETSC_DIR}/lib" -L"/opt/cray/pe/lib64"
LDLIBS = -lcuda -lcudart -lcublas -lcusolver -lcusparse -lmpich -lpetsc -lm

SOURCES= main_petsc.cpp utils.cpp utils_gpu.cu petsc_implementations_to_compare.cpp

CPP_SOURCES = $(filter %.cpp, $(SOURCES))
CU_SOURCES = $(filter %.cu, $(SOURCES))

CPP_OBJ_FILES=$(CPP_SOURCES:.cpp=.o)
CU_OBJ_FILES =$(CU_SOURCES:.cu=.o)
BINARY = main

.PHONY: all
all: $(BINARY)

$(BINARY): $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	CC $(CCFLAGS) $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $@ $(LDFLAGS) $(LDLIBS)


# Rule for compiling C++ source files
%.o: %.cpp
	CC $(CCFLAGS) -c $< -o $@

# Rule for compiling CUDA source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@


.PHONY: flags
flags:
	@echo CCFLAGS=$(CCFLAGS)
	@echo NVCCFLAGS=$(NVCCFLAGS)
	@echo LDFLAGS=$(LDFLAGS)
	@echo LDLIBS=$(LDLIBS)


.PHONY: clean
clean:
	rm -f $(BINARY) *.o
