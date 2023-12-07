# CXX = g++
# NVCC = nvcc
# CUDA_ROOT = /usr/local/cuda
# MPI_ROOT = /usr/ela/local/linux-local/mpich-3.4.2/gcc/
# MKL_ROOT = /usr/pack/intel_compiler-2020-af/x64/compilers_and_libraries_2019.0.117/linux/mkl
# PETSC_ROOT = /usr/scratch/mont-fort17/almaeder/libraries/petsc
# PETSC_ARCH_ROOT = /usr/scratch/mont-fort17/almaeder/libraries/petsc/arch-linux-c-debug

# CXXFLAGS = -Wall -Wextra -fopenmp -std=c++14 -I"${MKL_ROOT}/include" -I"${MPI_ROOT}/include" -I"${PETSC_ROOT}/include" -I"${PETSC_ARCH_ROOT}/include"
# NVCCFLAGS = -arch=sm_60 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -std=c++14 -Xcompiler -fopenmp -I${CUDA_ROOT}/include -I"${MKL_ROOT}/include" -I"${PETSC_ROOT}/include"
# LDFLAGS = -L"${CUDA_ROOT}/lib64" -L"${MPI_ROOT}/lib" -L"${MKL_ROOT}/lib/intel64" -L"${PETSC_ROOT}/lib" -L"${PETSC_ARCH_ROOT}/lib" -lpetsc -lcuda -lcudart -lcublas -lcusolver -lcusparse -lmkl_intel_lp64 -lmkl_gnu_thread -lm -lmkl_core -lmpi



# # The following variable must either be a path to petsc.pc or just "petsc" if petsc.pc
# # has been installed to a system location or can be found in PKG_CONFIG_PATH.
# petsc.pc := $(PETSC_DIR)/$(PETSC_ARCH)/lib/pkgconfig/PETSc.pc

# # Additional libraries that support pkg-config can be added to the list of PACKAGES below.
# PACKAGES := $(petsc.pc)

# CXX = g++
# NVCC = nvcc

# CFLAGS_OTHER := $(shell pkg-config --cflags-only-other $(PACKAGES))
# CFLAGS := $(shell pkg-config --variable=cflags_extra $(PACKAGES)) $(CFLAGS_OTHER)
# CXXFLAGS := $(shell pkg-config --variable=cxxflags_extra $(PACKAGES)) $(CFLAGS_OTHER)
# CXXFLAGS += -Wall -Wextra -fopenmp -std=c++14 -I"${MPICH_DIR}/include"
# NVCCFLAGS = -arch=sm_60 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -std=c++14 -Xcompiler -fopenmp -I${CUDA_HOME}/include -I"${PETSC_DIR}/include"

# LDFLAGS := $(shell pkg-config --libs-only-L --libs-only-other $(PACKAGES))
# LDFLAGS += $(patsubst -L%, $(shell pkg-config --variable=ldflag_rpath $(PACKAGES))%, $(shell pkg-config --libs-only-L $(PACKAGES)))
# LDFLAGS += -L"${CUDA_HOME}/lib64" -L"${MPICH_DIR}/lib" -L"${MKL_ROOT}/lib/intel64"
# LDLIBS := $(shell pkg-config --libs-only-l $(PACKAGES)) -lm
# LDLIBS += -lcuda -lcudart -lcublas -lcusolver -lcusparse -lmpich


CXX = g++
NVCC = nvcc

CXXFLAGS = -Wall -Wextra -fopenmp -std=c++14 -I"${MPICH_DIR}/include" -I"${PETSC_DIR}/include"
NVCCFLAGS = -arch=sm_60 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -std=c++14 -Xcompiler -fopenmp -I${CUDA_HOME}/include -I"${PETSC_DIR}/include"
LDFLAGS = -L"${CUDA_HOME}/lib64" -L"${MPICH_DIR}/lib" -L"${PETSC_DIR}/lib" -L"/opt/cray/pe/lib64"
LDLIBS = -lcuda -lcudart -lcublas -lcusolver -lcusparse -lmpich -lcraypetsc_crayclang_real

SOURCES= main.cpp utils.cpp test.cu

CPP_SOURCES = $(filter %.cpp, $(SOURCES))
CU_SOURCES = $(filter %.cu, $(SOURCES))

CPP_OBJ_FILES=$(CPP_SOURCES:.cpp=.o)
CU_OBJ_FILES =$(CU_SOURCES:.cu=.o)
BINARY = main

.PHONY: all
all: $(BINARY)

$(BINARY): $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	$(CXX) $(CXXFLAGS) $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $@ $(LDFLAGS) $(LDLIBS)


# Rule for compiling C++ source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for compiling CUDA source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@


.PHONY: flags
flags:
	@echo CXXFLAGS=$(CXXFLAGS)
	@echo NVCCFLAGS=$(NVCCFLAGS)
	@echo LDFLAGS=$(LDFLAGS)
	@echo LDLIBS=$(LDLIBS)


.PHONY: clean
clean:
	rm -f $(BINARY) *.o
