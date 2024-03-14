# *********************************************
# *** UNCOMMENT ONE OF THE FOLLOWING BLOCKS *** 
# *********************************************

# USE TO COMPILE CPU-ONLY CODE, remember to make clean when switching
# ***************************************************
# *** IIS - Intel Compiler ***
# CXX = /usr/sepp/bin/icc-2020-af 
# MKLROOT = /usr/pack/intel_compiler-2020-af/x64/compilers_and_libraries_2019.0.117/linux/mkl
# OMPROOT = /usr/pack/intel_compiler-2020-af/x64/compilers_and_libraries_2019.0.117/linux/compiler/lib/intel64_lin
# CUDA_ROOT = /usr/local/cuda
# CXXFLAGS = -O3 -std=c++11 -I$(OMPROOT) -I${CUDA_ROOT}/include -I$(MKLROOT)/include -Wl, -liomp5 -lpthread -ldl -mkl -qopenmp -fopenmp
# ***************************************************

# USE TO COMPILE GPU CODE ON ATTELAS, remember to make clean when switching
# ***************************************************
# *** IIS - GNU C++ compiler (with C++17) + nvcc *** 
#   CXX = /usr/sepp/bin/g++ 
#   MKLROOT = /usr/pack/intel_compiler-2020-af/x64/compilers_and_libraries_2019.0.117/linux/mkl
#   OMPROOT = /usr/pack/intel_compiler-2020-af/x64/compilers_and_libraries_2019.0.117/linux/compiler/lib/intel64_lin
#   CUDA_ROOT = /usr/local/cuda
#   CXXFLAGS = -std=c++17 -O2 -m64 -DMKL_ILP64 -I${CUDA_ROOT}/include -I"${MKLROOT}/include" -fopenmp -lpthread -lm -ldl
#   CXXFLAGS += -I"${MPICH_DIR}/include"
#   NVCC = nvcc
#   NVCCFLAGS = -O2 -std=c++17 -arch=sm_60 -ccbin "/usr/sepp/bin/g++" --extended-lambda #-G -lineinfo # Last two are for the visual profiler # To use visual profiler: nvprof --export-profile profile.nvvp ./bin/runKMC parameters.txt 
#   NVCCFLAGS += -I"${MPICH_DIR}/include" 
#   LDFLAGS = -L"${CUDA_ROOT}/lib64" -L"${MPICH_DIR}/lib" -lcuda -lcudart -lcublas -lcusolver -lcusparse -lmpi
#   CXXFLAGS += -DUSE_CUDA 
#   COMPILE_WITH_CUDA = -DCUDA 
# ***************************************************

# ***************************************************
# *** IIS - MPI GNU C++ compiler (with C++17) + nvcc for CUDA+MPI support *** 
#  CXX = /usr/sepp/bin/g++ 
#  MKLROOT = /usr/pack/intel_compiler-2020-af/x64/compilers_and_libraries_2019.0.117/linux/mkl
#  OMPROOT = /usr/pack/intel_compiler-2020-af/x64/compilers_and_libraries_2019.0.117/linux/compiler/lib/intel64_lin
#  CUDA_ROOT = /usr/local/cuda
#  MPI_ROOT = /usr/ela/local/linux-local/mpich-3.4.2/gcc/
#  CXXFLAGS = -std=c++17 -O3 -m64 -DMKL_ILP64 -I${CUDA_ROOT}/include -I"${MKLROOT}/include" -I"${MPI_ROOT}/include" -fopenmp -lpthread -lm -ldl
#  NVCC = nvcc
#  NVCCFLAGS = -O3 -std=c++17 -arch=sm_60 -ccbin "/usr/sepp/bin/g++" --extended-lambda #-G -lineinfo # Last two are for the visual profiler # To use visual profiler: nvprof --export-profile profile.nvvp ./bin/runKMC parameters.txt 
#  LDFLAGS = -L"${CUDA_ROOT}/lib64" -lcuda -lcudart -lcublas -lcusolver -lcusparse -L"${MPI_ROOT}/lib" -lmpi
#  CXXFLAGS += -DUSE_CUDA 
#  COMPILE_WITH_CUDA = -DCUDA 
# ***************************************************

# USE TO COMPILE GPU CODE WITH MPI ON PIZ DAINT, remember to make clean when switching
# ***************************************************
# *** Piz Daint *** 
#NVCC = nvcc

#CUDA_DIR = /opt/nvidia/hpc_sdk/Linux_x86_64/21.5/math_libs/11.3

#CXXFLAGS =-O3 -std=c++14 -I$(OMPROOT) -I"${CUDA_HOME}/include" -I"${CUDA_DIR}/include" -I"${MKLROOT}/include" -L${MKLROOT}/lib/intel64 -I"${MPICH_DIR}/include"
#CXXFLAGS += -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -fopenmp

#NVCCFLAGS = -O3 -arch=sm_60 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -std=c++14 -Xcompiler -fopenmp
#NVCCFLAGS += -I"${CUDA_HOME}/include" -I"${CUDA_DIR}/include" -I"${MPICH_DIR}/include"
#LDFLAGS = -L"${MPICH_DIR}/lib"  -L"${CUDA_HOME}/lib64" 
#LDFLAGS += -Wl,--copy-dt-needed-entries -lcuda -lcudart -lcublas -lcusolver -lcusparse -lm -lmpich
#CXXFLAGS += -DUSE_CUDA 
#COMPILE_WITH_CUDA = -DCUDA 

# ***************************************************

# USE TO COMPILE GPU CODE WITH MPI ON LUMI, remember to make clean when switching
# ***************************************************
# *** LUMI ***
# CXX = CC 
CXX = hipcc 
CXXFLAGS = --offload-arch=gfx90a --std=c++17 -O3 -I"${MPICH_DIR}/include"  -I"/opt/rocm-5.2.3/include/" #-I"/opt/rocm/include" -I"/opt/rocm/rocprim/include/rocprim" -I"/opt/rocm/hipcub/include/hipcub/"
CXXFLAGS += -fopenmp -w #-fno-unroll-loops # added -fno-unroll-loops
CXXFLAGS += -DUSE_CUDA
LDFLAGS = -L"${MPICH_DIR}/lib" -lmpi -L"/opt/rocm-5.2.3/lib/" -lhipblas -lhipsparse -lhipsolver  -L"/opt/rocm-5.2.3/rocprim/lib"

# ***************************************************

SRCDIR = src
SRCDIR_CG = dist_iterative
OBJDIR = obj
BINDIR = bin

TARGET = $(BINDIR)/runKMC																
all: $(TARGET) 

CUFILES = $(wildcard $(SRCDIR)/*.cu)
CPPFILES = $(wildcard $(SRCDIR)/*.cpp)		
CUFILES_CG = $(wildcard $(SRCDIR_CG)/*.cu)
CPPFILES_CG = $(wildcard $(SRCDIR_CG)/*.cpp)

CU_OBJ_FILES = $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(CUFILES))
CPP_OBJ_FILES = $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(CPPFILES))
CU_OBJ_FILES_CG = $(patsubst $(SRCDIR_CG)/%.cu, $(OBJDIR)/%.o, $(CUFILES_CG))
CPP_OBJ_FILES_CG = $(patsubst $(SRCDIR_CG)/%.cpp, $(OBJDIR)/%.o, $(CPPFILES_CG))

DEPS = $(SRCDIR)/random_num.h $(SRCDIR)/input_parser.h

# $(shell mkdir -p $(OBJDIR))
# $(shell mkdir -p $(BINDIR))

# COMPILES ONLY THE CPP FILES --> CPU-only code
$(info MAKE INFO: Compile target - KMC Simulation ./bin/runKMC)
$(TARGET): $(CU_OBJ_FILES) $(CPP_OBJ_FILES) $(CU_OBJ_FILES_CG) $(CPP_OBJ_FILES_CG)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS) # $(SRCDIR)/gpu_solvers.h
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: $(SRCDIR_CG)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(DEPS) # $(SRCDIR)/gpu_solvers.h
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: $(SRCDIR_CG)/%.cu
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c -o $@ $<


# # COMPILES CU AND CPP FILES --> GPU-accelerated code
# else
# $(info MAKE INFO: Compiling with CUDA)
# $(info MAKE INFO: Compile target - KMC Simulation ./bin/runKMC)

# $(TARGET): $(CU_OBJ_FILES) $(CPP_OBJ_FILES) $(CU_OBJ_FILES_CG) $(CPP_OBJ_FILES_CG)
# 	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# $(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS) #$(SRCDIR)/gpu_solvers.h
# 	@mkdir -p $(@D)
# 	$(CXX) $(CXXFLAGS) -c -o $@ $<
# $(OBJDIR)/%.o: $(SRCDIR_CG)/%.cpp #$(SRCDIR)/gpu_solvers.h
# 	@mkdir -p $(@D)
# 	$(CXX) $(CXXFLAGS) -c -o $@ $<	

# $(OBJDIR)/%.o: $(SRCDIR)/%.cu #$(SRCDIR)/gpu_solvers.h
# 	@mkdir -p $(@D)
# 	$(NVCC) $(NVCCFLAGS) -c $< -o $@
# $(OBJDIR)/%.o: $(SRCDIR_CG)/%.cu
# 	@mkdir -p $(@D)
# 	$(NVCC) $(NVCCFLAGS) -c $< -o $@	

# endif
	
clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all clean tests
