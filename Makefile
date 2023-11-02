# ***************************************************
# *** IIS - Intel Compiler -- USE TO COMPILE CPU CODE
# CXX = /usr/sepp/bin/icc-2020-af 
# MKLROOT = /usr/pack/intel_compiler-2020-af/x64/compilers_and_libraries_2019.0.117/linux/mkl
# OMPROOT = /usr/pack/intel_compiler-2020-af/x64/compilers_and_libraries_2019.0.117/linux/compiler/lib/intel64_lin
# CUDA_ROOT = /usr/local/cuda
# CXXFLAGS = -O3 -std=c++11 -I$(OMPROOT) -I${CUDA_ROOT}/include -I$(MKLROOT)/include -Wl, -liomp5 -lpthread -ldl -mkl -qopenmp -fopenmp
# ***************************************************

# ***************************************************
# *** IIS - GNU C++ compiler (with C++17) + nvcc *** -- USE TO COMPILE GPU CODE
CXX = /usr/sepp/bin/g++ 
MKLROOT = /usr/pack/intel_compiler-2020-af/x64/compilers_and_libraries_2019.0.117/linux/mkl
OMPROOT = /usr/pack/intel_compiler-2020-af/x64/compilers_and_libraries_2019.0.117/linux/compiler/lib/intel64_lin
CUDA_ROOT = /usr/local/cuda
CXXFLAGS = -std=c++17 -O3 -m64 -DMKL_ILP64 -I${CUDA_ROOT}/include -I"${MKLROOT}/include" -fopenmp -lpthread -lm -ldl
NVCC = nvcc
NVCCFLAGS = -O3 -std=c++17 -arch=sm_60 -ccbin "/usr/sepp/bin/g++" --extended-lambda #-G -lineinfo # Last two are for the visual profiler. To use visual profiler: nvprof --export-profile profile.nvvp ./bin/runKMC parameters.txt 
LDFLAGS = -L"${CUDA_ROOT}/lib64" -lcuda -lcudart -lcublas -lcusolver -lcusparse
CXXFLAGS += -DUSE_CUDA # uncomment to compile cuda code
COMPILE_WITH_CUDA = -DCUDA # uncomment to compile cuda code
# ***************************************************

# ***************************************************
# *** Piz Daint *** 
# CXX = gcc
# CXXFLAGS =-O3 -std=c++11 -I$(OMPROOT) -I"${CUDA_ROOT}/include" -I"${MKLROOT}/include" -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
# NVCC = nvcc
# NVCCFLAGS = -O3 -arch=sm_60 -ccbin "/usr/sepp/bin/g++" #-G -lineinfo
# ***************************************************

# ************************************
# *** DO NOT CHANGE ANYTHING BELOW *** 
# ************************************

SRCDIR = src
OBJDIR = obj
BINDIR = bin

TARGET = $(BINDIR)/runKMC
all: $(TARGET)

CUFILES = $(wildcard $(SRCDIR)/*.cu)
CPPFILES = $(wildcard $(SRCDIR)/*.cpp)
CU_OBJ_FILES = $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(CUFILES))
CPP_OBJ_FILES = $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(CPPFILES))
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SOURCES))

DEPS = $(SRCDIR)/random_num.h $(SRCDIR)/input_parser.h

$(shell mkdir -p $(OBJDIR))
$(shell mkdir -p $(BINDIR))

# COMPILES CU AND CPP FILES
ifeq ($(COMPILE_WITH_CUDA), )
$(info MAKE INFO: Compiling without CUDA)

$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $^
	
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# COMPILES ONLY THE CPP FILES
else
$(info MAKE INFO: Compiling with CUDA)

$(TARGET): $(CU_OBJ_FILES) $(CPP_OBJ_FILES)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
		
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS) $(SRCDIR)/cuda_wrapper.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<
		
$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(SRCDIR)/cuda_wrapper.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

endif
	
clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all clean
