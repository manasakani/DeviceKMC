
NVCC = nvcc

CCFLAGS = -O3 -g -Wall -Wextra -fopenmp -pthread -std=c++14 -I"${MPICH_DIR}/include" -I"${GINKGO_DIR}/include" -I"${HYPRE_DIR}/include" -I"${CUDA_HOME}/include" -I"${PETSC_DIR}/${PETSC_ARCH}/include"
NVCCFLAGS = -O3 -arch=sm_60 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -std=c++14 -Xcompiler -fopenmp -I"${PETSC_DIR}/${PETSC_ARCH}/include" -I"${PETSC_DIR}/include"
LDFLAGS = -L"${MPICH_DIR}/lib" -L"${HYPRE_DIR}/lib64" -L"${GINKGO_DIR}/lib64" -L"${CUDA_HOME}/lib64" -L"${PETSC_DIR}/${PETSC_ARCH}/lib"
LDLIBS = -Wl,--copy-dt-needed-entries -lcuda -lcudart -lcublas -lcusolver -lcusparse -lm -lmpi

SOURCES = main_test_cg_split.cpp
SOURCES += utils.cpp utils_gpu.cu 
OWN_DIR = ../dist_iterative
SOURCES += $(wildcard $(OWN_DIR)/*.cpp) $(wildcard $(OWN_DIR)/*.cu)

CPP_SOURCES = $(filter %.cpp, $(SOURCES))
CU_SOURCES = $(filter %.cu, $(SOURCES))

CPP_OBJ_FILES=$(CPP_SOURCES:.cpp=.o)
CU_OBJ_FILES =$(CU_SOURCES:.cu=.o)
BINARY = main




.PHONY: all
all: $(BINARY)

$(BINARY): $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	$(CC) $(CCFLAGS) $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $@ $(LDFLAGS) $(LDLIBS)


# Rule for compiling C++ source files
%.o: %.cpp
	$(CC) $(CCFLAGS) -c $< -o $@

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
	rm $(OWN_DIR)/*.o
