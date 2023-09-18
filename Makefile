CXXFLAGS = -O3 -std=c++11 -m64 -I"${CUDA_ROOT}/include" -I"${MKLROOT}/include"  -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
LDFLAGS = -L ${CUDA_ROOT}/lib64 -lcuda -lcudart -lcublas

# If compiling with GPU
NVCC = nvcc
NVCCFLAGS = -O3
CXXFLAGS += -DUSE_CUDA

SRCDIR = src
OBJDIR = obj
BINDIR = bin

TARGET = $(BINDIR)/runKMC

CUFILES = $(wildcard $(SRCDIR)/*.cu)
CPPFILES = $(wildcard $(SRCDIR)/*.cpp)

CU_OBJ_FILES = $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(CUFILES))
CPP_OBJ_FILES = $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(CPPFILES))

DEPS = $(SRCDIR)/random_num.h $(SRCDIR)/input_parser.h

$(shell mkdir -p $(OBJDIR))
$(shell mkdir -p $(BINDIR))

all: $(TARGET)

$(TARGET): $(CU_OBJ_FILES) $(CPP_OBJ_FILES)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS) $(SRCDIR)/cuda_wrapper.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<
	
$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(SRCDIR)/cuda_wrapper.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
	
clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all clean
