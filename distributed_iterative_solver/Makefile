CXX = g++
NVCC = nvcc
CUDA_ROOT = /usr/local/cuda
MPI_ROOT = /usr/pack/mpich-3.2.1-af/linux-x64
LDFLAGS = -L"${CUDA_ROOT}/lib64" -L"${MPI_ROOT}/lib" -lcuda -lmpi

main: main.o test.o
	$(NVCC) -o main main.o test.o $(LDFLAGS)

test.o: test.cu
	$(NVCC)  -c $< -o $@ -I ${MPI_ROOT}/include -L"${MPI_ROOT}/lib" -lcuda -lmpi

main.o: main.cpp
	$(CXX) -c $< -o $@ -I ${MPI_ROOT}/include -L"${MPI_ROOT}/lib" -lcuda -lmpi

clean:
	rm -f *.o main
