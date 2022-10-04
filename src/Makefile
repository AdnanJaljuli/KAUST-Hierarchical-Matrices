
USE_COUNTERS=0
EXPAND_MATRIX=0 # //TODO: rename
DEFS = -DUSE_COUNTERS=$(USE_COUNTERS)
DEFS += -DEXPAND_MATRIX=$(EXPAND_MATRIX)

NVCC_FLAGS = -O3 -G -g -maxrregcount=32 -w -rdc=true -gencode arch=compute_70,code=sm_70 -std=c++11 -I/apps/sw/magma-2.5.4-gcc-7.2.0-cuda-10.1/include -I"/apps/sw/cub-1.8.0" -I/scratch/7701501-aaj35/kblas-gpu/include -L /apps/sw/cuda/cuda_10.1.168_418.67/lib64 -lcuda -lcurand -lcusolver -L /apps/sw/magma-2.5.4-gcc-7.2.0-cuda-10.1/lib/ -lmagma_sparse -lmagma /scratch/7701501-aaj35/kblas-gpu/lib/libkblas-gpu.a -L /apps/sw/openblas-0.3.13-gcc-7.2.0-dynamic-arch/lib64 -lopenblas -lcublas -L"../../thrust" -lcudadevrt
# NVCC_FLAGS = -O3 -G -g -maxrregcount=32 -w -rdc=true -gencode arch=compute_70,code=sm_70 -std=c++11 -I/apps/sw/magma-2.5.4-gcc-7.2.0-cuda-10.1/include -I/scratch/7701501-aaj35/kblas-gpu/include -L /apps/sw/cuda/cuda_10.1.168_418.67/lib64 -lcuda -lcurand -lcusolver -L /apps/sw/magma-2.5.4-gcc-7.2.0-cuda-10.1/lib/ -lmagma_sparse -lmagma /scratch/7701501-aaj35/kblas-gpu/lib/libkblas-gpu.a -L /apps/sw/openblas-0.3.13-gcc-7.2.0-dynamic-arch/lib64 -lopenblas -lcublas -L"../../thrust" -lcudadevrt

output: main.o
	nvcc $(NVCC_FLAGS) main.o -o output

main.o: main.cu
	nvcc -c $(NVCC_FLAGS) $(DEFS) main.cu

clean:
	rm -fv *.o output
