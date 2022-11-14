
CC = nvcc

# specify the gpu architecture (volta, ampere)
GPU_ARCH = volta
ifeq (${GPU_ARCH}, volta)
	NVOPTS = arch=compute_70,code=sm_70
else ifeq (${GPU_ARCH}, ampere)
	NVOPTS = arch=compute_80,code=sm_80
endif

NVCC_FLAGS = -O3 -maxrregcount=32 -w -rdc=true -gencode $(NVOPTS) -std=c++11
DEBUG_FLAGS = -g -G -lineinfo

INC = -I/apps/sw/magma-2.5.4-gcc-7.2.0-cuda-10.1/include
INC += -I/scratch/7701501-aaj35/lr-kblas-gpu/include
INC += -I/apps/sw/cub-1.8.0

LIB_DIR = -L/apps/sw/cuda/cuda_10.1.168_418.67/lib64
LIB_DIR += -L/apps/sw/magma-2.5.4-gcc-7.2.0-cuda-10.1/lib
LIB_DIR += -L/apps/sw/openblas-0.3.13-gcc-7.2.0-dynamic-arch/lib64
LIB_DIR += -L/scratch/7701501-aaj35/lr-kblas-gpu/lib

LIB = -lcublas -lcudadevrt -lcuda -lcurand -lcusolver -lkblas-gpu -lmagma_sparse -lmagma -lopenblas

USE_COUNTERS = 0
EXPAND_MATRIX = 0
DEFS = -DUSE_COUNTERS=$(USE_COUNTERS)
DEFS += -DEXPAND_MATRIX=$(EXPAND_MATRIX)

OBJS = main.o

output: $(OBJS)
	$(CC) $(NVCC_FLAGS) $(DEBUG_FLAGS) $(LIB_DIR) $(LIB) main.o -o output

main.o: main.cu
	$(CC) -c $(NVCC_FLAGS) $(DEBUG_FLAGS) $(INC) $(DEFS) main.cu

clean:
	rm -fv *.o output *.csv
