
# specify the gpu architecture (volta, ampere)
GPU_ARCH = volta
ifeq (${GPU_ARCH}, volta)
	NVOPTS = arch=compute_70,code=sm_70
else ifeq (${GPU_ARCH}, ampere)
	NVOPTS = arch=compute_80,code=sm_80
endif # TODO: What if neither are true?

_KBLAS_ROOT_?=$(KBLAS_HOME)
_MAGMA_ROOT_?=$(MAGMA_HOME)
_CUDA_ROOT_?=$(CUDA_HOME)
_OPENBLAS_ROOT_?=$(OPENBLAS_HOME)

INC = -I$(_MAGMA_ROOT_)/include
INC += -I$(_KBLAS_ROOT_)/include
INC += -I/apps/sw/cub-1.8.0

LIB_DIR = -L$(_CUDA_ROOT_)/lib64
LIB_DIR += -L$(_MAGMA_ROOT_)/lib
LIB_DIR += -L$(_OPENBLAS_ROOT_)/lib64
LIB_DIR += -L$(_KBLAS_ROOT_)/lib

LIB = -lcublas -lcudadevrt -lcuda -lcurand -lcusolver -lkblas-gpu -lmagma_sparse -lmagma -lopenblas

CC = nvcc
NVCC_FLAGS = -O3 -maxrregcount=32 -w -rdc=true -gencode $(NVOPTS) -std=c++11
DEBUG_FLAGS = -g -G -lineinfo
OBJS = main.o

USE_COUNTERS = 0
EXPAND_MATRIX = 0
DEFS = -DUSE_COUNTERS=$(USE_COUNTERS)
DEFS += -DEXPAND_MATRIX=$(EXPAND_MATRIX)

# TODO: fix the compilation so that each file is compiled independently
output: $(OBJS)
	$(CC) $(NVCC_FLAGS) $(DEBUG_FLAGS) $(LIB_DIR) $(LIB) main.o -o output

main.o: main.cu
	$(CC) -c $(NVCC_FLAGS) $(DEBUG_FLAGS) $(INC) $(DEFS) main.cu

clean:
	rm -fv *.o output *.csv
