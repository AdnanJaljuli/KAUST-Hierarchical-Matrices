
# specify the gpu architecture (volta, ampere)
GPU_ARCH = ampere
ifeq (${GPU_ARCH}, volta)
	NVOPTS = arch=compute_70,code=sm_70
else ifeq (${GPU_ARCH}, ampere)
	NVOPTS = arch=compute_80,code=sm_80
else
$(error $(GPU_ARCH) GPU architecture is not supported. Supported architectures are volta and ampere)
endif

_KBLAS_ROOT_?=$(KBLAS_HOME)
_MAGMA_ROOT_?=$(MAGMA_HOME)
_CUDA_ROOT_?=$(CUDA_HOME)
_OPENBLAS_ROOT_?=$(OPENBLAS_HOME)

INC = -I$(_MAGMA_ROOT_)/include
INC += -I$(_KBLAS_ROOT_)/include
INC += -I$(CUB_HOME)

LIB_DIR = -L$(_CUDA_ROOT_)/lib64
LIB_DIR += -L$(_MAGMA_ROOT_)/lib
LIB_DIR += -L$(_OPENBLAS_ROOT_)/lib64
LIB_DIR += -L$(_KBLAS_ROOT_)/lib

LIB = -lcublas -lcudadevrt -lcuda -lcurand -lcusolver -lkblas-gpu -lmagma_sparse -lmagma -lopenblas

CC = nvcc
NVCC_FLAGS = -O3 -maxrregcount=32 -w -rdc=true -gencode $(NVOPTS) -std=c++11
DEBUG_FLAGS = -g -G -lineinfo
OBJ = main.o kDTree.o kDTreeHelpers.o TLRMatrix.o
EXE = output

USE_COUNTERS = 0
EXPAND_MATRIX = 0
DEFS = -DUSE_COUNTERS=$(USE_COUNTERS)
DEFS += -DEXPAND_MATRIX=$(EXPAND_MATRIX)

# TODO: fix the compilation so that each file is compiled independently
output: $(OBJ)
	$(CC) $(NVCC_FLAGS) $(DEBUG_FLAGS) $(LIB_DIR) $(LIB) $(OBJ) -o $(EXE)

main.o: main.cu kDTree.cuh config.h counters.h
	$(CC) -c $(NVCC_FLAGS) $(DEBUG_FLAGS) $(INC) $(DEFS) main.cu

kDTree.o: kDTree.cu kDTree.cuh
	$(CC) -c $(NVCC_FLAGS) $(DEBUG_FLAGS) $(INC) $(DEFS) kDTree.cu

kDTreeHelpers.o: kDTreeHelpers.cu kDTreeHelpers.cuh
	$(CC) -c $(NVCC_FLAGS) $(DEBUG_FLAGS) $(INC) $(DEFS) kDTreeHelpers.cu

TLRMatrix.o: TLRMatrix.cu TLRMatrix.cuh
	$(CC) -c $(NVCC_FLAGS) $(DEBUG_FLAGS) $(INC) $(DEFS) TLRMatrix.cu

clean:
	rm -fv *.o output *.csv
