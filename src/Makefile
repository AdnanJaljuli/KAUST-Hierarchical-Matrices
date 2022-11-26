
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
OBJ = main.o kDTree.o kDTreeConstruction.o kDTreeHelpers.o TLRMatrix.o TLRMatrixHelpers.o HMatrix.o HMatrixHelpers.o generateDataset.o helperFunctions.o constructTLRMatrix.o constructHMatrixFromStruct.o expandMatrix.o HMatrixVectorMultiplication.o HMatrixVectorMultHelpers.o
HDR = config.h counters.h HMatrix.cuh HMatrixHelpers.cuh constructHMatrixFromStruct.cuh TLRMatrix.cuh TLRMatrixHelpers.cuh constructTLRMatrix.cuh expandMatrix.cuh generateDataset.cuh kDTree.cuh kDTreeHelpers.cuh kDTreeConstruction.cuh helperFunctions.cuh helperKernels.cuh HMatrixVectorMultiplication.cuh HMatrixVectorMultHelpers.cuh
EXE = output

USE_COUNTERS = 0
EXPAND_MATRIX = 0
DEFS = -DUSE_COUNTERS=$(USE_COUNTERS)
DEFS += -DEXPAND_MATRIX=$(EXPAND_MATRIX)

%.o: %.cu $(HDR)
	$(CC) -c $(NVCC_FLAGS) $(DEBUG_FLAGS) $(INC) $(DEFS) -o $@ $<

$(EXE): $(OBJ)
	$(CC) $(NVCC_FLAGS) $(DEBUG_FLAGS) $(LIB_DIR) $(LIB) $(OBJ) -o $(EXE)

clean:
	rm -fv $(OBJ) $(EXE) *.csv
