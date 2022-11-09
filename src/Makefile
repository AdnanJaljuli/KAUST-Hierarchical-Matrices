
CC = nvcc

# specify the gpu architecture (volta, ampere)
GPU_ARCH = ampere
ifeq (${GPU_ARCH}, volta)
	NVOPTS = arch=compute_70,code=sm_70
else ifeq (${GPU_ARCH}, ampere)
	NVOPTS = arch=compute_80,code=sm_80
endif

NVCC_FLAGS = -O3 -maxrregcount=32 -w -rdc=true -gencode $(NVOPTS) -std=c++11
DEBUG_FLAGS = -g -G -lineinfo

INC = -I/opt/ecrc/spack-modules/spack/opt/spack/linux-ubuntu18.04-sandybridge/gcc-7.5.0/magma-2.6.1-q7xz6egxrpucofghryjerq743sv2oeff/include
INC += -I/home/jaljula/kblas-gpu/include

LINK = -L/sw/workstations/apps/linux-ubuntu18.04-ivybridge/cuda/11.5.0/gcc-7.5.0/syen6pj6ss3cw66zlj4wkfhtixh5i4ei/lib64
LINK += -L/opt/ecrc/spack-modules/spack/opt/spack/linux-ubuntu18.04-sandybridge/gcc-7.5.0/magma-2.6.1-q7xz6egxrpucofghryjerq743sv2oeff/lib
LINK += -L/sw/workstations/apps/linux-ubuntu18.04-ivybridge/openblas/0.3.12/gcc-7.5.0/t2ymetzabwriavmrtcwpsc5vd77xe4bg/lib
LINK += -L/home/jaljula/kblas-gpu/lib

LIB = -lcublas -lcudadevrt -lcuda -lcurand -lcusolver  -lkblas-gpu -lmagma_sparse -lmagma -lopenblas

OBJS = main.o

USE_COUNTERS = 0
EXPAND_MATRIX = 0
DEFS = -DUSE_COUNTERS = $(USE_COUNTERS)
DEFS += -DEXPAND_MATRIX = $(EXPAND_MATRIX)

output: $(OBJS)
	$(CC) $(NVCC_FLAGS) $(DEBUG_FLAGS) $(LINK) $(LIB) main.o -o output

main.o: main.cu
	$(CC) -c $(NVCC_FLAGS) $(DEBUG_FLAGS) $(INC) main.cu

clean:
	rm -fv *.o output *.csv
