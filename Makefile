# include make.inc.gpu

NVCC_FLAGS= -O3 -maxrregcount=31 -w -gencode arch=compute_70,code=sm_70 -std=c++11 -I/apps/sw/magma-2.5.4-gcc-7.2.0-cuda-10.1/include -I/scratch/7701501-aaj35/kblas-gpu/include -L /apps/sw/cuda/cuda_10.1.168_418.67/lib64 -lcuda -lcurand -L /apps/sw/magma-2.5.4-gcc-7.2.0-cuda-10.1/lib/ -lmagma_sparse -lmagma /scratch/7701501-aaj35/kblas-gpu/lib/libkblas-gpu.a -L /apps/sw/openblas-0.3.13-gcc-7.2.0-dynamic-arch/lib64 -lopenblas -lcublas -lcusolver -I"kdtree/cub-1.8.0"

# NVCC_FLAGS= -O3 -maxrregcount=31 -w -gencode arch=compute_70,code=sm_70 -std=c++11 -I/apps/sw/magma-2.5.4-gcc-7.2.0-cuda-10.1/include -I/scratch/7701501-aaj35/kblas-gpu/include -L /apps/sw/cuda/cuda_10.1.168_418.67/lib64 -lcuda -lcurand -L /apps/sw/magma-2.5.4-gcc-7.2.0-cuda-10.1/lib/ -lmagma_sparse -lmagma /scratch/7701501-aaj35/kblas-gpu/lib/libkblas-gpu.a -L /apps/sw/openblas-0.3.13-gcc-7.2.0-dynamic-arch/lib64 -lopenblas -lcublas

#-O3 -maxrregcount=31 -w -gencode arch=compute_70,code=sm_70 -std=c++11 -I$(MAGMA_INCDIR) -I$(KBLAS_INCDIR) -lcuda -lcudart -lcublas -lcusparse -lcurand -I$(MAGMA_LIBS) -I$(KBLAS_LIBS) -I"/apps/sw/magma-2.5.4-gcc-7.2.0-cuda-10.1/lib/libmagma_sparse.so"
output: kblas-test.o
	nvcc $(NVCC_FLAGS) kblas-test.o -o output

kblas-test.o: kblas-test.cu
	nvcc -c $(NVCC_FLAGS) kblas-test.cu

clean:
	rm -fv *.o output
