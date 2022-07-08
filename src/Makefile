
# NVCC_FLAGS= -O3 -maxrregcount=31 -w -gencode arch=compute_70,code=sm_70 -std=c++11 -I/apps/sw/magma-2.5.4-gcc-7.2.0-cuda-10.1/include -I/scratch/7701501-aaj35/kblas-gpu/include -L /apps/sw/cuda/cuda_10.1.168_418.67/lib64 -lcuda -lcurand -lcusolver -L /apps/sw/magma-2.5.4-gcc-7.2.0-cuda-10.1/lib/ -lmagma_sparse -lmagma /scratch/7701501-aaj35/kblas-gpu/lib/libkblas-gpu.a -L /apps/sw/openblas-0.3.13-gcc-7.2.0-dynamic-arch/lib64 -lopenblas -lcublas -I"/apps/sw/cub-1.8.0" -L"../../thrust"
NVCC_FLAGS= -O3 -G -g -maxrregcount=31 -w -gencode arch=compute_70,code=sm_70 -std=c++11 -I/apps/sw/magma-2.5.4-gcc-7.2.0-cuda-10.1/include -I/scratch/7701501-aaj35/kblas-gpu/include -L /apps/sw/cuda/cuda_10.1.168_418.67/lib64 -lcuda -lcurand -lcusolver -L /apps/sw/magma-2.5.4-gcc-7.2.0-cuda-10.1/lib/ -lmagma_sparse -lmagma /scratch/7701501-aaj35/kblas-gpu/lib/libkblas-gpu.a -L /apps/sw/openblas-0.3.13-gcc-7.2.0-dynamic-arch/lib64 -lopenblas -lcublas -I"/apps/sw/cub-1.8.0" -L"../../thrust"

# NVCC_FLAGS= -O3 -maxrregcount=31 -w -gencode arch=compute_70,code=sm_70 -std=c++11 -I"cub-1.8.0" -lcusolver -lcublas

output: main.o utilities.o
	nvcc $(NVCC_FLAGS) main.o utilities.o -o output

main.o: main.cu
	nvcc -c $(NVCC_FLAGS) main.cu

utilities.o: utilities.cu
	nvcc -c $(NVCC_FLAGS) utilities.cu

clean:
	rm -fv *.o output
