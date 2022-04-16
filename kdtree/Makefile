NVCC_FLAGS= -O3 -w -gencode arch=compute_70,code=sm_70 -std=c++11 -I"cub-1.8.0"

output: test.o
	nvcc $(NVCC_FLAGS) test.o -o output

test.o: test.cu
	nvcc -c $(NVCC_FLAGS) test.cu

clean:
	rm -fv *.o output
