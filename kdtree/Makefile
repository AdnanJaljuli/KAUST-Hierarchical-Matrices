NVCC_FLAGS= -O3 -w -G -g -gencode arch=compute_70,code=sm_70 -std=c++11 -I"cub-1.8.0" -lcusolver

output: main.o utilities.o
	nvcc $(NVCC_FLAGS) main.o utilities.o -o output

main.o: main.cu
	nvcc -c $(NVCC_FLAGS) main.cu

utilities.o: utilities.cu
	nvcc -c $(NVCC_FLAGS) utilities.cu

clean:
	rm -fv *.o output
