NVCC_FLAGS= -O3 -w -gencode arch=compute_70,code=sm_70 -std=c++11 -I"cub-1.8.0"

output: main.o
	nvcc $(NVCC_FLAGS) main.o -o output

main.o: main.cu
	nvcc -c $(NVCC_FLAGS) main.cu

clean:
	rm -fv *.o output
