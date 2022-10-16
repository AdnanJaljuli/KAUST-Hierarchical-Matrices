
NVCC_FLAGS = -O3 -G -g -maxrregcount=32 -w -rdc=true -gencode arch=compute_70,code=sm_70 -std=c++11 -lcuda -lcurand

output: main.o
	nvcc $(NVCC_FLAGS) main.o -o output

main.o: main.cu
	nvcc -c $(NVCC_FLAGS) main.cu

clean:
	rm -fv *.o output
