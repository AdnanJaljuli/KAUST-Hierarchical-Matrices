#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <iostream>
#include <iomanip>
#include <curand_kernel.h>

__global__ void RandomMatrixGen_kernel(float* array, unsigned int N){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<N){
        unsigned int seed = i;
        curandState s;
        curand_init(seed, 0, 0, &s);
        float random_n = curand_uniform(&s);
        array[i]=random_n*10 + 10;
    }
}