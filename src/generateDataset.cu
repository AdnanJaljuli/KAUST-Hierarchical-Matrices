
#include "generateDataset.cuh"
#include <curand_kernel.h>

__global__ void generateDataset_kernel(int numberOfInputPoints, int dimensionOfInputPoints, H2Opus_Real* pointCloud) {
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < numberOfInputPoints*dimensionOfInputPoints) {
        unsigned int seed = i;
        curandState s;
        curand_init(seed, 0, 0, &s);
        pointCloud[i] = curand_uniform(&s);
    }
}

void generateDataset(int numberOfInputPoints, int dimensionOfInputPoints, H2Opus_Real* d_pointCloud) {
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (numberOfInputPoints*dimensionOfInputPoints + numThreadsPerBlock - 1)/numThreadsPerBlock;
    generateDataset_kernel <<< numBlocks, numThreadsPerBlock >>> (numberOfInputPoints, dimensionOfInputPoints, d_pointCloud);
}
