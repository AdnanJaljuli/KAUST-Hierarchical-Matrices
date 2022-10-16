#ifndef __BATCHED_SAMPLING__
#define __BATCHED_SAMPLING__

#include <bits/stdc++.h>

static __global__ void batchedSampling(int tileSize, int batchSize, int batchUnitSize, double** U, double** V, int* scanRanks, double* samplingVectors, int samplingVectorsWidth, double* output, double* bufferMemory) {
    unsigned int batch = blockIdx.x/batchUnitSize;
    unsigned int blockInBatch = blockIdx.x%batchUnitSize;
    if(threadIdx.x < samplingVectorsWidth) {
        output[batch*batchUnitSize*tileSize*samplingVectorsWidth + threadIdx.x*batchUnitSize*tileSize + blockInBatch*tileSize + threadIdx.y] = 0;
    }
    __syncthreads();

    for(unsigned int tile = 0; tile < batchUnitSize; ++tile) {
        int rank = (blockInBatch == 0) ? scanRanks[batch*batchUnitSize*batchUnitSize + tile*batchUnitSize + blockInBatch] : scanRanks[batch*batchUnitSize*batchUnitSize + tile*batchUnitSize + blockInBatch] - scanRanks[batch*batchUnitSize*batchUnitSize + tile*batchUnitSize + blockInBatch - 1];
        // TODO: think about loading U, V and scanRanks into shared memory
        // multiply V by the sampling vector and store it in buffer memory
        double sum = 0;
        int scanRankVal = (blockInBatch == 0) ? 0 : scanRanks[batch*batchUnitSize*batchUnitSize + tile*batchUnitSize + blockInBatch - 1];

        if(threadIdx.x < samplingVectorsWidth && threadIdx.y < rank) {
            for(unsigned int j = 0; j < tileSize; ++j) {
                double x = V[batch][scanRankVal*tileSize + threadIdx.y*tileSize + j];
                double y = samplingVectors[batch*batchUnitSize*tileSize*samplingVectorsWidth + threadIdx.x*batchUnitSize*tileSize + tile*tileSize + j];
                sum += x*y;
            }
            bufferMemory[batch*batchUnitSize*tileSize*samplingVectorsWidth + threadIdx.x*batchUnitSize*tileSize + blockInBatch*tileSize + threadIdx.y] = sum;
        }
        __syncthreads();

        // multiply U by the result and add it to output
        sum = 0;
        if(threadIdx.x < samplingVectorsWidth) {
            for(unsigned int j = 0; j < rank; ++j) {
                double x = U[batch][scanRankVal*tileSize + j*tileSize + threadIdx.y];
                double y = bufferMemory[batch*batchUnitSize*tileSize*samplingVectorsWidth + threadIdx.x*batchUnitSize*tileSize + blockInBatch*tileSize + j];
                sum += x*y;
            }
            output[batch*batchUnitSize*tileSize*samplingVectorsWidth + threadIdx.x*batchUnitSize*tileSize + blockInBatch*tileSize + threadIdx.y] += sum;
        }
        __syncthreads();
    }
}

#endif