#ifndef __BATCHED_SAMPLING__
#define __BATCHED_SAMPLING__

#include <bits/stdc++.h>

static __host__ __device__ int getMOfromXY_h(unsigned int x, unsigned int y){
    static const unsigned int B[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};
    static const unsigned int S[] = {1, 2, 4, 8};
    x = (x | (x << S[3])) & B[3];
    x = (x | (x << S[2])) & B[2];
    x = (x | (x << S[1])) & B[1];
    x = (x | (x << S[0])) & B[0];
    y = (y | (y << S[3])) & B[3];
    y = (y | (y << S[2])) & B[2];
    y = (y | (y << S[1])) & B[1];
    y = (y | (y << S[0])) & B[0];

    int z = x | (y << 1);
    return z;
}

static __host__ __device__ int IndextoMOIndex_h(int numSegments, int n){
    unsigned int i = n%numSegments;
    unsigned int j = n/numSegments;
    return getMOfromXY_h(j, i);
}

static __global__ void batchedSampling(int tileSize, int batchSize, int batchUnitSize, double** U, double** V, int* scanRanks, double* samplingVectors, int samplingVectorsWidth, double* output, double* bufferMemory) {
    unsigned int batch = blockIdx.x/batchUnitSize;
    unsigned int blockInBatch = blockIdx.x%batchUnitSize;
    if(threadIdx.x < samplingVectorsWidth) {
        output[batch*batchUnitSize*tileSize*samplingVectorsWidth + threadIdx.x*batchUnitSize*tileSize + blockInBatch*tileSize + threadIdx.y] = 0.0f;
    }
    __syncthreads();

    for(unsigned int tile = 0; tile < batchUnitSize; ++tile) {
        // TODO: think about loading U, V and scanRanks into shared memory
        // multiply V by the sampling vector and store it in buffer memory
        int MOIndex = IndextoMOIndex_h(batchUnitSize, tile*batchUnitSize + blockInBatch);
        double sum = 0;
        int rank = (blockInBatch == 0 && tile == 0) ? (scanRanks[batch*batchUnitSize*batchUnitSize]) : (scanRanks[batch*batchUnitSize*batchUnitSize + MOIndex] - scanRanks[batch*batchUnitSize*batchUnitSize + MOIndex - 1]);
        int scanRankVal = (blockInBatch == 0 && tile == 0) ? 0 : (scanRanks[batch*batchUnitSize*batchUnitSize + MOIndex - 1]);

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