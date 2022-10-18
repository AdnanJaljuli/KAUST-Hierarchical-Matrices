#ifndef __BATCHED_SAMPLING__
#define __BATCHED_SAMPLING__

#include <bits/stdc++.h>

static __host__ __device__ int getMOIndexfromXY(unsigned int x, unsigned int y){
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

static __host__ __device__ int IndextoMOIndex(int numSegments, int n){
    unsigned int i = n%numSegments;
    unsigned int j = n/numSegments;
    return getMOIndexfromXY(j, i);
}

template <typename T>
static __global__ void batchedSampling(int tileSize, int batchSize, int batchUnitSize, T** U, T** V, int* scanRanks, T* samplingVectors, int samplingVectorsWidth, T* output, T* bufferMemory) {
    unsigned int batch = blockIdx.y;
    unsigned int blockInBatch = blockIdx.x;
    T outputValue = 0;
    T sum = 0;

    for(unsigned int tile = 0; tile < batchUnitSize; ++tile) {
        // multiply V by the sampling vector and store it in buffer
        int MOIndex = IndextoMOIndex(batchUnitSize, tile*batchUnitSize + blockInBatch);
        int rank, scanRankVal;
        if(blockInBatch == 0 && tile == 0) {
            rank = scanRanks[batch*batchUnitSize*batchUnitSize];
            scanRankVal = 0;
        }
        else {
            rank = scanRanks[batch*batchUnitSize*batchUnitSize + MOIndex] - scanRanks[batch*batchUnitSize*batchUnitSize + MOIndex - 1];
            scanRankVal = scanRanks[batch*batchUnitSize*batchUnitSize + MOIndex - 1];
        }

        sum = 0;
        if(threadIdx.y < rank) {
            // TODO: use warp shuffling: allocate multiple threadblocks per output element and let each do a part of the multiplication and then use shuffling
            // TODO: load U and V into shared memory: if k < half of 16, we can load both U and V with the same threadBlock
            for(unsigned int j = 0; j < tileSize; ++j) {
                T x = V[batch][scanRankVal*tileSize + threadIdx.y*tileSize + j];
                T y = samplingVectors[batch*batchUnitSize*tileSize*samplingVectorsWidth + threadIdx.x*batchUnitSize*tileSize + tile*tileSize + j];
                sum += x*y;
            }
            bufferMemory[batch*batchUnitSize*tileSize*samplingVectorsWidth + threadIdx.x*batchUnitSize*tileSize + blockInBatch*tileSize + threadIdx.y] = sum;
        }
        __syncthreads();

        // multiply U by the result and add it to output
        sum = 0;
        if(threadIdx.x < samplingVectorsWidth) {
            for(unsigned int j = 0; j < rank; ++j) {
                T x = U[batch][scanRankVal*tileSize + j*tileSize + threadIdx.y];
                T y = bufferMemory[batch*batchUnitSize*tileSize*samplingVectorsWidth + threadIdx.x*batchUnitSize*tileSize + blockInBatch*tileSize + j];
                sum += x*y;
            }
            outputValue += sum;
        }
        __syncthreads();
    }
    output[batch*batchUnitSize*tileSize*samplingVectorsWidth + threadIdx.x*batchUnitSize*tileSize + blockInBatch*tileSize + threadIdx.y] = outputValue;
}

#endif