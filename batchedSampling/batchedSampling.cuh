#ifndef __BATCHED_SAMPLING__
#define __BATCHED_SAMPLING__

static __global__ void batchedSampling(int tileSize, int batchSize, int batchUnitSize, double** U, double** V, int* scanRanks, double* samplingVectors, int samplingVectorsWidth, double* output, double* bufferMemory) {
    unsigned int batch = blockIdx.x/batchSize;
    unsigned int blockInBatch = blockIdx.x%batchSize;
    int rank = (blockInBatch == 0) ? scanRanks[batch][tile*batchUnitSize + blockInBatch] : scanRanks[batch][tile*batchUnitSize + blockInBatch] - scanRanks[batch][tile*batchUnitSize + blockInBatch - 1];
    output[batch*batchUnitSize*tileSize*samplingVectorsWidth + threadIdx.x*batchUnitSize*tileSize + blockInBatch*tileSize + threadIdx.y] = 0;

    for(unsigned int tile = 0; tile < batchUnitSize; ++tile) {
        // TODO: think about loading U, V and scanRanks into shared memory
        // multiply V by the sampling vector and store it in buffer memory
        int scanRanksIndex = (blockInBatch == 0) ? tile*batchUnitSize + blockInBatch : tile*batchUnitSize + blockInBatch - 1;
        double sum = 0;

        if(threadIdx.x < samplingVectorsWidth && threadIdx.y < rank) {
            for(unsigned int j = 0; j < tileSize; ++j) {
                sum += V[batch][scanRanks[batch][scanRanksIndex]*tileSize + threadIdx.x*tileSize + j]*samplingVectors[batch*batchUnitSize*tileSize*samplingVectorsWidth + threadIdx.x*batchUnitSize*tileSize + blockInBatch*tileSize + j];
            }
            bufferMemory[batch*batchUnitSize*tileSize*samplingVectorsWidth + threadIdx.x*batchUnitSize*tileSize + blockInBatch*tileSize + threadIdx.y] = sum;
        }
        __syncthreads();

        // multiply U by the result and add it to output
        sum = 0;
        if(threadIdx.x < samplingVectorsWidth) {
            for(unsigned int j = 0; j < rank; ++j) {
                sum += U[batch][scanRanks[batch][scanRanksIndex]*tileSize + j*tileSize + threadIdx.y]*bufferMemory[batch*batchUnitSize*tileSize*samplingVectorsWidth + threadIdx.x*batchUnitSize*tileSize + blockInBatch*tileSize + j];
            }
            output[batch*batchUnitSize*tileSize*samplingVectorsWidth + threadIdx.x*batchUnitSize*tileSize + blockInBatch*tileSize + threadIdx.y] += sum;
        }
        __syncthreads();
    }
}

#endif