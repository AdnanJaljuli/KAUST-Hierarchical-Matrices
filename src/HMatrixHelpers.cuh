
#ifndef __HELPERS_HIERARCHICALMATRIX_H__
#define __HELPERS_HIERARCHICALMATRIX_H__

#include "HMatrix.h"

__global__ void fillBatchedPtrs(H2Opus_Real **d_UPtrs, H2Opus_Real **d_VPtrs, TLR_Matrix mortonOrderedMatrix, int batchSize, int segmentSize, int batchUnitSize, int* tileIndices, int level) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < batchSize) {
        if(blockIdx.y == 0) {
            d_UPtrs[i] = &mortonOrderedMatrix.U[mortonOrderedMatrix.blockOffsets[tileIndices[i]*batchUnitSize*batchUnitSize]*segmentSize];
        }
        else {
            d_VPtrs[i] = &mortonOrderedMatrix.V[mortonOrderedMatrix.blockOffsets[tileIndices[i]*batchUnitSize*batchUnitSize]*segmentSize];
        }
    }
}

__global__ void fillScanRankPtrs(int **d_scanRanksPtrs, int *d_scanRanks, int batchUnitSize, int batchSize) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < batchSize) {
        d_scanRanksPtrs[i] = &d_scanRanks[i*batchUnitSize*batchUnitSize];
    }
}

__global__ void fillLRARAArrays(int batchSize, int maxRows, int* d_rowsBatch, int* d_colsBatch, int* d_LDABatch, int* d_LDBBatch){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < batchSize){
        d_rowsBatch[i] = maxRows;
        d_colsBatch[i] = maxRows;
        d_LDABatch[i] = maxRows;
        d_LDBBatch[i] = maxRows;
    }
}

void generateScanRanks(int batchSize, int batchUnitSize, int *ranks, int *scanRanks, int **scanRanksPtrs, int *levelTileIndices) {
    // loop over and do inclusiveSum
    // TODO: think about replacing this with a single inclusiveSumByKey thats called before the for loop
    for(unsigned int batch = 0; batch < batchSize; ++batch) {
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, ranks + levelTileIndices[batch]*batchUnitSize*batchUnitSize, scanRanks + batch*batchUnitSize*batchUnitSize, batchUnitSize*batchUnitSize);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, ranks + levelTileIndices[batch]*batchUnitSize*batchUnitSize, scanRanks + batch*batchUnitSize*batchUnitSize, batchUnitSize*batchUnitSize);
        cudaFree(d_temp_storage);
    }

    // fillScanRanksPtrs
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (batchSize + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillScanRankPtrs <<< numBlocks, numThreadsPerBlock >>> (scanRanksPtrs, scanRanks, batchUnitSize, batchSize);
}

__global__ void expandMatrix(H2Opus_Real **A, H2Opus_Real **B, int size, H2Opus_Real* output, int* ranks) {
    unsigned int batch = blockIdx.z;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    if(col < size && row < size) {
        H2Opus_Real sum = 0;
        for(unsigned int i = 0; i < ranks[batch]; ++i) {
            sum += A[batch][i*size + row]*B[batch][i*size + col];
        }
        output[batch*size*size + col*size + row] = sum;
    }
}

__global__ void compareResults(unsigned int numberOfInputPoints, double* denseMatrix, double* output, int* tileIndices, int batchSize, int batchUnitSize, double* error, double* tmp) {
    unsigned int batch = blockIdx.z;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;

    int diff;
    if(batch%2 == 0) {
        diff = 1;
    }
    else {
        diff = -1;
    }

    if(col < batchUnitSize*32 && row < batchUnitSize*32) {
        double x = denseMatrix[(col + batchUnitSize*32*(batch + diff))*numberOfInputPoints + batchUnitSize*32*batch + row];
        double y = output[batch*batchUnitSize*32*batchUnitSize*32 + col*batchUnitSize*32 + row];
        atomicAdd(tmp, x*x);
        atomicAdd(error, (x - y)*(x - y));
    }
}

#endif