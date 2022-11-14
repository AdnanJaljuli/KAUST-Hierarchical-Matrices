
#ifndef __HELPERS_HIERARCHICALMATRIX_H__
#define __HELPERS_HIERARCHICALMATRIX_H__

struct WeakAdmissibility {
    int numLevels;
    int* numTiles;
    int** tileIndices;
};

void allocateWeakAdmissibilityStruct(WeakAdmissibility &WAStruct, unsigned int numberOfInputPoints, unsigned int bucketSize) {
    // TODO: parallelize
    WAStruct.numLevels = __builtin_ctz(numberOfInputPoints/bucketSize) + 1;
    WAStruct.numTiles = (int*)malloc((WAStruct.numLevels - 1)*sizeof(int));
    WAStruct.tileIndices = (int**)malloc((WAStruct.numLevels - 1)*sizeof(int*));

    unsigned int dim = 2;
    for(unsigned int level = 0; level < WAStruct.numLevels - 1; ++level) {
        unsigned int numTiles = 1 << (level + 1);
        WAStruct.numTiles[level] = numTiles;
        
        WAStruct.tileIndices[level] = (int*)malloc(numTiles*sizeof(int));
        for(unsigned int j = 0; j < numTiles; ++j) {
            int x;
            if(j%2 == 0) {
                x = 1;
            }
            else {
                x = -1;
            }
            unsigned int tileIndex = j*dim + j + x;
            WAStruct.tileIndices[level][j + x] = IndextoMOIndex_h(dim, tileIndex);
        }
        
        dim <<= 1;
    }
}

void freeWeakAdmissbilityStruct(WeakAdmissibility WAStruct) {
    free(WAStruct.numTiles);
    for(unsigned int i = 0; i < WAStruct.numLevels - 1; ++i) {
        free(WAStruct.tileIndices[i]);
    }
    free(WAStruct.tileIndices);
}

__global__ void fillBatchPtrs(H2Opus_Real **d_UPtrs, H2Opus_Real **d_VPtrs, TLR_Matrix mortonOrderedMatrix, int batchSize, int segmentSize, int batchUnitSize, int* tileIndices, int level) {
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

struct LevelTilesPtrs {
    H2Opus_Real **U;
    H2Opus_Real **V;
};

void allocateAndFillLevelTilesPtrs(int batchSize, int batchUnitSize, int segmentSize, int level, int *tileIndices, LevelTilesPtrs &tilePtrs, TLR_Matrix mortonOrderedMatrix) {
    cudaMalloc((void**) &tilePtrs.U, batchSize*sizeof(H2Opus_Real*));
    cudaMalloc((void**) &tilePtrs.V, batchSize*sizeof(H2Opus_Real*));

    dim3 numThreadsPerBlock(1024);
    dim3 numBlocks((batchSize + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x, 2);
    fillBatchPtrs <<< numBlocks, numThreadsPerBlock >>> (tilePtrs.U, tilePtrs.V, mortonOrderedMatrix, batchSize, segmentSize, batchUnitSize, tileIndices, level);
}

void freeLevelTilesPtrs(LevelTilesPtrs tilePtrs) {
    cudaFree(tilePtrs.U);
    cudaFree(tilePtrs.V);
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

#endif