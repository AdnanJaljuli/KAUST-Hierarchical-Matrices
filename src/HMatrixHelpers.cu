
#include <cub/cub.cuh>
#include "helperKernels.cuh"
#include "HMatrixHelpers.cuh"

void allocateHMatrixStructure(HMatrixStructure &HMatrixStruct, unsigned int numberOfInputPoints, unsigned int bucketSize) {
    // TODO: parallelize
    HMatrixStruct.numLevels = __builtin_ctz(numberOfInputPoints/bucketSize) + 1;
    HMatrixStruct.numTiles = (int*)malloc((HMatrixStruct.numLevels - 1)*sizeof(int));
    HMatrixStruct.tileIndices = (int**)malloc((HMatrixStruct.numLevels - 1)*sizeof(int*));

    unsigned int dim = 2;
    for(unsigned int level = 0; level < HMatrixStruct.numLevels - 1; ++level) {
        unsigned int numTiles = 1 << (level + 1);
        HMatrixStruct.numTiles[level] = numTiles;
        
        HMatrixStruct.tileIndices[level] = (int*)malloc(numTiles*sizeof(int));
        for(unsigned int j = 0; j < numTiles; ++j) {
            int x;
            if(j%2 == 0) {
                x = 1;
            }
            else {
                x = -1;
            }
            unsigned int tileIndex = j*dim + j + x;
            HMatrixStruct.tileIndices[level][j + x] = CMIndextoMOIndex(dim, tileIndex);
        }
        
        dim <<= 1;
    }
}

void freeHMatrixStructure(HMatrixStructure &HMatrixStruct) {
    free(HMatrixStruct.numTiles);
    for(unsigned int i = 0; i < HMatrixStruct.numLevels - 1; ++i) {
        free(HMatrixStruct.tileIndices[i]);
    }
    free(HMatrixStruct.tileIndices);
}

__global__ void fillBatchPtrs(H2Opus_Real **d_UPtrs, H2Opus_Real **d_VPtrs, TLR_Matrix mortonOrderedMatrix, int batchSize, int segmentSize, int batchUnitSize, int* tileIndices, int level) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < batchSize) {
        if(blockIdx.y == 0) {
            d_UPtrs[i] = &mortonOrderedMatrix.U[static_cast<uint64_t>(mortonOrderedMatrix.blockOffsets[tileIndices[i]*batchUnitSize*batchUnitSize])*segmentSize];
        }
        else {
            d_VPtrs[i] = &mortonOrderedMatrix.V[static_cast<uint64_t>(mortonOrderedMatrix.blockOffsets[tileIndices[i]*batchUnitSize*batchUnitSize])*segmentSize];
        }
    }
}

void allocateTilePtrs(int batchSize, int batchUnitSize, int segmentSize, int level, int *tileIndices, LevelTilePtrs &tilePtrs, TLR_Matrix mortonOrderedMatrix) {
    cudaMalloc((void**) &tilePtrs.U, batchSize*sizeof(H2Opus_Real*));
    cudaMalloc((void**) &tilePtrs.V, batchSize*sizeof(H2Opus_Real*));

    dim3 numThreadsPerBlock(1024);
    dim3 numBlocks((batchSize + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x, 2);
    fillBatchPtrs <<< numBlocks, numThreadsPerBlock >>> (tilePtrs.U, tilePtrs.V, mortonOrderedMatrix, batchSize, segmentSize, batchUnitSize, tileIndices, level);
}

void freeLevelTilePtrs(LevelTilePtrs tilePtrs) {
    cudaFree(tilePtrs.U);
    cudaFree(tilePtrs.V);
}