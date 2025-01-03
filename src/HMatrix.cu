
#include "HMatrix.cuh"
#include "admissibilityCondition.cuh"
#include "boundingBoxes.h"
#include "config.h"
#include "helperKernels.cuh"
#include "HMatrixStructure.cuh"
#include "kDTreeHelpers.cuh"
#include <cub/cub.cuh>
#include <functional>

void allocateAndCopyToHMatrixLevel(
    HMatrixLevel &matrixLevel, 
    int* ranks, 
    HMatrixStructure HMatrixStruct, 
    unsigned int level, 
    H2Opus_Real *A, H2Opus_Real *B, 
    int maxRows, int maxRank) {
        
        // matrixLevel.numTiles = HMatrixStruct.numTiles[level - 1];
        matrixLevel.level = level;

        // scan ranks array
        cudaMalloc((void**) &matrixLevel.tileScanRanks, HMatrixStruct.numTiles[level - 1]*sizeof(int));
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, ranks, matrixLevel.tileScanRanks, HMatrixStruct.numTiles[level - 1]);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, ranks, matrixLevel.tileScanRanks, HMatrixStruct.numTiles[level - 1]);

        int *scanRanks = (int*)malloc(HMatrixStruct.numTiles[level - 1]*sizeof(int));
        cudaMemcpy(scanRanks, matrixLevel.tileScanRanks, HMatrixStruct.numTiles[level - 1]*sizeof(int), cudaMemcpyDeviceToHost);

        // allocate U and V
        cudaMalloc((void**) &matrixLevel.U, static_cast<uint64_t>(scanRanks[HMatrixStruct.numTiles[level - 1] - 1])*maxRows*sizeof(H2Opus_Real));
        cudaMalloc((void**) &matrixLevel.V, static_cast<uint64_t>(scanRanks[HMatrixStruct.numTiles[level - 1] - 1])*maxRows*sizeof(H2Opus_Real));

        // copy A and B to U and V
        for(unsigned int tile = 0; tile < HMatrixStruct.numTiles[level - 1]; ++tile) {
            int tileRank = (tile == 0) ? scanRanks[tile] : scanRanks[tile] - scanRanks[tile - 1];
            cudaMemcpy(&matrixLevel.U[static_cast<uint64_t>(scanRanks[tile] - tileRank)*maxRows], &A[static_cast<uint64_t>(tile)*maxRows*maxRank], static_cast<uint64_t>(tileRank)*maxRows*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&matrixLevel.V[static_cast<uint64_t>(scanRanks[tile] - tileRank)*maxRows], &B[static_cast<uint64_t>(tile)*maxRows*maxRank], static_cast<uint64_t>(tileRank)*maxRows*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
        }

        // copy tile indices from HMatrixStruct to here
        // cudaMalloc((void**) &matrixLevel.tileIndices, matrixLevel.numTiles*sizeof(int));
        // cudaMemcpy(matrixLevel.tileIndices, HMatrixStruct.tileIndices[level - 1], matrixLevel.numTiles*sizeof(int), cudaMemcpyHostToDevice);
}

void freeHMatrixLevel(HMatrixLevel matrixLevel){
    // cudaFree(matrixLevel.tileIndices);
    cudaFree(matrixLevel.tileScanRanks);
    cudaFree(matrixLevel.U);
    cudaFree(matrixLevel.V);
}

void allocateHMatrix(HMatrix &matrix, TLR_Matrix mortonOrderedMatrix, int segmentSize, int numSegments, unsigned int numberOfInputPoints, unsigned int bucketSize, HMatrixStructure HMatrixStruct) {
    // TODO: consolidate bucket size and segment size
    cudaMalloc((void**) &matrix.diagonalBlocks, segmentSize*segmentSize*numSegments*sizeof(H2Opus_Real));
    cudaMemcpy(matrix.diagonalBlocks, mortonOrderedMatrix.diagonal, segmentSize*segmentSize*numSegments*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
    // matrix.matrixStructure.numLevels = __builtin_ctz(numberOfInputPoints/bucketSize) + 1;
    matrix.levels = (HMatrixLevel*)malloc((matrix.matrixStructure.numLevels - 1)*sizeof(HMatrixLevel));

    // copy tlr tiles to HMatrix bottom level
    int *h_ranks = (int*)malloc(numSegments*numSegments*sizeof(int));
    int *h_scanRanks = (int*)malloc(numSegments*numSegments*sizeof(int));
    int *h_levelRanks = (int*)malloc(HMatrixStruct.numTiles[matrix.matrixStructure.numLevels - 2]*sizeof(int));

    cudaMemcpy(h_ranks, mortonOrderedMatrix.blockRanks, numSegments*numSegments*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_scanRanks, mortonOrderedMatrix.blockOffsets, numSegments*numSegments*sizeof(int), cudaMemcpyDeviceToHost);
    // matrix.levels[matrix.matrixStructure.numLevels - 2].numTiles = HMatrixStruct.numTiles[matrix.matrixStructure.numLevels - 2];
    matrix.levels[matrix.matrixStructure.numLevels - 2].level = matrix.matrixStructure.numLevels - 1;

    int rankSum = 0;
    for(unsigned int i = 0; i < matrix.matrixStructure.numTiles[matrix.matrixStructure.numLevels - 2]; ++i) {
        rankSum += h_ranks[HMatrixStruct.tileIndices[matrix.matrixStructure.numLevels - 2][i]];
        h_levelRanks[i] = h_ranks[HMatrixStruct.tileIndices[matrix.matrixStructure.numLevels - 2][i]];
    }

    // cudaMalloc((void**) &matrix.levels[matrix.matrixStructure.numLevels - 2].tileIndices, matrix.levels[matrix.matrixStructure.numLevels - 2].numTiles*sizeof(int));
    // cudaMemcpy(matrix.levels[matrix.matrixStructure.numLevels - 2].tileIndices, HMatrixStruct.tileIndices[matrix.matrixStructure.numLevels - 2], matrix.levels[matrix.matrixStructure.numLevels - 2].numTiles*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &matrix.levels[matrix.matrixStructure.numLevels - 2].U, rankSum*bucketSize*sizeof(H2Opus_Real));
    cudaMalloc((void**) &matrix.levels[matrix.matrixStructure.numLevels - 2].V, rankSum*bucketSize*sizeof(H2Opus_Real));

    cudaMalloc((void**) &matrix.levels[matrix.matrixStructure.numLevels - 2].tileScanRanks, matrix.matrixStructure.numTiles[matrix.matrixStructure.numLevels - 2]*sizeof(int));    
    int *d_levelRanks;
    cudaMalloc((void**) &d_levelRanks, HMatrixStruct.numTiles[matrix.matrixStructure.numLevels - 2]*sizeof(int));
    cudaMemcpy(d_levelRanks, h_levelRanks, HMatrixStruct.numTiles[matrix.matrixStructure.numLevels - 2]*sizeof(int), cudaMemcpyHostToDevice);
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_levelRanks, matrix.levels[matrix.matrixStructure.numLevels - 2].tileScanRanks, matrix.matrixStructure.numTiles[matrix.matrixStructure.numLevels - 2]);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_levelRanks, matrix.levels[matrix.matrixStructure.numLevels - 2].tileScanRanks, matrix.matrixStructure.numTiles[matrix.matrixStructure.numLevels - 2]);

    int tmp = 0;
    for(unsigned int i = 0; i < matrix.matrixStructure.numTiles[matrix.matrixStructure.numLevels - 2]; ++i) {
        cudaMemcpy(&matrix.levels[matrix.matrixStructure.numLevels - 2].U[tmp*bucketSize], &mortonOrderedMatrix.U[h_scanRanks[HMatrixStruct.tileIndices[matrix.matrixStructure.numLevels - 2][i]]*bucketSize], h_levelRanks[i]*bucketSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&matrix.levels[matrix.matrixStructure.numLevels - 2].V[tmp*bucketSize], &mortonOrderedMatrix.V[h_scanRanks[HMatrixStruct.tileIndices[matrix.matrixStructure.numLevels - 2][i]]*bucketSize], h_levelRanks[i]*bucketSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);

        tmp += h_levelRanks[i];
    }
}

void freeHMatrix(HMatrix &matrix) {
    cudaFree(matrix.diagonalBlocks);
    for(unsigned int level = 1; level < matrix.matrixStructure.numLevels - 1; ++level) {
        freeHMatrixLevel(matrix.levels[level - 1]);
    }
}