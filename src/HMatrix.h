#ifndef __HIERARCHICALMATRIX__
#define __HIERARCHICALMATRIX__

#include "HMatrixHelpers.cuh"

struct HMatrixLevel {
    int numTiles, level;
    int* tileIndices;
    int* tileScanRanks;
    H2Opus_Real* U;
    H2Opus_Real* V;
};

void allocateAndCopyToHMatrixLevel(HMatrixLevel &matrixLevel, int* ranks, WeakAdmissibility WAStruct, unsigned int level, H2Opus_Real *A, H2Opus_Real *B, int maxRows, int maxRank) {
    // TODO: make a double pointer array to U and V
    matrixLevel.numTiles = WAStruct.numTiles[level - 1];
    matrixLevel.level = level;

    // scan ranks array
    cudaMalloc((void**) &matrixLevel.tileScanRanks, matrixLevel.numTiles*sizeof(int));
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, ranks, matrixLevel.tileScanRanks, matrixLevel.numTiles);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, ranks, matrixLevel.tileScanRanks, matrixLevel.numTiles);

    int *scanRanks = (int*)malloc(matrixLevel.numTiles*sizeof(int));
    cudaMemcpy(scanRanks, matrixLevel.tileScanRanks, matrixLevel.numTiles*sizeof(int), cudaMemcpyDeviceToHost);

    // allocate U and V
    cudaMalloc((void**) &matrixLevel.U, scanRanks[matrixLevel.numTiles - 1]*maxRows*sizeof(H2Opus_Real));
    cudaMalloc((void**) &matrixLevel.V, scanRanks[matrixLevel.numTiles - 1]*maxRows*sizeof(H2Opus_Real));

    // copy A and B to U and V
    for(unsigned int tile = 0; tile < matrixLevel.numTiles; ++tile) {
        int tileRank = (tile == 0) ? scanRanks[tile] : scanRanks[tile] - scanRanks[tile - 1];
        cudaMemcpy(&matrixLevel.U[(scanRanks[tile] - tileRank)*maxRows], &A[tile*maxRows*maxRank], tileRank*maxRows*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&matrixLevel.V[(scanRanks[tile] - tileRank)*maxRows], &B[tile*maxRows*maxRank], tileRank*maxRows*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
    }

    // copy tile indices from WAStruct to here
    cudaMalloc((void**) &matrixLevel.tileIndices, matrixLevel.numTiles*sizeof(int));
    cudaMemcpy(matrixLevel.tileIndices, WAStruct.tileIndices[level - 1], matrixLevel.numTiles*sizeof(int), cudaMemcpyHostToDevice);
}

void freeHMatrixLevel(HMatrixLevel matrixLevel){
    cudaFree(matrixLevel.tileIndices);
    cudaFree(matrixLevel.tileScanRanks);
    cudaFree(matrixLevel.U);
    cudaFree(matrixLevel.V);
}

struct HMatrix {
    int numLevels;
    H2Opus_Real* diagonalBlocks;
    HMatrixLevel* levels;
};

void allocateHMatrix(HMatrix &matrix, int segmentSize, int numSegments, unsigned int numberOfInputPoints, unsigned int bucketSize) {
    cudaMalloc((void**) &matrix.diagonalBlocks, segmentSize*segmentSize*numSegments*sizeof(H2Opus_Real));
    // TODO: copy diagonal blocks from MOMatrix to HMatrix
    matrix.numLevels = __builtin_ctz(numberOfInputPoints/bucketSize) + 1;
    matrix.levels = (HMatrixLevel*)malloc((matrix.numLevels - 2)*sizeof(HMatrixLevel));
}

void freeHMatrix(HMatrix &matrix) {
    // TODO: loop over hmatrix levels and free thtem
}

#endif