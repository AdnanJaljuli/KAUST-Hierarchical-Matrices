
#include "HMatrix.cuh"
#include "admissibilityFunctions.cuh"
#include "boundingBoxes.h"
#include "config.h"
#include "helperKernels.cuh"
#include "HMatrixStructure.cuh"
#include "kDTreeHelpers.cuh"
#include <cub/cub.cuh>
#include <functional>

#if 0
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

#endif

template <class T>
void freeHMatrixLevel(HMatrixLevel <T> matrixLevel) { 
    // TODO
}

template <class T>
void allocateHMatrix(
    HMatrix <T> &matrix,
    unsigned int lowestLevelTileSize,
    unsigned int numLeaves) {

        cudaMalloc((void**) &matrix.diagonalBlocks, lowestLevelTileSize*lowestLevelTileSize*numLeaves*sizeof(T));
        matrix.levels = (HMatrixLevel <T> *)malloc((matrix.matrixStructure.numLevels - 1)*sizeof(T));
}

template void allocateHMatrix <H2Opus_Real> (
    HMatrix <H2Opus_Real> &matrix,
    unsigned int lowestLevelTileSize,
    unsigned int numLeaves);


template <class T>
void freeHMatrix(HMatrix <T> &matrix) {
    cudaFree(matrix.diagonalBlocks);
    for(unsigned int level = 1; level < matrix.matrixStructure.numLevels - 1; ++level) {
        freeHMatrixLevel <T> (matrix.levels[level - 1]);
    }
}