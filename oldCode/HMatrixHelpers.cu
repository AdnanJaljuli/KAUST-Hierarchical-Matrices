
#include <cub/cub.cuh>
#include "helperKernels.cuh"
#include "HMatrixHelpers.cuh"

#if 0

__global__ void fillBatchPtrs(H2Opus_Real **d_UPtrs, H2Opus_Real **d_VPtrs, TLR_Matrix mortonOrderedMatrix, int batchSize, int segmentSize, int batchUnitSize, int* tileIndices, int level) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < batchSize) {
        if(blockIdx.y == 0) {
            // printf("tile index: %d      batchUnitSize: %d       level: %d       batchSize: %d\n", tileIndices[i], batchUnitSize, level, batchSize);
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

#endif