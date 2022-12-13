
#ifndef HMATRIX_HELPERS_H
#define HMATRIX_HELPERS_H

#include "TLRMatrix.cuh"
#include "precision.h"

struct LevelTilePtrs {
    H2Opus_Real **U;
    H2Opus_Real **V;
};

void allocateTilePtrs(int batchSize, int batchUnitSize, int segmentSize, int level, int *tileIndices, LevelTilePtrs &tilePtrs, TLR_Matrix mortonOrderedMatrix);
__global__ void fillBatchPtrs(H2Opus_Real **d_UPtrs, H2Opus_Real **d_VPtrs, TLR_Matrix mortonOrderedMatrix, int batchSize, int segmentSize, int batchUnitSize, int* tileIndices, int level);
void freeLevelTilePtrs(LevelTilePtrs tilePtrs);

#endif