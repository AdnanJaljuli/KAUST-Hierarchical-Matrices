
#ifndef __HELPERS_HIERARCHICALMATRIX_H__
#define __HELPERS_HIERARCHICALMATRIX_H__

#include "TLRMatrix.cuh"

typedef double H2Opus_Real;

struct WeakAdmissibility {
    int numLevels;
    int* numTiles;
    int** tileIndices;
};

void allocateWeakAdmissibilityStruct(WeakAdmissibility &WAStruct, unsigned int numberOfInputPoints, unsigned int bucketSize);
void freeWeakAdmissbilityStruct(WeakAdmissibility WAStruct);

struct LevelTilePtrs {
    H2Opus_Real **U;
    H2Opus_Real **V;
};

void allocateTilePtrs(int batchSize, int batchUnitSize, int segmentSize, int level, int *tileIndices, LevelTilePtrs &tilePtrs, TLR_Matrix mortonOrderedMatrix);
__global__ void fillBatchPtrs(H2Opus_Real **d_UPtrs, H2Opus_Real **d_VPtrs, TLR_Matrix mortonOrderedMatrix, int batchSize, int segmentSize, int batchUnitSize, int* tileIndices, int level);
void freeLevelTilePtrs(LevelTilePtrs tilePtrs);

#endif