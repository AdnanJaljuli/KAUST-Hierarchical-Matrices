
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
void freeLevelTilePtrs(LevelTilePtrs tilePtrs);

__global__ void fillBatchPtrs(H2Opus_Real **d_UPtrs, H2Opus_Real **d_VPtrs, TLR_Matrix mortonOrderedMatrix, int batchSize, int segmentSize, int batchUnitSize, int* tileIndices, int level);
__global__ void fillScanRankPtrs(int **d_scanRanksPtrs, int *d_scanRanks, int batchUnitSize, int batchSize);
__global__ void fillLRARAArrays(int batchSize, int maxRows, int* d_rowsBatch, int* d_colsBatch, int* d_LDABatch, int* d_LDBBatch);
void generateScanRanks(int batchSize, int batchUnitSize, int *ranks, int *scanRanks, int **scanRanksPtrs, int *levelTileIndices);

#endif