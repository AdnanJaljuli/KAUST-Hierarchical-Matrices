
#ifndef HMATRIX_H
#define HMATRIX_H

#include "boundingBoxes.h"
#include "HMatrixStructure.cuh"

struct HMatrixLevel {
    int level;
    unsigned int rankSum;
    int* scannedBLockRanks; // inclusive scan
    H2Opus_Real* U;
    H2Opus_Real* V;
    // TODO: make a double pointer array to U and V
};

void copyToHMatrixLevel(); // TODO

struct HMatrix {
    H2Opus_Real* diagonalBlocks;
    HMatrixLevel* levels;
    HMatrixStructure matrixStructure;
};

template <class T>
void allocateHMatrix(
    HMatrix &matrix,
    unsigned int lowestLevelTileSize,
    unsigned int numLeaves);

void freeHMatrix(HMatrix &matrix);

#endif
