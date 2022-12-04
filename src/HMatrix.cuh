#ifndef HMATRIX_H
#define HMATRIX_H

#include "HMatrixHelpers.cuh"

struct HMatrixLevel {
    int numTiles, level;
    int* tileIndices;
    int* tileScanRanks; // inclusive scan
    H2Opus_Real* U;
    H2Opus_Real* V;
    // TODO: make a double pointer array to U and V
};

struct HMatrix {
    int numLevels;
    H2Opus_Real* diagonalBlocks;
    HMatrixLevel* levels;
};

void allocateAndCopyToHMatrixLevel(HMatrixLevel &matrixLevel, int* ranks, WeakAdmissibility WAStruct, unsigned int level, H2Opus_Real *A, H2Opus_Real *B, int maxRows, int maxRank);
void freeHMatrixLevel(HMatrixLevel matrixLevel);

void allocateHMatrix(HMatrix &matrix, TLR_Matrix mortonOrderedMatrix, int segmentSize, int numSegments, unsigned int numberOfInputPoints, unsigned int bucketSize, WeakAdmissibility WAStruct);
void freeHMatrix(HMatrix &matrix);

#endif
