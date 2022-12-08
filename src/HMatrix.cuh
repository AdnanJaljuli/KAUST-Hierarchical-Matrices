
#ifndef HMATRIX_H
#define HMATRIX_H

#include "HMatrixHelpers.cuh"

struct HMatrixStructure {
    int numLevels;
    int* numTiles;
    int** tileIndices;
};

void constructHMatrixStructure(
    HMatrixStructure *HMatrixStruct,
    unsigned int numberOfInputPoints, 
    unsigned int bucketSize,
    ADMISSIBILITY_CONDITION admissibilityCondition);
void freeHMatrixStructure(HMatrixStructure &HMatrixStruct);


struct HMatrixLevel {
    int numTiles, level;
    int* tileIndices;
    int* tileScanRanks; // inclusive scan
    H2Opus_Real* U;
    H2Opus_Real* V;
    // TODO: make a double pointer array to U and V
};

void allocateAndCopyToHMatrixLevel(
    HMatrixLevel &matrixLevel, 
    int* ranks, 
    HMatrixStructure HMatrixStruct, 
    unsigned int level, 
    H2Opus_Real *A, H2Opus_Real *B, 
    int maxRows, int maxRank);
void freeHMatrixLevel(HMatrixLevel matrixLevel);


struct HMatrix {
    H2Opus_Real* diagonalBlocks;
    HMatrixLevel* levels;
    HMatrixStructure matrixStructure;
};

void allocateHMatrix(HMatrix &matrix, 
    TLR_Matrix mortonOrderedMatrix, 
    int segmentSize, int numSegments, 
    unsigned int numberOfInputPoints, unsigned int bucketSize, 
    HMatrixStructure HMatrixStruct);
void freeHMatrix(HMatrix &matrix);

#endif
