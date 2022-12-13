
#ifndef HMATRIX_H
#define HMATRIX_H

#include "HMatrixHelpers.cuh"
#include "boundingBoxes.h"
#include "HMatrixStructure.cuh"

struct HMatrixLevel {
    // TODO: get rid of tileIndices and numTiles here because the information is redundant with HMatrixStructure
    // int numTiles;
    int level;
    // int* tileIndices;
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
    unsigned int numberOfInputPoints, unsigned int leafSize, 
    HMatrixStructure HMatrixStruct);
void freeHMatrix(HMatrix &matrix);

#endif
