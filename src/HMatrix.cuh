
#ifndef HMATRIX_H
#define HMATRIX_H

#include "boundingBoxes.h"
#include "HMatrixStructure.cuh"

struct HMatrixLevel {
    int level;
    int* tileScanRanks; // inclusive scan
    H2Opus_Real* U;
    H2Opus_Real* V;
    // TODO: make a double pointer array to U and V
};

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
