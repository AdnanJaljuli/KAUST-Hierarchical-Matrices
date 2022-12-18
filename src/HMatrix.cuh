
#ifndef HMATRIX_H
#define HMATRIX_H

#include "boundingBoxes.h"
#include "HMatrixStructure.cuh"

template <class T>
struct HMatrixLevel {
    int level;
    unsigned int rankSum;
    int* scannedBLockRanks; // inclusive scan
    T* U;
    T* V;
    // TODO: make a double pointer array to U and V
};

void copyToHMatrixLevel(); // TODO

template <class T>
struct HMatrix {
    T* diagonalBlocks;
    HMatrixLevel <T> *levels;
    HMatrixStructure matrixStructure;
};

template <class T>
void allocateHMatrix(
    HMatrix <T> &matrix,
    unsigned int lowestLevelTileSize,
    unsigned int numLeaves);

template <class T>
void freeHMatrix(HMatrix <T> &matrix);

#endif
