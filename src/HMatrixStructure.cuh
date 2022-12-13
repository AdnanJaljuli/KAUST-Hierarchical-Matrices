
#ifndef HMATRIX_STRUCTURE_H
#define HMATRIX_STRUCTURE_H

#include "boundingBoxes.h"
#include "config.h"
#include "kDTree.cuh"

struct HMatrixStructure {
    int numLevels;
    int* numTiles;
    // TODO: make this a vector instead of a malloced array
    int** tileIndices; // TODO: do they have to be sorted?
};

void constructHMatrixStructure(
    HMatrixStructure *HMatrixStruct,
    ADMISSIBILITY_CONDITION admissibilityCondition,
    KDTree rowTree,
    KDTree columnTree);
void freeHMatrixStructure(HMatrixStructure &HMatrixStruct);

#endif