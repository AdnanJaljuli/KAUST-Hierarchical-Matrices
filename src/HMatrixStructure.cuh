
#ifndef HMATRIX_STRUCTURE_H
#define HMATRIX_STRUCTURE_H

#include "boundingBoxes.h"
#include "config.h"
#include "kDTree.cuh"

#include <functional>

struct HMatrixStructure {
    int numLevels;
    int* numTiles;
    // TODO: make this a vector instead of a malloced array
    int** tileIndices; // TODO: do they have to be sorted?
};

void constructHMatrixStructure(
    HMatrixStructure *HMatrixStruct,
    std::function<bool(
        BoundingBox,
        BoundingBox,
        unsigned int,
        unsigned int,
        unsigned int,
        float)> isAdmissible,
    KDTree rowTree,
    KDTree columnTree);
void freeHMatrixStructure(HMatrixStructure &HMatrixStruct);

#endif