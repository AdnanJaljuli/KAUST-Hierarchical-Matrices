
#ifndef HMATRIX_STRUCTURE_H
#define HMATRIX_STRUCTURE_H

#include "boundingBoxes.h"
#include "config.h"
#include "kDTree.cuh"

#include <functional>
#include <vector>

struct HMatrixStructure {
    int numLevels;
    std::vector<int> numTiles;
    // TODO: make this a vector instead of a malloced array
    std::vector<std::vector<int>> tileIndices; // TODO: do they have to be sorted?
};

void constructHMatrixStructure(
    HMatrixStructure *HMatrixStruct,
    std::function<bool(
        BoundingBox,
        BoundingBox,
        unsigned int,
        float)> isAdmissible,
    KDTree rowTree,
    KDTree columnTree);

void allocateHMatrixStructure(HMatrixStructure *HMatrixStruct, unsigned int numLevels);

void freeHMatrixStructure(HMatrixStructure &HMatrixStruct);

#endif