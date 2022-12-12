
#ifndef HMATRIX_STRUCTURE_H
#define HMATRIX_STRUCTURE_H

#include "config.h"
#include "boundingBoxes.h"

struct HMatrixStructure {
    int numLevels;
    int* numTiles;
    // TODO: make this a vector instead of a malloced array
    int** tileIndices; // TODO: do they have to be sorted?
};

void constructHMatrixStructure(
    HMatrixStructure *HMatrixStruct,
    unsigned int numberOfInputPoints,
    unsigned int dimensionOfInputPoints, 
    unsigned int bucketSize,
    ADMISSIBILITY_CONDITION admissibilityCondition,
    KDTreeBoundingBoxes BBox1,
    KDTreeBoundingBoxes BBox2);
void freeHMatrixStructure(HMatrixStructure &HMatrixStruct);

#endif