#ifndef KD_TREE_H
#define KD_TREE_H

#include <stdint.h>
#include "boundingBoxes.h"
#include "config.h"

struct KDTree {
    unsigned int numLevels;
    unsigned int N;
    unsigned int nDim;
    unsigned int numLeaves;
    unsigned int leafSize;
    int *segmentIndices;
    int *leafOffsets;
    KDTreeBoundingBoxes boundingBoxes;
};

void allocateKDTree(
    KDTree &tree, 
    unsigned int numberOfInputPoints, 
    unsigned int dimensionOfInputPoints,
    unsigned int leafSize,
    DIVISION_METHOD divMethod);

void freeKDTree(KDTree tree);

#endif