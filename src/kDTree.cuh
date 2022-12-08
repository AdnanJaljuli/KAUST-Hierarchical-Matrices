#ifndef KD_TREE_H
#define KD_TREE_H

#include <stdint.h>
#include "boundingBoxes.h"
#include "config.h"

struct KDTree {
    unsigned int numSegments;
    unsigned int segmentSize;
    int *segmentIndices;
    int *segmentOffsets;
    KDTreeBoundingBoxes boundingBoxes;
};

void allocateKDTree(
    KDTree &tree, 
    unsigned int numberOfInputPoints, 
    unsigned int dimensionOfInputPoints,
    unsigned int bucketSize,
    DIVISION_METHOD divMethod);
void freeKDTree(KDTree tree);

#endif