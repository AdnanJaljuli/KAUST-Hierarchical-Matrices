#ifndef KD_TREE_H
#define KD_TREE_H

#include <stdint.h>

struct KDTree{
    unsigned int numSegments;
    unsigned int segmentSize;
    int *segmentIndices;
    int *segmentOffsets;
};

void allocateKDTree(KDTree &tree, unsigned int numberOfInputPoints, unsigned int bucketSize);
void freeKDTree(KDTree tree);

#endif