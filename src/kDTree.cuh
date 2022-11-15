#ifndef __KD_TREE_H__
#define __KD_TREE_H__

#include <stdint.h>

struct KDTree{
    uint64_t numSegments;
    uint64_t segmentSize;
    int *segmentIndices;
    int *segmentOffsets;
};

void allocateKDTree(KDTree &tree, unsigned int numberOfInputPoints, unsigned int bucketSize);
void freeKDTree(KDTree tree);

#endif