
#ifndef BOUNDING_BOXES_H
#define BOUNDING_BOXES_H

#include <stdint.h>
#include "precision.h"

struct BoundingBox {
    H2Opus_Real *dimMax;
    H2Opus_Real *dimMin;
    unsigned int index;
    unsigned int level;
};

struct KDTreeLevelBoundingBoxes {
    BoundingBox *boundingBoxes;
    H2Opus_Real *maxBBData;
    H2Opus_Real *minBBData;
};

struct KDTreeBoundingBoxes {
    KDTreeLevelBoundingBoxes *levels;
};

void allocateKDTreeBoundingBoxes(
    KDTreeBoundingBoxes *boxes,
    unsigned int numberOfInputPoints,
    unsigned int bucketSize,
    unsigned int dimensionOfInputPoints);

void freeBoundingBoxes(); // TODO

void copyMaxandMinToBoundingBoxes(KDTreeLevelBoundingBoxes BBlevel, 
    H2Opus_Real *d_maxSegmentItem,
    H2Opus_Real *d_minSegmentItem,
    unsigned int level,
    unsigned int dimensionOfInputPoints,
    unsigned int currentNumSegments,
    H2Opus_Real *d_bufferBBMax, H2Opus_Real *d_bufferBBMin);

#endif