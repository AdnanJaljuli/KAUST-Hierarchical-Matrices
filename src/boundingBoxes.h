
#ifndef BOUNDING_BOXES_H
#define BOUNDING_BOXES_H

#include <stdint.h>
#include "precision.h"

struct BoundingBox {
    H2Opus_Real *dimMax;
    H2Opus_Real *dimMin;
};

struct kdtreeBoundingBoxes {
    BoundingBox **levels;
};

void allocateBoundingBoxes(unsigned int numberOfInputPoints, 
    unsigned int bucketSize, 
    unsigned int dimensionOfInputPoints);
void freeBoundingBoxes();

#endif