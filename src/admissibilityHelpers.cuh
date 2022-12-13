
#ifndef ADMISSIBIILITY_HELPERS_H
#define ADMISSIBIILITY_HELPERS_H

#include "boundingBoxes.h"

H2Opus_Real BBoxCenterDistance(BoundingBox node_u, BoundingBox node_v, unsigned int dimension);

H2Opus_Real BBoxDiameter(BoundingBox node, unsigned int dimension);

#endif