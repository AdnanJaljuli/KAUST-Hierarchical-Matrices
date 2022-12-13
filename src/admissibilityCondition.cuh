
#ifndef ADMISSIBIILITY_CONDITION_H
#define ADMISSIBIILITY_CONDITION_H

#include "boundingBoxes.h"

bool BBoxCenterAdmissibility(
    BoundingBox node_u,
    BoundingBox node_v,
    unsigned int dimensionOfInputPoints,
    float eta);

bool weakAdmissibility(
    BoundingBox node_u,
    BoundingBox node_v,
    unsigned int dimensionOfInputPoints,
    float eta);

#endif