
#include "admissibilityHelpers.cuh"
#include "boundingBoxes.h"
#include "precision.h"
#include <stdio.h>

H2Opus_Real BBoxCenterDistance(BoundingBox node_u, BoundingBox node_v, unsigned int dimension) {
    H2Opus_Real distance = 0;
    for(unsigned int i = 0; i < dimension; ++i) {
        H2Opus_Real centerDiff = 0.5*((node_u.dimMax[i] + node_u.dimMin[i]) - (node_v.dimMax[i] + node_v.dimMin[i]));
        distance += centerDiff*centerDiff;
    }

    return sqrt(distance);
}

H2Opus_Real BBoxDiameter(BoundingBox node, unsigned int dimension) {
    H2Opus_Real diameter = 0;
    for(unsigned int i = 0; i < dimension; ++i) {
        H2Opus_Real dimensionDiff = node.dimMax[i] - node.dimMin[i];
        diameter += dimensionDiff*dimensionDiff;
    }

    return sqrt(diameter);
}
