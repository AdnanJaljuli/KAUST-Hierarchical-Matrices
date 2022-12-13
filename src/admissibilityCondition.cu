
#include "admissibilityCondition.cuh"
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

bool BBoxCenterAdmissibility(
    BoundingBox node_u,
    BoundingBox node_v,
    unsigned int dimensionOfInputPoints,
    float eta) {

        if(node_u.index == node_v.index) {
            return false;
        }
        else {
            // get BBox center distance
            H2Opus_Real distance = BBoxCenterDistance(node_u, node_v, dimensionOfInputPoints);
            // get diameter of each box
            H2Opus_Real diameter_u = BBoxDiameter(node_u, dimensionOfInputPoints);
            H2Opus_Real diameter_v = BBoxDiameter(node_v, dimensionOfInputPoints);

            if(diameter_u == 0 || diameter_v == 0) {
                return false;
            }
            else {
                return (0.5*(diameter_u + diameter_v) <= eta*distance);
            }
        }
}

bool weakAdmissibility(
    BoundingBox node_u,
    BoundingBox node_v,
    unsigned int dimensionOfInputPoints,
    float eta) {

        if(node_u.index == node_v.index) {
            return false;
        }
        else {
            return true;
        }
}