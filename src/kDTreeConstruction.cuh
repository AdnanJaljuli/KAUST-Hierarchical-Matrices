
#ifndef KDTREE_CONSTRUCTION_H
#define KDTREE_CONSTRUCTION_H

#include "kDTree.cuh"
#include "config.h"

typedef double H2Opus_Real;

void constructKDTree(
    unsigned int numberOfInputPoints, 
    unsigned int dimensionOfInputPoints, 
    unsigned int bucketSize, 
    KDTree &kDTree, 
    H2Opus_Real* 
    d_pointCloud, 
    DIVISION_METHOD divMethod);

#endif