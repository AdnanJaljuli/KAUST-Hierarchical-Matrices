
#ifndef KDTREE_CONSTRUCTION_H
#define KDTREE_CONSTRUCTION_H

#include "kDTree.cuh"
#include "config.h"

typedef double H2Opus_Real;

void constructKDTree(
    KDTree &kDTree, 
    H2Opus_Real* 
    d_pointCloud, 
    DIVISION_METHOD divMethod);

#endif