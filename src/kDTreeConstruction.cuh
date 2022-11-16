
#ifndef __KDTREECONSTRUCTION_CUH__
#define __KDTREECONSTRUCTION_CUH__

#include "kDTree.cuh"

typedef double H2Opus_Real;

void constructKDTree(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, unsigned int bucketSize, KDTree &kDTree, H2Opus_Real* d_pointCloud);

#endif