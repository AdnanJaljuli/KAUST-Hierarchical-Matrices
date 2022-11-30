
#ifndef CUTLASS_HMATRIX_X_VECTORS_H
#define CUTLASS_HMATRIX_X_VECTORS_H

#include "HMatrix.cuh"

typedef double H2Opus_Real;

cudaError_t cutlassHierarchicalXVec(unsigned int numberOfInputPoints, unsigned int  bucketSize, unsigned int  numSegments, unsigned int  vectorWidth, HMatrix hierarchicalMatrix, H2Opus_Real *inputVectors, H2Opus_Real *resultVectors);

#endif