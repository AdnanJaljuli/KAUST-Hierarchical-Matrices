
#ifndef HMATRIX_VECTOR_MULTIPLICATION_H
#define HMATRIX_VECTOR_MULTIPLICATION_H

typedef double H2Opus_Real;

#include "HMatrix.cuh"

cudaError_t HMatrixVecMult(unsigned int numberOfInputPoints, unsigned int bucketSize, unsigned int numSegments, unsigned int vectorWidth, HMatrix hierarchicalMatrix, H2Opus_Real* inpuVectors, H2Opus_Real* resultVectors);

#endif