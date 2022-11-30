
#ifndef CUTLASS_DIAGONAL_X_VECTORS_H
#define CUTLASS_DIAGONAL_X_VECTORS_H

typedef double H2Opus_Real;

cudaError_t cutlassDiagonalXVec(unsigned int numberOfInputPoints, unsigned int  bucketSize, unsigned int  numSegments, unsigned int  vectorWidth, H2Opus_Real *diagonal, H2Opus_Real *inpuVectors, H2Opus_Real *resultVectors);

#endif