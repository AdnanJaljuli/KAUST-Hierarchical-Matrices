
#ifndef HMATRIX_VECTOR_MULT_HELPERS_H
#define HMATRIX_VECTOR_MULT_HELPERS_H

typedef double H2Opus_Real;

__global__ void diagonalXVec(unsigned int numberOfInputPoints, unsigned int  bucketSize, unsigned int  numSegments, unsigned int  vectorWidth, H2Opus_Real *diagonal, H2Opus_Real *inpuVectors, H2Opus_Real *resultVectors);
cudaError_t cutlassDiagonalXVec(unsigned int numberOfInputPoints, unsigned int  bucketSize, unsigned int  numSegments, unsigned int  vectorWidth, H2Opus_Real *diagonal, H2Opus_Real *inpuVectors, H2Opus_Real *resultVectors);

#endif