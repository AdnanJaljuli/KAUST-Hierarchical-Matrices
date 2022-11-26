
#include "HMatrixVectorMultiplication.cuh"
#include "HMatrix.cuh"

void HMatrixVecMult(unsigned int numberOfInputPoints, unsigned int bucketSize, unsigned int numSegments, unsigned int vectorWidth, HMatrix hierarchicalMatrix, H2Opus_Real* inpuVectors, H2Opus_Real* resultVectors) {
    // multiply diagonal blocks
    // diagonalXVec <<< >>> (numberOfInputPoints, bucketSize, numSegments, vectorWidth, hierarchicalMatrix.diagonal, inpuVectors);
}