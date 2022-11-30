
#include "HMatrixVectorMultiplication.cuh"
#include "HMatrixVectorMultHelpers.cuh"
#include "HMatrix.cuh"
#include <assert.h>

cudaError_t HMatrixVecMult(unsigned int numberOfInputPoints, unsigned int bucketSize, unsigned int numSegments, unsigned int vectorWidth, HMatrix hierarchicalMatrix, H2Opus_Real* inpuVectors, H2Opus_Real* resultVectors) {
    
    assert((bucketSize & (bucketSize - 1)) == 0);
    assert((vectorWidth & (vectorWidth - 1)) == 0);

    // multiply diagonal blocks first
    cudaError_t result = cutlassDiagonalXVec(numberOfInputPoints, bucketSize, numSegments, vectorWidth, hierarchicalMatrix.diagonalBlocks, inpuVectors, resultVectors);
    // TODO: add error checking and return error message if result != success
    
    // multiply rest of hierarchical matrix by the vector
    

    return result;
}