
#include "HMatrixVectorMultiplication.cuh"
#include "cutlassDiagonalXVector.cuh"
#include "HMatrix.cuh"
#include <assert.h>

cudaError_t HMatrixVecMult(unsigned int numberOfInputPoints, unsigned int bucketSize, unsigned int numSegments, unsigned int vectorWidth, HMatrix hierarchicalMatrix, H2Opus_Real* inpuVectors, H2Opus_Real* resultVectors) {
    
    assert((bucketSize & (bucketSize - 1)) == 0);
    assert((vectorWidth & (vectorWidth - 1)) == 0);
    cudaError_t result;
    // multiply diagonal blocks first
    // TODO: replace this with cuBLAS batched gemm
    result = cutlassDiagonalXVec(numberOfInputPoints, bucketSize, numSegments, vectorWidth, hierarchicalMatrix.diagonalBlocks, inpuVectors, resultVectors);
    // TODO: add error checking and return error message if result != success
    
    // multiply rest of hierarchical matrix by the vector
    // result = cutlassHierarchicalXVec(numberOfInputPoints, bucketSize, numSegments, vectorWidth, hierarchicalMatrix, inpuVectors, resultVectors);

    return result;
}