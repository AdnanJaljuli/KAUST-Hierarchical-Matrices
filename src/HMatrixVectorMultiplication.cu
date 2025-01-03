
#include "HMatrixVectorMultiplication.cuh"
#include "cublas_v2.h"
#include "cutlassDiagonalXVector.cuh"
#include "cutlassHMatrixXVector.cuh"
#include "HMatrix.cuh"
#include <assert.h>
#include <stdio.h>

cudaError_t HMatrixVecMult(unsigned int numberOfInputPoints, unsigned int bucketSize, unsigned int numSegments, unsigned int vectorWidth, HMatrix hierarchicalMatrix, H2Opus_Real* inpuVectors, H2Opus_Real* resultVectors) {

    assert((bucketSize & (bucketSize - 1)) == 0);
    assert((vectorWidth & (vectorWidth - 1)) == 0);
    cudaError_t result;
    // multiply diagonal blocks first
    // TODO: replace this with cuBLAS batched gemm
    result = cutlassDiagonalXVec(numberOfInputPoints, bucketSize, numSegments, vectorWidth, hierarchicalMatrix.diagonalBlocks, inpuVectors, resultVectors);
    // TODO: add error checking and return error message if result != success

    // multiply rest of hierarchical matrix by the vector
    H2Opus_Real *d_bufferVectors;
    cudaMalloc((void**) &d_bufferVectors, numberOfInputPoints*vectorWidth*sizeof(H2Opus_Real));
    result = cutlassHierarchicalXVec(numberOfInputPoints, bucketSize, numSegments, vectorWidth, hierarchicalMatrix, inpuVectors, d_bufferVectors, resultVectors);
    cudaFree(d_bufferVectors);

    return result;
}