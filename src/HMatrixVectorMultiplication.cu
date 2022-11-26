
#include "HMatrixVectorMultiplication.cuh"
#include "HMatrixVectorMultHelpers.cuh"
#include "HMatrix.cuh"
#include <assert.h>

void HMatrixVecMult(unsigned int numberOfInputPoints, unsigned int bucketSize, unsigned int numSegments, unsigned int vectorWidth, HMatrix hierarchicalMatrix, H2Opus_Real* inpuVectors, H2Opus_Real* resultVectors) {
    // multiply diagonal blocks
    assert((bucketSize & (bucketSize - 1)) == 0);
    assert((vectorWidth & (vectorWidth - 1)) == 0); // TODO: is vectorWidth always a power of two?

    dim3 numThreadsPerBlock(min(32, vectorWidth), min(32, bucketSize));
    dim3 numBlocks(vectorWidth/numThreadsPerBlock.x, bucketSize/numThreadsPerBlock.y, numSegments);
    unsigned int sharedMemSize = numThreadsPerBlock.x*numThreadsPerBlock.y*sizeof(H2Opus_Real);
    diagonalXVec <<< numBlocks, numThreadsPerBlock, sharedMemSize >>> (numberOfInputPoints, bucketSize, numSegments, vectorWidth, hierarchicalMatrix.diagonalBlocks, inpuVectors, resultVectors);
}