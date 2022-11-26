
#include "HMatrixVectorMultiplication.cuh"
#include "HMatrixVectorMultHelpers.cuh"
#include "HMatrix.cuh"
#include <assert.h>

void HMatrixVecMult(unsigned int numberOfInputPoints, unsigned int bucketSize, unsigned int numSegments, unsigned int vectorWidth, HMatrix hierarchicalMatrix, H2Opus_Real* inpuVectors, H2Opus_Real* resultVectors) {
    // multiply diagonal blocks
    assert((bucketSize & (bucketSize - 1)) == 0);
    assert((vectorWidth & (vectorWidth - 1)) == 0); // TODO: is vectorWidth always a power of two?
    cudaError_t result = cutlassDiagonalXVec(numberOfInputPoints, bucketSize, numSegments, vectorWidth, hierarchicalMatrix.diagonalBlocks, inpuVectors, resultVectors);

    // loop over levels and for each level multiply the tiles

    // TODO: optimize grid allocation. what if bucketSize = 16 and vectorWidth = 64? we can allocate two blocks instead of four
    // dim3 numThreadsPerBlock(min(32, vectorWidth), min(32, bucketSize));
    // dim3 numBlocks(vectorWidth/numThreadsPerBlock.x, bucketSize/numThreadsPerBlock.y, numSegments);
    // unsigned int sharedMemSize = 2*numThreadsPerBlock.x*numThreadsPerBlock.y*sizeof(H2Opus_Real);
    // diagonalXVec <<< numBlocks, numThreadsPerBlock, sharedMemSize >>> (numberOfInputPoints, bucketSize, numSegments, vectorWidth, hierarchicalMatrix.diagonalBlocks, inpuVectors, resultVectors);
}