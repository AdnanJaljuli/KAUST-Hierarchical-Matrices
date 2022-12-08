
#include "boundingBoxes.h"
#include "kDTreeHelpers.cuh"

void allocateKDTreeLevelBoundingBox(
    KDTreeLevelBoundingBoxes *boundingBoxLevel, 
    unsigned int numNodes, 
    unsigned int dimensionOfInputPoints) {

        boundingBoxLevel->maxBBData = (H2Opus_Real*)malloc(numNodes*dimensionOfInputPoints*sizeof(H2Opus_Real));
        boundingBoxLevel->minBBData = (H2Opus_Real*)malloc(numNodes*dimensionOfInputPoints*sizeof(H2Opus_Real));
        boundingBoxLevel->boundingBoxes = (BoundingBox*)malloc(numNodes*sizeof(BoundingBox));

        for(unsigned int node = 0; node < numNodes; ++node) {
            boundingBoxLevel->boundingBoxes[node].dimMax = &boundingBoxLevel->maxBBData[node*dimensionOfInputPoints];
            boundingBoxLevel->boundingBoxes[node].dimMin = &boundingBoxLevel->minBBData[node*dimensionOfInputPoints];
        }
}

void allocateKDTreeBoundingBoxes(
    KDTreeBoundingBoxes *boxes,
    unsigned int numberOfInputPoints,
    unsigned int bucketSize,
    unsigned int dimensionOfInputPoints) {

        unsigned int numLevels = 1 + __builtin_ctz(upperPowerOfTwo(numberOfInputPoints)/bucketSize);
        boxes->levels = (KDTreeLevelBoundingBoxes*)malloc(numLevels*sizeof(KDTreeLevelBoundingBoxes));

        for(unsigned int level = 0; level < numLevels; ++level) {
            unsigned int numNodes = 1<<level;
            allocateKDTreeLevelBoundingBox(&boxes->levels[level], numNodes, dimensionOfInputPoints);
        }
}

__global__ void resortSegmentedScan(
    H2Opus_Real *maxSegmentItem, H2Opus_Real *minSegmentItem,
    H2Opus_Real *bufferBBMax, H2Opus_Real *bufferBBMin,
    unsigned int dimensionOfInputPoints,
    unsigned int currentNumSegments) {

        unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
        if(i < currentNumSegments) {
            bufferBBMax[i*dimensionOfInputPoints + blockIdx.y] = maxSegmentItem[blockIdx.y*currentNumSegments + i];
            bufferBBMin[i*dimensionOfInputPoints + blockIdx.y] = minSegmentItem[blockIdx.y*currentNumSegments + i];
        }
}

// TODO: rename function
void copyMaxandMinToBoundingBoxes(
    KDTreeLevelBoundingBoxes BBlevel,
    H2Opus_Real *d_maxSegmentItem,
    H2Opus_Real *d_minSegmentItem,
    unsigned int level,
    unsigned int dimensionOfInputPoints,
    unsigned int currentNumSegments,
    H2Opus_Real *d_bufferBBMax, H2Opus_Real *d_bufferBBMin) {

        // resort maxsegmentitem and minsegmentitem into buffer arrays
        dim3 numThreadsPerBlock(1024);
        dim3 numBlocks((currentNumSegments + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x, dimensionOfInputPoints);
        resortSegmentedScan <<< numBlocks, numThreadsPerBlock >>> (
            d_maxSegmentItem, d_minSegmentItem, 
            d_bufferBBMax, d_bufferBBMin,
            dimensionOfInputPoints, currentNumSegments);
        
        // copy buffer arrays to bblevel in host memory
        cudaMemcpy(BBlevel.maxBBData, d_bufferBBMax, dimensionOfInputPoints*currentNumSegments*sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
        cudaMemcpy(BBlevel.minBBData, d_bufferBBMin, dimensionOfInputPoints*currentNumSegments*sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
}
