
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
