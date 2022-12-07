
#include "boundingBoxes.h"
#include "kDTreeHelpers.cuh"

template<typename T>
void allocateBoundingBox(BoundingBox *box, unsigned int dimensionOfInputPoints) {
    cudaMalloc((void**) &box->dimMax, dimensionOfInputPoints*sizeof(T));
    cudaMalloc((void**) &box->dimMin, dimensionOfInputPoints*sizeof(T));
}

template<typename T>
void allocateKDTreeBoundingBoxes(
    KDTreeBoundingBoxes *boxes,
    unsigned int numberOfInputPoints,
    unsigned int bucketSize,
    unsigned int dimensionOfInputPoints) {

        unsigned int numLevels = 1 + __builtin_ctz(upperPowerOfTwo(numberOfInputPoints)/bucketSize);
        boxes->levels = (BoundingBox**)malloc(numLevels*sizeof(BoundingBox*));

        for(unsigned int level = 0; level < numLevels; ++level) {
            unsigned int numNodes = 1<<level;
            boxes->levels[level] = (BoundingBox*)malloc(numNodes*sizeof(BoundingBox));

            for(unsigned int box = 0; box < numNodes; ++box) {
                allocateBoundingBox < T > (boxes->levels[level][box], dimensionOfInputPoints);
            }
        }
}
