#include "kDTree.cuh"
#include "boundingBoxes.h"
#include "config.h"
#include "kDTreeHelpers.cuh"

void allocateKDTree(
    KDTree &tree, 
    unsigned int numberOfInputPoints, 
    unsigned int dimensionOfInputPoints, 
    unsigned int bucketSize, 
    DIVISION_METHOD divMethod) {

        tree.numSegments = 1;
        tree.segmentSize = bucketSize;
        int maxNumSegments;
        if(divMethod == FULL_TREE) {
            maxNumSegments = 1<<(getMaxSegmentSize(numberOfInputPoints, bucketSize).second);
        }
        else {
            maxNumSegments = (numberOfInputPoints + bucketSize - 1)/bucketSize;
        }
        
        cudaMalloc((void**) &tree.segmentIndices, numberOfInputPoints*sizeof(int)); // TODO: rename to indexMap
        cudaMalloc((void**) &tree.segmentOffsets, (maxNumSegments + 1)*sizeof(int));

        allocateKDTreeBoundingBoxes(
            &tree.boundingBoxes,
            numberOfInputPoints,
            bucketSize,
            dimensionOfInputPoints);
}

void freeKDTree(KDTree tree){
    cudaFree(tree.segmentIndices);
    cudaFree(tree.segmentOffsets);
}