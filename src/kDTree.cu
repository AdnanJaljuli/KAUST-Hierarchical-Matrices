#include "kDTree.cuh"
#include "boundingBoxes.h"
#include "config.h"
#include "kDTreeHelpers.cuh"

void allocateKDTree(
    KDTree &tree, 
    unsigned int numberOfInputPoints, 
    unsigned int dimensionOfInputPoints, 
    unsigned int leafSize, 
    DIVISION_METHOD divMethod) {

        tree.N = numberOfInputPoints;
        tree.nDim = dimensionOfInputPoints;
        tree.numLeaves = 1;
        tree.leafSize = leafSize;
        int maxNumSegments;
        if(divMethod == FULL_TREE) {
            maxNumSegments = 1<<(getMaxSegmentSize(numberOfInputPoints, leafSize).second);
        }
        else {
            maxNumSegments = (numberOfInputPoints + leafSize - 1)/leafSize;
        }

        cudaMalloc((void**) &tree.leafIndices, numberOfInputPoints*sizeof(int)); // TODO: rename to indexMap
        cudaMalloc((void**) &tree.leafOffsets, (maxNumSegments + 1)*sizeof(int));

        allocateKDTreeBoundingBoxes(
            &tree.boundingBoxes,
            numberOfInputPoints,
            leafSize,
            dimensionOfInputPoints);
}

void freeKDTree(KDTree tree){
    cudaFree(tree.leafIndices);
    cudaFree(tree.leafOffsets);
}