#ifndef __KD_TREE_H__
#define __KD_TREE_H__

struct KDTree{
    uint64_t numSegments;
    uint64_t segmentSize;
    int *segmentIndices;
    int *segmentOffsets;
};

void allocateKDTree(KDTree &tree, unsigned int numberOfInputPoints, unsigned int bucketSize){
    tree.numSegments = (numberOfInputPoints + bucketSize - 1)/bucketSize;
    tree.segmentSize = bucketSize;
    cudaMalloc((void**) &tree.segmentIndices, numberOfInputPoints*sizeof(int)); // TODO: rename to indexMap
    cudaMalloc((void**) &tree.segmentOffsets, (tree.numSegments + 1)*sizeof(int));
}

void cudaFreeKDTree(KDTree tree){
    cudaFree(tree.segmentIndices);
    cudaFree(tree.segmentOffsets);
}   

#endif