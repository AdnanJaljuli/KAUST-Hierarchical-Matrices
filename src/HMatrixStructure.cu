
#include "HMatrixStructure.cuh"
#include "admissibilityCondition.cuh"
#include "helperKernels.cuh"
#include "kDTreeHelpers.cuh"

#include <functional>

#define DEFAULT_ETA 1.0

void constructMatrixStruct_recursive(
    HMatrixStructure *HMatrixStruct,
    KDTreeBoundingBoxes BBox_u,
    KDTreeBoundingBoxes BBox_v,
    BoundingBox node_u,
    BoundingBox node_v,
    unsigned int dimensionOfInputPoints,
    unsigned int currentLevel,
    float eta,
    std::function<bool(
        BoundingBox,
        BoundingBox,
        unsigned int,
        unsigned int,
        unsigned int,
        float)> isAdmissible) {

            unsigned int maxDepth = HMatrixStruct->numLevels - 1;

            bool isDiagonal = (node_u.index == node_v.index);
            bool isLeafNode = (currentLevel == maxDepth);

            if(isDiagonal && isLeafNode) {
                return;
            }
            else if(isLeafNode || isAdmissible(node_u, node_v, dimensionOfInputPoints, currentLevel, maxDepth, eta)) {
                // TODO: write to HMatrixStruct
                unsigned int numRows = 1<<currentLevel;
                unsigned int tileIndex = CMIndextoMOIndex(numRows, node_u.index*numRows + node_v.index);
                HMatrixStruct->tileIndices[currentLevel - 1][HMatrixStruct->numTiles[currentLevel - 1]++] = tileIndex;
                return;
            }
            else {
                constructMatrixStruct_recursive(
                    HMatrixStruct,
                    BBox_u,
                    BBox_v,
                    BBox_u.levels[currentLevel + 1].boundingBoxes[2*node_u.index],
                    BBox_v.levels[currentLevel + 1].boundingBoxes[2*node_v.index],
                    dimensionOfInputPoints,
                    currentLevel + 1,
                    eta,
                    isAdmissible);

                constructMatrixStruct_recursive(
                    HMatrixStruct,
                    BBox_u,
                    BBox_v,
                    BBox_u.levels[currentLevel + 1].boundingBoxes[2*node_u.index + 1],
                    BBox_v.levels[currentLevel + 1].boundingBoxes[2*node_v.index],
                    dimensionOfInputPoints,
                    currentLevel + 1,
                    eta,
                    isAdmissible);

                constructMatrixStruct_recursive(
                    HMatrixStruct,
                    BBox_u,
                    BBox_v,
                    BBox_u.levels[currentLevel + 1].boundingBoxes[2*node_u.index],
                    BBox_v.levels[currentLevel + 1].boundingBoxes[2*node_v.index + 1],
                    dimensionOfInputPoints,
                    currentLevel + 1,
                    eta,
                    isAdmissible);

                constructMatrixStruct_recursive(
                    HMatrixStruct,
                    BBox_u,
                    BBox_v,
                    BBox_u.levels[currentLevel + 1].boundingBoxes[2*node_u.index + 1],
                    BBox_v.levels[currentLevel + 1].boundingBoxes[2*node_v.index + 1],
                    dimensionOfInputPoints,
                    currentLevel + 1,
                    eta,
                    isAdmissible);

            }
}

void constructMatrixStructure(
    HMatrixStructure *HMatrixStruct,
    std::function<bool(
        BoundingBox,
        BoundingBox,
        unsigned int,
        unsigned int,
        unsigned int,
        float)> isAdmissible,
    KDTreeBoundingBoxes BBoxTree1,
    KDTreeBoundingBoxes BBoxTree2,
    unsigned int dimensionOfInputPoints,
    float eta = DEFAULT_ETA ) {

        constructMatrixStruct_recursive(
            HMatrixStruct,
            BBoxTree1,
            BBoxTree2,
            BBoxTree1.levels[0].boundingBoxes[0],
            BBoxTree2.levels[0].boundingBoxes[0],
            dimensionOfInputPoints,
            0,
            eta,
            isAdmissible);
}

void constructHMatrixStructure(
    HMatrixStructure *HMatrixStruct,
    std::function<bool(
        BoundingBox,
        BoundingBox,
        unsigned int,
        unsigned int,
        unsigned int,
        float)> isAdmissible,
    KDTree rowTree,
    KDTree columnTree) {

        // TODO: place the mallocs below in their own allocateHMatrixStructure function
        HMatrixStruct->numLevels = rowTree.numLevels;
        HMatrixStruct->numTiles = (int*)malloc((HMatrixStruct->numLevels)*sizeof(int));
        HMatrixStruct->tileIndices = (int**)malloc((HMatrixStruct->numLevels)*sizeof(int*));
        for(unsigned int level = 0; level < HMatrixStruct->numLevels; ++level) {
            HMatrixStruct->numTiles[level] = 0;
            unsigned int numTiles = 1<<(level + 1);
            HMatrixStruct->tileIndices[level] = (int*)malloc(numTiles*(numTiles - 1)*sizeof(int));
        }

        constructMatrixStructure(
            HMatrixStruct,
            isAdmissible,
            rowTree.boundingBoxes,
            columnTree.boundingBoxes,
            rowTree.nDim);
}

void freeHMatrixStructure(HMatrixStructure &HMatrixStruct) {
    free(HMatrixStruct.numTiles);
    for(unsigned int i = 0; i < HMatrixStruct.numLevels - 1; ++i) {
        free(HMatrixStruct.tileIndices[i]);
    }
    free(HMatrixStruct.tileIndices);
}