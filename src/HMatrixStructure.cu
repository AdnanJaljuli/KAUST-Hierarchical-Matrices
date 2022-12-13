
#include "HMatrixStructure.cuh"
#include "admissibilityFunctions.cuh"
#include "helperKernels.cuh"
#include "kDTreeHelpers.cuh"

#include <functional>
#include <vector>

void constructMatrixStruct_recursive(
    HMatrixStructure *HMatrixStruct,
    KDTreeBoundingBoxes BBox_u,
    KDTreeBoundingBoxes BBox_v,
    BoundingBox node_u,
    BoundingBox node_v,
    unsigned int dimensionOfInputPoints,
    unsigned int currentLevel,
    float eta,
    Admissibility &admissibility) {

            unsigned int maxDepth = HMatrixStruct->numLevels - 1;

            bool isDiagonal = (node_u.index == node_v.index);
            bool isLeafNode = (currentLevel == maxDepth);

            if(isDiagonal && isLeafNode) {
                return;
            }
            else if(isLeafNode || admissibility(node_u, node_v)) {
                // write to HMatrixStruct
                unsigned int numRows = 1<<currentLevel;
                unsigned int tileIndex = CMIndextoMOIndex(numRows, node_u.index*numRows + node_v.index);
                HMatrixStruct->tileIndices.at(currentLevel - 1).push_back(tileIndex);
                ++HMatrixStruct->numTiles[currentLevel - 1];
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
                    admissibility);

                constructMatrixStruct_recursive(
                    HMatrixStruct,
                    BBox_u,
                    BBox_v,
                    BBox_u.levels[currentLevel + 1].boundingBoxes[2*node_u.index + 1],
                    BBox_v.levels[currentLevel + 1].boundingBoxes[2*node_v.index],
                    dimensionOfInputPoints,
                    currentLevel + 1,
                    eta,
                    admissibility);

                constructMatrixStruct_recursive(
                    HMatrixStruct,
                    BBox_u,
                    BBox_v,
                    BBox_u.levels[currentLevel + 1].boundingBoxes[2*node_u.index],
                    BBox_v.levels[currentLevel + 1].boundingBoxes[2*node_v.index + 1],
                    dimensionOfInputPoints,
                    currentLevel + 1,
                    eta,
                    admissibility);

                constructMatrixStruct_recursive(
                    HMatrixStruct,
                    BBox_u,
                    BBox_v,
                    BBox_u.levels[currentLevel + 1].boundingBoxes[2*node_u.index + 1],
                    BBox_v.levels[currentLevel + 1].boundingBoxes[2*node_v.index + 1],
                    dimensionOfInputPoints,
                    currentLevel + 1,
                    eta,
                    admissibility);
            }
}

void constructMatrixStructure(
    HMatrixStructure *HMatrixStruct,
    Admissibility &admissibility,
    KDTreeBoundingBoxes BBoxTree1,
    KDTreeBoundingBoxes BBoxTree2,
    unsigned int dimensionOfInputPoints,
    float eta = 1.0f ) {

        constructMatrixStruct_recursive(
            HMatrixStruct,
            BBoxTree1,
            BBoxTree2,
            BBoxTree1.levels[0].boundingBoxes[0],
            BBoxTree2.levels[0].boundingBoxes[0],
            dimensionOfInputPoints,
            0,
            eta,
            admissibility);
}

void constructHMatrixStructure(
    HMatrixStructure *HMatrixStruct,
    Admissibility &admissibility,
    KDTree rowTree,
    KDTree columnTree) {

        // TODO: place the mallocs below in their own allocateHMatrixStructure function
        constructMatrixStructure(
            HMatrixStruct,
            admissibility,
            rowTree.boundingBoxes,
            columnTree.boundingBoxes,
            rowTree.nDim);
}

void allocateHMatrixStructure(HMatrixStructure *HMatrixStruct, unsigned int numLevels) {
    HMatrixStruct->numLevels = numLevels;
    HMatrixStruct->numTiles.resize(HMatrixStruct->numLevels);
    HMatrixStruct->tileIndices.resize(HMatrixStruct->numLevels);
    memset(&HMatrixStruct->numTiles[0], 0, sizeof(HMatrixStruct->numTiles[0]) * HMatrixStruct->numTiles.size());
}

void freeHMatrixStructure(HMatrixStructure &HMatrixStruct) {
}