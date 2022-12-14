
#include "HMatrixStructure.cuh"
#include "admissibilityFunctions.cuh"
#include "helperKernels.cuh"
#include "kDTreeHelpers.cuh"
#include "precision.h"

#include <functional>
#include <vector>

template <class T>
void constructMatrixStruct_recursive(
    HMatrixStructure *HMatrixStruct,
    KDTreeBoundingBoxes BBoxTree_u,
    KDTreeBoundingBoxes BBoxTree_v,
    BoundingBox node_u,
    BoundingBox node_v,
    unsigned int dimensionOfInputPoints,
    unsigned int currentLevel,
    Admissibility<T> &admissibility) {

        unsigned int maxDepth = HMatrixStruct->numLevels - 1;

        bool isDiagonal = (node_u.index == node_v.index);
        bool isLeafNode = (currentLevel == maxDepth);

        if(isDiagonal && isLeafNode) {
            return;
        }
        else if(isLeafNode || admissibility(node_u, node_v)) {
            // write to HMatrixStruct
            unsigned int numRows = 1<<currentLevel;
            unsigned int tileIndex = columnMajor2Morton(numRows, node_u.index*numRows + node_v.index);
            HMatrixStruct->tileIndices.at(currentLevel - 1).push_back(tileIndex);
            ++HMatrixStruct->numTiles[currentLevel - 1];
            return;
        }
        else {
            for(unsigned int i = 0; i < 2; ++i) {
                for(unsigned int j = 0; j < 2; ++j) {
                    constructMatrixStruct_recursive(
                        HMatrixStruct,
                        BBoxTree_u,
                        BBoxTree_v,
                        BBoxTree_u.levels[currentLevel + 1].boundingBoxes[2*node_u.index + i],
                        BBoxTree_v.levels[currentLevel + 1].boundingBoxes[2*node_v.index + j],
                        dimensionOfInputPoints,
                        currentLevel + 1,
                        admissibility);
                }
            }
        }
}

template <class T>
void constructHMatrixStructure(
    HMatrixStructure *HMatrixStruct,
    Admissibility<T> &admissibility,
    KDTree rowTree,
    KDTree columnTree) {

        constructMatrixStruct_recursive(
            HMatrixStruct,
            rowTree.boundingBoxes,
            columnTree.boundingBoxes,
            columnTree.boundingBoxes.levels[0].boundingBoxes[0],
            columnTree.boundingBoxes.levels[0].boundingBoxes[0],
            rowTree.nDim,
            0,
            admissibility);
}

void allocateHMatrixStructure(HMatrixStructure *HMatrixStruct, unsigned int numLevels) {
    HMatrixStruct->numLevels = numLevels;
    HMatrixStruct->numTiles.resize(HMatrixStruct->numLevels);
    HMatrixStruct->tileIndices.resize(HMatrixStruct->numLevels);
    memset(&HMatrixStruct->numTiles[0], 0, sizeof(HMatrixStruct->numTiles[0]) * HMatrixStruct->numTiles.size());
}

void freeHMatrixStructure(HMatrixStructure *HMatrixStruct) {
    HMatrixStruct->numTiles.clear();
    for(unsigned int level = 0; level < HMatrixStruct->numLevels; ++level) {
        HMatrixStruct->tileIndices[level].clear();
    }
    HMatrixStruct->tileIndices.clear();
}

template void constructHMatrixStructure<H2Opus_Real>(
    HMatrixStructure *HMatrixStruct,
    Admissibility<H2Opus_Real> &admissibility,
    KDTree rowTree,
    KDTree columnTree);