
#include "HMatrixStructure.cuh"
#include "admissibilityCondition.cuh"
#include "helperKernels.cuh"
#include "kDTreeHelpers.cuh"

#define DEFAULT_ETA 1.0

void constructMatrixStruct_recursive(
    HMatrixStructure *HMatrixStruct,
    KDTreeBoundingBoxes BBox_u,
    KDTreeBoundingBoxes BBox_v,
    BoundingBox node_u,
    BoundingBox node_v,
    unsigned int dimensionOfInputPoints,
    unsigned int currentLevel,
    unsigned int maxDepth,
    float eta,
    std::function<bool(
        BoundingBox,
        BoundingBox,
        unsigned int,
        unsigned int,
        unsigned int,
        float)> isAdmissible) {

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
                    maxDepth,
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
                    maxDepth,
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
                    maxDepth,
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
                    maxDepth,
                    eta,
                    isAdmissible);

            }
}

void constructMatrixStruct(
    HMatrixStructure *HMatrixStruct,
    ADMISSIBILITY_CONDITION admissibilityCondition,
    KDTreeBoundingBoxes BBox1,
    KDTreeBoundingBoxes BBox2,
    unsigned int numberOfInputPoints,
    unsigned int dimensionOfInputPoints,
    unsigned int bucketSize,
    float eta = DEFAULT_ETA ) {

        unsigned int maxDepth = __builtin_ctz(upperPowerOfTwo(numberOfInputPoints)/bucketSize);

        // call recursive function
        if(admissibilityCondition == BOX_CENTER_ADMISSIBILITY) {
            constructMatrixStruct_recursive(
                HMatrixStruct,
                BBox1,
                BBox2,
                BBox1.levels[0].boundingBoxes[0],
                BBox2.levels[0].boundingBoxes[0],
                dimensionOfInputPoints,
                0,
                maxDepth,
                eta,
                &BBoxCenterAdmissibility);
        }
        else if(admissibilityCondition == WEAK_ADMISSIBILITY) {
            constructMatrixStruct_recursive(
                HMatrixStruct,
                BBox1,
                BBox2,
                BBox1.levels[0].boundingBoxes[0],
                BBox2.levels[0].boundingBoxes[0],
                dimensionOfInputPoints,
                0,
                maxDepth,
                eta,
                &weakAdmissibility);
        }
}

void constructHMatrixStructure(
    HMatrixStructure *HMatrixStruct,
    unsigned int numberOfInputPoints,
    unsigned int dimensionOfInputPoints,
    unsigned int bucketSize,
    ADMISSIBILITY_CONDITION admissibilityCondition,
    KDTreeBoundingBoxes BBox1,
    KDTreeBoundingBoxes BBox2) {

        // TODO: place the mallocs below in their own allocateHMatrixStructure function
        HMatrixStruct->numLevels = __builtin_ctz(numberOfInputPoints/bucketSize) + 1;
        HMatrixStruct->numTiles = (int*)malloc((HMatrixStruct->numLevels)*sizeof(int));
        HMatrixStruct->tileIndices = (int**)malloc((HMatrixStruct->numLevels)*sizeof(int*));
        for(unsigned int level = 0; level < HMatrixStruct->numLevels; ++level) {
            HMatrixStruct->numTiles[level] = 0;
            unsigned int numTiles = 1<<(level + 1);
            HMatrixStruct->tileIndices[level] = (int*)malloc(numTiles*(numTiles - 1)*sizeof(int));
        }

        constructMatrixStruct(
            HMatrixStruct,
            admissibilityCondition,
            BBox1,
            BBox2,
            numberOfInputPoints,
            dimensionOfInputPoints,
            bucketSize);
}

void freeHMatrixStructure(HMatrixStructure &HMatrixStruct) {
    free(HMatrixStruct.numTiles);
    for(unsigned int i = 0; i < HMatrixStruct.numLevels - 1; ++i) {
        free(HMatrixStruct.tileIndices[i]);
    }
    free(HMatrixStruct.tileIndices);
}