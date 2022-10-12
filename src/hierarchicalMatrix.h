#ifndef __HIERARCHICALMATRIX__
#define __HIERARCHICALMATRIX__

struct HMatrixLevel {
    int numTiles;
    int* tileIndices;
    int* tileRanks;
    int* tileOffsets;
    H2Opus_Real* U;
    H2Opus_Real* V;
};

struct HMatrix {
    int numLevels;
    H2Opus_Real* diagonalBlocks;
    HMatrixLevel* levels;
};

void allocateHMatrix(HMatrix &matrix, int segmentSize, int numSegments, unsigned int numberOfInputPoints, unsigned int bucketSize) {
    matrix.numLevels = __builtin_ctz(numberOfInputPoints/bucketSize) + 1;
    cudaMalloc((void**) &matrix.diagonalBlocks, segmentSize*segmentSize*numSegments*sizeof(H2Opus_Real));
    matrix.levels = (HMatrixLevel*)malloc(matrix.numLevels*sizeof(HMatrixLevel));
}

void allocateHMatrixLevel() {
}

void freeHMatrix(HMatrix &matrix) {
    // TODO
}

void freeHMatrixLevel(){
    // TODO
}
#endif