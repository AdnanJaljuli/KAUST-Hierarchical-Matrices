#ifndef __HIERARCHICALMATRIX__
#define __HIERARCHICALMATRIX__

struct WeakAdmissibility {
    int* numTiles;
    int** tileIndices;
};

void allocateWeakAdmissibilityStruct(WeakAdmissibility &WAStruct) {
    // TODO: generalize
    WAStruct.numTiles = (int*)malloc(2*sizeof(int));
    WAStruct.numTiles[0] = 2;
    WAStruct.numTiles[1] = 8;
    WAStruct.tileIndices = (int**)malloc(2*sizeof(int*));
    WAStruct.tileIndices[0] = (int*)malloc(2*sizeof(int));
    WAStruct.tileIndices[1] = (int*)malloc(8*sizeof(int));
    WAStruct.tileIndices[0][0] = 1;
    WAStruct.tileIndices[0][1] = 2;

    WAStruct.tileIndices[1][0] = 0;
    WAStruct.tileIndices[1][1] = 1;
    WAStruct.tileIndices[1][2] = 2;
    WAStruct.tileIndices[1][3] = 3;
    WAStruct.tileIndices[1][4] = 12;
    WAStruct.tileIndices[1][5] = 13;
    WAStruct.tileIndices[1][6] = 14;
    WAStruct.tileIndices[1][7] = 15;
}

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

void allocateHMatrixLevel(HMatrixLevel matrixLevel) {
}

void freeHMatrix(HMatrix &matrix) {
    // TODO
}

void freeHMatrixLevel(){
    // TODO
}
#endif