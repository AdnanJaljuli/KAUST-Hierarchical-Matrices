#ifndef __HIERARCHICALMATRIX__
#define __HIERARCHICALMATRIX__

struct WeakAdmissibility {
    int* numTiles;
    int** tileIndices;
};

void allocateWeakAdmissibilityStruct(WeakAdmissibility &WAStruct, unsigned int numberOfInputPoints, unsigned int bucketSize) {
    // TODO: generalize on any n size
    int numLevels = __builtin_ctz(numberOfInputPoints/bucketSize) + 1;
    WAStruct.numTiles = (int*)malloc((numLevels - 1)*sizeof(int));
    WAStruct.tileIndices = (int**)malloc((numLevels - 1)*sizeof(int*));
    
    int dim = 2;
    for(unsigned int level = 0; level < numLevels - 1; ++level) {
        unsigned int numTiles = 1 << (level + 1);
        WAStruct.numTiles[level] = numTiles;
        
        WAStruct.tileIndices[level] = (int*)malloc(numTiles*sizeof(int));
        for(unsigned int j = 0; j < numTiles; ++j) {
            int x;
            if(j%2 == 0){
                x = 1;
            }
            else{
                x = -1;
            }
            unsigned int tileIndex = j*dim + j + x;
            WAStruct.tileIndices[level][j + x] = IndextoMOIndex_h(dim, tileIndex);
        }
        for(unsigned int j = 0; j < numTiles; ++j) {
            printf("%d ", WAStruct.tileIndices[level][j]);
        }
        printf("\n");
        dim *= 2;

    }
}

struct HMatrixLevel {
    int numTiles;
    int* tileIndices;
    int* tileScanRanks;
    int* tileOffsets;
    H2Opus_Real* U;
    H2Opus_Real* V;
};

void allocateHMatrixLevel(HMatrixLevel matrixLevel) {
    // TODO
}

void freeHMatrixLevel(){
    // TODO
}

struct HMatrix {
    int numLevels;
    H2Opus_Real* diagonalBlocks;
    HMatrixLevel* levels;
};

void allocateHMatrix(HMatrix &matrix, int segmentSize, int numSegments, unsigned int numberOfInputPoints, unsigned int bucketSize) {
    
    cudaMalloc((void**) &matrix.diagonalBlocks, segmentSize*segmentSize*numSegments*sizeof(H2Opus_Real));
    matrix.levels = (HMatrixLevel*)malloc(matrix.numLevels*sizeof(HMatrixLevel));
}

void freeHMatrix(HMatrix &matrix) {
    // TODO
}

#endif