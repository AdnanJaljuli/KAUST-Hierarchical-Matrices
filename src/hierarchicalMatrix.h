#ifndef __HIERARCHICALMATRIX__
#define __HIERARCHICALMATRIX__

struct WeakAdmissibility {
    int numLevels;
    int* numTiles;
    int** tileIndices;
};

void allocateWeakAdmissibilityStruct(WeakAdmissibility &WAStruct, unsigned int numberOfInputPoints, unsigned int bucketSize) {
    // TODO: parallelize
    WAStruct.numLevels = __builtin_ctz(numberOfInputPoints/bucketSize) + 1;
    WAStruct.numTiles = (int*)malloc((WAStruct.numLevels - 1)*sizeof(int));
    WAStruct.tileIndices = (int**)malloc((WAStruct.numLevels - 1)*sizeof(int*));

    unsigned int dim = 2;
    for(unsigned int level = 0; level < WAStruct.numLevels - 1; ++level) {
        unsigned int numTiles = 1 << (level + 1);
        WAStruct.numTiles[level] = numTiles;
        
        WAStruct.tileIndices[level] = (int*)malloc(numTiles*sizeof(int));
        for(unsigned int j = 0; j < numTiles; ++j) {
            int x;
            if(j%2 == 0) {
                x = 1;
            }
            else {
                x = -1;
            }
            unsigned int tileIndex = j*dim + j + x;
            WAStruct.tileIndices[level][j + x] = IndextoMOIndex_h(dim, tileIndex);
        }
        for(unsigned int j = 0; j < numTiles; ++j) {
            printf("%d ", WAStruct.tileIndices[level][j]);
        }
        printf("\n");
        dim <<= 1;
    }
}

void freeWeakAdmissbilityStruct() {
    // TODO
}

struct HMatrixLevel {
    int numTiles;
    int* tileIndices;
    int* tileScanRanks;
    H2Opus_Real* U;
    H2Opus_Real* V;
};

void allocateHMatrixLevel(HMatrixLevel &matrixLevel, int* ranks, WeakAdmissibility WAStruct, unsigned int level, H2Opus_Real* A, H2Opus_Real* B, int maxRows, int maxRank) {
    // TODO: dont copy all of A and B to U and V. Instead, copy only rank*maxRows of each tile
    // TODO: make a double pointer array to U and V
    matrixLevel.numTiles = WAStruct.numTiles[level - 1];
    // allocate U and V
    cudaMalloc((void**) &matrixLevel.U, maxRows*maxRank*matrixLevel.numTiles*sizeof(H2Opus_Real));
    cudaMalloc((void**) &matrixLevel.V, maxRows*maxRank*matrixLevel.numTiles*sizeof(H2Opus_Real));

    // copy A and B to U and V
    cudaMemcpy(matrixLevel.U, A, maxRows*maxRank*matrixLevel.numTiles*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
    cudaMemcpy(matrixLevel.V, B, maxRows*maxRank*matrixLevel.numTiles*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);

    //  do a scan of the ranks
    cudaMalloc((void**) &matrixLevel.tileScanRanks, matrixLevel.numTiles*sizeof(int));
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, ranks, matrixLevel.tileScanRanks, matrixLevel.numTiles);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, ranks, matrixLevel.tileScanRanks, matrixLevel.numTiles);
    
    // copy tile indices from WAStruct to here
    cudaMalloc((void**) &matrixLevel.tileIndices, matrixLevel.numTiles*sizeof(int));
    cudaMemcpy(matrixLevel.tileIndices, WAStruct.tileIndices[level - 1], matrixLevel.numTiles*sizeof(int), cudaMemcpyHostToDevice);
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
    matrix.numLevels = __builtin_ctz(numberOfInputPoints/bucketSize) + 1;
    matrix.levels = (HMatrixLevel*)malloc((matrix.numLevels - 2)*sizeof(HMatrixLevel));
}

void freeHMatrix(HMatrix &matrix) {
    // TODO
}

#endif