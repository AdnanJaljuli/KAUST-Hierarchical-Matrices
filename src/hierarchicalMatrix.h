#ifndef __HIERARCHICALMATRIX__
#define __HIERARCHICALMATRIX__

struct HierarchicalMatrix {
    H2Opus_Real* U;
    H2Opus_Real* V;
    H2Opus_Real* diagonal;
    int** HMatrixExistingTiles;
    int** HMatrixExistingTileRanks;
    int** HMatrixExistingTileOffsets;
};

void allocateHierarchicalMatrix(HierarchicalMatrix &matrix, int numHMatrixLevels) {
    // TODO: cudaMalloc U and V
    matrix.HMatrixExistingTiles = (int**)malloc((numHMatrixLevels - 1)*sizeof(int*));
    matrix.HMatrixExistingTileRanks = (int**)malloc((numHMatrixLevels - 1)*sizeof(int*));
    matrix.HMatrixExistingTileOffsets = (int**)malloc((numHMatrixLevels - 1)*sizeof(int*));
}

void cudaFreeHierarchicalMatrix(HierarchicalMatrix &matrix) {
    // TODO: free arrays
}   

#endif