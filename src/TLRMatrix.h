#ifndef __TLR_MATRIX_H__
#define __TLR_MATRIX_H__

typedef double H2Opus_Real;
enum Ordering {COLUMN_MAJOR, MORTON};

struct TLR_Matrix{
    Ordering ordering;
    unsigned int n;
    unsigned int blockSize;
    unsigned int numBlocks;
    int *blockRanks;
    int *blockOffsets;
    H2Opus_Real *U;
    H2Opus_Real *V;
    H2Opus_Real *diagonal;

    void cudaFreeMatrix();
};

void cudaFreeMatrix(TLR_Matrix matrix){
    cudaFree(matrix.blockRanks);
    cudaFree(matrix.blockOffsets);
    cudaFree(matrix.U);
    cudaFree(matrix.V);
    cudaFree(matrix.diagonal);
}

#endif