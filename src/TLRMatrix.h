#ifndef __TLR_MATRIX_H__
#define __TLR_MATRIX_H__

typedef double H2Opus_Real;
enum Ordering {COLUMN_MAJOR, MORTON};

struct TLR_Matrix{
    Ordering type;
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

void TLR_Matrix::cudaFreeMatrix(){
    cudaFree(blockRanks);
    cudaFree(blockOffsets);
    cudaFree(U);
    cudaFree(V);
    cudaFree(diagonal);
}

#endif