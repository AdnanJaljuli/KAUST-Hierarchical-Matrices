#ifndef TLR_Matrix_H
#define TLR_Matrix_H
typedef double H2Opus_Real;

struct TLR_Matrix{
    unsigned int n;
    unsigned int blockSize;
    unsigned int numBlocks;
    int *blockRanks;
    int *blockOffsets;
    H2Opus_Real *U;
    H2Opus_Real *V;
    H2Opus_Real *diagonal;

    void del();
};

#endif