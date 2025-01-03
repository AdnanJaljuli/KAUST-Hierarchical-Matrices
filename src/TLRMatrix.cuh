#ifndef TLR_MATRIX_H
#define TLR_MATRIX_H

#include <cstdint>
#include "kDTree.cuh"

typedef double H2Opus_Real;
enum Ordering {COLUMN_MAJOR, MORTON};

struct TLR_Matrix{
    Ordering ordering;
    unsigned int n;
    unsigned int blockSize;
    unsigned int numBlocks;
    // TODO: stop storing blockRanks because it is redundant with blockOffsets
    int *blockRanks;
    int *blockOffsets;
    H2Opus_Real *U;
    H2Opus_Real *V;
    H2Opus_Real *diagonal;
};

// TODO: have an allocateMatrix function

void freeMatrix(TLR_Matrix matrix);

#endif