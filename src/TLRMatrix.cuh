#ifndef TLR_MATRIX_H
#define TLR_MATRIX_H

#include "kDTree.cuh"

#include <cstdint>
#include <vector>

typedef double H2Opus_Real;
enum Ordering {COLUMN_MAJOR, MORTON};

struct TLR_Matrix {
    Ordering ordering;
    unsigned int blockSize;
    unsigned int numBlocksInAxis;
    thrust::device_vector<int> d_blockOffsets;
    thrust::device_vector<H2Opus_Real> d_U;
    thrust::device_vector<H2Opus_Real> d_V;
    thrust::device_vector<H2Opus_Real> d_diagonal;
};

// TODO: have an allocateMatrix function

void freeMatrix(TLR_Matrix *matrix);

#endif