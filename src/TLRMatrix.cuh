#ifndef TLR_MATRIX_H
#define TLR_MATRIX_H

#include "kDTree.cuh"
#include "precision.h"

#include <cstdint>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

typedef double H2Opus_Real;
enum Ordering {COLUMN_MAJOR, MORTON};

struct TLR_Matrix {
    Ordering ordering;
    unsigned int tileSize;
    unsigned int numTilesInAxis;
    thrust::device_vector<int> d_tileOffsets;
    thrust::device_vector<H2Opus_Real> d_U;
    thrust::device_vector<H2Opus_Real> d_V;
};

// TODO: have an allocateMatrix function

void freeMatrix(TLR_Matrix *matrix);

#endif