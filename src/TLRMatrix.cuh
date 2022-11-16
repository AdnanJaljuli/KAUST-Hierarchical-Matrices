#ifndef __TLR_MATRIX_H__
#define __TLR_MATRIX_H__

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

void freeMatrix(TLR_Matrix matrix);

__device__ H2Opus_Real getCorrelationLength(int dim);
__device__ H2Opus_Real interaction(int n, int dim, int col, int row, H2Opus_Real* pointCloud);
__global__ void generateDenseBlockColumn(uint64_t numberOfInputPoints, uint64_t maxSegmentSize, int dimensionOfInputPoints, H2Opus_Real* matrix, H2Opus_Real* pointCloud, KDTree kDTree, int columnIndex, H2Opus_Real* diagonal);

#endif