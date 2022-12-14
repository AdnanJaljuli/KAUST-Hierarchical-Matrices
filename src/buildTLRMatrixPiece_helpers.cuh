
#ifndef BUILD_TLR_MATRIX_PIECE_HELPERS_H
#define BUILD_TLR_MATRIX_PIECE_HELPERS_H

#include "kDTree.cuh"

bool isPieceDiagonal(unsigned int pieceMortonIndex);

void fillARAHelpers(
    unsigned int batchCount, int tileSize,
    int *d_rowsBatch, int *d_colsBatch,
    int *d_LDMBatch, int *d_LDABatch, int *d_LDBBatch);

template <class T>
void generateDenseTileCol(
    T *d_denseTileCol, 
    T *d_pointCloud, 
    KDTree kdtree, 
    unsigned int tileColIdx, 
    unsigned int tileSize, 
    unsigned int batchCount,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    unsigned int numTilesInAxis,
    bool isDiagonal);

__global__ void fillSortBits(
    unsigned int totalNumElements, 
    unsigned int numElementsInTile, 
    int *sortBits, 
    int *ranks);


#endif