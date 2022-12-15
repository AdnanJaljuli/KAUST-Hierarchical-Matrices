
#ifndef BUILD_TLR_MATRIX_PIECE_HELPERS_H
#define BUILD_TLR_MATRIX_PIECE_HELPERS_H

#include "kDTree.cuh"
#include "TLRMatrix.cuh"

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

template <class T>
void copyTiles(
    TLR_Matrix *matrix, 
    T *d_UOutput, T *d_VOutput, 
    int *d_scannedRanks, 
    int maxRank, 
    int batchCount);

__global__ void copyScannedRanks(
    unsigned int numElements,
    int* fromScannedRanks,
    int* toScannedRanks,
    unsigned int tileColIdx,
    bool isDiagonal);

template <class T>
void generateDensePiece(
    T *d_densePiece, 
    KDTree kdtree, 
    T *d_pointCloud, 
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    unsigned int numTilesInAxis, unsigned int numTilesInCol,
    bool isDiagonal);

#endif