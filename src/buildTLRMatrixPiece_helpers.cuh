
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
    unsigned int totalRankSum,
    int batchCount);

__global__ void copyRanks(
    unsigned int numElements,
    int* fromRanks,
    int* toRanks,
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

template <class T>
__global__ void expandTLRPiece(
    T *UPtr, T *VPtr,
    int *tileOffsets,
    T* expandedPiece,
    unsigned int numTilesInAxis,
    unsigned int numTilesInCol,
    unsigned int tileSize,
    bool isDiagonal);

template <class T>
__global__ void calcErrorInPiece (
    T *expandedTLRPiece,
    T *densePiece,
    unsigned int numTilesInCol,
    unsigned int tileSize,
    T *error, T *tmp);

#endif