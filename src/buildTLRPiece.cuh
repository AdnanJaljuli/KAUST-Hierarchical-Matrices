
#ifndef BUILD_TLR_MATRIX_PIECE_H
#define BUILD_TLR_MATRIX_PIECE_H

#include "TLRMatrix.cuh"
#include "kDTree.cuh"
#include "precision.h"

template <class T>
void buildTLRPiece(
    TLR_Matrix *matrix,
    KDTree kdtree,
    T* d_pointDataset,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    T tol);

template <class T>
void checkErrorInTLRPiece(
    TLR_Matrix matrix,
    KDTree kdtree,
    T* d_pointCloud,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis);

#endif