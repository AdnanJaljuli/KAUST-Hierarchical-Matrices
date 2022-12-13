
#ifndef BUILD_TLR_MATRIX_PIECE_H
#define BUILD_TLR_MATRIX_PIECE_H

#include "TLRMatrix.cuh"
#include "kDTree.cuh"
#include "precision.h"

template <class T>
void buildTLRMatrixPiece(
    TLR_Matrix *matrix,
    KDTree kdtree,
    T* d_dataset,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    T tolerance);

template void buildTLRMatrixPiece<H2Opus_Real>(
    TLR_Matrix *matrix,
    KDTree kdtree,
    H2Opus_Real* d_dataset,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    H2Opus_Real tolerance);

#endif