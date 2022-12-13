
#ifndef CONSTRUCT_TLR_MATRIX_H
#define CONSTRUCT_TLR_MATRIX_H

#include "kDTree.cuh"
#include "TLRMatrix.cuh"

typedef double H2Opus_Real;

uint64_t createColumnMajorLRMatrix(unsigned int numberOfInputPoints, unsigned int leafSize, unsigned int dimensionOfInputPoints, TLR_Matrix &matrix, KDTree kDTree, H2Opus_Real* &d_dataset, float tolerance, int ARA_R);

#endif
