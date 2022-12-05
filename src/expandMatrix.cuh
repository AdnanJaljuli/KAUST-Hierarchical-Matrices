
#ifndef EXPAND_MATRIX_H
#define EXPAND_MATRIX_H

#include "HMatrix.cuh"

__global__ void expandLRMatrix(int num_segments, int maxSegmentSize, H2Opus_Real* expandedMatrix, TLR_Matrix matrix);
__global__ void errorInLRMatrix(int num_segments, int maxSegmentSize, H2Opus_Real* denseMatrix, H2Opus_Real* expandedMatrix, H2Opus_Real* error, H2Opus_Real* tmp);
void checkErrorInLRMatrix(unsigned int numSegments, unsigned int maxSegmentSize, TLR_Matrix matrix, H2Opus_Real* d_denseMatrix);
__global__ void expandHMatrix(HMatrixLevel matrixLevel, H2Opus_Real* output, int tileSize);
__global__ void errorInHMatrix(unsigned int numberOfInputPoints, double* denseMatrix, double* output, int* tileIndices, int batchSize, int batchUnitSize, double* error, double* tmp);
void checkErrorInHMatrixLevel(int numberOfInputPoints, int batchSize, int batchUnitSize, int bucketSize, HMatrixLevel matrixLevel, H2Opus_Real *denseMatrix);
void checkErrorInHMatrix(int numberOfInputPoints, int bucketSize, HMatrix hierarchicalMatrix, H2Opus_Real* d_denseMatrix);
__global__ void generateDenseMatrix_kernel(unsigned int numberOfInputPoints, unsigned int numSegments, unsigned int maxSegmentSize, int dimensionOfInputPoints, H2Opus_Real* denseMatrix, int* indexMap, int* offsetsSort, H2Opus_Real* pointCloud);
void generateDenseMatrix(int numberOfInputPoints, int numSegments, int maxSegmentSize, int dimensionOfInputPoints, H2Opus_Real* d_denseMatrix, int* &d_valuesIn, int* &d_offsetsSort, H2Opus_Real* &d_dataset);
void checkErrorInHmatrixVecMult(unsigned int numberOfInputPoints, unsigned int vectorWidth, int numSegments, H2Opus_Real *d_denseMatrix, H2Opus_Real *d_inputVectors, H2Opus_Real *d_resultVectors);

#endif