
#ifndef __EXPAND_MATRIX_HELPERS_CUH__
#define __EXPAND_MATRIX_HELPERS_CUH__

#include "HMatrix.cuh"

__global__ void expandLRMatrix(int num_segments, int maxSegmentSize, H2Opus_Real* expandedMatrix, TLR_Matrix matrix);
__global__ void errorInLRMatrix(int num_segments, int maxSegmentSize, H2Opus_Real* denseMatrix, H2Opus_Real* expandedMatrix, H2Opus_Real* error, H2Opus_Real* tmp);
void checkErrorInLRMatrix(uint64_t numSegments, uint64_t maxSegmentSize, TLR_Matrix matrix, H2Opus_Real* d_denseMatrix);
__global__ void expandHMatrix(HMatrixLevel matrixLevel, H2Opus_Real* output, int tileSize);
__global__ void errorInHMatrix(unsigned int numberOfInputPoints, double* denseMatrix, double* output, int* tileIndices, int batchSize, int batchUnitSize, double* error, double* tmp);
void checkErrorInHMatrixLevel(int numberOfInputPoints, int batchSize, int batchUnitSize, int bucketSize, HMatrixLevel matrixLevel, H2Opus_Real *denseMatrix);
void checkErrorInHMatrix(int numberOfInputPoints, int bucketSize, HMatrix hierarchicalMatrix, H2Opus_Real* d_denseMatrix);
__global__ void generateDenseMatrix_kernel(uint64_t numberOfInputPoints, uint64_t numSegments, uint64_t maxSegmentSize, int dimensionOfInputPoints, H2Opus_Real* denseMatrix, int* indexMap, int* offsetsSort, H2Opus_Real* pointCloud);
void generateDenseMatrix(int numberOfInputPoints, int numSegments, int maxSegmentSize, int dimensionOfInputPoints, H2Opus_Real* d_denseMatrix, int* &d_valuesIn, int* &d_offsetsSort, H2Opus_Real* &d_dataset);

#endif