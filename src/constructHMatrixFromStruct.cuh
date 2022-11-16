
#ifndef __GENERATE_HIERARCHICALMATRIX_H__
#define __GENERATE_HIERARCHICALMATRIX_H__

#include "HMatrix.cuh"
#include "TLRMatrix.cuh"

void generateHMatrixFromStruct(unsigned int numberOfInputPoints, unsigned int bucketSize, unsigned int numSegments, unsigned int segmentSize, TLR_Matrix mortonOrderedMatrix, int ARA_R, float tolerance, HMatrix hierarchicalMatrix, WeakAdmissibility WAStruct);
__global__ void fillScanRankPtrs(int **d_scanRanksPtrs, int *d_scanRanks, int batchUnitSize, int batchSize);
__global__ void fillLRARAArrays(int batchSize, int maxRows, int* d_rowsBatch, int* d_colsBatch, int* d_LDABatch, int* d_LDBBatch);
void generateScanRanks(int batchSize, int batchUnitSize, int *ranks, int *scanRanks, int **scanRanksPtrs, int *levelTileIndices);
#endif
