#ifndef TLR_MATRIX_HELPERS_H
#define TLR_MATRIX_HELPERS_H

#include "TLRMatrix.cuh"

__global__ void fillARAArrays(int batchCount, int maxSegmentSize, int* d_rows_batch, int* d_cols_batch, int* d_ldm_batch, int* d_lda_batch, int* d_ldb_batch);
__global__ void copyTiles(int batchCount, int maxSegmentSize, int* d_ranks, int* d_scan_k, H2Opus_Real* d_U_tiled_segmented, H2Opus_Real* d_A, H2Opus_Real* d_V_tiled_segmented, H2Opus_Real* d_B, unsigned int maxRank);
__global__ void copyRanks(int num_segments, int maxSegmentSize, int* from_ranks, int* to_ranks);
__device__ H2Opus_Real getCorrelationLength(int dim);
__device__ H2Opus_Real interaction(int n, int dim, int col, int row, H2Opus_Real* pointCloud);
__global__ void generateDenseBlockColumn(unsigned int numberOfInputPoints, unsigned int maxSegmentSize, int dimensionOfInputPoints, H2Opus_Real* matrix, H2Opus_Real* pointCloud, KDTree kDTree, int columnIndex, H2Opus_Real* diagonal);

#endif
