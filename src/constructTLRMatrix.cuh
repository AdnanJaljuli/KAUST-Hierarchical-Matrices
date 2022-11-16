
#ifndef __CREATE_LR_MATRIX_CUH__
#define __CREATE_LR_MATRIX_CUH__

// TODO: ask Izzat about the order of the below 3 header files. If their order is different, it throus a compilation error.
#include "kDTree.cuh"
#include "TLRMatrix.cuh"

typedef double H2Opus_Real;

__global__ void getTotalMem(uint64_t* totalMem, int* K, int* scan_K, int num_segments);
__global__ void fillARAArrays(int batchCount, int maxSegmentSize, int* d_rows_batch, int* d_cols_batch, int* d_ldm_batch, int* d_lda_batch, int* d_ldb_batch);
__global__ void copyTiles(int batchCount, int maxSegmentSize, int* d_ranks, int* d_scan_k, H2Opus_Real* d_U_tiled_segmented, H2Opus_Real* d_A, H2Opus_Real* d_V_tiled_segmented, H2Opus_Real* d_B);
__global__ void copyRanks(int num_segments, int maxSegmentSize, int* from_ranks, int* to_ranks);
uint64_t createColumnMajorLRMatrix(unsigned int numberOfInputPoints, unsigned int bucketSize, unsigned int dimensionOfInputPoints, TLR_Matrix &matrix, KDTree kDTree, H2Opus_Real* &d_dataset, float tolerance, int ARA_R);
__device__ H2Opus_Real getCorrelationLength(int dim);
__device__ H2Opus_Real interaction(int n, int dim, int col, int row, H2Opus_Real* pointCloud);
__global__ void generateDenseBlockColumn(uint64_t numberOfInputPoints, uint64_t maxSegmentSize, int dimensionOfInputPoints, H2Opus_Real* matrix, H2Opus_Real* pointCloud, KDTree kDTree, int columnIndex, H2Opus_Real* diagonal);

#endif
