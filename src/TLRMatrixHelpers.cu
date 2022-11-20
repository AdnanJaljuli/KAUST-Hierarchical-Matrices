
#include "TLRMatrixHelpers.cuh"
#include "TLRMatrix.cuh"
#include <assert.h>

__global__ void fillARAArrays(int batchCount, int maxSegmentSize, int* d_rows_batch, int* d_cols_batch, int* d_ldm_batch, int* d_lda_batch, int* d_ldb_batch){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < batchCount){
        d_rows_batch[i] = maxSegmentSize;
        d_cols_batch[i] = maxSegmentSize;
        d_ldm_batch[i] = maxSegmentSize;
        d_lda_batch[i] = maxSegmentSize;
        d_ldb_batch[i] = maxSegmentSize;
    }
}

__global__ void copyTiles(int batchCount, int maxSegmentSize, int* d_ranks, unsigned int* d_scan_k, H2Opus_Real* d_U_tiled_segmented, H2Opus_Real* d_A, H2Opus_Real* d_V_tiled_segmented, H2Opus_Real* d_B, unsigned int maxRank){
    if(threadIdx.x < d_ranks[blockIdx.x]) {
        unsigned int scanRanks = d_scan_k[blockIdx.x] - d_ranks[blockIdx.x];
        for(unsigned int i = 0; i < maxSegmentSize; ++i) {
            d_U_tiled_segmented[scanRanks*maxSegmentSize + threadIdx.x*maxSegmentSize + i] = d_A[blockIdx.x*maxSegmentSize*maxRank + threadIdx.x*maxSegmentSize + i];
            d_V_tiled_segmented[scanRanks*maxSegmentSize + threadIdx.x*maxSegmentSize + i] = d_B[blockIdx.x*maxSegmentSize*maxRank + threadIdx.x*maxSegmentSize + i];
        }
    }
}

__global__ void copyRanks(int num_segments, int maxSegmentSize, int* from_ranks, int* to_ranks){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < num_segments*(num_segments-1)){
        int row = i%(num_segments-1);
        int col = i/(num_segments-1);
        int diff = (row>=col) ? 1 : 0;
        to_ranks[i + col + diff] = from_ranks[i];
    }
    if(i < num_segments){
        to_ranks[i*num_segments + i] = 0;
    }
}

__device__ H2Opus_Real getCorrelationLength(int dim){
    return dim == 3 ? 0.2 : 0.1;
}

__device__ H2Opus_Real interaction(int n, int dim, int col, int row, H2Opus_Real* pointCloud){
    assert(col<n);
    assert(row<n);

    H2Opus_Real diff = 0;
    H2Opus_Real x, y;
    for (int d = 0; d < dim; ++d){
        x = pointCloud[d*n + col];
        y = pointCloud[d*n + row];
        diff += (x - y)*(x - y);
    }

    H2Opus_Real dist = sqrt(diff);
    return exp(-dist/getCorrelationLength(dim));
}

__global__ void generateDenseBlockColumn(unsigned int numberOfInputPoints, unsigned int maxSegmentSize, int dimensionOfInputPoints, H2Opus_Real* matrix, H2Opus_Real* pointCloud, KDTree kDTree, int columnIndex, H2Opus_Real* diagonal) {
    for(unsigned int i = 0; i < (maxSegmentSize/blockDim.x); ++i) {
        for(unsigned int j = 0; j < (maxSegmentSize/blockDim.x); ++j) {
            unsigned int row = blockIdx.y*maxSegmentSize + i*blockDim.x + threadIdx.y;
            unsigned int col = columnIndex*maxSegmentSize + j*blockDim.x + threadIdx.x;

            if(blockIdx.y == columnIndex) {
                diagonal[columnIndex*maxSegmentSize*maxSegmentSize + j*maxSegmentSize*blockDim.x + threadIdx.x*maxSegmentSize + i*blockDim.y + threadIdx.y] = interaction(numberOfInputPoints, dimensionOfInputPoints, kDTree.segmentIndices[kDTree.segmentOffsets[columnIndex] + blockDim.x*j + threadIdx.x], kDTree.segmentIndices[kDTree.segmentOffsets[blockIdx.y] + i*blockDim.x + threadIdx.y], pointCloud);
            }
            else {
                unsigned int diff = (blockIdx.y > columnIndex) ? 1 : 0;
                unsigned int matrixIndex = blockIdx.y*maxSegmentSize*maxSegmentSize - diff*maxSegmentSize*maxSegmentSize + j*blockDim.x*maxSegmentSize + threadIdx.x*maxSegmentSize + i*blockDim.y + threadIdx.y;
                int xDim = kDTree.segmentOffsets[columnIndex + 1] - kDTree.segmentOffsets[columnIndex];
                int yDim = kDTree.segmentOffsets[blockIdx.y + 1] - kDTree.segmentOffsets[blockIdx.y];

                if(threadIdx.x + j*blockDim.x >= xDim || threadIdx.y + i*blockDim.x >= yDim) {
                    if(col == row) {
                        matrix[matrixIndex] = 1;
                    }
                    else {
                        matrix[matrixIndex] = 0;
                    }
                }
                else {
                    matrix[matrixIndex] = interaction(numberOfInputPoints, dimensionOfInputPoints, kDTree.segmentIndices[kDTree.segmentOffsets[columnIndex] + blockDim.x*j + threadIdx.x], kDTree.segmentIndices[kDTree.segmentOffsets[blockIdx.y] + i*blockDim.x + threadIdx.y], pointCloud);
                }
            }
        }
    }
}