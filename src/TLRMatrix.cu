
#include <assert.h>
#include "TLRMatrix.cuh"

void freeMatrix(TLR_Matrix matrix){
    cudaFree(matrix.blockRanks);
    cudaFree(matrix.blockOffsets);
    cudaFree(matrix.U);
    cudaFree(matrix.V);
    cudaFree(matrix.diagonal);
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

__global__ void generateDenseBlockColumn(uint64_t numberOfInputPoints, uint64_t maxSegmentSize, int dimensionOfInputPoints, H2Opus_Real* matrix, H2Opus_Real* pointCloud, KDTree kDTree, int columnIndex, H2Opus_Real* diagonal) {
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