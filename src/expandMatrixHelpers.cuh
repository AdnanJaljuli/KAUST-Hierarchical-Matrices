
#ifndef __EXPAND_MATRIX_HELPERS_CUH__
#define __EXPAND_MATRIX_HELPERS_CUH__

#include <assert.h>
#include <ctype.h>
#include <cub/cub.cuh>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#include <curand_kernel.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

static __global__ void expandMatrix(int num_segments, int maxSegmentSize, H2Opus_Real* expandedMatrix, TLR_Matrix matrix){
    if(blockIdx.x == blockIdx.y) {
        expandedMatrix[blockIdx.x*num_segments*maxSegmentSize*maxSegmentSize + blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y] = matrix.diagonal[blockIdx.x*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y];
    }
    else{
        unsigned int index;
        if(matrix.ordering == MORTON) {
            index = IndextoMOIndex_h(num_segments, blockIdx.x*num_segments + blockIdx.y);
        }
        else if(matrix.ordering == COLUMN_MAJOR) {
            index = blockIdx.x*num_segments + blockIdx.y;
        }

        H2Opus_Real sum = 0;
        for(unsigned int i=0; i<matrix.blockRanks[index]; ++i) {
            sum += matrix.U[matrix.blockOffsets[index]*maxSegmentSize + i*maxSegmentSize + threadIdx.y]*matrix.V[matrix.blockOffsets[index]*maxSegmentSize + i*maxSegmentSize + threadIdx.x];
        }
        expandedMatrix[blockIdx.x*num_segments*maxSegmentSize*maxSegmentSize + blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y] = sum;
    }
}

static __global__ void errorInMatrix(int num_segments, int maxSegmentSize, H2Opus_Real* denseMatrix, H2Opus_Real* expandedMatrix, H2Opus_Real* error, H2Opus_Real* tmp){
    H2Opus_Real x = denseMatrix[(blockIdx.x*maxSegmentSize + threadIdx.x)*maxSegmentSize*num_segments + blockIdx.y*maxSegmentSize + threadIdx.y];
    H2Opus_Real y = expandedMatrix[blockIdx.x*num_segments*maxSegmentSize*maxSegmentSize + blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y];
    
    atomicAdd(tmp, x*x);
    atomicAdd(error, (x-y)*(x-y));
}

static void checkErrorInLRMatrix(uint64_t numSegments, uint64_t maxSegmentSize, TLR_Matrix matrix, H2Opus_Real* d_denseMatrix){
    H2Opus_Real* d_expandedMatrix;
    cudaMalloc((void**) &d_expandedMatrix, numSegments*maxSegmentSize*numSegments*maxSegmentSize*sizeof(H2Opus_Real));

    dim3 mm_numBlocks(numSegments, numSegments);
    dim3 mm_numThreadsPerBlock(32, 32);
    expandMatrix <<< mm_numBlocks, mm_numThreadsPerBlock >>> (numSegments, maxSegmentSize, d_expandedMatrix, matrix);

    H2Opus_Real* d_error;
    H2Opus_Real* d_tmp;
    cudaMalloc((void**) &d_error, sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_tmp, sizeof(H2Opus_Real));
    cudaMemset(d_error, 0, sizeof(H2Opus_Real));
    cudaMemset(d_tmp, 0, sizeof(H2Opus_Real));

    errorInMatrix <<< mm_numBlocks, mm_numThreadsPerBlock >>> (numSegments, maxSegmentSize, d_denseMatrix, d_expandedMatrix, d_error, d_tmp);

    H2Opus_Real h_error;
    H2Opus_Real h_tmp;
    cudaMemcpy(&h_error, d_error, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_tmp, d_tmp, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    printf("error in matrix: %lf\n", sqrt(h_error)/sqrt(h_tmp));
    cudaFree(d_tmp);
    cudaFree(d_error);
    cudaFree(d_expandedMatrix);
}

static __global__ void generateDenseMatrix_kernel(uint64_t numberOfInputPoints, uint64_t numSegments, uint64_t maxSegmentSize, int dimensionOfInputPoints, H2Opus_Real* denseMatrix, int* indexMap, int* offsetsSort, H2Opus_Real* pointCloud ){
    for(unsigned int i = 0; i < (maxSegmentSize/blockDim.x); ++i){
        for(unsigned int j = 0; j < (maxSegmentSize/blockDim.x); ++j){
            
            unsigned int row = blockIdx.y*maxSegmentSize + i*blockDim.x + threadIdx.y;
            unsigned int col = blockIdx.x*maxSegmentSize + j*blockDim.x + threadIdx.x;

            int xDim = offsetsSort[blockIdx.x + 1] - offsetsSort[blockIdx.x];
            int yDim = offsetsSort[blockIdx.y + 1] - offsetsSort[blockIdx.y];
            unsigned int matrixIndex = col*maxSegmentSize*numSegments + row;

            if (blockIdx.y == blockIdx.x) {
                    denseMatrix[matrixIndex] = interaction(numberOfInputPoints, dimensionOfInputPoints, indexMap[offsetsSort[blockIdx.x] + blockDim.x*j + threadIdx.x], indexMap[offsetsSort[blockIdx.y] + i*blockDim.x + threadIdx.y], pointCloud);
            }
            else {
                if(threadIdx.x + j*blockDim.x >= xDim || threadIdx.y + i*blockDim.x >= yDim) {
                    if (col == row) {
                        denseMatrix[matrixIndex] = 1;
                    }
                    else {
                        denseMatrix[matrixIndex] = 0;
                    }
                }
                else {
                    denseMatrix[matrixIndex] = interaction(numberOfInputPoints, dimensionOfInputPoints, indexMap[offsetsSort[blockIdx.x] + blockDim.x*j + threadIdx.x], indexMap[offsetsSort[blockIdx.y] + i*blockDim.x + threadIdx.y], pointCloud);
                }
            }
        }
    }
}

static void generateDenseMatrix(int numberOfInputPoints, int numSegments, int maxSegmentSize, int dimensionOfInputPoints, H2Opus_Real* d_denseMatrix, int* &d_valuesIn, int* &d_offsetsSort, H2Opus_Real* &d_dataset) {
    dim3 m_numThreadsPerBlock(min(32, (int)maxSegmentSize), min(32, (int)maxSegmentSize));
    dim3 m_numBlocks(numSegments, numSegments);
    generateDenseMatrix_kernel <<< m_numBlocks, m_numThreadsPerBlock >>> (numberOfInputPoints, numSegments, maxSegmentSize, dimensionOfInputPoints, d_denseMatrix, d_valuesIn, d_offsetsSort, d_dataset);
}

#endif