
#ifndef __EXPAND_MATRIX_HELPERS_CUH__
#define __EXPAND_MATRIX_HELPERS_CUH__

#include <ctype.h>
#include <cub/cub.cuh>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#include <assert.h>
#include <curand_kernel.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

static __global__ void generateDenseMatrix_kernel(uint64_t numberOfInputPoints, uint64_t numSegments, uint64_t maxSegmentSize, int dimensionOfInputPoints, H2Opus_Real* denseMatrix, int* indexMap, int* offsetsSort, H2Opus_Real* pointCloud ){
    for(unsigned int i = 0; i < (maxSegmentSize/blockDim.x); ++i){
        for(unsigned int j = 0; j < (maxSegmentSize/blockDim.x); ++j){
            unsigned int row = blockIdx.y*maxSegmentSize + i*blockDim.x + threadIdx.y;
            unsigned int col = blockIdx.x*maxSegmentSize + j*blockDim.x + threadIdx.x;

            int xDim = offsetsSort[blockIdx.x + 1] - offsetsSort[blockIdx.x];
            int yDim = offsetsSort[blockIdx.y + 1] - offsetsSort[blockIdx.y];
            
            if(blockIdx.y == blockIdx.x){
                    denseMatrix[(blockIdx.x*maxSegmentSize + threadIdx.x)*maxSegmentSize*numSegments + blockIdx.y*maxSegmentSize + threadIdx.y] = interaction(numberOfInputPoints, dimensionOfInputPoints, indexMap[offsetsSort[blockIdx.x] + blockDim.x*j + threadIdx.x], indexMap[offsetsSort[blockIdx.y] + i*blockDim.x + threadIdx.y], pointCloud);
            }
            else{
                if(threadIdx.x + j*blockDim.x >= xDim || threadIdx.y + i*blockDim.x >= yDim) {
                    if(col == row){
                        denseMatrix[(blockIdx.x*maxSegmentSize + threadIdx.x)*maxSegmentSize*numSegments + blockIdx.y*maxSegmentSize + threadIdx.y] = 1;
                    }
                    else{
                        denseMatrix[(blockIdx.x*maxSegmentSize + threadIdx.x)*maxSegmentSize*numSegments + blockIdx.y*maxSegmentSize + threadIdx.y] = 0;
                    }
                }
                else {
                    denseMatrix[(blockIdx.x*maxSegmentSize + threadIdx.x)*maxSegmentSize*numSegments + blockIdx.y*maxSegmentSize + threadIdx.y] = interaction(numberOfInputPoints, dimensionOfInputPoints, indexMap[offsetsSort[blockIdx.x] + blockDim.x*j + threadIdx.x], indexMap[offsetsSort[blockIdx.y] + i*blockDim.x + threadIdx.y], pointCloud);
                }
            }
        }
    }
}

static void generateDenseMatrix(int numberOfInputPoints, int numSegments, int maxSegmentSize, int dimensionOfInputPoints, H2Opus_Real* d_denseMatrix, int* &d_valuesIn, int* &d_offsetsSort, H2Opus_Real* &d_dataset){
    dim3 m_numThreadsPerBlock(min(32, (int)maxSegmentSize), min(32, (int)maxSegmentSize));
    dim3 m_numBlocks(numSegments, numSegments);
    generateDenseMatrix_kernel<<<m_numBlocks, m_numThreadsPerBlock>>>(numberOfInputPoints, numSegments, maxSegmentSize, dimensionOfInputPoints, d_denseMatrix, d_valuesIn, d_offsetsSort, d_dataset);
}

#endif