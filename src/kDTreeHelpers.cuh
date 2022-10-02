
#ifndef __KD_TREE_HELPER_FUNCTIONS_H__
#define __KD_TREE_HELPER_FUNCTIONS_H__

#include <utility>
#include <cstdint> 
#include "cublas_v2.h"
#include "kblas.h"
#include "TLRMatrix.h"
#include "helperKernels.cuh"
#include <cub/cub.cuh>

static __global__ void initializeArrays(unsigned int numberOfInputPoints, int* valuesIn, int* currentDim, uint64_t numSegments) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numberOfInputPoints) {
        valuesIn[i] = i;
    }
    if(i < numSegments) {
        currentDim[i] = -1;
    }
}

static __global__ void fillOffsets(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, unsigned int numSegments, unsigned int segmentSize, int* offsets_sort, int* offsets_reduce){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numSegments + 1){
        offsets_sort[i] = (i < numSegments) ? (i*segmentSize) : numberOfInputPoints;

        if(threadIdx.x == 0 && blockIdx.x == 0){
            offsets_reduce[0] = 0;
        }

        for(unsigned int j = 0; j < dimensionOfInputPoints; ++j){
            if(i < numSegments){
                offsets_reduce[j*numSegments + i + 1] = (i + 1 < numSegments) ? ((i + 1)*segmentSize + numberOfInputPoints*j) : numberOfInputPoints*(j + 1);
            }
        }
    }
}

static __global__ void fillReductionArray(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, H2Opus_Real* pointCloud, int* values_in, H2Opus_Real* reduce_in){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numberOfInputPoints*dimensionOfInputPoints) {
        reduce_in[i] = pointCloud[values_in[i - (i/numberOfInputPoints)*numberOfInputPoints] + (i/numberOfInputPoints)*numberOfInputPoints];
    }
}

static __global__ void findSpan(int n, unsigned int dim, unsigned int num_segments, H2Opus_Real* reduce_min_out, H2Opus_Real* reduce_max_out, H2Opus_Real* span, int* span_offsets){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < num_segments) {
        for(unsigned int j=0; j < dim; ++j) {
            span[i*dim + j] = reduce_max_out[j*num_segments + i] - reduce_min_out[j*num_segments + i];
        }
        span_offsets[i] = i*dim;
    }

    if(threadIdx.x == 0 && blockIdx.x == 0) {
        span_offsets[num_segments] = num_segments*dim;
    }
}

static __global__ void fillCurrDim(int n, unsigned int num_segments, int* currentDim, cub::KeyValuePair<int, H2Opus_Real>* spanReduced){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < num_segments) {
        currentDim[i] = spanReduced[i].key;
    }
}

static __global__ void fillKeysIn(int n, unsigned int segmentSize, H2Opus_Real* keys_in, int* currentDim, int* values_in, H2Opus_Real* pointCloud){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if( i < n) {
        keys_in[i] = pointCloud[currentDim[i/segmentSize]*n + values_in[i]];
    }
}

#endif