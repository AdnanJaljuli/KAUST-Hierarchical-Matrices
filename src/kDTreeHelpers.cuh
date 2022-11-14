
#ifndef __KD_TREE_HELPER_FUNCTIONS_H__
#define __KD_TREE_HELPER_FUNCTIONS_H__

#include <utility>
#include <cstdint> 
#include "cublas_v2.h"
#include "kblas.h"
#include "kDTree.h"
#include "helperKernels.cuh"

static __global__ void initIndexMap(unsigned int numberOfInputPoints, KDTree kDTree) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numberOfInputPoints) {
        kDTree.segmentIndices[i] = i;
    }
}

static __global__ void fillOffsets(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, unsigned int currentNumSegments, unsigned int currentSegmentSize, KDTree kDTree, int* dimxNSegmentOffsets) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < currentNumSegments + 1){
        kDTree.segmentOffsets[i] = (i < currentNumSegments) ? (i*currentSegmentSize) : numberOfInputPoints;

        if(threadIdx.x == 0 && blockIdx.x == 0){
            dimxNSegmentOffsets[0] = 0;
        }

        for(unsigned int j = 0; j < dimensionOfInputPoints; ++j){
            if(i < currentNumSegments){
                dimxNSegmentOffsets[j*currentNumSegments + i + 1] = (i + 1 < currentNumSegments) ? ((i + 1)*currentSegmentSize + numberOfInputPoints*j) : numberOfInputPoints*(j + 1);
            }
        }
    }
}

static __global__ void fillReductionArray(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, H2Opus_Real* pointCloud, int* values_in, H2Opus_Real* reduce_in) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numberOfInputPoints*dimensionOfInputPoints) {
        reduce_in[i] = pointCloud[values_in[i - (i/numberOfInputPoints)*numberOfInputPoints] + (i/numberOfInputPoints)*numberOfInputPoints];
    }
}

static __global__ void findSpan(int n, unsigned int dim, unsigned int num_segments, H2Opus_Real* reduce_min_out, H2Opus_Real* reduce_max_out, H2Opus_Real* span, int* span_offsets) {
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

static __global__ void fillKeysIn(int n, unsigned int segmentSize, H2Opus_Real* keys_in, cub::KeyValuePair<int, H2Opus_Real>* spanReduced, int* values_in, H2Opus_Real* pointCloud) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if( i < n) {
        keys_in[i] = pointCloud[spanReduced[i/segmentSize].key*n + values_in[i]];
    }
}

#endif