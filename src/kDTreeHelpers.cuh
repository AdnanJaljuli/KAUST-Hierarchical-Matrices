
#ifndef KDTREE_HELPERS_H
#define KDTREE_HELPERS_H

#include "kDTree.cuh"
#include "helperKernels.cuh"

__global__ void initIndexMap(unsigned int numberOfInputPoints, KDTree kDTree);
__global__ void fillOffsets(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, unsigned int currentNumSegments, unsigned int currentSegmentSize, KDTree kDTree, int* dimxNSegmentOffsets);
__global__ void fillReductionArray(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, H2Opus_Real* pointCloud, int* values_in, H2Opus_Real* reduce_in);
__global__ void findSpan(int n, unsigned int dim, unsigned int num_segments, H2Opus_Real* reduce_min_out, H2Opus_Real* reduce_max_out, H2Opus_Real* span, int* span_offsets);
__global__ void fillKeysIn(int n, unsigned int segmentSize, H2Opus_Real* keys_in, cub::KeyValuePair<int, H2Opus_Real>* spanReduced, int* values_in, H2Opus_Real* pointCloud);
__device__ __host__ int upperPowerOfTwo(int v);

#endif