
#ifndef KDTREE_HELPERS_H
#define KDTREE_HELPERS_H

#include "kDTree.cuh"
#include "helperKernels.cuh"

__global__ void initIndexMap(unsigned int numberOfInputPoints, KDTree kDTree);
__global__ void initIndexMap(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, KDTree tree, int* input_search, int *d_dimxNLeafOffsets);
__global__ void fillOffsets(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, unsigned int currentNumSegments, unsigned int currentSegmentSize, KDTree kDTree, int* dimxNLeafOffsets);
__global__ void fillReductionArray(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, H2Opus_Real* pointCloud, int* values_in, H2Opus_Real* reduce_in);
__global__ void findSpan(int n, unsigned int dim, unsigned int num_segments, H2Opus_Real* reduce_min_out, H2Opus_Real* reduce_max_out, H2Opus_Real* span, int* span_offsets);
__global__ void fillKeysIn(int n, unsigned int segmentSize, H2Opus_Real* keys_in, cub::KeyValuePair<int, H2Opus_Real>* spanReduced, int* values_in, H2Opus_Real* pointCloud);
__global__ void fillKeysIn(int n, H2Opus_Real* keys_in, cub::KeyValuePair<int, H2Opus_Real>* spanReduced, int* values_in, H2Opus_Real* pointCloud, int* offsets_sort, int* output);
__global__ void fillOffsetsSort(int n, unsigned int dim, unsigned int num_segments, int* offsets_sort, int* aux_offsets_sort);
__global__ void fillOffsetsReduce(int n, int dim, unsigned int num_segments, int* offsets_sort, int* offsets_reduce);
std::pair<int, int> getMaxSegmentSize(int n, int bucket_size);

#endif