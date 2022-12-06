
#include "kDTreeHelpers.cuh"

__global__ void initIndexMap(unsigned int numberOfInputPoints, KDTree kDTree) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numberOfInputPoints) {
        kDTree.segmentIndices[i] = i;
    }
}

__global__ void initIndexMap(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, KDTree tree, int* input_search, int *d_dimxNSegmentOffsets) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numberOfInputPoints) {
        tree.segmentIndices[i] = i;
        input_search[i] = i;
    }
    
    if(threadIdx.x == 0 && blockIdx.x == 0){
        tree.segmentOffsets[0] = 0;
        tree.segmentOffsets[1] = numberOfInputPoints;
        for(unsigned int j = 0; j < dimensionOfInputPoints + 1; ++j) { //TODO: might have to be <dim+1
            d_dimxNSegmentOffsets[j] = j*numberOfInputPoints;
        }
    }
}

__global__ void fillOffsets(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, unsigned int currentNumSegments, unsigned int currentSegmentSize, KDTree kDTree, int* dimxNSegmentOffsets) {
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

__global__ void fillReductionArray(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, H2Opus_Real* pointCloud, int* values_in, H2Opus_Real* reduce_in) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numberOfInputPoints*dimensionOfInputPoints) {
        reduce_in[i] = pointCloud[values_in[i - (i/numberOfInputPoints)*numberOfInputPoints] + (i/numberOfInputPoints)*numberOfInputPoints];
    }
}

__global__ void findSpan(int n, unsigned int dim, unsigned int num_segments, H2Opus_Real* reduce_min_out, H2Opus_Real* reduce_max_out, H2Opus_Real* span, int* span_offsets) {
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

__global__ void fillKeysIn(int n, unsigned int segmentSize, H2Opus_Real* keys_in, cub::KeyValuePair<int, H2Opus_Real>* spanReduced, int* values_in, H2Opus_Real* pointCloud) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if( i < n) {
        keys_in[i] = pointCloud[spanReduced[i/segmentSize].key*n + values_in[i]];
    }
}

__global__ void fillKeysIn(int n, H2Opus_Real* keys_in, cub::KeyValuePair<int, H2Opus_Real>* spanReduced, int* values_in, H2Opus_Real* pointCloud, int* offsets_sort, int* output) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if( i < n) {
        int segmentIndex = output[i] - 1;
        keys_in[i] = pointCloud[spanReduced[segmentIndex].key*n + values_in[i]];
    }
}

__global__ void fillOffsetsSort(int n, unsigned int dim, unsigned int num_segments, int* offsets_sort, int* aux_offsets_sort){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;

    if(i==0){
        aux_offsets_sort[num_segments*2] = n;
    }

    if(i < num_segments){
        unsigned int index = i*2; 
        aux_offsets_sort[index] = offsets_sort[i];
        aux_offsets_sort[index + 1] = (offsets_sort[i+1] - offsets_sort[i] + 1)/2 + offsets_sort[i];
    }
}

__global__ void fillOffsetsReduce(int n, int dim, unsigned int num_segments, int* offsets_sort, int* offsets_reduce){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i==0){
        offsets_reduce[0] = 0;
    }
    if(i < num_segments){
        for(unsigned int j=0; j<dim; ++j){
            offsets_reduce[j*num_segments + i + 1] = offsets_sort[i + 1] + (n*j);
        }
    }
}

__device__ __host__ int upperPowerOfTwo(int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}