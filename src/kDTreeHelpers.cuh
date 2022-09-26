
#ifndef __KD_TREE_HELPER_FUNCTIONS_H__
#define __KD_TREE_HELPER_FUNCTIONS_H__

#include <utility>
#include <cstdint> 
#include "cublas_v2.h"
#include "kblas.h"
#include "TLRMatrix.h"
#include "helperKernels.cuh"
#include <cub/cub.cuh>

__global__ void initializeArrays(int numberOfInputPoints, int* valuesIn, int* currentDim, int numSegments){
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < numberOfInputPoints){
        valuesIn[i] = i;
    }
    if(i < numSegments){
        currentDim[i] = -1;
    }
}

__global__ void fillOffsets(int numberOfInputPoints, unsigned int dimensionOfInputPoints, unsigned int numSegments, unsigned int segmentSize, int* offsets_sort, int* offsets_reduce){
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
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

__global__ void fillOffsetsSort(int n, unsigned int dim, unsigned int num_segments, int* offsets_sort, int* aux_offsets_sort, uint64_t* bit_vector, short* popc_scan, unsigned int* new_num_segments, bool* workDone, int bucket_size){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;

    if(i < num_segments){
        *new_num_segments = num_segments + popc_scan[(num_segments + sizeof(uint64_t)*8-1)/(sizeof(uint64_t)*8) - 1] + __popcll(bit_vector[(num_segments + sizeof(uint64_t)*8-1)/(sizeof(uint64_t)*8) - 1]);

        if(threadIdx.x==0 && blockIdx.x==0){
            aux_offsets_sort[*new_num_segments] = n;
        }

        unsigned int pos = i%(sizeof(uint64_t)*8);
        unsigned int sub = i/(sizeof(uint64_t)*8);
        unsigned int onesToLeft = __popcll(bit_vector[sub]>>(sizeof(uint64_t)*8 - pos));

        unsigned int index = (popc_scan[sub] + onesToLeft)*2 + (sizeof(uint64_t)*8*sub - popc_scan[sub] + pos - onesToLeft);
 
        aux_offsets_sort[index] = offsets_sort[i];
        if(offsets_sort[i+1] - offsets_sort[i] > bucket_size){
            aux_offsets_sort[index + 1] = (offsets_sort[i+1] - offsets_sort[i] + 1)/2 + offsets_sort[i];
        }
    }

    if(threadIdx.x==0 && blockIdx.x==0){
        if(aux_offsets_sort[1] - aux_offsets_sort[0] <= bucket_size){
            *workDone = true;
        }
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

__global__ void fillReductionArray(int n, unsigned int dim, H2Opus_Real* pointCloud, int* values_in, H2Opus_Real* reduce_in){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<(long long)n*(long long)dim){
        reduce_in[i] = pointCloud[(long long)values_in[i - (i/n)*n] + (long long)(i/n)*n];
    }
}

__global__ void findSpan(int n, unsigned int dim, unsigned int num_segments, H2Opus_Real* reduce_min_out, H2Opus_Real* reduce_max_out, H2Opus_Real* span, int* span_offsets){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<num_segments){
        for(unsigned int j=0; j<dim; ++j){
            span[i*dim + j] = reduce_max_out[j*num_segments + i] - reduce_min_out[j*num_segments + i];
        }
        span_offsets[i] = i*dim;
    }

    if(threadIdx.x==0 && blockIdx.x==0){
        span_offsets[num_segments] = num_segments*dim;
    }
}

__global__ void fillCurrDim(int n, unsigned int num_segments, int* currentDim, cub::KeyValuePair<int, H2Opus_Real>* spanReduced){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<num_segments){
        currentDim[i] = spanReduced[i].key;
    }
}

__global__ void fillCurrDim(int n, unsigned int num_segments, int* currentDim, cub::KeyValuePair<int, H2Opus_Real>* spanReduced, uint64_t* bit_vector){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<num_segments){
        currentDim[i] = spanReduced[i].key;
    }

    unsigned int last_index = (num_segments+sizeof(uint64_t)*8-1)/(sizeof(uint64_t)*8);
    if(i<last_index){
        bit_vector[i] = 0ULL;
    }
}

__global__ void fillKeysIn(int n, unsigned int segmentSize, H2Opus_Real* keys_in, int* currentDim, int* values_in, H2Opus_Real* pointCloud){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<n){
        keys_in[i] = pointCloud[(long long)currentDim[i/segmentSize]*n + (long long)values_in[i]];
    }
}

__global__ void fillKeysIn(int n, H2Opus_Real* keys_in, int* currentDim, int* values_in, H2Opus_Real* pointCloud, int* offsets_sort, int* output){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<n){
        int segment_index = output[i] - 1;
        keys_in[i] = pointCloud[(long long)currentDim[segment_index]*n + (long long)values_in[i]];
    }
}

#endif