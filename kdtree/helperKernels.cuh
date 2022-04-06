#include <cub/cub.cuh>
#define BUCKET_SIZE 1<<4

typedef double H2Opus_Real;

__global__ void fillOffsetsArrays(int n, unsigned int dim, unsigned int num_segments, unsigned int segment_size, int* offsets_sort, int* offsets_reduce){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<num_segments+1){
        offsets_sort[i] = i*segment_size;

        if(threadIdx.x==0 && blockIdx.x==0){
            offsets_reduce[0] = 0;
        }

        for(unsigned int j=0; j<dim; ++j){
            if(i < num_segments){
                offsets_reduce[j*num_segments + i + 1] = (i+1)*segment_size + n*j;
            }
        }
    }
}

__global__ void initializeArrays(int n, int* values_in, int* currDimArray){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<n){
        values_in[i] = i;
        // currDimArray[i] = -1;
    }
    if(i<((n+BUCKET_SIZE-1)/BUCKET_SIZE)){
        currDimArray[i] = -1;
    }
}

__global__ void fillReductionArray(int n, unsigned int dim, H2Opus_Real* dataset, int* values_in, H2Opus_Real* reduce_in){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<(long long)n*(long long)dim){
        reduce_in[i] = dataset[(long long)values_in[i - (i/n)*n] + (long long)(i/n)*n];
    }
}

__global__ void findSpan(int n, unsigned int dim, unsigned int num_segments, unsigned int segment_size, int* reduce_min_out, int* reduce_max_out, H2Opus_Real* span, int* span_offsets){
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

__global__ void fillCurDimArray(int n, unsigned int num_segments, int* currDimArray, cub::KeyValuePair<int, H2Opus_Real>* spanReduced){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<num_segments){
        currDimArray[i] = spanReduced[i].key;
    }
}

__global__ void fillKeysIn(int n, unsigned int segment_size, H2Opus_Real* keys_in, int* currDimArray, int* values_in, H2Opus_Real* dataset){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<n){
        keys_in[i] = dataset[(long long)currDimArray[i/segment_size]*n + (long long)values_in[i]];
    }
}

__global__ void fillPaddedElements(int n, int padded_n, int dim, H2Opus_Real* dataset, H2Opus_Real* maxValue){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<padded_n-n){
        for(unsigned int j=0; j<dim; ++j){
            dataset[j*padded_n + n+i] = maxValue[j];
        }

    }
}

__device__ H2Opus_Real interaction(int padded_n, int col, int row, H2Opus_Real* dataset){
    H2Opus_Real ans=0;
    for(unsigned int i=0; i<dim; ++i){
        for(unsigned int j=0; j<dim; ++j){
            ans += dataset[i*padded_n + row]*dataset[j*padded_n + col];
        }
    }
    return ans;
}

__global__ void generateInputMatrix(int n, int padded_n, int dim, int* index_map, H2Opus_Real* matrix, H2Opus_Real* dataset){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<n*n){
        int col = i%n;
        int row = i/n;
        matrix[col*n+row] = interaction(padded_n, index_map[col], index_map[row], dataset);
    }
}
