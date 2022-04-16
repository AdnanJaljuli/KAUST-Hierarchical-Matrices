#include <cub/cub.cuh>
#include "helperFunctions.h"
#define BUCKET_SIZE 1<<3
typedef double H2Opus_Real;

__global__ void initializeArrays(int n, int* values_in, int* currDimArray){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<n){
        values_in[i] = i;
    }
    if(i<((n+BUCKET_SIZE-1)/BUCKET_SIZE)){
        currDimArray[i] = -1;
    }
}

__global__ void initializeArrays(int n, int dim, int* values_in, int* currDimArray, int* offsets_sort, int* offsets_reduce, int* input_search){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<n){
        values_in[i] = i;
        input_search[i] = i;
    }
    if(i<((n+BUCKET_SIZE-1)/BUCKET_SIZE)){
        currDimArray[i] = -1;
    }
    if(threadIdx.x==0 && blockIdx.x==0){
        offsets_sort[0] = 0;
        offsets_sort[1] = n;
        for(unsigned int j=0; j<dim+1; ++j){ //TODO: might have to be <dim+1
            offsets_reduce[j] = j*n;
        }
    }
}

__global__ void fillOffsets(int n, unsigned int dim, unsigned int num_segments, unsigned int segment_size, int* offsets_sort, int* offsets_reduce){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;

    if(i<num_segments+1){
        offsets_sort[i] = (i<num_segments) ? (i*segment_size) : n;

        if(threadIdx.x==0 && blockIdx.x==0){
            offsets_reduce[0] = 0;
        }

        for(unsigned int j=0; j<dim; ++j){
            if(i < num_segments){
                offsets_reduce[j*num_segments + i + 1] = (i+1<num_segments) ? ((i+1)*segment_size + n*j) : n*(j+1);
            }
        }
    }
}

__global__ void fillOffsetsSort(int n, unsigned int dim, unsigned int num_segments, int* offsets_sort, int* offsets_reduce, int* aux_offsets_sort, uint64_t* bit_vector, short* popc_scan, unsigned int* new_num_segments){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;

    if(i<num_segments){
        *new_num_segments = num_segments + popc_scan[num_segments - 1] + __popcll(bit_vector[((((num_segments+BUCKET_SIZE-1)/BUCKET_SIZE) + sizeof(uint64_t)*8-1)/(sizeof(uint64_t)*8)) - 1]);
        if(threadIdx.x==0 && blockIdx.x==0){
            aux_offsets_sort[*new_num_segments] = n;
            
        }

        unsigned int pos = i%(sizeof(uint64_t)*8);
        unsigned int sub = i/(sizeof(uint64_t)*8);
        unsigned int onesToLeft = __popcll(bit_vector[sub]>>(sizeof(uint64_t)*8 - pos));

        if(threadIdx.x==0 && blockIdx.x==0){
            printf("ones to left: %d\n", onesToLeft);
        }

        unsigned int index = (popc_scan[i] + onesToLeft)*2 + (sizeof(uint64_t)*8*sub - popc_scan[i] + pos - onesToLeft);
        if(threadIdx.x==0 && blockIdx.x==0){
            printf("index: %d\n", index);
        }
        aux_offsets_sort[index] = offsets_sort[i];
        if(offsets_sort[i+1] - offsets_sort[i] > BUCKET_SIZE){
            aux_offsets_sort[index + 1] = (offsets_sort[i+1] - offsets_sort[i] + 1)/2 + offsets_sort[i];
        }

        // for(unsigned int j=0; j<dim; ++j){
        //     offsets_reduce[j*(*new_num_segments) + i + 1] = ((offsets_sort[i+1] - offsets_sort[i] + 1)/2 + offsets_sort[i])*(n+j);
        // }
    }

    if(threadIdx.x==0 && blockIdx.x==0){
        printf("new num segment: %d\n", *new_num_segments);
        printf("offsets sort\n");
        for(unsigned int j=0; j<num_segments+1; ++j){
            printf("%d ", offsets_sort[j]);
        }
        printf("\n");
        printf(" newoffsets sort\n");
        for(unsigned int j=0; j<*new_num_segments+1; ++j){
            printf("%d ", aux_offsets_sort[j]);
        }
        printf("\n");
    }
}

__global__ void fillOffsetsReduce(int n, int dim, unsigned int num_segments, int* offsets_sort, int* offsets_reduce){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(threadIdx.x==0 && blockIdx.x==0){
        offsets_reduce[0] = 0;
    }
    for(unsigned int j=0; j<dim; ++j){
            offsets_reduce[j*num_segments + i + 1] = offsets_sort[i + 1] + (n*j);
    }
    if(threadIdx.x==0 && blockIdx.x==0){
        printf(" newoffsets reduce\n");
        for(unsigned int j=0; j<num_segments*dim+1; ++j){
            printf("%d ", offsets_reduce[j]);
        }
        printf("\n");
    }
}

__global__ void fillReductionArray(int n, unsigned int dim, H2Opus_Real* dataset, int* values_in, H2Opus_Real* reduce_in){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<(long long)n*(long long)dim){
        reduce_in[i] = dataset[(long long)values_in[i - (i/n)*n] + (long long)(i/n)*n];
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

__global__ void fillCurrDim(int n, unsigned int num_segments, int* currDimArray, cub::KeyValuePair<int, H2Opus_Real>* spanReduced){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<num_segments){
        currDimArray[i] = spanReduced[i].key;
    }
}

// TODO: change the name of this kernel because it also changes the bit vector
__global__ void fillCurrDim(int n, unsigned int num_segments, int* currDimArray, cub::KeyValuePair<int, H2Opus_Real>* spanReduced, uint64_t* bit_vector){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<num_segments){
        currDimArray[i] = spanReduced[i].key;
    }

    unsigned int last_index = (num_segments+sizeof(uint64_t)*8-1)/(sizeof(uint64_t)*8);
    if(i<last_index){
        bit_vector[i] = 0ULL;
    }
}

__global__ void fillKeysIn(int n, unsigned int segment_size, H2Opus_Real* keys_in, int* currDimArray, int* values_in, H2Opus_Real* dataset){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<n){
        keys_in[i] = dataset[(long long)currDimArray[i/segment_size]*n + (long long)values_in[i]];
    }
}

__global__ void fillKeysIn(int n, H2Opus_Real* keys_in, int* currDimArray, int* values_in, H2Opus_Real* dataset, int* offsets_sort, int* output){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<n){
        int segment_index = output[i] - 1;
        keys_in[i] = dataset[(long long)currDimArray[segment_index]*n + (long long)values_in[i]];
    }
    // if(threadIdx.x==0 && blockIdx.x==0){
    //     printf("output\n");
    //     for(unsigned int j=0; j<n; ++j){
    //         printf("%d ", output[j]);
    //     }
    // }
    // if(threadIdx.x==0 && blockIdx.x==0){
    //     printf("keys in\n");
    //     for(unsigned int j=0; j<n; ++j){
    //         printf("%f ", keys_in[j]);
    //     }
    // }
}

__device__ H2Opus_Real interaction(int n, int dim, int col, int row, H2Opus_Real* dataset){
    H2Opus_Real ans=0;
    for(unsigned int i=0; i<dim; ++i){
        for(unsigned int j=0; j<dim; ++j){
            ans += dataset[i*n + row]*dataset[j*n + col];
        }
    }
    return ans;
}

__global__ void generateInputMatrix(int n, int dim, int* index_map, H2Opus_Real* matrix, H2Opus_Real* dataset){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<n*n){
        int col = i%n;
        int row = i/n;
        matrix[col*n+row] = interaction(n, dim, index_map[col], index_map[row], dataset);
    }
}

// __global__ void calcMemNeeded(int n, int padded_n, int* K, H2Opus_Real* S, float eps){
//     unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
//     __shared__ int k=0;
//     if(i<BUCKET_SIZE){
//         if((S[BUCKET_SIZE*blockIdx.x + i]/S[BUCKET_SIZE*blockIdx.x]) > eps){
//             atomicAdd(k, 1);
//         }
//     }
//     __syncthreads();
//     K[blockIdx.x] = k;
// }

__global__ void tileMatrix(int n, int padded_n, H2Opus_Real* S, H2Opus_Real* U, H2Opus_Real* V, H2Opus_Real* STiled, H2Opus_Real* UTiled, H2Opus_Real* VTiled, int* K, int* KScan){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<K[blockIdx.x]){
        STiled[blockIdx.x + i] = S[blockIdx.x*BUCKET_SIZE + i];
        for(unsigned int j=0; j<BUCKET_SIZE;++j){
            UTiled[KScan[blockIdx.x]*BUCKET_SIZE + i*BUCKET_SIZE + j]= U[blockIdx.x*BUCKET_SIZE*BUCKET_SIZE + i*BUCKET_SIZE + j];
            VTiled[KScan[blockIdx.x]*BUCKET_SIZE + j*K[blockIdx.x] + i]= V[blockIdx.x*BUCKET_SIZE*BUCKET_SIZE + i*BUCKET_SIZE + j];
        }
    }
}

__global__ void fillBitVector(int num_segments, uint64_t* bit_vector, int* offsets_sort){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;

    if(i<num_segments){
        unsigned int pos = i%(sizeof(uint64_t)*8);
        unsigned int sub = i/(sizeof(uint64_t)*8);
        if(offsets_sort[i+1] - offsets_sort[i] > BUCKET_SIZE){
            atomicOr((unsigned long long*)&bit_vector[sub], 1ULL<<(sizeof(uint64_t)*8-1-pos));
        }
    }
}

__global__ void  fillPopCount(int num_segments, uint64_t* bit_vector, short int* popc_bit_vector){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<num_segments){
        popc_bit_vector[i] = __popcll(bit_vector[i]);
    }
    // if(threadIdx.x==0 && blockIdx.x==0){
    //     printf("pop count\n");
    //     for(unsigned int j=0; j<num_segments; ++j){
    //         printf("%hi", popc_bit_vector[j]);
    //     }
    // }
}

// __global__ void printPopcScan(int num_segments, short int* popc_scan){
//     if(threadIdx.x==0 && blockIdx.x==0){
//         printf("popc scan\n");
//         for(unsigned int j=0; j<num_segments; ++j){
//             printf("%hi", popc_scan[j]);
//         }
//     }
// }