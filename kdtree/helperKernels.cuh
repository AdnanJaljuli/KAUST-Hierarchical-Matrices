#include "helperFunctions.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stddef.h>
#include <cub/cub.cuh>
#include <assert.h>
#include <curand_kernel.h>

__global__ void initializeArrays(int n, int* values_in, int* currDimArray, int max_num_segments){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<n){
        values_in[i] = i;
    }
    if(i<max_num_segments){
        currDimArray[i] = -1;
    }
}

__global__ void initializeArrays(int n, int dim, int* values_in, int* currDimArray, int* offsets_sort, int* offsets_reduce, int* input_search, int max_num_segments){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<n){
        values_in[i] = i;
        input_search[i] = i;
    }
    if(i<max_num_segments){
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
    // if(i==0){
    //     printf("num segments fill offsets sort: %d\n", num_segments);
    // }
    if(i==0){
        aux_offsets_sort[num_segments*2] = n;
    }

    if(i < num_segments){
        unsigned int index = i*2; 
        aux_offsets_sort[index] = offsets_sort[i];
        aux_offsets_sort[index + 1] = (offsets_sort[i+1] - offsets_sort[i] + 1)/2 + offsets_sort[i];
    }

    // if(threadIdx.x==0 && blockIdx.x==0){
    //     printf("new num segment: %d\n", num_segments*2);
    //     printf("offsets sort\n");
    //     for(unsigned int j=0; j<num_segments+1; ++j){
    //         printf("%d ", offsets_sort[j]);
    //     }
    //     printf("\n");
    //     printf(" newoffsets sort\n");
    //     for(unsigned int j=0; j<num_segments*2+1; ++j){
    //         printf("%d ", aux_offsets_sort[j]);
    //     }
    //     printf("\n");
    // }
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

    // if(threadIdx.x==0 && blockIdx.x==0){
    //     printf(" newoffsets reduce\n");
    //     for(unsigned int j=0; j<num_segments*dim+1; ++j){
    //         printf("%d ", offsets_reduce[j]);
    //     }
    //     printf("\n");
    // }
}

__global__ void fillReductionArray(int n, unsigned int dim, H2Opus_Real* dataset, int* values_in, H2Opus_Real* reduce_in){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<(long long)n*(long long)dim){
        reduce_in[i] = dataset[(long long)values_in[i - (i/n)*n] + (long long)(i/n)*n];
    }

    // if(i==0){
    //     printf("reduction array\n");
    //     for(unsigned int j=0; j<n*dim; ++j){
    //         printf("%lf ", reduce_in[j]);
    //     }
    //     printf("\n");
    // }
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

    // if(i==0){
    //     printf("num segments %u\n", num_segments);
    //     printf("spans\n");
    //     for(unsigned int j=0;j<num_segments; ++j){
    //         for(int k=0; k<dim;++k){
    //             printf("%lf ", span[j*dim + k]);
    //             printf("  %lf %lf      ", reduce_max_out[k*num_segments + j], reduce_min_out[k*num_segments + j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");

        // printf("span offsets\n");
        // for(unsigned int j=0;j<num_segments + 1; ++j){
        //     printf("%d ", span_offsets[j]);
        // }
        // printf("\n");
    // }
}

__global__ void fillCurrDim(int n, unsigned int num_segments, int* currDimArray, cub::KeyValuePair<int, H2Opus_Real>* spanReduced){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<num_segments){
        currDimArray[i] = spanReduced[i].key;
    }
    // if(threadIdx.x==0 && blockIdx.x==0){
    //     printf("curr dim\n");
    //     for(int j=0; j<num_segments; ++j){
    //         printf("%d ", currDimArray[j]);
    //     }
    //     printf("\n");
    // }
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
    // if(threadIdx.x==0 && blockIdx.x==0){
    //     printf("curr dim\n");
    //     for(int j=0; j<num_segments; ++j){
    //         printf("%d ", currDimArray[j]);
    //     }
    //     printf("\n");
    // }
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
    // if(i==0){
    //     printf("fill keys in\n");
    //     for(unsigned int j=0; j<n; ++j){
    //         printf("%lf ", keys_in[j]);
    //     }
    //     printf("\n");
    // }
    // if(threadIdx.x==0 && blockIdx.x==0){
    //     printf("output\n");
    //     for(unsigned int j=0; j<n; ++j){
    //         printf("%d ", output[j]);
    //     }
    // }
    // if(threadIdx.x==0 && blockIdx.x==0){
    //     printf("keys in\n");
    //     for(unsigned int j=0; j<n; ++j){
    //         printf("%lf ", keys_in[j]);
    //     }
    // }
}

__device__ H2Opus_Real getCorrelationLength(int dim){
    return dim == 3 ? 0.2 : 0.1;
}

__device__ H2Opus_Real interaction(int n, int dim, int col, int row, H2Opus_Real* dataset){
    H2Opus_Real diff = 0;
    H2Opus_Real x, y;
    for (int d = 0; d < dim; ++d){
        x = dataset[d*n + col];
        y = dataset[d*n + row];
        diff += (x - y)*(x - y);
    }

    H2Opus_Real dist = sqrt(diff);
    return exp(-dist / getCorrelationLength(dim));
}

__global__ void generateInputMatrix(uint64_t n, uint64_t num_segments, uint64_t maxSegmentSize, uint64_t dim, int* index_map, H2Opus_Real* matrix, H2Opus_Real* dataset, int* offsets_sort, int segment, H2Opus_Real* diagonal){
    if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0){
        // printf("generate input matrix\n");
    }
    unsigned int row = blockIdx.y*maxSegmentSize + threadIdx.y;
    unsigned int col = segment*maxSegmentSize + threadIdx.x;

    uint64_t num_elements = num_segments*maxSegmentSize;
    int xDim = offsets_sort[segment + 1] - offsets_sort[segment];
    int yDim = offsets_sort[blockIdx.y + 1] - offsets_sort[blockIdx.y];

    if(threadIdx.x < maxSegmentSize && threadIdx.y < maxSegmentSize){
        if(((uint64_t)col*num_elements + (uint64_t)row) >= (maxSegmentSize*maxSegmentSize*num_segments*num_segments)){
            assert(0);
        }
        if(threadIdx.x >= xDim || threadIdx.y >= yDim) {
            if(col == row){
                matrix[blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y] = 1;
            }
            else{
                matrix[blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y] = 0;
            }
        }
        else{
            // matrix[(uint64_t)col*num_elements + (uint64_t)row] = interaction(n, dim, index_map[offsets_sort[blockIdx.x] + threadIdx.x], index_map[offsets_sort[blockIdx.y] + threadIdx.y], dataset);
            matrix[blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y] = interaction(n, dim, index_map[offsets_sort[segment] + threadIdx.x], index_map[offsets_sort[blockIdx.y] + threadIdx.y], dataset);
        }
        if(blockIdx.y == segment){
            diagonal[segment*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y] = matrix[blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y];
        }
    }
}

__global__ void calcMemNeeded(int maxSegmentSize, unsigned int* K, H2Opus_Real* S, float eps){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;

    __shared__ int k;
    if(threadIdx.x == 0){
        k = 0;
    }
    __syncthreads();

    if(threadIdx.x < maxSegmentSize){
        if((S[maxSegmentSize*blockIdx.x + threadIdx.x] / S[maxSegmentSize*blockIdx.x]) > eps){
            atomicAdd(&k, 1);
        }
    }
    __syncthreads();

    if(threadIdx.x == 0){
        K[blockIdx.x] = k;
    }
}

__global__ void tileMatrix(int n, int num_segments, int maxSegmentSize, H2Opus_Real* S, H2Opus_Real* U, H2Opus_Real* V, H2Opus_Real* U_tiled, H2Opus_Real* V_tiled, unsigned int* K, int* K_scan, int segment){
    if(threadIdx.x < K[blockIdx.y] && threadIdx.y < maxSegmentSize){
        U_tiled[K_scan[blockIdx.y]*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y]
          = U[blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y]
          * S[blockIdx.y*maxSegmentSize + threadIdx.x];
    }

    if(threadIdx.x < K[blockIdx.y] && threadIdx.y < maxSegmentSize){
        V_tiled[K_scan[blockIdx.y]*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y] 
      = V[blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y];
    }

    // __syncthreads();
    // if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0){
    //     printf("U\n");
    //     for(int i=0; i<maxSegmentSize; ++i){
    //         for(int j=0; j<K[blockIdx.x*num_segments + blockIdx.y]; ++j){
    //             printf("%lf ", U_tiled[j*maxSegmentSize + i]);
    //         }
    //         printf("\n");
    //     }
    // }
    // if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0){
    //     printf("V\n");
    //     for(int i=0; i<K[blockIdx.x*num_segments + blockIdx.y]; ++i){
    //         for(int j=0; j<maxSegmentSize; ++j){
    //             printf("%lf ", V_tiled[i*maxSegmentSize + j]);
    //         }
    //         printf("\n");
    //     }
    // }
}

__global__ void fillBitVector(int num_segments, uint64_t* bit_vector, int* offsets_sort, int bucket_size){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;

    if(i<num_segments){
        unsigned int pos = i%(sizeof(uint64_t)*8);
        unsigned int sub = i/(sizeof(uint64_t)*8);
        if(offsets_sort[i+1] - offsets_sort[i] > bucket_size){
            atomicOr((unsigned long long*)&bit_vector[sub], 1ULL<<(sizeof(uint64_t)*8-1-pos));
        }
    }
}

__global__ void  fillPopCount(int num_threads, uint64_t* bit_vector, short int* popc_bit_vector){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i<num_threads){
        popc_bit_vector[i] = __popcll(bit_vector[i]);
    }
}

__global__ void isWorkDone(int num_segments, uint64_t* bit_vector, bool* workDone){
    if(bit_vector[0] == 0ULL){
        *workDone = true;
    }
}

__global__ void printMatrix(int num_segments, int maxSegmentSize, H2Opus_Real* matrix){
    for(unsigned int i=0; i<num_segments; ++i){
        for(unsigned int j=0; j<maxSegmentSize; ++j){
            for(unsigned int k=0; k<maxSegmentSize; ++k){
                printf("%lf ", matrix[i*maxSegmentSize*maxSegmentSize + j*maxSegmentSize + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

__global__ void printSort(int n, H2Opus_Real* keys_out, int* values_out){
    printf("keys out\n");
    for(int i=0; i<n; ++i){
        printf("%lf ", keys_out[i]);
    }
    printf("\n");
    printf("values out\n");
    for(int i=0; i<n; ++i){
        printf("%d ", values_out[i]);
    }
    printf("\n");
}

__global__ void printK(int num_segments, unsigned int* K){
    printf("ks\n");
    for(int i=0; i<num_segments; ++i){
        printf("%d ", K[i]);
    }
    printf("\n");
}

// __global__ void printK(int num_segments, unsigned int* K){
//     printf("ks\n");
//     for(int i=0; i<num_segments; ++i){
//         for(int j=0; j<num_segments; ++j){
//             printf("%d ", K[j*num_segments + i]);
//         }
//         printf("\n");
//     }
// }

__global__ void printOffsetsSort(int num_segments, int* offsets_sort){
    printf("print offsets sort\n");
    for(int i=0; i<num_segments+1; ++i){
        printf("%d ", offsets_sort[i]);
    }
    printf("\n");
}

__global__ void getTotalMem(int* totalMem, unsigned int* K, int* scan_K, int num_segments){
    // printf("k scan\n");
    // for(int i=0; i<num_segments*num_segments; ++i){
    //     printf("%d ", scan_K[i]);
    // }
    // printf("\n");
    *totalMem = scan_K[num_segments - 1] + K[num_segments - 1];
    // printf("total mem: %d\n", *totalMem);
}

__global__ void expandMatrix(int num_segments, int maxSegmentSize, unsigned int* K, int* scan_K, H2Opus_Real* U_tiled, H2Opus_Real* V_tiled, H2Opus_Real* expMatrix){
    if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0){
        // printf("expand matrix\n");
    }

    if(threadIdx.x < maxSegmentSize && threadIdx.y < maxSegmentSize){
        int col = blockIdx.x*maxSegmentSize + threadIdx.x;
        int row = blockIdx.y*maxSegmentSize + threadIdx.y;
        int index = blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y;
        expMatrix[index] = 0;

        for(unsigned int i = 0; i<K[blockIdx.y]; ++i){
            expMatrix[index] += (U_tiled[scan_K[blockIdx.y]*maxSegmentSize + i*maxSegmentSize + threadIdx.y]
                              * V_tiled[scan_K[blockIdx.y]*maxSegmentSize + i*maxSegmentSize + threadIdx.x]);
        }
    }
}

__global__ void calcError(int num_segments, int maxSegmentSize, H2Opus_Real* expMatrix, H2Opus_Real* input_matrix, H2Opus_Real* error, H2Opus_Real* tmp){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < num_segments*maxSegmentSize*maxSegmentSize){
        H2Opus_Real x = input_matrix[i];
        H2Opus_Real y = expMatrix[i];
        atomicAdd(tmp, x*x);
        atomicAdd(error, (x-y)*(x-y));
        // printf("x: %lf   y: %lf\n", x, y);
    }
}

__global__ void printExpM(uint64_t num_segments, uint64_t maxSegmentSize, H2Opus_Real* expMatrix, H2Opus_Real* input_matrix){
    // printf("exp matrix\n");
    for(unsigned int i=0; i<maxSegmentSize; ++i){
        for(unsigned int j=0; j<num_segments*maxSegmentSize; ++j){
            int xCoord = i/maxSegmentSize;
            int yCoord = j/maxSegmentSize;
            int index = yCoord*maxSegmentSize*maxSegmentSize + i*maxSegmentSize + (j%maxSegmentSize);
            // printf("%lf ", expMatrix[index]);
        }
        // printf("\n");
    }
}

__global__ void fillVector(int num_segments, int maxSegmentSize, H2Opus_Real* input_vector, H2Opus_Real* output_vector, H2Opus_Real* output_vector_org){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < num_segments*maxSegmentSize){
        unsigned int seed = i;
        curandState s;
        curand_init(seed, 0, 0, &s);
        H2Opus_Real random_n = curand_uniform(&s);
        input_vector[i]= random_n;
        output_vector[i]= 0;
        output_vector_org[i]= 0;
    }
}

__global__ void PrintVector(unsigned int num_segments, unsigned int maxSegmentSize, H2Opus_Real * d_output_vector){
    for (unsigned int i = 0; i < num_segments*maxSegmentSize; ++i){
        printf("%lf ", d_output_vector[i]);
    }
    printf("\n");
}

__global__ void calcError_vector (int num_segments, int maxSegmentSize, H2Opus_Real* output_vector, H2Opus_Real* output_vector_org, H2Opus_Real* error_vector, H2Opus_Real* tmp_vector){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < num_segments*maxSegmentSize){
        H2Opus_Real x = output_vector_org[i];
        H2Opus_Real y = output_vector[i];
        atomicAdd(tmp_vector, x*x);
        atomicAdd(error_vector, (x-y)*(x-y));
        // printf("x: %lf   y: %lf\n", x, y);
    }
}

__global__ void GEMV(int num_segments, int maxSegmentSize, unsigned int* K, unsigned int* scan_k, H2Opus_Real* U_tiled, H2Opus_Real* V_tiled, H2Opus_Real* diagonal, H2Opus_Real* input_vector, H2Opus_Real* output_vector, H2Opus_Real* buffer_vector){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    // __shared__ H2Opus_Real input_vector_s[maxSegmentSize];
    // if(threadIdx.x < maxSegmentSize){
    //     input_vector_s[threadIdx.x] = input_vector[threadIdx.x + maxSegmentSize*blockIdx.x];
    // }
    // __syncthreads();
    unsigned int warpIdx = threadIdx.x/32;
    unsigned int blockIndex = (blockIdx.x*2 + warpIdx);
    if(threadIdx.x==0 && blockIdx.x==0){
        printf("started GEMV\n");
    }
    for(unsigned int tile = 0; tile < num_segments; ++tile){
        if(tile == blockIndex){
            H2Opus_Real tmp_sum = 0;
            for(unsigned int index=0; index<maxSegmentSize; ++index){
                tmp_sum += diagonal[index*maxSegmentSize + threadIdx.x]*input_vector[maxSegmentSize*blockIndex + index];
            }
            output_vector[maxSegmentSize*(blockIdx.x*2) + threadIdx.x] += tmp_sum;
        }
        else{
            if(threadIdx.x < K[tile*num_segments + blockIndex]){
                H2Opus_Real tmp_sum = 0;
                for(unsigned int v_tile_index=0; v_tile_index<maxSegmentSize; ++v_tile_index){
                    tmp_sum += V_tiled[scan_k[tile*num_segments + blockIndex]*maxSegmentSize + maxSegmentSize*threadIdx.x + v_tile_index]*input_vector[maxSegmentSize*blockIndex + v_tile_index];
                }
                assert(K[tile*num_segments + blockIndex] <= 32);
                buffer_vector[K[tile*num_segments + blockIndex] - (threadIdx.x+1) + maxSegmentSize*(blockIdx.x*2)] = tmp_sum;
                // buffer_vector[threadIdx.x + maxSegmentSize*blockIdx.x] = tmp_sum;
            }
            __syncthreads();

            if(threadIdx.x < maxSegmentSize){
                H2Opus_Real tmp_sum = 0;
                for(unsigned int u_tile_index=0; u_tile_index<K[tile*num_segments + blockIndex]; ++u_tile_index){
                    tmp_sum += U_tiled[scan_k[tile*num_segments + blockIndex]*maxSegmentSize + u_tile_index*maxSegmentSize + threadIdx.x]*buffer_vector[maxSegmentSize*blockIndex + u_tile_index];
                }
                output_vector[maxSegmentSize*(blockIdx.x*2) + threadIdx.x] += tmp_sum;
            }
        }
        __syncthreads();
        // if(threadIdx.x==0 && blockIdx.x==0){
        //     printf("tile #: %d\n", tile);
        // }
    }
    if(threadIdx.x==0 && blockIdx.x==0){
        printf("finished GEMV\n");
    }
}

__global__ void GEMM(int num_segments, int maxSegmentSize, H2Opus_Real* U_tiled_1, H2Opus_Real* V_tiled_1, H2Opus_Real* diagonal_1, unsigned int* K_1, unsigned int* scan_k_1, H2Opus_Real* U_tiled_2, H2Opus_Real* V_tiled_2, H2Opus_Real* diagonal_2, unsigned int* K_2, unsigned int* scan_k_2, H2Opus_Real* d_gemm_matrix_segmented, unsigned int segment, int bucket_size){
    // __shared__ H2Opus_Real first_matrix[bucket_size][bucket_size];
    // __shared__ H2Opus_Real second_matrix[bucket_size][bucket_size];
    extern __shared__ H2Opus_Real shmem[];
    H2Opus_Real *first_matrix = (H2Opus_Real*) shmem;
    H2Opus_Real *second_matrix = (H2Opus_Real*)&shmem[bucket_size*bucket_size];
    H2Opus_Real sum = 0;

    for(unsigned int tile = 0; tile<num_segments; ++tile){
        // dense*dense
        if(blockIdx.y == tile && segment == tile){
            first_matrix[threadIdx.x*maxSegmentSize + threadIdx.y] = diagonal_1[blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y];
            second_matrix[threadIdx.x*maxSegmentSize + threadIdx.y] = diagonal_2[blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y];
            __syncthreads();

            for(unsigned int i=0; i<maxSegmentSize; ++i){
                sum += first_matrix[i*maxSegmentSize + threadIdx.y]*second_matrix[threadIdx.x*maxSegmentSize + i];
            }
            __syncthreads();
        }
        // lowRank*lowRank
        else if(tile!=blockIdx.y && tile!=segment){
            H2Opus_Real temp = 0;
            unsigned int matrix1_rank = K_1[tile*num_segments + segment];
            unsigned int matrix2_rank = K_2[segment*num_segments + tile];

            // loads Av_s into shared memory
            if(threadIdx.x < matrix1_rank){
                first_matrix[threadIdx.x*bucket_size + threadIdx.y] = V_tiled_1[scan_k_1[tile*num_segments + segment] + threadIdx.x*maxSegmentSize + threadIdx.y];
            }
            // loads Bu_s into shared memory
            if(threadIdx.x < matrix2_rank){
                second_matrix[threadIdx.x*bucket_size + threadIdx.y] = U_tiled_2[scan_k_2[segment*num_segments + tile] + threadIdx.x*maxSegmentSize + threadIdx.y];
            }
            __syncthreads();

            // multiplies Av_s*Bu_s
            // stores result in place of Av_s
            if(threadIdx.x < matrix2_rank && threadIdx.y < matrix1_rank){
                for(unsigned int i = 0; i < maxSegmentSize; ++i){
                    temp += first_matrix[threadIdx.y*bucket_size + i] * second_matrix[bucket_size*threadIdx.x + i];
                }
            }
            __syncthreads();

            if(threadIdx.x < matrix2_rank && threadIdx.y < matrix1_rank){
                first_matrix[threadIdx.x*bucket_size + threadIdx.y] = temp;
            }
            __syncthreads();

            // loads Bv_s into shared memory. Replaces Bu_s
            if(threadIdx.x < matrix2_rank){
                second_matrix[threadIdx.y + bucket_size*threadIdx.x] = V_tiled_2[scan_k_2[segment*num_segments + tile] + threadIdx.x*maxSegmentSize + threadIdx.y];
            }
            __syncthreads();

            temp = 0.0f;
            // multiplies (Av_s*Bu_s)*Bv_s and stores result in Av_s space
            if(threadIdx.y < matrix1_rank){
                for(unsigned int i = 0; i < matrix2_rank; ++i){
                    temp += first_matrix[i*bucket_size + threadIdx.y]*second_matrix[threadIdx.x + bucket_size*i];
                }
            }
            __syncthreads();
            if(threadIdx.y < matrix1_rank){
                first_matrix[threadIdx.x*bucket_size + threadIdx.y] = temp;
            }
            __syncthreads();
            
            // loads Au_s into shared memory
            if(threadIdx.x < matrix1_rank){
                second_matrix[threadIdx.x*bucket_size + threadIdx.y] = U_tiled_1[scan_k_1[tile*num_segments + segment] + threadIdx.x*maxSegmentSize + threadIdx.y];
            }
            __syncthreads();
            // multiplies Au_s*(Av_s*Bu_s*Bv_s) and stores result in sum
            for(unsigned int i = 0; i < matrix1_rank; ++i){
                sum += second_matrix[threadIdx.y + bucket_size*i]*first_matrix[i + bucket_size*threadIdx.x];
            }
            __syncthreads();
        }
        // lowRank*dense
        else if(tile!=segment){
            H2Opus_Real temp = 0;
            unsigned int matrix1_rank = K_1[tile*num_segments + segment];
            // loads Av_s into shared memory
            if(threadIdx.x < matrix1_rank){
                first_matrix[threadIdx.x*bucket_size + threadIdx.y] = V_tiled_1[scan_k_1[tile*num_segments + segment] + threadIdx.x*maxSegmentSize + threadIdx.y];
            }
            if(threadIdx.x<maxSegmentSize && threadIdx.y<maxSegmentSize){
                second_matrix[threadIdx.x*maxSegmentSize + threadIdx.y] = diagonal_2[threadIdx.x*maxSegmentSize + threadIdx.y];
            }
            __syncthreads();

            if(threadIdx.x < matrix1_rank){
                for(unsigned int i=0; i<maxSegmentSize; ++i){
                    temp += first_matrix[threadIdx.x*maxSegmentSize + i]*second_matrix[threadIdx.x*maxSegmentSize + i];
                }

            }
            __syncthreads();

            if(threadIdx.x < matrix1_rank && threadIdx.y < maxSegmentSize){
                second_matrix[threadIdx.x*maxSegmentSize + threadIdx.y] = temp;
            }
            __syncthreads();

            // loads Au_s into shared memory
            if(threadIdx.x < matrix1_rank && threadIdx.y<maxSegmentSize){
                first_matrix[threadIdx.x*maxSegmentSize + threadIdx.y] = U_tiled_1[scan_k_1[tile*num_segments + segment] + threadIdx.x*maxSegmentSize + threadIdx.y];
            }
            __syncthreads();
            if(threadIdx.x<maxSegmentSize && threadIdx.y<maxSegmentSize){
                for(unsigned int i=0; i<matrix1_rank; ++i){
                    sum+=first_matrix[threadIdx.y*maxSegmentSize + i]*second_matrix[threadIdx.x*maxSegmentSize + i];
                }
            }
            __syncthreads();
        }
        // dense*lowrank
        else{
            H2Opus_Real temp = 0;
            unsigned int matrix2_rank = K_2[segment*num_segments + tile];
            // loads Bu_s into shared memory
            if(threadIdx.x < matrix2_rank){
                first_matrix[threadIdx.x*maxSegmentSize + threadIdx.y] = U_tiled_2[scan_k_2[segment*num_segments + tile] + threadIdx.x*maxSegmentSize + threadIdx.y];
            }
            if(threadIdx.x<maxSegmentSize && threadIdx.y<maxSegmentSize){
                second_matrix[threadIdx.x*maxSegmentSize + threadIdx.y] = diagonal_1[threadIdx.x*maxSegmentSize + threadIdx.y];
            }
            __syncthreads();

            if(threadIdx.x<matrix2_rank && threadIdx.y<maxSegmentSize){
                for(unsigned int i=0; i<maxSegmentSize; ++i){
                    temp += first_matrix[i*maxSegmentSize + threadIdx.y]*second_matrix[threadIdx.x*maxSegmentSize + i];
                }
            }
            __syncthreads();

            if(threadIdx.x<matrix2_rank && threadIdx.y<maxSegmentSize){
                first_matrix[threadIdx.x*maxSegmentSize + threadIdx.y] = temp;
            }
            __syncthreads();

            if(threadIdx.x<matrix2_rank && threadIdx.y<maxSegmentSize){
                second_matrix[threadIdx.y + maxSegmentSize*threadIdx.x] = V_tiled_2[scan_k_2[segment*num_segments + tile] + threadIdx.x*maxSegmentSize + threadIdx.y];
            }
            __syncthreads();
            if(threadIdx.x<maxSegmentSize && threadIdx.y<maxSegmentSize){
                for(unsigned int i=0; i<matrix2_rank; ++i){
                    sum += first_matrix[i*maxSegmentSize + threadIdx.y]*second_matrix[i*maxSegmentSize + threadIdx.x];
                }
            }
            __syncthreads();
        }
    }
    d_gemm_matrix_segmented[blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y] = sum;
}

__global__ void fillDenseMatrix(int n, H2Opus_Real* matrix){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int seed = i;
    if(i<n){
        curandState s;
        for(unsigned int j=0;j<n; ++j){
            curand_init(seed, 0, 0, &s);
            H2Opus_Real random_n = curand_uniform(&s);
            matrix[j*n + i] = random_n;
        }
    }
}

__global__ void filltmpVector(int n, H2Opus_Real* vector){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int seed = i;
    if(i<n){
        curandState s;
        curand_init(seed, 0, 0, &s);
        H2Opus_Real random_n = curand_uniform(&s);
        vector[i] = random_n;
    }
}

__global__ void fillBatch(int num_segments, int* rows_batch, int* cols_batch, int maxSegmentSize){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<num_segments){
        rows_batch[i] = maxSegmentSize;
        cols_batch[i] = maxSegmentSize;
    }
}