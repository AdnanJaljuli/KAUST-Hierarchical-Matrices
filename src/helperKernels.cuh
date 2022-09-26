
// TODO: split the functions in this file into two. Those which are called in kdtreeConstruction.cuh and others
#ifndef HELPERKERNELS_CUH
#define HELPERKERNELS_CUH

#include "TLR_Matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stddef.h>
#include <cub/cub.cuh>

#include <assert.h>
#include <curand_kernel.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

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

__device__ H2Opus_Real getCorrelationLength(int dim){
    return dim == 3 ? 0.2 : 0.1;
}

__device__ H2Opus_Real interaction(int n, int dim, int col, int row, H2Opus_Real* pointCloud){
    assert(col<n);
    assert(row<n);

    H2Opus_Real diff = 0;
    H2Opus_Real x, y;
    for (int d = 0; d < dim; ++d){
        x = pointCloud[d*n + col];
        y = pointCloud[d*n + row];
        diff += (x - y)*(x - y);
    }

    H2Opus_Real dist = sqrt(diff);
    return exp(-dist / getCorrelationLength(dim));
}

__global__ void generateInputMatrix(uint64_t n, uint64_t num_segments, uint64_t maxSegmentSize, int dim, int* index_map, H2Opus_Real* matrix, H2Opus_Real* pointCloud, int* offsets_sort, int segment, H2Opus_Real* diagonal, H2Opus_Real* denseMatrix, int expand_matrix){
    for(unsigned int i=0; i<(maxSegmentSize/blockDim.x); ++i){
        for(unsigned int j=0; j<(maxSegmentSize/blockDim.x); ++j){
            unsigned int row = blockIdx.y*maxSegmentSize + i*blockDim.x + threadIdx.y;
            unsigned int col = segment*maxSegmentSize + j*blockDim.x + threadIdx.x;

            int xDim = offsets_sort[segment + 1] - offsets_sort[segment];
            int yDim = offsets_sort[blockIdx.y + 1] - offsets_sort[blockIdx.y];

            unsigned int diff = (blockIdx.y > segment) ? 1 : 0;

            if(((uint64_t)col*num_segments*maxSegmentSize + (uint64_t)row) >= (maxSegmentSize*maxSegmentSize*num_segments*num_segments)){
                assert(0);
            }
            if(blockIdx.y == segment){
                diagonal[segment*maxSegmentSize*maxSegmentSize + j*maxSegmentSize*blockDim.x + threadIdx.x*maxSegmentSize + i*blockDim.x + threadIdx.y] = interaction(n, dim, index_map[offsets_sort[segment] + blockDim.x*j + threadIdx.x], index_map[offsets_sort[blockIdx.y] + i*blockDim.x + threadIdx.y], pointCloud);
                if(expand_matrix == 1){
                    denseMatrix[(segment*maxSegmentSize + threadIdx.x)*maxSegmentSize*num_segments + blockIdx.y*maxSegmentSize + threadIdx.y] = interaction(n, dim, index_map[offsets_sort[segment] + blockDim.x*j + threadIdx.x], index_map[offsets_sort[blockIdx.y] + i*blockDim.x + threadIdx.y], pointCloud);
                }
            }
            else{
                if(threadIdx.x + j*blockDim.x >= xDim || threadIdx.y + i*blockDim.x >= yDim) {
                    if(col == row){
                        matrix[blockIdx.y*maxSegmentSize*maxSegmentSize  + (-1*diff*maxSegmentSize*maxSegmentSize) + j*blockDim.x*maxSegmentSize + threadIdx.x*maxSegmentSize + i*blockDim.x + threadIdx.y] = 1;
                        if(expand_matrix == 1){
                            denseMatrix[(segment*maxSegmentSize + threadIdx.x)*maxSegmentSize*num_segments + blockIdx.y*maxSegmentSize + threadIdx.y] = 1;
                        }
                    }
                    else{
                        matrix[blockIdx.y*maxSegmentSize*maxSegmentSize - diff*maxSegmentSize*maxSegmentSize + j*blockDim.x*maxSegmentSize + threadIdx.x*maxSegmentSize + i*blockDim.x + threadIdx.y] = 0;
                        if(expand_matrix == 1){
                            denseMatrix[(segment*maxSegmentSize + threadIdx.x)*maxSegmentSize*num_segments + blockIdx.y*maxSegmentSize + threadIdx.y] = 0;
                        }
                    }
                }
                else {
                    matrix[blockIdx.y*maxSegmentSize*maxSegmentSize - diff*maxSegmentSize*maxSegmentSize + j*blockDim.x*maxSegmentSize + threadIdx.x*maxSegmentSize + i*blockDim.x + threadIdx.y] = interaction(n, dim, index_map[offsets_sort[segment] + blockDim.x*j + threadIdx.x], index_map[offsets_sort[blockIdx.y] + i*blockDim.x + threadIdx.y], pointCloud);
                    if(expand_matrix == 1){
                        denseMatrix[(segment*maxSegmentSize + threadIdx.x)*maxSegmentSize*num_segments + blockIdx.y*maxSegmentSize + threadIdx.y] = interaction(n, dim, index_map[offsets_sort[segment] + blockDim.x*j + threadIdx.x], index_map[offsets_sort[blockIdx.y] + i*blockDim.x + threadIdx.y], pointCloud);
                    }
                }
            }
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
}

__global__ void fillBitVector(int num_segments, uint64_t* bit_vector, int* offsets_sort, int bucket_size){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;

    if(i < num_segments){
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

__global__ void printK(int* K, int num_segments){
    printf("ks\n");
    for(int i=0; i<num_segments; ++i){
        printf("%d ", K[i]);
    }
    printf("\n");
}

__global__ void getTotalMem(uint64_t* totalMem, int* K, int* scan_K, int num_segments){
    *totalMem = (uint64_t)scan_K[num_segments - 1] + (uint64_t)K[num_segments - 1];
}

__global__ void calcError(int num_segments, int maxSegmentSize, H2Opus_Real* expMatrix, H2Opus_Real* input_matrix, H2Opus_Real* error, H2Opus_Real* tmp){
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < num_segments*maxSegmentSize*maxSegmentSize){
        H2Opus_Real x = input_matrix[i];
        H2Opus_Real y = expMatrix[i];
        atomicAdd(tmp, x*x);
        atomicAdd(error, (x-y)*(x-y));
    }
}

__global__ void fillDenseMatrix(int n, H2Opus_Real* matrix) {
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

__global__ void fillARAArrays(int batchCount, int max_rows, int max_cols, int* d_rows_batch, int* d_cols_batch, int* d_ldm_batch, int* d_lda_batch, int* d_ldb_batch){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < batchCount){
        d_rows_batch[i] = max_rows;
        d_cols_batch[i] = max_cols;
        d_ldm_batch[i] = max_rows;
        d_lda_batch[i] = max_rows;
        d_ldb_batch[i] = max_rows;
    }
}

__global__ void fillLRARAArrays(int batchCount, int max_rows, int max_cols, int* d_rows_batch, int* d_cols_batch, int* d_lda_batch, int* d_ldb_batch){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < batchCount){
        d_rows_batch[i] = max_rows;
        d_cols_batch[i] = max_rows;
        d_lda_batch[i] = max_rows;
        d_ldb_batch[i] = max_rows;
    }
}

template<class T>
struct UnaryAoAAssign : public thrust::unary_function<int, T*>
{
    T* original_array;
    int stride;
    UnaryAoAAssign(T* original_array, int stride) { this->original_array = original_array; this->stride = stride; }
    __host__ __device__
    T* operator()(const unsigned int& thread_id) const { return original_array + thread_id * stride; }
};

template<class T>
void generateArrayOfPointersT(T* original_array, T** array_of_arrays, int stride, int num_arrays, cudaStream_t stream)
{
    thrust::device_ptr<T*> dev_data(array_of_arrays);
    thrust::transform(
        thrust::cuda::par.on(stream),
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(num_arrays),
        dev_data,
        UnaryAoAAssign<T>(original_array, stride)
    );
    cudaGetLastError();
}

__global__ void printARAOutput(H2Opus_Real* d_A, H2Opus_Real* d_B, int* k, int batchCount, int max_rows, int max_rank){
    printf("ks\n");
    for(unsigned int i=0;i<batchCount; ++i){
        printf("%d ", k[i]);
    }
    printf("\n");
}

__global__ void copyTiles(int batchCount, int maxSegmentSize, int* d_ranks, int* d_scan_k, H2Opus_Real* d_U_tiled_segmented, H2Opus_Real* d_A, H2Opus_Real* d_V_tiled_segmented, H2Opus_Real* d_B){
    if(threadIdx.x < d_ranks[blockIdx.x]){
        for(unsigned int i = 0; i < maxSegmentSize; ++i) {
            d_U_tiled_segmented[d_scan_k[blockIdx.x]*maxSegmentSize + threadIdx.x*maxSegmentSize + i] = d_A[blockIdx.x*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + i];
            d_V_tiled_segmented[d_scan_k[blockIdx.x]*maxSegmentSize + threadIdx.x*maxSegmentSize + i] = d_B[blockIdx.x*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + i];
        }
    }
}

__global__ void copyRanks(int num_segments, int maxSegmentSize, int* from_ranks, int* to_ranks){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < num_segments*(num_segments-1)){
        int row = i%(num_segments-1);
        int col = i/(num_segments-1);
        int diff = (row>=col) ? 1 : 0;
        to_ranks[i + col + diff] = from_ranks[i];
    }
    if(i < num_segments){
        to_ranks[i*num_segments + i] = 0;
    }
}

__host__ __device__ int getMOfromXY_h(unsigned int x, unsigned int y){
    static const unsigned int B[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};
    static const unsigned int S[] = {1, 2, 4, 8};

    x = (x | (x << S[3])) & B[3];
    x = (x | (x << S[2])) & B[2];
    x = (x | (x << S[1])) & B[1];
    x = (x | (x << S[0])) & B[0];

    y = (y | (y << S[3])) & B[3];
    y = (y | (y << S[2])) & B[2];
    y = (y | (y << S[1])) & B[1];
    y = (y | (y << S[0])) & B[0];

    int z = x | (y << 1);
    return z;
}

__host__ __device__ int IndextoMOIndex_h(int num_segments, int n){
    unsigned int i = n%num_segments;
    unsigned int j = n/num_segments;
    return getMOfromXY_h(j, i);
}

__global__ void copyCMRanksToMORanks(int num_segments, int maxSegmentSize, int* matrixRanks, int* mortonMatrixRanks){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<num_segments*num_segments){
        int MOIndex = IndextoMOIndex_h(num_segments, i);
        mortonMatrixRanks[MOIndex] = matrixRanks[i];
    }
}

__global__ void copyTilestoMO(int n, H2Opus_Real* toArray, H2Opus_Real* fromArray, int offset_1, int offset_2){
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n){
        toArray[offset_1 + i] = fromArray[offset_2 + i];
    }
}

__device__ uint32_t morton1(uint32_t x)
{
    x = x & 0x55555555;
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF;
    return x;
}

__global__ void fillFirstLevelExistingArrays(int num_segments, int* existingArrays, int* existingRanks, int* matrixRanks){
    int tmp = 0;
    for(unsigned int i=0; i<num_segments*num_segments; ++i){
        int x = morton1(i);
        int y = morton1(i >> 1);
        if(x != y){
            existingRanks[tmp] = matrixRanks[i];
            existingArrays[tmp++] = i;
        }
    }
}

__global__ void fillActiveTiles(int numExistingArrays, int* activeArrays, int* existingArrays, int* activeRanks, int* existingRanks){
    int tmp = 0;
    for(int i=0; i<numExistingArrays; ++i){
        if(existingArrays[i]%4 == 0 && i<=(numExistingArrays-4)){
            bool flag = true;
            for(int j=1; j<4; ++j){
                if(existingArrays[i + j] != existingArrays[i] + j){
                    flag = false;
                    break;
                }
            }
            if(flag){
                for(int j=0; j<4; ++j){
                    activeArrays[tmp] = existingArrays[i + j];
                    activeRanks[tmp++] = existingRanks[i + j];
                }
                i += 3;
            }
        }
    }
}

__global__ void fillBitVector(int num_ops, int tile_size, int* new_ranks, int* old_ranks, int* new_level_bit_vector, int* old_level_bit_vector){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < num_ops){
        if(2*tile_size*(old_ranks[i*4] + old_ranks[i*4 + 1] + old_ranks[i*4 + 2] + old_ranks[i*4 + 3]) > 4*tile_size*new_ranks[i]){
            new_level_bit_vector[i] = 1;
            old_level_bit_vector[i] = 0;
        }
        else{
            new_level_bit_vector[i] = 0;
            old_level_bit_vector[i] = 1;
        }
    }
}

__global__ void fillNewLevel(int num_ops, int* bit_vector, int* bit_vector_scan, int* ranks_output, int* new_ranks, int* old_active_tiles, int* new_active_tiles){
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < num_ops){
        if(bit_vector[i] == 1){
            new_ranks[bit_vector_scan[i]] = ranks_output[i];
            new_active_tiles[bit_vector_scan[i]] = old_active_tiles[i*4]/4;
        }
    }
}

__global__ void expandMatrix(int num_segments, int maxSegmentSize, H2Opus_Real* expandedMatrix, TLR_Matrix matrix){
    if(blockIdx.x == blockIdx.y){
        expandedMatrix[blockIdx.x*num_segments*maxSegmentSize*maxSegmentSize + blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y] = matrix.diagonal[blockIdx.x*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y];
    }
    else{
        unsigned int index;
        if(matrix.type == MORTON){
            index = IndextoMOIndex_h(num_segments, blockIdx.x*num_segments + blockIdx.y);
        }
        else if(matrix.type == COLUMN_MAJOR){
            index = blockIdx.x*num_segments + blockIdx.y;
        }

        H2Opus_Real sum = 0;
        for(unsigned int i=0; i<matrix.blockRanks[index]; ++i){
            sum += matrix.U[matrix.blockOffsets[index]*maxSegmentSize + i*maxSegmentSize + threadIdx.y]*matrix.V[matrix.blockOffsets[index]*maxSegmentSize + i*maxSegmentSize + threadIdx.x];
        }

        expandedMatrix[blockIdx.x*num_segments*maxSegmentSize*maxSegmentSize + blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y] = sum;
    }
}

__global__ void errorInMatrix(int num_segments, int maxSegmentSize, H2Opus_Real* denseMatrix, H2Opus_Real* expandedMatrix, H2Opus_Real* error, H2Opus_Real* tmp){
    H2Opus_Real x = denseMatrix[(blockIdx.x*maxSegmentSize + threadIdx.x)*maxSegmentSize*num_segments + blockIdx.y*maxSegmentSize + threadIdx.y];
    H2Opus_Real y = expandedMatrix[blockIdx.x*num_segments*maxSegmentSize*maxSegmentSize + blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y];
    
    atomicAdd(tmp, x*x);
    atomicAdd(error, (x-y)*(x-y));
}

__global__ void compareMOwithCM(int num_segments, int maxSegmentSize, H2Opus_Real* expandedCMMatrix, H2Opus_Real* expandedMOMatrix){
    H2Opus_Real x = expandedMOMatrix[blockIdx.x*num_segments*maxSegmentSize*maxSegmentSize + blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y];
    H2Opus_Real y = expandedCMMatrix[blockIdx.x*num_segments*maxSegmentSize*maxSegmentSize + blockIdx.y*maxSegmentSize*maxSegmentSize + threadIdx.x*maxSegmentSize + threadIdx.y];
    assert(x == y);
}

__global__ void getNewLevelCount(int num_ops, int* d_new_bit_vector, int* d_new_bit_vector_scan, int* d_newLevelCount){
    d_newLevelCount[0] = d_new_bit_vector_scan[num_ops - 1] + d_new_bit_vector[num_ops - 1];
}

__global__ void copyTilesToNewLevel(int num_ops, int* bit_vector, TLR_Matrix mortonMatrix, H2Opus_Real* d_A, H2Opus_Real* d_B, int* new_ranks, int* old_active_tiles, int row, int col){
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < num_ops){
        if(bit_vector[i] == 1){
            // TODO: fix the address to where the function will copy
            // TODO: use multiple streams
            cudaMemcpyAsync(&mortonMatrix.U[mortonMatrix.blockOffsets[old_active_tiles[i*4]]*32], &d_A[row*col*i], new_ranks[i]*32*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice, 0);
            cudaMemcpyAsync(&mortonMatrix.V[mortonMatrix.blockOffsets[old_active_tiles[i*4]]*32], &d_B[row*col*i], new_ranks[i]*32*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice, 0);
        }
    }
}

__global__ void calcNumOps(int num_existing_tiles, int* num_ops, int* availableTiles){
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < num_existing_tiles){
        if(availableTiles[i]%4 == 0){
            bool flag = true;
            for(int j=1; j<4; ++j){
                if(availableTiles[i+j] != availableTiles[i]+j){
                    flag = false;
                    break;
                }
            }
            if(flag){
                atomicAdd(num_ops, 1);
            }
        }
    }
}

__global__ void expandHMatrixLevel(int num_ops, int max_rows, int max_cols, H2Opus_Real* d_A, H2Opus_Real* d_B, int* d_ranks, H2Opus_Real* expandedMatrix){
    int col = threadIdx.x + blockIdx.x*(max_cols/2);
    int row = threadIdx.y + (blockIdx.y%2)*(max_rows/2);
    int block = blockIdx.y/2;
    H2Opus_Real sum = 0;

    for(unsigned int i=0; i<d_ranks[block]; ++i){
        sum += d_A[max_rows*max_cols*block + i*max_rows + row]*d_B[max_rows*max_cols*block + i*max_rows + col];
    }
    expandedMatrix[block*max_rows*max_cols + col*max_rows + row] = sum;
}

__global__ void errorInHMatrix(int num_segments, int maxSegmentSize, int num_ops, int max_rows, int max_cols, H2Opus_Real* expandedMatrix, H2Opus_Real* d_denseMatrix, int* activeTiles, H2Opus_Real* d_error, H2Opus_Real* d_tmp){
    int col = threadIdx.x + blockIdx.x*(max_cols/2);
    int row = threadIdx.y + (blockIdx.y%2)*(max_rows/2);
    int block = blockIdx.y/2;

    int MOIndex = activeTiles[block*4]/4;
    int i = morton1(MOIndex);
    int j = morton1(MOIndex >> 1);

    H2Opus_Real x = d_denseMatrix[(col + i*max_cols)*num_segments*maxSegmentSize + j*max_rows + row];
    H2Opus_Real y = expandedMatrix[block*max_rows*max_cols + col*max_rows + row];
    atomicAdd(d_tmp, x*x);
    atomicAdd(d_error, (x-y)*(x-y));
}

#endif
