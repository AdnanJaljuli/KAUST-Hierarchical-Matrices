#include "kdtree/tlr_example.h"
#include "kdtree/TLR_Matrix.cuh"
#include "kdtree/helperFunctions.h"
#include "kdtree/helperKernels.cuh"
#include "kdtree/SVD.cuh"
#include "kdtree/config.h"

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include "cublas_v2.h"
#include <iostream>
#include <utility>
#include <time.h>
#include <assert.h>
#include <typeinfo>

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include "kblas.h"

#include "batch_rand.h"
#include "batch_pstrf.h"
#include "batch_block_copy.h"
#include "batch_ara.h"
#include "helperKernels.cuh"
#include "magma_auxiliary.h"

#define BLOCK_SIZE 32
#define PRINT_OUTPUT 0
using namespace std;
// typedef float Real;
#define ARA_TOLERANCE 1e-5

// typedef Real*		RealArray;
typedef int*		IntArray;
// typedef Real**		RealPtrArray;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char** argv)
{
    float SVDTotalTime = 0, ARATotalTime = 0;
    cudaEvent_t startCode, stopCode;
    cudaEventCreate(&startCode);
    cudaEventCreate(&stopCode);
    cudaEventRecord(startCode);

    Config config = parseArgs(argc,argv);
    printf("n: %d\n", config.n);
    printf("bucket size: %d\n", config.bucket_size);
    printf("epsilon: %f\n", config.epsilon);
    printf("dim: %d\n", config.dim);

    float* timer_arr = (float*)malloc(numTimers*sizeof(float));
    timer_arr[0] = (float)config.n;
    timer_arr[1] = (float)config.bucket_size;
    timer_arr[2] = (float)config.dim;
    timer_arr[3] = (float)config.epsilon;

    // Create point cloud
    PointCloud<H2Opus_Real> pt_cloud(config.dim, (size_t)config.n);
    generateGrid<H2Opus_Real>(pt_cloud, config.n);

    #if PRINT_OUTPUT
    printf("created point cloud\n");
    for(int i=0; i<config.n; ++i){
        for(int j=0; j<config.dim;++j){
            printf("%lf ", pt_cloud.pts[j][i]);
        }
        printf("\n");
    }
    printf("\n\n");
    #endif

    H2Opus_Real *dataset;
    dataset = (H2Opus_Real*)malloc(config.n*config.dim*(uint64_t)sizeof(H2Opus_Real));
    assert(dataset != NULL);

    // TODO: move this to a kernel
    for (unsigned long long i = 0; i < config.dim; ++i){
        for(unsigned long long j = 0; j < config.n; ++j){
            dataset[i*config.n+j] = pt_cloud.getDataPoint((size_t)j, (int)i);
        }
    }

    H2Opus_Real *d_dataset;
    cudaError_t cudaErr = cudaMalloc((void**) &d_dataset, config.n*config.dim*(uint64_t)sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMemcpy(d_dataset, dataset, config.n*config.dim*(uint64_t)sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    free(dataset);

    uint64_t num_segments = 1;
    uint64_t num_segments_reduce = num_segments*config.dim;
    uint64_t segment_size = upper_power_of_two(config.n);

    int *d_offsets_sort;         // e.g., [0, 3, 3, 7]
    int *d_offsets_reduce;
    H2Opus_Real *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
    H2Opus_Real *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
    int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
    int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
    int *d_curr_dim;
    H2Opus_Real *d_reduce_in;
    H2Opus_Real *d_reduce_min_out;
    H2Opus_Real *d_reduce_max_out;
    int *d_temp;
    H2Opus_Real *d_span;
    int* d_span_offsets;
    cub::KeyValuePair<int, H2Opus_Real> *d_span_reduce_out;

    bool workDone= false;
    bool* d_workDone;
    uint64_t* d_bit_vector;
    short int* d_popc_bit_vector;
    short int* d_popc_scan;
    unsigned int* d_new_num_segments;
    unsigned int* new_num_segments;
    int max_num_segments;

    int* A;
    int* B;
    int* d_bin_search_output;
    int* d_thrust_v_bin_search_output;
    int* d_input_search;
    int* d_aux_offsets_sort;

    if(config.div_method != POWER_OF_TWO_ON_LEFT){
        max_num_segments = 1<<(getMaxSegmentSize(config.n, config.bucket_size).second);
    } else {
        max_num_segments = (config.n+config.bucket_size-1)/config.bucket_size;
    }
    printf("max num segments: %d\n", max_num_segments);

    unsigned int largest_segment_size = config.n;

    // TODO: fix memory allocated
    cudaErr = cudaMalloc((void**) &d_offsets_sort, (max_num_segments + 1)*sizeof(int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_offsets_reduce, (long long)((max_num_segments*config.dim + 1)*(long long)sizeof(int)));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_keys_in, config.n*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_keys_out, config.n*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_values_in, config.n*sizeof(int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_values_out, config.n*sizeof(int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_curr_dim, (max_num_segments + 1)*sizeof(int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_reduce_in, (long long)config.n*(long long)config.dim*(long long)sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_reduce_min_out, (long long)((max_num_segments+1)*config.dim)*(long long)sizeof(int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_reduce_max_out, (long long)((max_num_segments+1)*config.dim)*(long long)sizeof(int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_span, (long long)((max_num_segments+1)*config.dim)*(long long)sizeof(int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_span_offsets, (max_num_segments + 1)*sizeof(int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_span_reduce_out, (max_num_segments+1)*sizeof(cub::KeyValuePair<int, H2Opus_Real>));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    if(config.div_method == DIVIDE_IN_HALF){
        cudaErr = cudaMalloc((void**) &d_bit_vector, (((max_num_segments+1) + sizeof(uint64_t)*8-1)/(sizeof(uint64_t)*8)) *sizeof(uint64_t));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaErr = cudaMalloc((void**) &d_popc_bit_vector, (((max_num_segments + 1) + sizeof(uint64_t)*8-1)/(sizeof(uint64_t)*8))*sizeof(short int));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaErr = cudaMalloc((void**) &d_popc_scan, (((max_num_segments + 1) + sizeof(uint64_t)*8-1)/(sizeof(uint64_t)*8))*sizeof(short int));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaErr = cudaMalloc((void**) &d_new_num_segments, sizeof(unsigned int));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaErr = cudaMalloc((void**) &d_workDone, sizeof(bool));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        new_num_segments = (unsigned int*)malloc(sizeof(unsigned int));
    }

    if(config.div_method != POWER_OF_TWO_ON_LEFT){
        cudaErr = cudaMalloc((void**) &d_aux_offsets_sort, (max_num_segments + 1) * sizeof(int));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaErr = cudaMalloc((void**) &A, (max_num_segments + 1)*sizeof(int));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaErr = cudaMalloc((void**) &B, config.n*sizeof(int));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaErr = cudaMalloc((void**) &d_bin_search_output, config.n*sizeof(int));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaErr = cudaMalloc((void**) &d_input_search, config.n*sizeof(int));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    }

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (config.n+numThreadsPerBlock-1)/numThreadsPerBlock;

    if(config.div_method != POWER_OF_TWO_ON_LEFT){
        initializeArrays<<<numBlocks, numThreadsPerBlock>>>(config.n, config.dim, d_values_in, d_curr_dim, d_offsets_sort, d_offsets_reduce, d_input_search, max_num_segments);
    } else {
        initializeArrays<<<numBlocks, numThreadsPerBlock>>>(config.n, d_values_in, d_curr_dim, max_num_segments);
    }
    cudaDeviceSynchronize();

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cudaEvent_t startKDtree, stopKDtree;
    cudaEventCreate(&startKDtree);
    cudaEventCreate(&stopKDtree);
    cudaEventRecord(startKDtree);

    unsigned int iteration = 0;

    // TODO: fix the while loop if statement
    #if 0
    if(config.div_method == DIVIDE_IN_HALF){
        while(!workDone)
    } else if (config.div_method == POWER_OF_TWO_ON_LEFT) {
        while(segment_size > config.bucket_size)
    } else {
        while(largest_segment_size > config.bucket_size)
    }
    #endif

    while(segment_size > config.bucket_size)
    {
        numThreadsPerBlock = 1024;
        numBlocks = (num_segments+1+numThreadsPerBlock-1)/numThreadsPerBlock;
        if(config.div_method == POWER_OF_TWO_ON_LEFT){
            fillOffsets<<<numBlocks, numThreadsPerBlock>>>(config.n, config.dim, num_segments, segment_size, d_offsets_sort, d_offsets_reduce);
        }
        cudaDeviceSynchronize();

        numThreadsPerBlock = 1024;
        numBlocks = (long long)((long long)config.n*(long long)config.dim + numThreadsPerBlock-1)/numThreadsPerBlock;

        fillReductionArray<<<numBlocks, numThreadsPerBlock>>> (config.n, config.dim, d_dataset, d_values_in, d_reduce_in);
        cudaDeviceSynchronize();

        d_temp_storage = NULL;
        temp_storage_bytes = 0;

        cub::DeviceSegmentedReduce::Min(d_temp_storage, temp_storage_bytes, d_reduce_in, d_reduce_min_out,
            num_segments_reduce, d_offsets_reduce, d_offsets_reduce + 1);

        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceSegmentedReduce::Min(d_temp_storage, temp_storage_bytes, d_reduce_in, d_reduce_min_out,
            num_segments_reduce, d_offsets_reduce, d_offsets_reduce + 1);
        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);

        d_temp_storage = NULL;
        temp_storage_bytes = 0;

        cub::DeviceSegmentedReduce::Max(d_temp_storage, temp_storage_bytes, d_reduce_in, d_reduce_max_out,
            num_segments_reduce, d_offsets_reduce, d_offsets_reduce + 1);

        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceSegmentedReduce::Max(d_temp_storage, temp_storage_bytes, d_reduce_in, d_reduce_max_out,
            num_segments_reduce, d_offsets_reduce, d_offsets_reduce + 1);
        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);

        numThreadsPerBlock = 1024;
        numBlocks = (num_segments+numThreadsPerBlock-1)/numThreadsPerBlock;

        findSpan<<<numBlocks, numThreadsPerBlock>>> (config.n, config.dim, num_segments, d_reduce_min_out, d_reduce_max_out, d_span, d_span_offsets);
        cudaDeviceSynchronize();

        d_temp_storage = NULL;
        temp_storage_bytes = 0;

        cub::DeviceSegmentedReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_span, d_span_reduce_out,
            num_segments, d_span_offsets, d_span_offsets + 1);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run argmax-reduction
        cub::DeviceSegmentedReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_span, d_span_reduce_out,
            num_segments, d_span_offsets, d_span_offsets + 1);
        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);

        numThreadsPerBlock = 1024;
        numBlocks = (num_segments+numThreadsPerBlock-1)/numThreadsPerBlock;

        if(config.div_method == DIVIDE_IN_HALF){
            fillCurrDim<<<numBlocks, numThreadsPerBlock>>> (config.n, num_segments, d_curr_dim, d_span_reduce_out, d_bit_vector);
        } else {
            fillCurrDim<<<numBlocks, numThreadsPerBlock>>> (config.n, num_segments, d_curr_dim, d_span_reduce_out);
        }
        cudaDeviceSynchronize();

        // fill keys_in array
        numThreadsPerBlock = 1024;
        numBlocks = (config.n+numThreadsPerBlock-1)/numThreadsPerBlock;

        if(config.div_method != POWER_OF_TWO_ON_LEFT){
            thrust::device_ptr<int> A = thrust::device_pointer_cast((int *)d_offsets_sort), B = thrust::device_pointer_cast((int *)d_input_search);
            thrust::device_vector<int> d_bin_search_output(config.n);
            thrust::upper_bound(A, A + num_segments + 1, B, B + config.n, d_bin_search_output.begin(), thrust::less<int>());
            d_thrust_v_bin_search_output = thrust::raw_pointer_cast(&d_bin_search_output[0]);
            fillKeysIn<<<numBlocks, numThreadsPerBlock>>> (config.n, d_keys_in, d_curr_dim, d_values_in, d_dataset, d_offsets_sort, d_thrust_v_bin_search_output);
        } else {
            fillKeysIn<<<numBlocks, numThreadsPerBlock>>> (config.n, segment_size, d_keys_in, d_curr_dim, d_values_in, d_dataset);
        }
        cudaDeviceSynchronize();

        d_temp_storage = NULL;
        temp_storage_bytes = 0;

        cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out,
        config.n, num_segments, d_offsets_sort, d_offsets_sort + 1);

        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out,
        config.n, num_segments, d_offsets_sort, d_offsets_sort + 1);
        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);

        d_temp = d_values_in;
        d_values_in = d_values_out;
        d_values_out = d_temp;
        ++iteration;

        // TODO: fix cuda-memcheck bug in divide_in_half
        if(config.div_method == DIVIDE_IN_HALF){
            numThreadsPerBlock = 1024;
            numBlocks = (num_segments+numThreadsPerBlock-1)/numThreadsPerBlock;
            fillBitVector<<<numBlocks, numThreadsPerBlock>>>(num_segments, d_bit_vector, d_offsets_sort, config.bucket_size);
            cudaDeviceSynchronize();

            unsigned int num_threads = (num_segments + sizeof(uint64_t)*8 - 1)/(sizeof(uint64_t)*8);
            numThreadsPerBlock = 1024;
            numBlocks = (num_threads + numThreadsPerBlock-1)/numThreadsPerBlock;
            fillPopCount<<<numBlocks, numThreadsPerBlock>>>(num_threads, d_bit_vector, d_popc_bit_vector);
            cudaDeviceSynchronize();

            d_temp_storage = NULL;
            temp_storage_bytes = 0;
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_popc_bit_vector, d_popc_scan, num_segments);
            // Allocate temporary storage
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            // Run exclusive prefix sum
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_popc_bit_vector, d_popc_scan, num_segments);
            cudaFree(d_temp_storage);

            numThreadsPerBlock = 1024;
            numBlocks = (num_segments+numThreadsPerBlock-1)/numThreadsPerBlock;
            fillOffsetsSort<<<numBlocks, numThreadsPerBlock>>>(config.n, config.dim, num_segments, d_offsets_sort, d_aux_offsets_sort, d_bit_vector, d_popc_scan, d_new_num_segments, d_workDone, config.bucket_size);
            cudaDeviceSynchronize();
            cudaMemcpy(new_num_segments, d_new_num_segments, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&workDone, d_workDone, sizeof(bool), cudaMemcpyDeviceToHost);
            num_segments = *new_num_segments;
            d_temp = d_aux_offsets_sort;
            d_aux_offsets_sort = d_offsets_sort;
            d_offsets_sort = d_temp;

            if(workDone){
                break;
            }

            numThreadsPerBlock = 1024;
            numBlocks = (num_segments+numThreadsPerBlock-1)/numThreadsPerBlock;
            fillOffsetsReduce<<<numBlocks, numThreadsPerBlock>>> (config.n, config.dim, num_segments, d_offsets_sort, d_offsets_reduce);
            cudaDeviceSynchronize();

        } else if(config.div_method == POWER_OF_TWO_ON_LEFT){
            segment_size /= 2;
            num_segments = (config.n+segment_size-1)/segment_size;

        } else if(config.div_method == FULL_TREE){
            numThreadsPerBlock = 1024;
            numBlocks = (num_segments+numThreadsPerBlock-1)/numThreadsPerBlock;
            fillOffsetsSort<<<numBlocks, numThreadsPerBlock>>> (config.n, config.dim, num_segments, d_offsets_sort, d_aux_offsets_sort);
            cudaDeviceSynchronize();

            d_temp = d_aux_offsets_sort;
            d_aux_offsets_sort = d_offsets_sort;
            d_offsets_sort = d_temp;
            num_segments *= 2;

            numThreadsPerBlock = 1024;
            numBlocks = (num_segments+numThreadsPerBlock-1)/numThreadsPerBlock;
            fillOffsetsReduce<<<numBlocks, numThreadsPerBlock>>> (config.n, config.dim, num_segments, d_offsets_sort, d_offsets_reduce);
            cudaDeviceSynchronize();

            ++largest_segment_size;
            largest_segment_size /= 2;
        }

        num_segments_reduce = num_segments*config.dim;
    }

    cudaEventRecord(stopKDtree);
    cudaEventSynchronize(stopKDtree);
    float KDtree_time = 0;
    cudaEventElapsedTime(&KDtree_time, startKDtree, stopKDtree);
    timer_arr[4] = KDtree_time;
    cudaEventDestroy(startKDtree);
    cudaEventDestroy(stopKDtree);

    #if 0
    int *index_map = (int*)malloc(config.n*sizeof(int));
    cudaMemcpy(index_map, d_values_in, config.n*sizeof(int), cudaMemcpyDeviceToHost);
    
    FILE *fp;
    fp = fopen("results/pointcloud.csv", "a");// "w" means that we are going to write on this file
    printf("bucket size: %d, n: %d, num segments: %d,\n", config.bucket_size, config.n, num_segments);
    for(int i=0; i<config.n; ++i){
        for(int j=0; j<config.dim; ++j){
            printf("%lf, ", pt_cloud.pts[j][index_map[i]]);
        }
        printf("\n");
    }
    fclose(fp); //Don't forget to close the file when finished    
    #endif

    #if 0
    printf("index map\n");
    for(int i=0; i<config.n; ++i){
        printf("%d ", index_map[i]);
    }
    printf("\n");
    free(index_map);
    #endif

    if(config.div_method == POWER_OF_TWO_ON_LEFT){
        printf("num segments :%d\n", num_segments);
        printf("segment size :%d\n", segment_size);
        fillOffsets<<<numBlocks, numThreadsPerBlock>>>(config.n, config.dim, num_segments, segment_size, d_offsets_sort, d_offsets_reduce);
        cudaDeviceSynchronize();
    }

    cudaFree(d_offsets_reduce);
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_values_out);
    cudaFree(d_curr_dim);
    cudaFree(d_reduce_in);
    cudaFree(d_reduce_min_out);
    cudaFree(d_reduce_max_out);
    cudaFree(d_span_reduce_out);
    cudaFree(d_span);
    cudaFree(d_span_offsets);
    if(config.div_method != POWER_OF_TWO_ON_LEFT){
        cudaFree(d_aux_offsets_sort);
        cudaFree(A);
        cudaFree(B);
        cudaFree(d_bin_search_output);
        cudaFree(d_input_search);
    }
    if(config.div_method == DIVIDE_IN_HALF){
        free(new_num_segments);
        cudaFree(d_bit_vector);
        cudaFree(d_popc_bit_vector);
        cudaFree(d_popc_scan);
        cudaFree(d_new_num_segments);
        cudaFree(d_workDone);
    }

    uint64_t maxSegmentSize;
    if(config.div_method != POWER_OF_TWO_ON_LEFT){
        maxSegmentSize = getMaxSegmentSize(config.n, config.bucket_size).first;
    } else {
        maxSegmentSize = config.bucket_size;
    }
    printf("max segment size: %lu\n", maxSegmentSize);

    H2Opus_Real* d_input_matrix_segmented;
    H2Opus_Real* input_matrix_segmented = (H2Opus_Real*)malloc(maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real));
    printf("mem allocated to input matrix: %lu\n", maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real));
    cudaErr = cudaMalloc((void**) &d_input_matrix_segmented, maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    H2Opus_Real *h_S = (H2Opus_Real *)malloc(maxSegmentSize * num_segments * sizeof(H2Opus_Real));
    H2Opus_Real *h_U = (H2Opus_Real *)malloc(maxSegmentSize * maxSegmentSize * num_segments * sizeof(H2Opus_Real));
    H2Opus_Real *h_V = (H2Opus_Real *)malloc(maxSegmentSize * maxSegmentSize * num_segments * sizeof(H2Opus_Real));

    H2Opus_Real* d_S;
    H2Opus_Real* d_U;
    H2Opus_Real* d_V;
    cudaErr = cudaMalloc((void**) &d_S, maxSegmentSize * num_segments * sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_U, maxSegmentSize * maxSegmentSize * num_segments * sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_V, maxSegmentSize * maxSegmentSize * num_segments * sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    int* d_scan_K_segmented;
    cudaErr = cudaMalloc((void**) &d_scan_K_segmented, num_segments*sizeof(int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    H2Opus_Real** d_U_tiled_temp = (H2Opus_Real**)malloc(num_segments*sizeof(H2Opus_Real*));
    H2Opus_Real** d_V_tiled_temp = (H2Opus_Real**)malloc(num_segments*sizeof(H2Opus_Real*));

    TLR_Matrix matrix;
    cudaErr = cudaMalloc((void**) &matrix.blockRanks, num_segments*num_segments*sizeof(unsigned int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &matrix.diagonal, num_segments*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    cudaEvent_t startGenerateInputMatrix, stopGenerateInputMatrix;
    cudaEventCreate(&startGenerateInputMatrix);
    cudaEventCreate(&stopGenerateInputMatrix);
    cudaEventRecord(startGenerateInputMatrix);
    cudaErr = cudaGetLastError(); if(cudaErr != cudaSuccess) { printf("CUDA error -3: %s\n", cudaGetErrorString(cudaErr)); exit(-1); }
    
    magma_init();

    float tol = 1e-5;
    const int ARA_R = 10;
    const int batchCount = num_segments;
    const int max_rows = 32;
    const int max_cols = 32;
    int max_rank = max_cols;

    int *d_rows_batch, *d_cols_batch, *d_ranks;
    int *d_ldm_batch, *d_lda_batch, *d_ldb_batch;
    H2Opus_Real *d_A, *d_B;
    H2Opus_Real** d_M_ptrs, **d_A_ptrs, **d_B_ptrs;

    cudaMalloc((void**) &d_rows_batch, batchCount*sizeof(int));
    cudaMalloc((void**) &d_cols_batch, batchCount*sizeof(int));
    cudaMalloc((void**) &d_ranks, batchCount*sizeof(int));
    cudaMalloc((void**) &d_ldm_batch, batchCount*sizeof(int));
    cudaMalloc((void**) &d_lda_batch, batchCount*sizeof(int));
    cudaMalloc((void**) &d_ldb_batch, batchCount*sizeof(int));
    cudaMalloc((void**) &d_A, batchCount*max_rows*max_rank*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_B, batchCount*max_rows*max_rank*sizeof(H2Opus_Real));

    cudaMalloc((void**) &d_M_ptrs, batchCount*sizeof(H2Opus_Real*));
    cudaMalloc((void**) &d_A_ptrs, batchCount*sizeof(H2Opus_Real*));
    cudaMalloc((void**) &d_B_ptrs, batchCount*sizeof(H2Opus_Real*));
    cudaErr = cudaGetLastError(); if(cudaErr != cudaSuccess) { printf("CUDA error -1: %s\n", cudaGetErrorString(cudaErr)); exit(-1); }

    numThreadsPerBlock = 1024;
    numBlocks = (batchCount + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillARAArrays<<<numBlocks, numThreadsPerBlock>>>(batchCount, max_rows, max_cols, d_rows_batch, d_cols_batch, d_M_ptrs, d_input_matrix_segmented, d_ldm_batch, d_lda_batch, d_ldb_batch);
    cudaDeviceSynchronize();
    cudaErr = cudaGetLastError(); if(cudaErr != cudaSuccess) { printf("CUDA error 0: %s\n", cudaGetErrorString(cudaErr)); exit(-1); }

    
    kblasHandle_t kblas_handle;
    kblasRandState_t rand_state;
    kblasCreate(&kblas_handle);
    cudaDeviceSynchronize();
    cudaErr = cudaGetLastError(); if(cudaErr != cudaSuccess) { printf("CUDA error here: %s\n", cudaGetErrorString(cudaErr)); exit(-1); }

    int errormsg = kblasInitRandState(kblas_handle, &rand_state, 1<<15, 0);
    // printf("error: %d: %s\n", errormsg, kblasGetErrorString(errormsg));
    cudaErr = cudaGetLastError(); if(cudaErr != cudaSuccess) { printf("CUDA error here: %s\n", cudaGetErrorString(cudaErr)); exit(-1); }

    kblasEnableMagma(kblas_handle);

    kblas_gemm_batch_strided_wsquery(kblas_handle, batchCount);
    kblas_gesvj_batch_wsquery<H2Opus_Real>(kblas_handle, max_rows, max_cols, batchCount);
    kblas_ara_batch_wsquery<H2Opus_Real>(kblas_handle, BLOCK_SIZE, batchCount);
    kblas_rsvd_batch_wsquery<H2Opus_Real>(kblas_handle, max_rows, max_cols, 64, batchCount);
    cudaErr = cudaGetLastError(); if(cudaErr != cudaSuccess) { printf("CUDA error here: %s\n", cudaGetErrorString(cudaErr)); exit(-1); }

    errormsg = kblasAllocateWorkspace(kblas_handle);
    // printf("error: %d: %s\n", errormsg, kblasGetErrorString(errormsg));
    cudaErr = cudaGetLastError(); if(cudaErr != cudaSuccess) { printf("CUDA error here: %s\n", cudaGetErrorString(cudaErr)); exit(-1); }


    for(unsigned int segment = 0; segment < num_segments; ++segment){

        // printf("for loop\n");
        dim3 m_numThreadsPerBlock(upper_power_of_two(maxSegmentSize), upper_power_of_two(maxSegmentSize));
        dim3 m_numBlocks(1, num_segments);

        generateInputMatrix<<<m_numBlocks, m_numThreadsPerBlock>>>(config.n, num_segments, maxSegmentSize, config.dim, d_values_in, d_input_matrix_segmented, d_dataset, d_offsets_sort, segment, matrix.diagonal);
        cudaDeviceSynchronize();
        cudaErr = cudaGetLastError(); if(cudaErr != cudaSuccess) { printf("CUDA error -2: %s\n", cudaGetErrorString(cudaErr)); exit(-1); }

        cudaEvent_t startSVD, stopSVD;
        cudaEventCreate(&startSVD);
        cudaEventCreate(&stopSVD);
        cudaEventRecord(startSVD);
        SVD(config.n, num_segments, input_matrix_segmented, maxSegmentSize, h_S, h_U, h_V);
        cudaEventRecord(stopSVD);
        cudaEventSynchronize(stopSVD);
        float SVD_time = 0;
        cudaEventElapsedTime(&SVD_time, startSVD, stopSVD);
        // printf("SVD time: %f\n", SVD_time);
        SVDTotalTime += SVD_time;
        cudaEventDestroy(startSVD);
        cudaEventDestroy(stopSVD);
        cudaDeviceSynchronize();

        generateArrayOfPointersT<H2Opus_Real>(d_input_matrix_segmented, d_M_ptrs, max_rows*max_cols, batchCount, 0);
        generateArrayOfPointersT<H2Opus_Real>(d_A, d_A_ptrs, max_rows*max_cols, batchCount, 0);
        cudaErr = cudaGetLastError(); if(cudaErr != cudaSuccess) { printf("CUDA error 1: %s\n", cudaGetErrorString(cudaErr)); exit(-1); }
        generateArrayOfPointersT<H2Opus_Real>(d_B, d_B_ptrs, max_rows*max_cols, batchCount, 0);
        cudaErr = cudaGetLastError(); if(cudaErr != cudaSuccess) { printf("CUDA error 1: %s\n", cudaGetErrorString(cudaErr)); exit(-1); }

        cudaSetDevice(0);
        cudaErr = cudaGetLastError(); if(cudaErr != cudaSuccess) { printf("CUDA error 2: %s\n", cudaGetErrorString(cudaErr)); exit(-1); }
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        cudaErr = cudaGetLastError(); if(cudaErr != cudaSuccess) { printf("CUDA error 3: %s\n", cudaGetErrorString(cudaErr)); exit(-1); }

        cudaDeviceSynchronize();
        cudaEvent_t startARA, stopARA;
        cudaEventCreate(&startARA);
        cudaEventCreate(&stopARA);
        cudaEventRecord(startARA);
        errormsg = kblas_ara_batch(
                            kblas_handle, d_rows_batch, d_cols_batch, d_M_ptrs, d_ldm_batch, 
                            d_A_ptrs, d_lda_batch, d_B_ptrs, d_ldb_batch, d_ranks, 
                            tol, max_rows, max_cols, max_rank, BLOCK_SIZE, ARA_R, rand_state, 0, batchCount
                        );
        cudaEventRecord(stopARA);
        cudaEventSynchronize(stopARA);
        float ARA_time = 0;
        cudaEventElapsedTime(&ARA_time, startARA, stopARA);
        // printf("ARA time: %f\n", ARA_time);
        ARATotalTime += ARA_time;
        cudaEventDestroy(startARA);
        cudaEventDestroy(stopARA);
        cudaDeviceSynchronize();

        // printf("error: %d: %s\n", errormsg, kblasGetErrorString(errormsg));

        // printf("ended for loop\n");
        // printOutput<<<1, 1>>>(d_A, d_B, d_ranks, batchCount);
        // cudaDeviceSynchronize();

    }
    cudaFree(d_rows_batch);
    cudaFree(d_cols_batch);
    cudaFree(d_ldm_batch);
    cudaFree(d_lda_batch);
    cudaFree(d_ldb_batch);
    cudaFree(d_M_ptrs);
    cudaFree(d_A_ptrs);
    cudaFree(d_B_ptrs);
    cudaFree(d_ranks);
    cudaFree(d_A);
    cudaFree(d_B);
    magma_finalize();
    printf("\nARA time %f\n SVD time %f\n", ARATotalTime, SVDTotalTime);

    return 0;
}
