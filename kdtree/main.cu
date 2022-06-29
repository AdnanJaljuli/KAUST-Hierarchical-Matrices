#include "tlr_example.h"
#include "TLR_Matrix.cuh"
#include "helperFunctions.h"
#include "helperKernels.cuh"
#include "SVD.cuh"
#include "config.h"

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include "cublas_v2.h"
#include <iostream>
#include <utility>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <typeinfo>

#define BLOCK_SIZE 32
#define PRINT_OUTPUT 0
using namespace std;

// TODO: create a struct for the tiled matrix that has u_tiled, v_tiled, k, k_scan
// TODO: generate pointcloud and copy values of the pointcloud to ptr on GPU
// TODO: fix makefile so main.cu depends on helperKerlens.cuh
// TODO: make sure that everything that is malloced is freed

int main(int argc, char *argv[]){
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
    cudaErr = cudaMemcpy(d_dataset, dataset, config.n*config.dim*(uint64_t)sizeof(H2Opus_Real*), cudaMemcpyHostToDevice);
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
    fprintf(fp, "bucket size: %d, n: %d, num segments: %d,\n", config.bucket_size, config.n, num_segments);
    for(int i=0; i<config.n; ++i){
        for(int j=0; j<config.dim; ++j){
            fprintf(fp, "%lf, ", pt_cloud.pts[j][index_map[i]]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp); //Don't forget to close the file when finished    
    free(index_map);
    #endif

    #if 0
    printf("index max\n");
    for(int i=0; i<config.n; ++i){
        printf("%d ", index_map[i]);
    }
    printf("\n");
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

    int k_sum = 0;

    cudaEvent_t startGenerateInputMatrix, stopGenerateInputMatrix;
    cudaEventCreate(&startGenerateInputMatrix);
    cudaEventCreate(&stopGenerateInputMatrix);
    cudaEventRecord(startGenerateInputMatrix);

    for(unsigned int segment = 0; segment < num_segments; ++segment){
        dim3 m_numThreadsPerBlock(upper_power_of_two(maxSegmentSize), upper_power_of_two(maxSegmentSize));
        dim3 m_numBlocks(1, num_segments);

        generateInputMatrix<<<m_numBlocks, m_numThreadsPerBlock>>>(config.n, num_segments, maxSegmentSize, config.dim, d_values_in, d_input_matrix_segmented, d_dataset, d_offsets_sort, segment, matrix.diagonal);
        cudaDeviceSynchronize();

        cudaMemcpy(input_matrix_segmented, d_input_matrix_segmented, maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);

        #if PRINT_OUTPUT
        printf("input matrix\n");
        for(unsigned int i=0; i<num_segments*maxSegmentSize*maxSegmentSize; ++i){
            printf("%lf ", input_matrix_segmented[i]);
        }
        #endif

        cudaEvent_t startSVD, stopSVD;
        cudaEventCreate(&startSVD);
        cudaEventCreate(&stopSVD);
        cudaEventRecord(startSVD);
        SVD(config.n, num_segments, input_matrix_segmented, maxSegmentSize, h_S, h_U, h_V);
        cudaEventRecord(stopSVD);
        cudaEventSynchronize(stopSVD);
        cudaEventDestroy(startSVD);
        cudaEventDestroy(stopSVD);
        cudaDeviceSynchronize();

        cudaMemcpy(d_S, h_S, maxSegmentSize * num_segments * sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_U, h_U, maxSegmentSize*maxSegmentSize*num_segments * sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V, maxSegmentSize*maxSegmentSize*num_segments * sizeof(H2Opus_Real), cudaMemcpyHostToDevice);

        // ----------------------------ARA----------------------------
        #if 0
        kblasHandle_t kblas_handle;
        kblasRandState_t rand_state;
        kblasCreate(&kblas_handle);
		kblasInitRandState(kblas_handle, &rand_state, 16384*2, 0);
		kblasEnableMagma(kblas_handle);
        kblas_ara_batch_wsquery<H2Opus_Real>(kblas_handle, BLOCK_SIZE, num_segments);
        check_kblas_error( kblasAllocateWorkspace(kblas_handle) );

        int* d_rows_batch, d_cols_batch;
        cudaMalloc((void**) &d_rows_batch, num_segments*sizeof(int));
        cudaMalloc((void**) &d_cols_batch, num_segments*sizeof(int));

        numThreadsPerBlock = 1024;
        numBlocks = (num_segments + numThreadsPerBlock - 1)/numThreadsPerBlock;
        fillBatch<<<numBlocks, numThreadsPerBlock>>>(num_segments, d_rows_batch, d_cols_batch);
        cudaDeviceSynchronize();

        cudaEvent_t startARA, stopARA;
        cudaEventCreate(&startARA);
        cudaEventCreate(&stopARA);
        cudaEventRecord(startARA);
        kblas_ara_batch( kblas_handles[g], d_rows_batch[g], d_cols_batch[g], d_M_ptrs[g], d_ldm_batch[g], 
								d_A_ptrs[g], d_lda_batch[g], d_B_ptrs[g], d_ldb_batch[g], d_ranks[g], 
								tol, max_rows, max_cols, max_rank, BLOCK_SIZE, ARA_R, rand_state[g], 0, batchCount_gpu);
        cudaEventRecord(stopARA);
        cudaEventSynchronize(stopARA);
        cudaEventDestroy(startARA);
        cudaEventDestroy(stopARA);
        cudaDeviceSynchronize();

        cudaFree(d_rows_batch);
        cudaFree(d_cols_batch);
        #endif

        #if 0
        printSigmas(h_S, num_segments, maxSegmentSize, config.bucket_size, config.n, segment);  
        #endif

        numThreadsPerBlock = maxSegmentSize; //TODO: make sure that bucket_size is less than 1024
        numBlocks = num_segments;
        calcMemNeeded<<<numBlocks, numThreadsPerBlock>>> (maxSegmentSize, matrix.blockRanks + segment*num_segments, d_S, config.epsilon);
        cudaDeviceSynchronize();

        #if 0
        unsigned int *h_k = (unsigned int *)malloc(num_segments*sizeof(unsigned int));
        cudaMemcpy(h_k, matrix.blockRanks + segment*num_segments, num_segments*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        printKs(h_k, num_segments, maxSegmentSize, config.bucket_size, config.n, segment, config.epsilon);
        free(h_k);
        #endif

        d_temp_storage = NULL;
        temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, matrix.blockRanks + segment*num_segments, d_scan_K_segmented, num_segments);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, matrix.blockRanks + segment*num_segments, d_scan_K_segmented, num_segments);
        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);

        int* totalMem = (int*)malloc(sizeof(int));
        int* d_totalMem;
        cudaMalloc((void**) &d_totalMem, sizeof(int));
        getTotalMem<<<1, 1>>> (d_totalMem, matrix.blockRanks + segment*num_segments, d_scan_K_segmented, num_segments);
        cudaDeviceSynchronize();

        cudaErr = cudaMemcpy(totalMem, d_totalMem, sizeof(int), cudaMemcpyDeviceToHost);
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaFree(d_totalMem);

        #if 0
        printf("max mem: %d\n", maxSegmentSize*num_segments);
        printf("total mem: %d\n", (*totalMem));
        #endif

        H2Opus_Real* d_U_tiled_segmented;
        cudaErr = cudaMalloc((void**) &d_U_tiled_segmented, maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        H2Opus_Real* d_V_tiled_segmented;
        cudaErr = cudaMalloc((void**) &d_V_tiled_segmented, maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

        dim3 d_numThreadsPerBlock(upper_power_of_two(maxSegmentSize), upper_power_of_two(maxSegmentSize));
        dim3 d_numBlocks(1, num_segments);
        tileMatrix<<<d_numBlocks, d_numThreadsPerBlock>>> (config.n, num_segments, maxSegmentSize, d_S, d_U, d_V, d_U_tiled_segmented, d_V_tiled_segmented, matrix.blockRanks + segment*num_segments, d_scan_K_segmented, segment);
        cudaDeviceSynchronize();

        #if 0
        H2Opus_Real* d_expMatrix;
        cudaErr = cudaMalloc((void**) &d_expMatrix, num_segments*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

        expandMatrix<<<d_numBlocks, d_numThreadsPerBlock>>> (num_segments, maxSegmentSize, matrix.blockRanks + segment*num_segments, d_scan_K_segmented, d_U_tiled_segmented, d_V_tiled_segmented, d_expMatrix);
        cudaDeviceSynchronize();

        // #if 0
        // printExpM<<<1, 1>>> (num_segments, maxSegmentSize, d_expMatrix, d_input_matrix_segmented);
        // cudaDeviceSynchronize();
        // #endif

        H2Opus_Real* d_error;
        H2Opus_Real* error = (H2Opus_Real*) malloc(sizeof(H2Opus_Real));
        cudaMalloc((void**) &d_error, sizeof(H2Opus_Real));

        H2Opus_Real* d_tmp;
        H2Opus_Real* tmp = (H2Opus_Real*) malloc(sizeof(H2Opus_Real));
        cudaMalloc((void**) &d_tmp, sizeof(H2Opus_Real));

        *error = 0;
        *tmp = 0;
        cudaMemcpy(d_error, error, sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tmp, tmp, sizeof(H2Opus_Real), cudaMemcpyHostToDevice);

        numThreadsPerBlock = 1024;
        numBlocks = (num_segments*maxSegmentSize*maxSegmentSize + numThreadsPerBlock-1)/numThreadsPerBlock;
        calcError<<<numBlocks, numThreadsPerBlock>>> (num_segments, maxSegmentSize, d_expMatrix, d_input_matrix_segmented, d_error, d_tmp);
        cudaDeviceSynchronize();
        cudaMemcpy(error, d_error, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
        cudaMemcpy(tmp, d_tmp, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
        cudaFree(d_error);
        cudaFree(d_tmp);
        // printf("error: %lf\n", sqrt(*error)/sqrt(*tmp));
        free(tmp);
        free(error);
        cudaFree(d_expMatrix);
        #endif

        cudaErr = cudaMalloc((void**) &d_U_tiled_temp[segment], maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaErr = cudaMalloc((void**) &d_V_tiled_temp[segment], maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaMemcpy(d_U_tiled_temp[segment], d_U_tiled_segmented, maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_V_tiled_temp[segment], d_V_tiled_segmented, maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);

        k_sum += (*totalMem);

        free(totalMem);
        cudaFree(d_U_tiled_segmented);
        cudaFree(d_V_tiled_segmented);
    }
    printf("total mem %d\n", k_sum);
    timer_arr[5] =k_sum;

    cudaEventRecord(stopGenerateInputMatrix);
    cudaEventSynchronize(stopGenerateInputMatrix);
    float GenMatrix_time = 0;
    cudaEventElapsedTime(&GenMatrix_time, startGenerateInputMatrix, stopGenerateInputMatrix);
    timer_arr[6] = GenMatrix_time;
    cudaEventDestroy(startGenerateInputMatrix);
    cudaEventDestroy(stopGenerateInputMatrix);

    free(h_S);
    free(h_U);
    free(h_V);
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_scan_K_segmented);
    free(input_matrix_segmented);
    cudaFree(d_values_in);
    cudaFree(d_offsets_sort);
    cudaFree(d_dataset);
    cudaFree(d_input_matrix_segmented);
    
    cudaErr = cudaMalloc((void**) &matrix.U, k_sum*maxSegmentSize*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &matrix.V, k_sum*maxSegmentSize*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &matrix.blockOffsets, num_segments*num_segments*sizeof(unsigned int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, matrix.blockRanks, matrix.blockOffsets, num_segments*num_segments);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, matrix.blockRanks, matrix.blockOffsets, num_segments*num_segments);
    cudaDeviceSynchronize();
    cudaFree(d_temp_storage);

    unsigned int* h_scan_K = (unsigned int*)malloc(num_segments*num_segments*sizeof(unsigned int));
    cudaErr = cudaMemcpy(h_scan_K, matrix.blockOffsets, num_segments*num_segments*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    for(unsigned int segment = 0; segment < num_segments-1; ++segment){
        cudaErr = cudaMemcpy(&matrix.U[h_scan_K[num_segments*segment]*maxSegmentSize], d_U_tiled_temp[segment], (h_scan_K[num_segments*(segment+1)] - h_scan_K[num_segments*segment])*maxSegmentSize*sizeof(int), cudaMemcpyDeviceToDevice);
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaErr = cudaMemcpy(&matrix.V[h_scan_K[num_segments*segment]*maxSegmentSize], d_V_tiled_temp[segment], (h_scan_K[num_segments*(segment+1)] - h_scan_K[num_segments*segment])*maxSegmentSize*sizeof(int), cudaMemcpyDeviceToDevice);
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    }

    printf("copied data\n");
    for(unsigned int segment = 0; segment < num_segments; ++segment){
        cudaFree(d_U_tiled_temp[segment]);
        cudaFree(d_V_tiled_temp[segment]);
    }
    free(h_scan_K);
    free(d_U_tiled_temp);
    free(d_V_tiled_temp);
    
    // TODO: cudafree d_U_tiled_temp and d_V_tiled_temp

    H2Opus_Real* d_buffer_vector;
    H2Opus_Real* d_input_vector;
    H2Opus_Real* d_output_vector;
    H2Opus_Real* d_output_vector_org;
    cudaErr = cudaMalloc((void**) &d_buffer_vector, maxSegmentSize*num_segments*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error d_buffer_vector: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_input_vector, maxSegmentSize*num_segments*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error d_input_vector: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_output_vector, maxSegmentSize*num_segments*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error d_output_vector: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_output_vector_org, maxSegmentSize*num_segments*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error d_output_vector_org: %s\n", cudaGetErrorString(cudaErr)); }

    numThreadsPerBlock = 1024;
    numBlocks = (num_segments*maxSegmentSize + numThreadsPerBlock-1)/numThreadsPerBlock;
    fillVector<<<numBlocks, numThreadsPerBlock>>> (num_segments, maxSegmentSize, d_input_vector, d_output_vector, d_output_vector_org);
    cudaErr = cudaGetLastError();
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error fillVector: %s\n", cudaGetErrorString(cudaErr)); }
    cudaDeviceSynchronize();
    printf("filled vector\n");

    numThreadsPerBlock = 2*upper_power_of_two(maxSegmentSize);
    numBlocks = (num_segments+1)/2;

    cudaEvent_t startGEMV, stopGEMV;
    cudaEventCreate(&startGEMV);
    cudaEventCreate(&stopGEMV);
    cudaEventRecord(startGEMV);
    GEMV<<<numBlocks, numThreadsPerBlock>>> (num_segments, maxSegmentSize, matrix.blockRanks, matrix.blockOffsets, matrix.U, matrix.V, matrix.diagonal, d_input_vector, d_output_vector, d_buffer_vector);
    cudaErr = cudaGetLastError();
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaDeviceSynchronize();
    cudaEventRecord(stopGEMV);
    cudaEventSynchronize(stopGEMV);
    float GEMV_time = 0;
    cudaEventElapsedTime(&GEMV_time, startGEMV, stopGEMV);
    cudaEventDestroy(startGEMV);
    cudaEventDestroy(stopGEMV);

    // dense matrix * vector
    #if 1
    // generate n*n dense matrix
    H2Opus_Real* d_denseMatrix;
    cudaErr = cudaMalloc((void**) &d_denseMatrix, config.n*config.n*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error d_output_vector_org: %s\n", cudaGetErrorString(cudaErr)); }
    H2Opus_Real* d_vector;
    cudaErr = cudaMalloc((void**) &d_vector, config.n*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error d_output_vector_org: %s\n", cudaGetErrorString(cudaErr)); }

    numThreadsPerBlock = 1024; //TODO: make sure that bucket_size is less than 1024   
    numBlocks = (config.n + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillDenseMatrix<<<numBlocks, numThreadsPerBlock>>>(config.n, d_denseMatrix);
    filltmpVector<<<numBlocks, numThreadsPerBlock>>>(config.n, d_vector);
    cudaDeviceSynchronize();

    cublasHandle_t handle;
    cublasStatus_t stat;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    H2Opus_Real alfa=1, beta=0;
    
    cudaEvent_t startDenseGEMV, stopDenseGEMV;
    cudaEventCreate(&startDenseGEMV);
    cudaEventCreate(&stopDenseGEMV);
    cudaEventRecord(startDenseGEMV);
    cublasDgemv(handle, CUBLAS_OP_T,
                           config.n, config.n,
                           &alfa,
                           d_denseMatrix, config.n,
                           d_vector, 1,
                           &beta,
                           d_vector, 1);
    cudaDeviceSynchronize();
    cudaEventRecord(stopDenseGEMV);
    cudaEventSynchronize(stopDenseGEMV);
    float DenseGEMV_time = 0;
    cudaEventElapsedTime(&DenseGEMV_time, startDenseGEMV, stopDenseGEMV);
    printf("DenseGEMV time: %f\n", DenseGEMV_time);
    timer_arr[8] = DenseGEMV_time;
    cudaEventDestroy(startDenseGEMV);
    cudaEventDestroy(stopDenseGEMV);

    cublasDestroy(handle);
    cudaFree(d_denseMatrix);
    cudaFree(d_vector);
    #endif

    printf("GEMV time: %f\n", GEMV_time);
    timer_arr[7] = GEMV_time;

    cudaFree(d_buffer_vector);
    cudaFree(d_input_vector);
    cudaFree(d_output_vector);
    cudaFree(d_output_vector_org);

    // H2Opus_Real* d_U_tiled_2;
    // H2Opus_Real* d_V_tiled_2;
    // int* d_scan_K_2;
    // int* d_K_2;
    TLR_Matrix matrix2;
    cudaErr = cudaMalloc((void**) &matrix2.blockRanks, num_segments*num_segments*sizeof(unsigned int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &matrix2.U, k_sum*maxSegmentSize*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &matrix2.V, k_sum*maxSegmentSize*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &matrix2.diagonal, maxSegmentSize*maxSegmentSize*num_segments*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &matrix2.blockOffsets, num_segments*num_segments*sizeof(unsigned int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    cudaMemcpy(matrix2.blockRanks, matrix.blockRanks, num_segments*num_segments*sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(matrix2.blockOffsets, matrix.blockOffsets, num_segments*num_segments*sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(matrix2.U, matrix.U, k_sum*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
    cudaMemcpy(matrix2.V, matrix.V, k_sum*maxSegmentSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
    cudaMemcpy(matrix2.diagonal, matrix.diagonal, maxSegmentSize*maxSegmentSize*num_segments*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);

    H2Opus_Real* d_U_tiled_output;
    H2Opus_Real* d_V_tiled_output;
    int* d_scan_K_output;
    TLR_Matrix matrix_gemm_output;

    cudaErr = cudaMalloc((void**) &matrix_gemm_output.blockRanks, num_segments*num_segments*sizeof(unsigned int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_U_tiled_output, k_sum*maxSegmentSize*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_V_tiled_output, k_sum*maxSegmentSize*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_scan_K_output, num_segments*num_segments*sizeof(unsigned int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    h_S = (H2Opus_Real *)malloc(maxSegmentSize * num_segments * sizeof(H2Opus_Real));
    h_U = (H2Opus_Real *)malloc(maxSegmentSize * maxSegmentSize * num_segments * sizeof(H2Opus_Real));
    h_V = (H2Opus_Real *)malloc(maxSegmentSize * maxSegmentSize * num_segments * sizeof(H2Opus_Real));

    d_S;
    d_U;
    d_V;
    cudaErr = cudaMalloc((void**) &d_S, maxSegmentSize * num_segments * sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_U, maxSegmentSize * maxSegmentSize * num_segments * sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_V, maxSegmentSize * maxSegmentSize * num_segments * sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    H2Opus_Real* d_gemm_matrix_segmented;
    H2Opus_Real* gemm_matrix_segmented = (H2Opus_Real*)malloc(maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real));
    cudaErr = cudaMalloc((void**) &d_gemm_matrix_segmented, maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    d_U_tiled_temp = (H2Opus_Real**)malloc(num_segments*sizeof(H2Opus_Real*));
    d_V_tiled_temp = (H2Opus_Real**)malloc(num_segments*sizeof(H2Opus_Real*));

    d_scan_K_segmented;
    cudaErr = cudaMalloc((void**) &d_scan_K_segmented, num_segments*sizeof(int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    k_sum=0;

    dim3 mm_numThreadsPerBlock(upper_power_of_two(maxSegmentSize), upper_power_of_two(maxSegmentSize));
    dim3 mm_numBlocks(1, num_segments);
    numThreadsPerBlock = maxSegmentSize; //TODO: make sure that bucket_size is less than 1024   
    numBlocks = num_segments;

    cudaEvent_t startGEMM, stopGEMM;
    cudaEventCreate(&startGEMM);
    cudaEventCreate(&stopGEMM);
    cudaEventRecord(startGEMM);

    for(unsigned int segment = 0; segment < num_segments; ++segment){
        GEMM<<<mm_numBlocks, mm_numThreadsPerBlock, 2*config.bucket_size*config.bucket_size*sizeof(H2Opus_Real)>>>(num_segments, maxSegmentSize, matrix.U, matrix.V, matrix.diagonal, matrix.blockRanks, matrix.blockOffsets, matrix2.U, matrix2.V, matrix2.diagonal, matrix2.blockRanks, matrix2.blockOffsets, d_gemm_matrix_segmented, segment, config.bucket_size);
        cudaDeviceSynchronize();
        cudaMemcpy(gemm_matrix_segmented, d_gemm_matrix_segmented, maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);

        // SVD on gemm_matrix
        SVD(config.n, num_segments, gemm_matrix_segmented, maxSegmentSize, h_S, h_U, h_V);
        cudaDeviceSynchronize();

        cudaMemcpy(d_S, h_S, maxSegmentSize * num_segments * sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_U, h_U, maxSegmentSize*maxSegmentSize*num_segments * sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V, maxSegmentSize*maxSegmentSize*num_segments * sizeof(H2Opus_Real), cudaMemcpyHostToDevice);

        // calcmemneeded and truncation on gemm matrix
        calcMemNeeded<<<numBlocks, numThreadsPerBlock>>> (maxSegmentSize, matrix_gemm_output.blockRanks + segment*num_segments, d_S, config.epsilon);
        cudaDeviceSynchronize();

        d_temp_storage = NULL;
        temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, matrix_gemm_output.blockRanks + segment*num_segments, d_scan_K_segmented, num_segments);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, matrix_gemm_output.blockRanks + segment*num_segments, d_scan_K_segmented, num_segments);
        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);

        int* totalMem = (int*)malloc(sizeof(int));
        int* d_totalMem;
        cudaMalloc((void**) &d_totalMem, sizeof(int));
        getTotalMem<<<1, 1>>> (d_totalMem, matrix_gemm_output.blockRanks + segment*num_segments, d_scan_K_segmented, num_segments);
        cudaDeviceSynchronize();

        cudaErr = cudaMemcpy(totalMem, d_totalMem, sizeof(int), cudaMemcpyDeviceToHost);
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaFree(d_totalMem);

        H2Opus_Real* d_U_tiled_segmented;
        cudaErr = cudaMalloc((void**) &d_U_tiled_segmented, maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        H2Opus_Real* d_V_tiled_segmented;
        cudaErr = cudaMalloc((void**) &d_V_tiled_segmented, maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

        dim3 d_numThreadsPerBlock(upper_power_of_two(maxSegmentSize), upper_power_of_two(maxSegmentSize));
        dim3 d_numBlocks(1, num_segments);
        tileMatrix<<<d_numBlocks, d_numThreadsPerBlock>>> (config.n, num_segments, maxSegmentSize, d_S, d_U, d_V, d_U_tiled_segmented, d_V_tiled_segmented, matrix_gemm_output.blockRanks + segment*num_segments, d_scan_K_segmented, segment);
        cudaDeviceSynchronize();

        cudaErr = cudaMalloc((void**) &d_U_tiled_temp[segment], maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaErr = cudaMalloc((void**) &d_V_tiled_temp[segment], maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaMemcpy(d_U_tiled_temp[segment], d_U_tiled_segmented, maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_V_tiled_temp[segment], d_V_tiled_segmented, maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);

        k_sum += (*totalMem);

        free(totalMem);
        cudaFree(d_U_tiled_segmented);
        cudaFree(d_V_tiled_segmented);
        // save output in a double pointer array
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stopGEMM);
    cudaEventSynchronize(stopGEMM);
    float GEMM_time = 0;
    cudaEventElapsedTime(&GEMM_time, startGEMM, stopGEMM);
    cudaEventDestroy(startGEMM);
    cudaEventDestroy(stopGEMM);
    printf("GEMM time: %f\n", GEMM_time);
    timer_arr[9] = GEMM_time;
    // TODO: copy from double pointer array to a single pointer array

    #if 1
    // generate n*n dense matrix
    H2Opus_Real* d_denseMatrix1;
    cudaErr = cudaMalloc((void**) &d_denseMatrix1, config.n*config.n*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error d_output_vector_org: %s\n", cudaGetErrorString(cudaErr)); }
    H2Opus_Real* d_denseMatrix2;
    cudaErr = cudaMalloc((void**) &d_denseMatrix2, config.n*config.n*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error d_output_vector_org: %s\n", cudaGetErrorString(cudaErr)); }
    H2Opus_Real* d_denseMatrix3;
    cudaErr = cudaMalloc((void**) &d_denseMatrix3, config.n*config.n*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error d_output_vector_org: %s\n", cudaGetErrorString(cudaErr)); }

    numThreadsPerBlock = 1024; //TODO: make sure that bucket_size is less than 1024   
    numBlocks = (config.n + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillDenseMatrix<<<numBlocks, numThreadsPerBlock>>>(config.n, d_denseMatrix1);
    fillDenseMatrix<<<numBlocks, numThreadsPerBlock>>>(config.n, d_denseMatrix2);
    cudaDeviceSynchronize();

    
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    
    cudaEvent_t startDenseGEMM, stopDenseGEMM;
    cudaEventCreate(&startDenseGEMM);
    cudaEventCreate(&stopDenseGEMM);
    cudaEventRecord(startDenseGEMM);
    cudaDeviceSynchronize();
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                           config.n, config.n, config.n,
                           &alfa,
                           d_denseMatrix1, config.n,
                           d_denseMatrix2, config.n,
                           &beta,
                           d_denseMatrix3, config.n);
    cudaDeviceSynchronize();
    cudaEventRecord(stopDenseGEMM);
    cudaEventSynchronize(stopDenseGEMM);
    float DenseGEMM_time = 0;
    cudaEventElapsedTime(&DenseGEMM_time, startDenseGEMM, stopDenseGEMM);
    printf("DenseGEMM time: %f\n", DenseGEMM_time);
    timer_arr[10] = DenseGEMM_time;
    cudaEventDestroy(startDenseGEMM);
    cudaEventDestroy(stopDenseGEMM);

    cublasDestroy(handle);
    cudaFree(d_denseMatrix1);
    cudaFree(d_denseMatrix2);
    cudaFree(d_denseMatrix3);
    #endif

    free(gemm_matrix_segmented);
    cudaFree(d_scan_K_segmented);
    cudaFree(matrix_gemm_output.blockRanks);
    cudaFree(d_gemm_matrix_segmented);
    
    cudaFreeMatrix(matrix);
    cudaFreeMatrix(matrix2);

    cudaDeviceSynchronize();
    cudaEventRecord(stopCode);
    cudaEventSynchronize(stopCode);
    float code_time = 0;
    float Code_time=0;
    cudaEventElapsedTime(&Code_time, startCode, stopCode);
    cudaEventDestroy(startCode);
    cudaEventDestroy(stopCode);
    printf("total time: %f\n", Code_time);
    timer_arr[11] = Code_time;
    printCountersInFile(timer_arr);
    free(timer_arr);
    
}