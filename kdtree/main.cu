#include "tlr_example.h"
#include "helperFunctions.h"
#include "helperKernels.cuh"
#include "SVD.cuh"

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <utility>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <typeinfo>

#define eps 1e-4
#define PRINT_OUTPUT 0
#define USE_SVD 0
#define DIVISION_METHOD 2
using namespace std;

// TODO: generate pointcloud and copy values of the pointcloud to ptr on GPU
// TODO: fix makefile so main.cu depends on helperKerlens.cuh

int main(){
    int n = 1<<6;
    int dim = 2;
    printf("N = %d\n", n);

    // Create point cloud
    PointCloud<H2Opus_Real> pt_cloud(dim, (size_t)n);
    generateGrid<H2Opus_Real>(pt_cloud, n);
    printf("dimension: %d\n", pt_cloud.getDimension());
    printf("bucket size: %d\n", BUCKET_SIZE);

    #if PRINT_OUTPUT
    printf("created point cloud\n");
    for(int i=0; i<n; ++i){
        for(int j=0; j<dim;++j){
            printf("%f ", pt_cloud.pts[j][i]);
        }
        printf("\n");
    }
    printf("\n\n");
    #endif

    H2Opus_Real *dataset;
    dataset = (H2Opus_Real*)malloc((long long)n*(long long)dim*(long long)sizeof(H2Opus_Real));
    assert(dataset != NULL);

    // TODO: move this to a kernel
    for (unsigned long long i = 0; i < dim; ++i){
        for(unsigned long long j = 0; j < n; ++j){
            dataset[i*n+j] = pt_cloud.getDataPoint((size_t)j, (int)i);
        }
    }

    H2Opus_Real *d_dataset;
    cudaMalloc((void**) &d_dataset, (long long)n*(long long)dim*(long long)sizeof(H2Opus_Real));
    cudaMemcpy(d_dataset, dataset, (long long)n*(long long)dim*(long long)sizeof(H2Opus_Real*), cudaMemcpyHostToDevice);

    unsigned int  num_segments = 1;
    unsigned long long num_segments_reduce = num_segments*dim;

    #if DIVISION_METHOD == 0
    unsigned int segment_size = upper_power_of_two(n);
    #endif

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
    float* timer_arr = (float*)malloc(numTimers*sizeof(float));

    #if DIVISION_METHOD == 1
    bool workDone= false;
    bool* d_workDone;
    uint64_t* d_bit_vector;
    short int* d_popc_bit_vector;
    short int* d_popc_scan;
    unsigned int* d_new_num_segments;
    unsigned int* new_num_segments = (unsigned int*)malloc(sizeof(unsigned int));
    #endif

    int max_num_segments;

    #if DIVISION_METHOD !=0 
    int* A;
    int* B;
    int* output;
    int* d_output;
    int* d_input_search;
    int* d_aux_offsets_sort;
    max_num_segments = 1<<(getMaxSegmentSize(n, BUCKET_SIZE).second);
    #else
    max_num_segments = (n+BUCKET_SIZE-1)/BUCKET_SIZE;
    #endif
    printf("max num segments: %d\n", max_num_segments);

    #if DIVISION_METHOD == 2
    int largest_segment_size = n;
    #endif

    // TODO: fix memory allocated
    cudaMalloc((void**) &d_temp, n*sizeof(int));
    cudaMalloc((void**) &d_offsets_sort, (max_num_segments + 1)*sizeof(int));
    cudaMalloc((void**) &d_offsets_reduce, (long long)((max_num_segments*dim + 1)*(long long)sizeof(int)));
    cudaMalloc((void**) &d_keys_in, n*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_keys_out, n*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_values_in, n*sizeof(int));
    cudaMalloc((void**) &d_values_out, n*sizeof(int));
    cudaMalloc((void**) &d_curr_dim, (max_num_segments + 1)*sizeof(int));
    cudaMalloc((void**) &d_reduce_in, (long long)n*(long long)dim*(long long)sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_reduce_min_out, (long long)((max_num_segments+1)*dim)*(long long)sizeof(int));
    cudaMalloc((void**) &d_reduce_max_out, (long long)((max_num_segments+1)*dim)*(long long)sizeof(int));
    cudaMalloc((void**) &d_span, (long long)((max_num_segments+1)*dim)*(long long)sizeof(int));
    cudaMalloc((void**) &d_span_offsets, (max_num_segments + 1)*sizeof(int));
    cudaMalloc((void**) &d_span_reduce_out, (max_num_segments+1)*sizeof(cub::KeyValuePair<int, H2Opus_Real>));

    #if DIVISION_METHOD == 1
    cudaMalloc((void**) &d_bit_vector, (((max_num_segments+1) + sizeof(uint64_t)*8-1)/(sizeof(uint64_t)*8)) *sizeof(uint64_t));
    cudaMalloc((void**) &d_popc_bit_vector, (((max_num_segments + 1) + sizeof(uint64_t)*8-1)/(sizeof(uint64_t)*8))*sizeof(short int));
    cudaMalloc((void**) &d_popc_scan, (((max_num_segments + 1) + sizeof(uint64_t)*8-1)/(sizeof(uint64_t)*8))*sizeof(short int));
    cudaMalloc((void**) &d_new_num_segments, sizeof(unsigned int));
    cudaMalloc((void**) &d_workDone, sizeof(bool));
    #endif

    #if DIVISION_METHOD !=0
    cudaMalloc((void**) &d_aux_offsets_sort, (max_num_segments + 1) * sizeof(int));
    cudaMalloc((void**) &A, (max_num_segments + 1)*sizeof(int));
    cudaMalloc((void**) &B, n*sizeof(int));
    cudaMalloc((void**) &output, n*sizeof(int));
    cudaMalloc((void**) &d_output, n*sizeof(int));
    cudaMalloc((void**) &d_input_search, n*sizeof(int));
    #endif

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (n+numThreadsPerBlock-1)/numThreadsPerBlock;

    #if DIVISION_METHOD != 0
    initializeArrays<<<numBlocks, numThreadsPerBlock>>>(n, dim, d_values_in, d_curr_dim, d_offsets_sort, d_offsets_reduce, d_input_search, max_num_segments);
    #elif DIVISION_METHOD == 0
    initializeArrays<<<numBlocks, numThreadsPerBlock>>>(n, d_values_in, d_curr_dim, max_num_segments);
    #endif
    cudaDeviceSynchronize();

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cudaEvent_t startWhileLoop, stopWhileLoop;
    cudaEventCreate(&startWhileLoop);
    cudaEventCreate(&stopWhileLoop);
    cudaEventRecord(startWhileLoop);

    unsigned int iteration = 0;

    #if DIVISION_METHOD == 1
    while(!workDone) {
    #elif DIVISION_METHOD == 0
    while(segment_size > BUCKET_SIZE) {
    #else
    while(largest_segment_size > BUCKET_SIZE) {
    #endif
        // printf("begin\n");
        for(unsigned int i=0; i<numTimers; ++i){
            timer_arr[i]=0;
        }

        numThreadsPerBlock = 1024;
        numBlocks = (num_segments+1+numThreadsPerBlock-1)/numThreadsPerBlock;
        cudaEvent_t startFillOffsets_k, stopFillOffsets_k;
        cudaEventCreate(&startFillOffsets_k);
        cudaEventCreate(&stopFillOffsets_k);
        cudaEventRecord(startFillOffsets_k);
        #if DIVISION_METHOD==0
        fillOffsets<<<numBlocks, numThreadsPerBlock>>>(n, dim, num_segments, segment_size, d_offsets_sort, d_offsets_reduce);
        #endif
        cudaEventRecord(stopFillOffsets_k);
        cudaEventSynchronize(stopFillOffsets_k);
        cudaEventElapsedTime(&timer_arr[0], startFillOffsets_k, stopFillOffsets_k);
        cudaEventDestroy(startFillOffsets_k);
        cudaEventDestroy(stopFillOffsets_k);
        cudaDeviceSynchronize();

        numThreadsPerBlock = 1024;
        numBlocks = (long long)((long long)n*(long long)dim + numThreadsPerBlock-1)/numThreadsPerBlock;

        cudaEvent_t startFillReduction_k, stopFillReduction_k;
        cudaEventCreate(&startFillReduction_k);
        cudaEventCreate(&stopFillReduction_k);
        cudaEventRecord(startFillReduction_k);
        fillReductionArray<<<numBlocks, numThreadsPerBlock>>> (n, dim, d_dataset, d_values_in, d_reduce_in);
        cudaEventRecord(stopFillReduction_k);
        cudaEventSynchronize(stopFillReduction_k);
        cudaEventElapsedTime(&timer_arr[1], startFillReduction_k, stopFillReduction_k);
        cudaEventDestroy(startFillReduction_k);
        cudaEventDestroy(stopFillReduction_k);
        cudaDeviceSynchronize();

        d_temp_storage = NULL;
        temp_storage_bytes = 0;

        cudaEvent_t startMinReduce_k, stopMinReduce_k;
        cudaEventCreate(&startMinReduce_k);
        cudaEventCreate(&stopMinReduce_k);
        cudaEventRecord(startMinReduce_k);
        cub::DeviceSegmentedReduce::Min(d_temp_storage, temp_storage_bytes, d_reduce_in, d_reduce_min_out,
            num_segments_reduce, d_offsets_reduce, d_offsets_reduce + 1);

        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceSegmentedReduce::Min(d_temp_storage, temp_storage_bytes, d_reduce_in, d_reduce_min_out,
            num_segments_reduce, d_offsets_reduce, d_offsets_reduce + 1);
        cudaEventRecord(stopMinReduce_k);
        cudaEventSynchronize(stopMinReduce_k);
        cudaEventElapsedTime(&timer_arr[2], startMinReduce_k, stopMinReduce_k);
        cudaEventDestroy(startMinReduce_k);
        cudaEventDestroy(stopMinReduce_k);
        cudaDeviceSynchronize();

        d_temp_storage = NULL;
        temp_storage_bytes = 0;

        cudaEvent_t startMaxReduce_k, stopMaxReduce_k;
        cudaEventCreate(&startMaxReduce_k);
        cudaEventCreate(&stopMaxReduce_k);
        cudaEventRecord(startMaxReduce_k);
        cub::DeviceSegmentedReduce::Max(d_temp_storage, temp_storage_bytes, d_reduce_in, d_reduce_max_out,
            num_segments_reduce, d_offsets_reduce, d_offsets_reduce + 1);

        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceSegmentedReduce::Max(d_temp_storage, temp_storage_bytes, d_reduce_in, d_reduce_max_out,
            num_segments_reduce, d_offsets_reduce, d_offsets_reduce + 1);
        cudaEventRecord(stopMaxReduce_k);
        cudaEventSynchronize(stopMaxReduce_k);
        cudaEventElapsedTime(&timer_arr[3], startMaxReduce_k, stopMaxReduce_k);
        cudaEventDestroy(startMaxReduce_k);
        cudaEventDestroy(stopMaxReduce_k);
        cudaDeviceSynchronize();

        numThreadsPerBlock = 1024;
        numBlocks = (num_segments+numThreadsPerBlock-1)/numThreadsPerBlock;

        cudaEvent_t startFindSpan_k, stopFindSpan_k;
        cudaEventCreate(&startFindSpan_k);
        cudaEventCreate(&stopFindSpan_k);
        cudaEventRecord(startFindSpan_k);
        findSpan<<<numBlocks, numThreadsPerBlock>>> (n, dim, num_segments, d_reduce_min_out, d_reduce_max_out, d_span, d_span_offsets);
        cudaEventRecord(stopFindSpan_k);
        cudaEventSynchronize(stopFindSpan_k);
        cudaEventElapsedTime(&timer_arr[4], startFindSpan_k, stopFindSpan_k);
        cudaEventDestroy(startFindSpan_k);
        cudaEventDestroy(stopFindSpan_k);
        cudaDeviceSynchronize();

        d_temp_storage = NULL;
        temp_storage_bytes = 0;

        cudaEvent_t startArgMaxReduce_k, stopArgMaxReduce_k;
        cudaEventCreate(&startArgMaxReduce_k);
        cudaEventCreate(&stopArgMaxReduce_k);
        cudaEventRecord(startArgMaxReduce_k);
        cub::DeviceSegmentedReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_span, d_span_reduce_out,
            num_segments, d_span_offsets, d_span_offsets + 1);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run argmax-reduction
        cub::DeviceSegmentedReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_span, d_span_reduce_out,
            num_segments, d_span_offsets, d_span_offsets + 1);
        cudaEventRecord(stopArgMaxReduce_k);
        cudaEventSynchronize(stopArgMaxReduce_k);
        cudaEventElapsedTime(&timer_arr[5], startArgMaxReduce_k, stopArgMaxReduce_k);
        cudaEventDestroy(startArgMaxReduce_k);
        cudaEventDestroy(stopArgMaxReduce_k);
        cudaDeviceSynchronize();

        numThreadsPerBlock = 1024;
        numBlocks = (num_segments+numThreadsPerBlock-1)/numThreadsPerBlock;

        cudaEvent_t startfillCurrDim_k, stopFillCurrDim_k;
        cudaEventCreate(&startfillCurrDim_k);
        cudaEventCreate(&stopFillCurrDim_k);
        cudaEventRecord(startfillCurrDim_k);
        #if DIVISION_METHOD == 1
        fillCurrDim<<<numBlocks, numThreadsPerBlock>>> (n, num_segments, d_curr_dim, d_span_reduce_out, d_bit_vector);
        #else
        fillCurrDim<<<numBlocks, numThreadsPerBlock>>> (n, num_segments, d_curr_dim, d_span_reduce_out);
        #endif
        cudaDeviceSynchronize();
        cudaEventRecord(stopFillCurrDim_k);
        cudaEventSynchronize(stopFillCurrDim_k);
        cudaEventElapsedTime(&timer_arr[6], startfillCurrDim_k, stopFillCurrDim_k);
        cudaEventDestroy(startfillCurrDim_k);
        cudaEventDestroy(stopFillCurrDim_k);

        // fill keys_in array
        numThreadsPerBlock = 1024;
        numBlocks = (n+numThreadsPerBlock-1)/numThreadsPerBlock;

        cudaEvent_t startfillKeysIn_k, stopKeysIn_k;
        cudaEventCreate(&startfillKeysIn_k);
        cudaEventCreate(&stopKeysIn_k);
        cudaEventRecord(startfillKeysIn_k);

        #if DIVISION_METHOD != 0
        thrust::device_ptr<int> A = thrust::device_pointer_cast((int *)d_offsets_sort), B = thrust::device_pointer_cast((int *)d_input_search);
        thrust::device_vector<int> output(n);
        thrust::upper_bound(A, A + num_segments + 1, B, B + n, output.begin(), thrust::less<int>());
        d_output = thrust::raw_pointer_cast(&output[0]);
        fillKeysIn<<<numBlocks, numThreadsPerBlock>>> (n, d_keys_in, d_curr_dim, d_values_in, d_dataset, d_offsets_sort, d_output);
        #else
        fillKeysIn<<<numBlocks, numThreadsPerBlock>>> (n, segment_size, d_keys_in, d_curr_dim, d_values_in, d_dataset);
        #endif

        cudaEventRecord(stopKeysIn_k);
        cudaEventSynchronize(stopKeysIn_k);
        cudaEventElapsedTime(&timer_arr[7], startfillKeysIn_k, stopKeysIn_k);
        cudaEventDestroy(startfillKeysIn_k);
        cudaEventDestroy(stopKeysIn_k);
        cudaDeviceSynchronize();

        d_temp_storage = NULL;
        temp_storage_bytes = 0;

        cudaEvent_t startSort_k, stopSort_k;
        cudaEventCreate(&startSort_k);
        cudaEventCreate(&stopSort_k);
        cudaEventRecord(startSort_k);
        cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out,
        n, num_segments, d_offsets_sort, d_offsets_sort + 1);

        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out,
        n, num_segments, d_offsets_sort, d_offsets_sort + 1);
        cudaEventRecord(stopSort_k);
        cudaEventSynchronize(stopSort_k);
        cudaEventElapsedTime(&timer_arr[8], startSort_k, stopSort_k);
        cudaEventDestroy(startSort_k);
        cudaEventDestroy(stopSort_k);
        cudaDeviceSynchronize();
        
        cudaFree(d_temp_storage);

        d_temp = d_values_in;
        d_values_in = d_values_out;
        d_values_out = d_temp;
        printCountersInFile(iteration, num_segments, timer_arr);
        ++iteration;

        #if DIVISION_METHOD == 1
        numThreadsPerBlock = 1024;
        numBlocks = (num_segments+numThreadsPerBlock-1)/numThreadsPerBlock;
        fillBitVector<<<numBlocks, numThreadsPerBlock>>>(num_segments, d_bit_vector, d_offsets_sort);
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

        numThreadsPerBlock = 1024;
        numBlocks = (num_segments+numThreadsPerBlock-1)/numThreadsPerBlock;
        fillOffsetsSort<<<numBlocks, numThreadsPerBlock>>>(n, dim, num_segments, d_offsets_sort, d_aux_offsets_sort, d_bit_vector, d_popc_scan, d_new_num_segments, d_workDone);
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
        fillOffsetsReduce<<<numBlocks, numThreadsPerBlock>>> (n, dim, num_segments, d_offsets_sort, d_offsets_reduce);
        cudaDeviceSynchronize();

        #elif DIVISION_METHOD == 0
        segment_size /= 2;
        num_segments = (n+segment_size-1)/segment_size;

        #elif DIVISION_METHOD == 2
        numThreadsPerBlock = 1024;
        numBlocks = (num_segments+numThreadsPerBlock-1)/numThreadsPerBlock;
        fillOffsetsSort<<<numBlocks, numThreadsPerBlock>>> (n, dim, num_segments, d_offsets_sort, d_aux_offsets_sort);
        cudaDeviceSynchronize();
        d_temp = d_aux_offsets_sort;
        d_aux_offsets_sort = d_offsets_sort;
        d_offsets_sort = d_temp;
        num_segments *= 2;

        numThreadsPerBlock = 1024;
        numBlocks = (num_segments+numThreadsPerBlock-1)/numThreadsPerBlock;
        fillOffsetsReduce<<<numBlocks, numThreadsPerBlock>>> (n, dim, num_segments, d_offsets_sort, d_offsets_reduce);
        cudaDeviceSynchronize();
        ++largest_segment_size;
        largest_segment_size /= 2;
        #endif

        num_segments_reduce = num_segments*dim;
    }

    cudaEventRecord(stopWhileLoop);
    cudaEventSynchronize(stopWhileLoop);
    float whileLoop_time = 0;
    cudaEventElapsedTime(&whileLoop_time, startWhileLoop, stopWhileLoop);
    printf("total time taken for while loop: %f\n", whileLoop_time);
    cudaEventDestroy(startWhileLoop);
    cudaEventDestroy(stopWhileLoop);

    #if DIVISION_METHOD == 0
    printf("num segments :%d\n", num_segments);
    printf("segment size :%d\n", segment_size);
    fillOffsets<<<numBlocks, numThreadsPerBlock>>>(n, dim, num_segments, segment_size, d_offsets_sort, d_offsets_reduce);
    cudaDeviceSynchronize();
    #endif

    #if PRINT_OUTPUT
    int *index_map = (int*)malloc(n*sizeof(int));
    cudaMemcpy(index_map, d_values_in, n*sizeof(int), cudaMemcpyDeviceToHost);
    printf("inex max\n");
    for(int i=0; i<n; ++i){
        printf("%d ", index_map[i]);
    }
    printf("\n");

    for(int i=0; i<n; ++i){
        for(int j=0; j<dim; ++j){
            printf("%f ", pt_cloud.pts[j][index_map[i]]);
        }
        printf("\n");
    }
    free(index_map);
    #endif

    // // _______________________________________________________________________________________________
    // #if 0
    // int maxSegmentSize;
    // #if DIVISION_METHOD != 0
    // maxSegmentSize = getMaxSegmentSize(n, BUCKET_SIZE).first;
    // printf("max segment size: %d\n", maxSegmentSize);
    // printf("n: %d\n", n);
    // #else
    // maxSegmentSize = BUCKET_SIZE;
    // #endif

    // H2Opus_Real* d_input_matrix;
    // H2Opus_Real* input_matrix = (H2Opus_Real*)malloc((long long)maxSegmentSize*num_segments*(long long)maxSegmentSize*num_segments*(long long)sizeof(H2Opus_Real));
    // cudaMalloc((void**) &d_input_matrix, (long long)maxSegmentSize*num_segments*(long long)maxSegmentSize*num_segments*(long long)sizeof(H2Opus_Real));
    
    // dim3 m_numThreadsPerBlock(upper_power_of_two(maxSegmentSize), upper_power_of_two(maxSegmentSize));
    // dim3 m_numBlocks(num_segments, num_segments);
    // generateInputMatrix<<<m_numBlocks, m_numThreadsPerBlock>>>(n, num_segments, maxSegmentSize, dim, d_values_in, d_input_matrix, d_dataset, d_offsets_sort);
    // cudaDeviceSynchronize();
    // cudaMemcpy(input_matrix, d_input_matrix, (long long)maxSegmentSize*num_segments*(long long)maxSegmentSize*num_segments*(long long)sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);

    // int NRows = maxSegmentSize;
    // int NCols = maxSegmentSize;
    // int numMatrices = num_segments*num_segments;

    // H2Opus_Real *h_S = (H2Opus_Real *)malloc(maxSegmentSize * numMatrices * sizeof(H2Opus_Real));
    // H2Opus_Real *h_U = (H2Opus_Real *)malloc(maxSegmentSize * maxSegmentSize * numMatrices * sizeof(H2Opus_Real));
    // H2Opus_Real *h_V = (H2Opus_Real *)malloc(maxSegmentSize * maxSegmentSize * numMatrices * sizeof(H2Opus_Real));
    // SVD(n, num_segments, input_matrix, NRows, NCols, maxSegmentSize, h_S, h_U, h_V);
    // #endif
    // // _______________________________________________________________________________________________

    int maxSegmentSize;
    #if DIVISION_METHOD != 0
    maxSegmentSize = getMaxSegmentSize(n, BUCKET_SIZE).first;
    printf("max segment size: %d\n", maxSegmentSize);
    printf("n: %d\n", n);
    #else
    maxSegmentSize = BUCKET_SIZE;
    #endif

    H2Opus_Real* d_input_matrix;
    H2Opus_Real* input_matrix = (H2Opus_Real*)malloc((long long)maxSegmentSize*num_segments*(long long)maxSegmentSize*num_segments*(long long)sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_input_matrix, (long long)maxSegmentSize*num_segments*(long long)maxSegmentSize*num_segments*(long long)sizeof(H2Opus_Real));
    
    dim3 m_numThreadsPerBlock(upper_power_of_two(maxSegmentSize), upper_power_of_two(maxSegmentSize));
    dim3 m_numBlocks(num_segments, num_segments);
    generateInputMatrix<<<m_numBlocks, m_numThreadsPerBlock>>>(n, num_segments, maxSegmentSize, dim, d_values_in, d_input_matrix, d_dataset, d_offsets_sort);
    cudaDeviceSynchronize();
    cudaMemcpy(input_matrix, d_input_matrix, (long long)maxSegmentSize*num_segments*(long long)maxSegmentSize*num_segments*(long long)sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);

    #if PRINT_OUTPUT
    for(unsigned int i=0; i<num_segments*maxSegmentSize; ++i){
        for(unsigned int j=0; j<num_segments*maxSegmentSize; ++j){
            printf("%f ", input_matrix[i*num_segments*maxSegmentSize + j]);
        }
        printf("\n");
    }
    #endif

    int nBlocks = num_segments;
    int NRows = maxSegmentSize;
    int NCols = maxSegmentSize;

    H2Opus_Real *h_S = (H2Opus_Real *)malloc(maxSegmentSize * num_segments*num_segments * sizeof(H2Opus_Real));
    H2Opus_Real *h_U = (H2Opus_Real *)malloc(maxSegmentSize * maxSegmentSize * num_segments*num_segments * sizeof(H2Opus_Real));
    H2Opus_Real *h_V = (H2Opus_Real *)malloc(maxSegmentSize * maxSegmentSize * num_segments*num_segments * sizeof(H2Opus_Real));

    SVD(n, num_segments, input_matrix, NRows, NCols, maxSegmentSize, h_S, h_U, h_V);
    cudaDeviceSynchronize();

    int* d_K;
    H2Opus_Real* d_S;
    H2Opus_Real* d_U;
    H2Opus_Real* d_V;
    cudaMalloc((void**) &d_S, maxSegmentSize * num_segments*num_segments * sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_U, maxSegmentSize * maxSegmentSize * num_segments*num_segments * sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_V, maxSegmentSize * maxSegmentSize * num_segments*num_segments * sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_K, num_segments*num_segments * sizeof(int));
    cudaMemcpy(d_S, h_S, maxSegmentSize * num_segments*num_segments * sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
    
    printf("num segments: %d\n", num_segments);
    // TODO: call a kernel to figure out size of each tile
    numThreadsPerBlock = maxSegmentSize; //TODO: make sure that bucket_size is less than 1024
    numBlocks = num_segments*num_segments;
    calcMemNeeded<<<numBlocks, numThreadsPerBlock>>> (n, maxSegmentSize, d_K, d_S, eps, d_offsets_sort, num_segments);
    cudaDeviceSynchronize();

    int* d_scan_K;
    cudaMalloc((void**) &d_scan_K, num_segments*num_segments*sizeof(int));
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_K, d_scan_K, num_segments*num_segments);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_K, d_scan_K, num_segments*num_segments);

    int* totalMem = (int*)malloc(sizeof(int));
    int* d_totalMem;
    cudaMalloc((void**) &d_totalMem, sizeof(int));
    getTotalMem<<<1, 1>>> (d_totalMem, d_K, d_scan_K, num_segments);
    cudaDeviceSynchronize();
    cudaMemcpy(totalMem, d_totalMem, sizeof(int), cudaMemcpyDeviceToHost);

    H2Opus_Real* d_U_tiled;
    H2Opus_Real* d_V_tiled;
    cudaMalloc((void**) &d_U_tiled, maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_V_tiled, maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real));

    // TODO: find the maximum rank using a max reduction
    dim3 d_numThreadsPerBlock(upper_power_of_two(maxSegmentSize), upper_power_of_two(maxSegmentSize));
    dim3 d_numBlocks(num_segments, num_segments);
    tileMatrix<<<d_numBlocks, d_numThreadsPerBlock>>> (n, num_segments, maxSegmentSize, d_S, d_U, d_V, d_U_tiled, d_V_tiled, d_K, d_scan_K);
    cudaDeviceSynchronize();

    H2Opus_Real* expMatrix;
    cudaMalloc((void**) &expMatrix, num_segments*num_segments*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real));
    expandMatrix<<<d_numBlocks, d_numThreadsPerBlock>>> (num_segments, maxSegmentSize, d_K, d_scan_K, d_U_tiled, d_V_tiled, expMatrix);
    cudaDeviceSynchronize();

    H2Opus_Real* d_error;
    H2Opus_Real* error = (H2Opus_Real*) malloc(sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_error, sizeof(H2Opus_Real));

    H2Opus_Real* d_tmp;
    H2Opus_Real* tmp = (H2Opus_Real*) malloc(sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_tmp, sizeof(H2Opus_Real));

    numThreadsPerBlock = 1024;
    numBlocks = (num_segments*num_segments*maxSegmentSize*maxSegmentSize + numThreadsPerBlock-1)/numThreadsPerBlock;
    calcError<<<d_numBlocks, d_numThreadsPerBlock>>> (num_segments, maxSegmentSize, expMatrix, d_input_matrix, d_error, d_tmp);
    cudaDeviceSynchronize();
    cudaMemcpy(error, d_error, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(tmp, d_tmp, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    printf("error: %f\n", sqrt(*error)/sqrt(*tmp));
    // free(dataset);
    // cudaFree(d_dataset);
    // cudaFree(d_offsets_sort);
    // cudaFree(d_offsets_reduce);
    // cudaFree(d_keys_in);
    // cudaFree(d_keys_out);
    // cudaFree(d_values_in);
    // cudaFree(d_values_out);
    // cudaFree(d_curr_dim);
    // cudaFree(d_reduce_in);
    // cudaFree(d_reduce_min_out);
    // cudaFree(d_reduce_max_out);
    // cudaFree(d_temp);
    // cudaFree(d_span_reduce_out);
    // cudaFree(d_span);
    // cudaFree(d_span_offsets);
    // cudaFree(d_input_matrix);
    // free(input_matrix);

    // #if DIVISION_METHOD == 1
    // cudaFree(A);
    // cudaFree(B);
    // cudaFree(output);
    // cudaFree(d_output);
    // cudaFree(d_input_search);
    // cudaFree(d_bit_vector);
    // cudaFree(d_aux_offsets_sort);
    // cudaFree(d_new_num_segments);
    // free(new_num_segments);
    // cudaFree(d_workDone);
    // #endif
}