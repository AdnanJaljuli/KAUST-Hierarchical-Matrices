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

#define eps 1e-2
#define PRINT_OUTPUT 0
#define USE_SVD 0
#define DIVISION_METHOD 0
using namespace std;

// TODO: generate pointcloud and copy values of the pointcloud to ptr on GPU
// TODO: fix makefile so main.cu depends on helperKerlens.cuh

int main(){

    cudaEvent_t startCode, stopCode;
    cudaEventCreate(&startCode);
    cudaEventCreate(&stopCode);
    cudaEventRecord(startCode);
    uint64_t n = 1<<15;
    uint64_t dim = 2;
    printf("N = %d\n", n);
    fflush(stdout);

    float* timer_arr = (float*)malloc(numTimers*sizeof(float));
    timer_arr[0] = (float)n;
    timer_arr[1] = (float)BUCKET_SIZE;
    timer_arr[2] = (float)dim;
    timer_arr[3] = (float)eps;

    // Create point cloud
    PointCloud<H2Opus_Real> pt_cloud(dim, (size_t)n);
    generateGrid<H2Opus_Real>(pt_cloud, n);
    printf("dimension: %d\n", pt_cloud.getDimension());
    printf("bucket size: %d\n", BUCKET_SIZE);

    #if PRINT_OUTPUT
    printf("created point cloud\n");
    for(int i=0; i<n; ++i){
        for(int j=0; j<dim;++j){
            printf("%lf ", pt_cloud.pts[j][i]);
        }
        printf("\n");
    }
    printf("\n\n");
    #endif

    H2Opus_Real *dataset;
    dataset = (H2Opus_Real*)malloc(n*dim*(uint64_t)sizeof(H2Opus_Real));
    assert(dataset != NULL);

    // TODO: move this to a kernel
    for (unsigned long long i = 0; i < dim; ++i){
        for(unsigned long long j = 0; j < n; ++j){
            dataset[i*n+j] = pt_cloud.getDataPoint((size_t)j, (int)i);
        }
    }

    H2Opus_Real *d_dataset;
    cudaMalloc((void**) &d_dataset, n*dim*(uint64_t)sizeof(H2Opus_Real));
    cudaMemcpy(d_dataset, dataset, n*dim*(uint64_t)sizeof(H2Opus_Real*), cudaMemcpyHostToDevice);

    uint64_t num_segments = 1;
    uint64_t num_segments_reduce = num_segments*dim;

    #if DIVISION_METHOD == 0
    uint64_t segment_size = upper_power_of_two(n);
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
    int* d_bin_search_output;
    int* d_thrust_v_bin_search_output;
    int* d_input_search;
    int* d_aux_offsets_sort;
    max_num_segments = 1<<(getMaxSegmentSize(n, BUCKET_SIZE).second);
    #else
    max_num_segments = (n+BUCKET_SIZE-1)/BUCKET_SIZE;
    #endif
    printf("max num segments: %d\n", max_num_segments);

    #if DIVISION_METHOD == 2
    unsigned int largest_segment_size = n;
    #endif

    // TODO: fix memory allocated
    // cudaMalloc((void**) &d_temp, n*sizeof(int));
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
    cudaMalloc((void**) &d_bin_search_output, n*sizeof(int));
    // cudaMalloc((void**) &d_output, n*sizeof(int));
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
    while(!workDone)
    #elif DIVISION_METHOD == 0
    while(segment_size > BUCKET_SIZE)
    #else
    while(largest_segment_size > BUCKET_SIZE)
    #endif
    {
        numThreadsPerBlock = 1024;
        numBlocks = (num_segments+1+numThreadsPerBlock-1)/numThreadsPerBlock;
        #if DIVISION_METHOD==0
        fillOffsets<<<numBlocks, numThreadsPerBlock>>>(n, dim, num_segments, segment_size, d_offsets_sort, d_offsets_reduce);
        #endif
        cudaDeviceSynchronize();

        numThreadsPerBlock = 1024;
        numBlocks = (long long)((long long)n*(long long)dim + numThreadsPerBlock-1)/numThreadsPerBlock;

        fillReductionArray<<<numBlocks, numThreadsPerBlock>>> (n, dim, d_dataset, d_values_in, d_reduce_in);
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

        findSpan<<<numBlocks, numThreadsPerBlock>>> (n, dim, num_segments, d_reduce_min_out, d_reduce_max_out, d_span, d_span_offsets);
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

        #if DIVISION_METHOD == 1
        fillCurrDim<<<numBlocks, numThreadsPerBlock>>> (n, num_segments, d_curr_dim, d_span_reduce_out, d_bit_vector);
        #else
        fillCurrDim<<<numBlocks, numThreadsPerBlock>>> (n, num_segments, d_curr_dim, d_span_reduce_out);
        #endif
        cudaDeviceSynchronize();

        // fill keys_in array
        numThreadsPerBlock = 1024;
        numBlocks = (n+numThreadsPerBlock-1)/numThreadsPerBlock;

        #if DIVISION_METHOD != 0
        thrust::device_ptr<int> A = thrust::device_pointer_cast((int *)d_offsets_sort), B = thrust::device_pointer_cast((int *)d_input_search);
        thrust::device_vector<int> d_bin_search_output(n);
        thrust::upper_bound(A, A + num_segments + 1, B, B + n, d_bin_search_output.begin(), thrust::less<int>());
        d_thrust_v_bin_search_output = thrust::raw_pointer_cast(&d_bin_search_output[0]);
        fillKeysIn<<<numBlocks, numThreadsPerBlock>>> (n, d_keys_in, d_curr_dim, d_values_in, d_dataset, d_offsets_sort, d_thrust_v_bin_search_output);
        #else
        fillKeysIn<<<numBlocks, numThreadsPerBlock>>> (n, segment_size, d_keys_in, d_curr_dim, d_values_in, d_dataset);
        #endif
        cudaDeviceSynchronize();

        d_temp_storage = NULL;
        temp_storage_bytes = 0;

        cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out,
        n, num_segments, d_offsets_sort, d_offsets_sort + 1);

        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out,
        n, num_segments, d_offsets_sort, d_offsets_sort + 1);
        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);

        d_temp = d_values_in;
        d_values_in = d_values_out;
        d_values_out = d_temp;
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
        cudaFree(d_temp_storage);

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
    timer_arr[4] = whileLoop_time;
    cudaEventDestroy(startWhileLoop);
    cudaEventDestroy(stopWhileLoop);

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
            printf("%lf ", pt_cloud.pts[j][index_map[i]]);
        }
        printf("\n");
    }
    free(index_map);
    #endif

    #if DIVISION_METHOD == 0
    printf("num segments :%d\n", num_segments);
    printf("segment size :%d\n", segment_size);
    fillOffsets<<<numBlocks, numThreadsPerBlock>>>(n, dim, num_segments, segment_size, d_offsets_sort, d_offsets_reduce);
    cudaDeviceSynchronize();
    #endif

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
    #if DIVISION_METHOD != 0
    cudaFree(d_aux_offsets_sort);
    cudaFree(A);
    cudaFree(B);
    cudaFree(d_bin_search_output);
    cudaFree(d_input_search);
    #endif
    #if DIVISION_METHOD == 1
    cudaFree(d_bit_vector);
    cudaFree(d_popc_bit_vector);
    cudaFree(d_popc_scan);
    cudaFree(d_new_num_segments);
    cudaFree(d_workDone);
    #endif

    uint64_t maxSegmentSize;
    #if DIVISION_METHOD != 0
    maxSegmentSize = getMaxSegmentSize(n, BUCKET_SIZE).first;
    #else
    maxSegmentSize = BUCKET_SIZE;
    #endif
    printf("num segments: %lu\n", num_segments);
    printf("max segment size: %lu\n", maxSegmentSize);

    H2Opus_Real* d_input_matrix;
    H2Opus_Real* input_matrix = (H2Opus_Real*)malloc(maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real));
    cudaError_t cudaErr = cudaMalloc((void**) &d_input_matrix, maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real));
    printf("mem allocated to input matrix: %lu\n", maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess )
    {
       printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr));
    }

    for(unsigned int segment = 0; segment < num_segments; ++segment){

        dim3 m_numThreadsPerBlock(upper_power_of_two(maxSegmentSize), upper_power_of_two(maxSegmentSize));
        dim3 m_numBlocks(1, num_segments);
        cudaEvent_t startGenerateInputMatrix, stopGenerateInputMatrix;
        cudaEventCreate(&startGenerateInputMatrix);
        cudaEventCreate(&stopGenerateInputMatrix);
        cudaEventRecord(startGenerateInputMatrix);
        generateInputMatrix<<<m_numBlocks, m_numThreadsPerBlock>>>(n, num_segments, maxSegmentSize, dim, d_values_in, d_input_matrix, d_dataset, d_offsets_sort, segment);
        cudaEventRecord(stopGenerateInputMatrix);
        cudaEventSynchronize(stopGenerateInputMatrix);
        cudaEventElapsedTime(&timer_arr[5], startGenerateInputMatrix, stopGenerateInputMatrix);
        cudaEventDestroy(startGenerateInputMatrix);
        cudaEventDestroy(stopGenerateInputMatrix);
        cudaDeviceSynchronize();
        cudaMemcpy(input_matrix, d_input_matrix, maxSegmentSize*maxSegmentSize*num_segments*(uint64_t)sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);

        // printMatrix<<<1, 1>>>(num_segments, maxSegmentSize, d_input_matrix);
        // free(dataset);
        // cudaFree(d_dataset);

        #if PRINT_OUTPUT
        printf("input matrix\n");
        for(unsigned int i=0; i<num_segments*maxSegmentSize; ++i){
            for(unsigned int j=0; j<num_segments*maxSegmentSize; ++j){
                int xCoord = i/maxSegmentSize;
                int yCoord = j/maxSegmentSize;
                int index = xCoord*maxSegmentSize*maxSegmentSize*num_segments + yCoord*maxSegmentSize*maxSegmentSize + (i%maxSegmentSize)*maxSegmentSize + (j%maxSegmentSize);
                printf("%lf ", input_matrix[index]);
            }
            printf("\n");
        }
        #endif

        H2Opus_Real *h_S = (H2Opus_Real *)malloc(maxSegmentSize * num_segments * sizeof(H2Opus_Real));
        H2Opus_Real *h_U = (H2Opus_Real *)malloc(maxSegmentSize * maxSegmentSize * num_segments * sizeof(H2Opus_Real));
        H2Opus_Real *h_V = (H2Opus_Real *)malloc(maxSegmentSize * maxSegmentSize * num_segments * sizeof(H2Opus_Real));

        cudaEvent_t startSVD, stopSVD;
        cudaEventCreate(&startSVD);
        cudaEventCreate(&stopSVD);
        cudaEventRecord(startSVD);
        SVD(n, num_segments, input_matrix, maxSegmentSize, h_S, h_U, h_V);
        cudaEventRecord(stopSVD);
        cudaEventSynchronize(stopSVD);
        cudaEventElapsedTime(&timer_arr[6], startSVD, stopSVD);
        cudaEventDestroy(startSVD);
        cudaEventDestroy(stopSVD);
        cudaDeviceSynchronize();

        int* d_K;
        H2Opus_Real* d_S;
        H2Opus_Real* d_U;
        H2Opus_Real* d_V;
        cudaMalloc((void**) &d_S, maxSegmentSize * num_segments * sizeof(H2Opus_Real));
        cudaMalloc((void**) &d_U, maxSegmentSize * maxSegmentSize * num_segments * sizeof(H2Opus_Real));
        cudaMalloc((void**) &d_V, maxSegmentSize * maxSegmentSize * num_segments * sizeof(H2Opus_Real));
        cudaMalloc((void**) &d_K, num_segments * sizeof(int));
        cudaMemcpy(d_S, h_S, maxSegmentSize * num_segments * sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_U, h_U, maxSegmentSize*maxSegmentSize*num_segments * sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V, maxSegmentSize*maxSegmentSize*num_segments * sizeof(H2Opus_Real), cudaMemcpyHostToDevice);

        // printf("num segments: %d\n", num_segments);
        numThreadsPerBlock = maxSegmentSize; //TODO: make sure that bucket_size is less than 1024   
        numBlocks = num_segments;
        calcMemNeeded<<<numBlocks, numThreadsPerBlock>>> (n, maxSegmentSize, d_K, d_S, eps, d_offsets_sort, num_segments);
        cudaDeviceSynchronize();

        #if PRINT_OUTPUT
        printK<<<1, 1>>> (num_segments, d_K);
        cudaDeviceSynchronize();
        #endif

        int* d_scan_K;
        cudaErr = cudaMalloc((void**) &d_scan_K, num_segments*sizeof(int));
        if ( cudaErr != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr));
        }
        // printf("num segments: %d\n", num_segments);

        d_temp_storage = NULL;
        temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_K, d_scan_K, num_segments);
        // Allocate temporary storage
        // printf("temp storage bytes: %zu\n", temp_storage_bytes);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run exclusive prefix sum
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_K, d_scan_K, num_segments);
        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);

        int* totalMem = (int*)malloc(sizeof(int));
        int* d_totalMem;
        cudaMalloc((void**) &d_totalMem, sizeof(int));
        getTotalMem<<<1, 1>>> (d_totalMem, d_K, d_scan_K, num_segments);
        cudaDeviceSynchronize();
        cudaErr = cudaMemcpy(totalMem, d_totalMem, sizeof(int), cudaMemcpyDeviceToHost);
        timer_arr[7] = maxSegmentSize*maxSegmentSize*num_segments;
        timer_arr[8] = (*totalMem)*maxSegmentSize;
        // printf("max mem: %d\n", maxSegmentSize*maxSegmentSize*num_segments);
        // printf("total mem: %d\n", (*totalMem)*maxSegmentSize);
        if ( cudaErr != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr));
        }

        H2Opus_Real* d_U_tiled;
        H2Opus_Real* d_V_tiled;
        cudaErr = cudaMalloc((void**) &d_U_tiled, maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real));
        if ( cudaErr != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr));
        }
        cudaErr = cudaMalloc((void**) &d_V_tiled, maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real));
        if ( cudaErr != cudaSuccess )
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr));
        }

        // TODO: find the maximum rank using a max reduction
        dim3 d_numThreadsPerBlock(upper_power_of_two(maxSegmentSize), upper_power_of_two(maxSegmentSize));
        dim3 d_numBlocks(1, num_segments);
        tileMatrix<<<d_numBlocks, d_numThreadsPerBlock>>> (n, num_segments, maxSegmentSize, d_S, d_U, d_V, d_U_tiled, d_V_tiled, d_K, d_scan_K, segment);
        cudaDeviceSynchronize();
    }
    H2Opus_Real* d_expMatrix;
    cudaErr = cudaMalloc((void**) &d_expMatrix, num_segments*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess )
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr));
    }
    expandMatrix<<<d_numBlocks, d_numThreadsPerBlock>>> (num_segments, maxSegmentSize, d_K, d_scan_K, d_U_tiled, d_V_tiled, d_expMatrix);
    cudaDeviceSynchronize();

    #if 0
    printExpM<<<1, 1>>> (num_segments, maxSegmentSize, d_expMatrix, d_input_matrix);
    cudaDeviceSynchronize();
    #endif

    H2Opus_Real* d_error;
    H2Opus_Real* error = (H2Opus_Real*) malloc(sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_error, sizeof(H2Opus_Real));

    H2Opus_Real* d_tmp;
    H2Opus_Real* tmp = (H2Opus_Real*) malloc(sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_tmp, sizeof(H2Opus_Real));

    *error = 0;
    *tmp = 0;

    cudaErr = cudaMemcpy(d_error, error, sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
    if ( cudaErr != cudaSuccess )
    {
    printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr));
    }
    cudaErr = cudaMemcpy(d_tmp, tmp, sizeof(H2Opus_Real), cudaMemcpyHostToDevice);
    if ( cudaErr != cudaSuccess )
    {
    printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr));
    }
    numThreadsPerBlock = 1024;
    numBlocks = (num_segments*maxSegmentSize*maxSegmentSize + numThreadsPerBlock-1)/numThreadsPerBlock;
    calcError<<<numBlocks, numThreadsPerBlock>>> (num_segments, maxSegmentSize, d_expMatrix, d_input_matrix, d_error, d_tmp);
    cudaDeviceSynchronize();


    cudaMemcpy(error, d_error, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(tmp, d_tmp, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    // printf("error: %lf\n", sqrt(*error)/sqrt(*tmp));
    timer_arr[10] = sqrt(*error)/sqrt(*tmp);

    // cudaFree(d_offsets_sort);
    // free(input_matrix);

    H2Opus_Real* d_buffer_vector;
    H2Opus_Real* d_input_vector;
    H2Opus_Real* d_output_vector;
    H2Opus_Real* d_output_vector_org;
    cudaMalloc((void**) &d_buffer_vector, maxSegmentSize*num_segments*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_input_vector, maxSegmentSize*num_segments*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_output_vector, maxSegmentSize*num_segments*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_output_vector_org, maxSegmentSize*num_segments*sizeof(H2Opus_Real));

    numThreadsPerBlock = 1024;
    numBlocks = (num_segments*maxSegmentSize + numThreadsPerBlock-1)/numThreadsPerBlock;
    fillVector<<<numBlocks, numThreadsPerBlock>>> (num_segments, maxSegmentSize, d_input_vector, d_output_vector, d_output_vector_org);
    cudaDeviceSynchronize();

    numThreadsPerBlock = upper_power_of_two(maxSegmentSize);
    numBlocks = num_segments;

    cudaEvent_t startGEMV, stopGEMV;
    cudaEventCreate(&startGEMV);
    cudaEventCreate(&stopGEMV);
    cudaEventRecord(startGEMV);
    GEMV<<<numBlocks, numThreadsPerBlock>>> (num_segments, maxSegmentSize, d_K, d_scan_K, d_U_tiled, d_V_tiled, d_input_vector, d_output_vector, d_buffer_vector, segment);
    cudaDeviceSynchronize();
    cudaEventRecord(stopGEMV);
    cudaEventSynchronize(stopGEMV);
    float GEMV_time = 0;
    cudaEventElapsedTime(&GEMV_time, startGEMV, stopGEMV);
    cudaEventDestroy(startGEMV);
    cudaEventDestroy(stopGEMV);
    timer_arr[9] = GEMV_time;
    // printf("total time taken for GEMV: %f\n", GEMV_time);

    free(h_S);
    free(h_U);
    free(h_V);
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_K);
    cudaFree(d_scan_K);
    cudaFree(d_U_tiled);
    cudaFree(d_V_tiled);
    cudaFree(d_expMatrix);
    cudaFree(d_buffer_vector);
    cudaFree(d_input_vector);
    cudaFree(d_output_vector);
    cudaFree(d_output_vector_org);
    // }
    cudaDeviceSynchronize();
    cudaEventRecord(stopCode);
    cudaEventSynchronize(stopCode);
    float code_time = 0;
    float Code_time=0;
    cudaEventElapsedTime(&Code_time, startCode, stopCode);
    cudaEventDestroy(startCode);
    cudaEventDestroy(stopCode);
    printf("total time: %f\n", Code_time);
    printf("done\n");

    printCountersInFile(timer_arr);
}