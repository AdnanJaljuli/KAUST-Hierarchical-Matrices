#include "tlr_example.h"
#include "helperFunctions.h"
#include "helperKernels.cuh"

#include <iostream>
#include <utility>
#include <time.h>
#include <assert.h>
#include <math.h>

#define eps 1e-4
#define PRINT_OUTPUT 1
#define USE_SVD 0
#define DIVIDE_IN_HALF 0
using namespace std;

// TODO: generate pointcloud and copy values of the pointcloud to ptr on GPU
// TODO: fix makefile so main.cu depends on helperKerlens.cuh

int main(){
    int n = 21;
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

    unsigned long long segment_size = upper_power_of_two(n);
    unsigned long long  num_segments = 1;
    unsigned long long num_segments_reduce = num_segments*dim;

    int *d_offsets_sort;         // e.g., [0, 3, 3, 7]
    int *d_offsets_reduce;
    H2Opus_Real *d_keys_in;         // e.g., [8, 6, 7, 5, 3, 0, 9]
    H2Opus_Real *d_keys_out;        // e.g., [-, -, -, -, -, -, -]
    int  *d_values_in;       // e.g., [0, 1, 2, 3, 4, 5, 6]
    int  *d_values_out;      // e.g., [-, -, -, -, -, -, -]
    int *currDimArray_d;
    H2Opus_Real *d_reduce_in;
    H2Opus_Real *d_reduce_min_out;
    H2Opus_Real *d_reduce_max_out;
    int *d_temp;
    H2Opus_Real *d_span;
    int* d_span_offsets;
    cub::KeyValuePair<int, H2Opus_Real> *d_span_reduce_out;
    float* timer_arr = (float*)malloc(numTimers*sizeof(float));
    H2Opus_Real* d_input_matrix;

    cudaMalloc((void**) &d_temp, n*sizeof(int));
    cudaMalloc((void**) &d_offsets_sort, ((n+BUCKET_SIZE-1)/BUCKET_SIZE)*sizeof(int));
    cudaMalloc((void**) &d_offsets_reduce, (long long)((n+BUCKET_SIZE-1)/BUCKET_SIZE)*dim*(long long)sizeof(int));
    cudaMalloc((void**) &d_keys_in, n*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_keys_out, n*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_values_in, n*sizeof(int));
    cudaMalloc((void**) &d_values_out, n*sizeof(int));
    cudaMalloc((void**) &currDimArray_d, ((n+BUCKET_SIZE-1)/BUCKET_SIZE)*sizeof(int));
    cudaMalloc((void**) &d_reduce_in, (long long)n*(long long)dim*(long long)sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_reduce_min_out, (long long)((n+BUCKET_SIZE-1)/BUCKET_SIZE)*dim*(long long)sizeof(int));
    cudaMalloc((void**) &d_reduce_max_out, (long long)((n+BUCKET_SIZE-1)/BUCKET_SIZE)*dim*(long long)sizeof(int));
    cudaMalloc((void**) &d_span, (long long)((n+BUCKET_SIZE-1)/BUCKET_SIZE)*dim*(long long)sizeof(int));
    cudaMalloc((void**) &d_span_offsets, ((n+BUCKET_SIZE-1)/BUCKET_SIZE)*sizeof(int));
    cudaMalloc((void**) &d_span_reduce_out, ((n+BUCKET_SIZE-1)/BUCKET_SIZE)*sizeof(cub::KeyValuePair<int, H2Opus_Real>));
    cudaMalloc((void**) &d_input_matrix, (long long)n*(long long)n*(long long)sizeof(H2Opus_Real));

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (n+numThreadsPerBlock-1)/numThreadsPerBlock;
    initializeArrays<<<numBlocks, numThreadsPerBlock>>>(n, d_values_in, currDimArray_d);
    cudaDeviceSynchronize();

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cudaEvent_t startWhileLoop, stopWhileLoop;
    cudaEventCreate(&startWhileLoop);
    cudaEventCreate(&stopWhileLoop);
    cudaEventRecord(startWhileLoop);

    unsigned int iteration = 0;

    while(segment_size > BUCKET_SIZE) {
        for(unsigned int i=0; i<numTimers; ++i){
            timer_arr[i]=0;
        }

        numThreadsPerBlock = 1024;
        numBlocks = (num_segments+1+numThreadsPerBlock-1)/numThreadsPerBlock;

        cudaEvent_t startFillOffsets_k, stopFillOffsets_k;
        cudaEventCreate(&startFillOffsets_k);
        cudaEventCreate(&stopFillOffsets_k);
        cudaEventRecord(startFillOffsets_k);
        fillOffsetsArrays<<<numBlocks, numThreadsPerBlock>>>(n, dim, num_segments, segment_size, d_offsets_sort, d_offsets_reduce);
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
        findSpan<<<numBlocks, numThreadsPerBlock>>> (n, dim, num_segments, segment_size, d_reduce_min_out, d_reduce_max_out, d_span, d_span_offsets);
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
        fillCurDimArray<<<numBlocks, numThreadsPerBlock>>> (n, num_segments, currDimArray_d, d_span_reduce_out);
        cudaEventRecord(stopFillCurrDim_k);
        cudaEventSynchronize(stopFillCurrDim_k);
        cudaEventElapsedTime(&timer_arr[6], startfillCurrDim_k, stopFillCurrDim_k);
        cudaEventDestroy(startfillCurrDim_k);
        cudaEventDestroy(stopFillCurrDim_k);
        cudaDeviceSynchronize();

        // fill keys_in array
        numThreadsPerBlock = 1024;
        numBlocks = (n+numThreadsPerBlock-1)/numThreadsPerBlock;

        cudaEvent_t startfillKeysIn_k, stopKeysIn_k;
        cudaEventCreate(&startfillKeysIn_k);
        cudaEventCreate(&stopKeysIn_k);
        cudaEventRecord(startfillKeysIn_k);
        fillKeysIn<<<numBlocks, numThreadsPerBlock>>> (n, segment_size, d_keys_in, currDimArray_d, d_values_in, d_dataset);
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

        printCountersInFile(iteration, segment_size, num_segments, timer_arr);

        ++iteration;
        segment_size /= 2;
        num_segments = (n/segment_size) + 1;
        num_segments_reduce = num_segments*dim;
    }

    cudaEventRecord(stopWhileLoop);
    cudaEventSynchronize(stopWhileLoop);
    float whileLoop_time = 0;
    cudaEventElapsedTime(&whileLoop_time, startWhileLoop, stopWhileLoop);
    printf("total time taken for while loop: %f\n", whileLoop_time);
    cudaEventDestroy(startWhileLoop);
    cudaEventDestroy(stopWhileLoop);

    #if 0
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (n*n+numThreadsPerBlock-1)/numThreadsPerBlock;
    // TODO: rewrite padded numbers as INT_MAX
    generateInputMatrix<<<numBlocks, numThreadsPerBlock>>>(n, dim, d_values_in, d_input_matrix, d_dataset);
    cudaDeivceSynchronize();

    int nBlocks = (n+BUCKET_SIZE-1)/BUCKET_SIZE;

    int NRows=BUCKET_SIZE;
    int NCols=BUCKET_SIZE;

    H2Opus_Real* S = (H2Opus_Real*)malloc(min(Nrows, Ncols) * nBlocks*nBlocks * sizeof(H2Opus_Real));
    H2Opus_Real* U = (H2Opus_Real*)malloc(Nrows * Ncols * nBlocks*nBlocks * sizeof(H2Opus_Real));
    H2Opus_Real* V = (H2Opus_Real*)malloc(Nrows * Ncols * nBlocks*nBlocks * sizeof(H2Opus_Real));

    SVD(n, BUCKET_SIZE, d_input_matrix, S, U, V);

    int* d_K;
    H2Opus_Real* d_S;
    H2Opus_Real* d_U;
    H2Opus_Real* d_V;
    cudaMalloc((void**) &d_S, min(Nrows, Ncols) * nBlocks*nBlocks * sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_U, Nrows*Ncols * nBlocks*nBlocks * sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_V, Nrows*Ncols * nBlocks*nBlocks * sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_K, nBlocks*nBlocks * sizeof(int));

    cudaMemcpy(d_S, S, min(Nrows, Ncols) * nBlocks*nBlocks * sizeof(H2Opus_Real), cudaMemcpyHostToDevice);

    // TODO: call a kernel to figure out size of each tile
    numThreadsPerBlock = BUCKET_SIZE; //TODO: make sure that bucket_size is less than 1024
    numBlocks = (nBlocks*nBlocks+numThreadsPerBlock-1)/numThreadsPerBlock;
    calcMemNeeded<<<numBlocks, numThreadsPerBlock>>> (n, d_K, d_S, eps);
    cudaDeviceSynchronize();

    int* totalMem = (int*)malloc(sizeof(int));
    int* d_totalMem;
    cudaMalloc((void**) &d_totalMem, sizeof(int));
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_K, d_totalMem, nBlocks*nBlocks);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sum-reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_K, d_totalMem, nBlocks*nBlocks);

    cudaMemcpy(totalMem, d_totalMem, sizeof(int), cudaMemcpyDeviceToHost);

    // TODO: call a kernel to allocate the tiles, copy them from Us, Ss and Vs
    H2Opus_Real* d_STiled;
    H2Opus_Real* d_UTiled;
    H2Opus_Real* d_VTiled;
    cudaMalloc((void**) &d_STiles, totalMemNeeded * sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_UTiles, Nrows*totalMemNeeded * sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_VTiled, Nrows*totalMemNeeded * sizeof(H2Opus_Real));

    int* maxMem = (int*)malloc(sizeof(int));
    int* d_maxMem;
    cudaMalloc((void**) &d_maxMem, sizeof(int));
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_K, d_maxMem, nBlocks*nBlocks);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run max-reduction
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_K, d_maxMem, nBlocks*nBlocks);
    cudaMemcpy(maxMem, d_maxMem, sizeof(int), cudaMemcpyDeviceToHost);

    int* d_KScan;
    cudaMalloc((void**) &d_maxMem, nBlocks*nBlocks*sizeof(int));
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_K, d_out, nBlocks*nBlocks);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_K, d_out, nBlocks*nBlocks);

    numThreadsPerBlock = maxMem;
    numBlocks = (nBlocks*nBlocks+numThreadsPerBlock-1)/numThreadsPerBlock;
    tileMatrix<<<numBlocks, numThreadsPerBlock>>> (n, d_S, d_U, d_V, d_Siled, d_UTiled, d_VTiled, d_K, d_KScan);

    free(Ss);
    free(Us);
    free(Vs);
    #endif

    #if PRINT_OUTPUT
    int *index_map = (int*)malloc(n*sizeof(int));
    cudaMemcpy(index_map, d_values_in, n*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0; i<n; ++i){
        for(int j=0; j<dim; ++j){
            printf("%f ", pt_cloud.pts[j][index_map[i]]);
        }
        printf("\n");
    }
    free(index_map);
    #endif

    free(dataset);

    cudaFree(d_dataset);
    cudaFree(d_offsets_sort);
    cudaFree(d_offsets_reduce);
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_values_in);
    cudaFree(d_values_out);
    cudaFree(currDimArray_d);
    cudaFree(d_reduce_in);
    cudaFree(d_reduce_min_out);
    cudaFree(d_reduce_max_out);
    cudaFree(d_temp);
    cudaFree(d_span_reduce_out);
    cudaFree(d_span);
    cudaFree(d_span_offsets);
    cudaFree(d_input_matrix);
}