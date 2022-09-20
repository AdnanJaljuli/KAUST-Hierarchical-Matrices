#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stddef.h>
#include <cub/cub.cuh>
#include <assert.h>
#include <curand_kernel.h>

// TODO: clean this file
void createKDTree(int n, int dim, int bucket_size, uint64_t &num_segments, DIVISION_METHOD div_method, int* &d_values_in, int* &d_offsets_sort, H2Opus_Real* d_dataset, int max_num_segments){
    cudaMalloc((void**) &d_values_in, n*sizeof(int));
    cudaMalloc((void**) &d_offsets_sort, (max_num_segments + 1)*sizeof(int));

    uint64_t num_segments_reduce = num_segments*dim;
    uint64_t segment_size = upper_power_of_two(n);

    int *d_offsets_reduce;
    H2Opus_Real *d_keys_in;
    H2Opus_Real *d_keys_out;
    int  *d_values_out;
    int *d_curr_dim;
    H2Opus_Real *d_reduce_in;
    H2Opus_Real *d_reduce_min_out;
    H2Opus_Real *d_reduce_max_out;
    int *d_temp;
    H2Opus_Real *d_span;
    int* d_span_offsets;
    cub::KeyValuePair<int, H2Opus_Real> *d_span_reduce_out;

    unsigned int largest_segment_size = n;

    // TODO: fix memory allocated
    cudaMalloc((void**) &d_offsets_reduce, (long long)((max_num_segments*dim + 1)*(long long)sizeof(int)));
    cudaMalloc((void**) &d_keys_in, n*sizeof(H2Opus_Real));    
    cudaMalloc((void**) &d_keys_out, n*sizeof(H2Opus_Real));    
    cudaMalloc((void**) &d_values_out, n*sizeof(int));    
    cudaMalloc((void**) &d_curr_dim, (max_num_segments + 1)*sizeof(int));    
    cudaMalloc((void**) &d_reduce_in, (long long)n*(long long)dim*(long long)sizeof(H2Opus_Real));    
    cudaMalloc((void**) &d_reduce_min_out, (long long)((max_num_segments+1)*dim)*(long long)sizeof(int));    
    cudaMalloc((void**) &d_reduce_max_out, (long long)((max_num_segments+1)*dim)*(long long)sizeof(int));    
    cudaMalloc((void**) &d_span, (long long)((max_num_segments+1)*dim)*(long long)sizeof(int));    
    cudaMalloc((void**) &d_span_offsets, (max_num_segments + 1)*sizeof(int));    
    cudaMalloc((void**) &d_span_reduce_out, (max_num_segments+1)*sizeof(cub::KeyValuePair<int, H2Opus_Real>));    

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (n + numThreadsPerBlock - 1)/numThreadsPerBlock;
    initializeArrays<<<numBlocks, numThreadsPerBlock>>>(n, d_values_in, d_curr_dim, max_num_segments);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    unsigned int iteration = 0;

    while(segment_size > bucket_size)
    {
        numThreadsPerBlock = 1024;
        numBlocks = (num_segments+1+numThreadsPerBlock-1)/numThreadsPerBlock;
        fillOffsets<<<numBlocks, numThreadsPerBlock>>>(n, dim, num_segments, segment_size, d_offsets_sort, d_offsets_reduce);

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
        cudaFree(d_temp_storage);

        d_temp_storage = NULL;
        temp_storage_bytes = 0;

        cub::DeviceSegmentedReduce::Max(d_temp_storage, temp_storage_bytes, d_reduce_in, d_reduce_max_out,
            num_segments_reduce, d_offsets_reduce, d_offsets_reduce + 1);

        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceSegmentedReduce::Max(d_temp_storage, temp_storage_bytes, d_reduce_in, d_reduce_max_out,
            num_segments_reduce, d_offsets_reduce, d_offsets_reduce + 1);
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
        cudaFree(d_temp_storage);

        numThreadsPerBlock = 1024;
        numBlocks = (num_segments+numThreadsPerBlock-1)/numThreadsPerBlock;
        fillCurrDim<<<numBlocks, numThreadsPerBlock>>> (n, num_segments, d_curr_dim, d_span_reduce_out);

        // fill keys_in array
        numThreadsPerBlock = 1024;
        numBlocks = (n+numThreadsPerBlock-1)/numThreadsPerBlock;
        fillKeysIn<<<numBlocks, numThreadsPerBlock>>> (n, segment_size, d_keys_in, d_curr_dim, d_values_in, d_dataset);
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
        cudaFree(d_temp_storage);

        d_temp = d_values_in;
        d_values_in = d_values_out;
        d_values_out = d_temp;
        ++iteration;

        segment_size /= 2;
        num_segments = (n+segment_size-1)/segment_size;
        num_segments_reduce = num_segments*dim;
    }


    fillOffsets<<<numBlocks, numThreadsPerBlock>>>(n, dim, num_segments, segment_size, d_offsets_sort, d_offsets_reduce);

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
}