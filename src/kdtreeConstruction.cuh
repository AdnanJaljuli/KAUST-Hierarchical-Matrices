#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stddef.h>
#include <cub/cub.cuh>
#include <assert.h>
#include <curand_kernel.h>

void createKDTree(int n, int dim, int bucket_size, uint64_t &num_segments, DIVISION_METHOD div_method, int* &d_values_in, int* &d_offsets_sort, H2Opus_Real* d_dataset, int max_num_segments){
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

    bool workDone= false;
    bool* d_workDone;
    uint64_t* d_bit_vector;
    short int* d_popc_bit_vector;
    short int* d_popc_scan;
    unsigned int* d_new_num_segments;
    unsigned int* new_num_segments;

    int* A;
    int* B;
    int* d_bin_search_output;
    int* d_thrust_v_bin_search_output;
    int* d_input_search;
    int* d_aux_offsets_sort;

    unsigned int largest_segment_size = n;

    // TODO: fix memory allocated
    
    cudaError_t cudaErr = cudaMalloc((void**) &d_offsets_reduce, (long long)((max_num_segments*dim + 1)*(long long)sizeof(int)));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_keys_in, n*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_keys_out, n*sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    
    cudaErr = cudaMalloc((void**) &d_values_out, n*sizeof(int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_curr_dim, (max_num_segments + 1)*sizeof(int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_reduce_in, (long long)n*(long long)dim*(long long)sizeof(H2Opus_Real));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_reduce_min_out, (long long)((max_num_segments+1)*dim)*(long long)sizeof(int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_reduce_max_out, (long long)((max_num_segments+1)*dim)*(long long)sizeof(int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_span, (long long)((max_num_segments+1)*dim)*(long long)sizeof(int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_span_offsets, (max_num_segments + 1)*sizeof(int));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    cudaErr = cudaMalloc((void**) &d_span_reduce_out, (max_num_segments+1)*sizeof(cub::KeyValuePair<int, H2Opus_Real>));
    if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }

    if(div_method == DIVIDE_IN_HALF){
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

    if(div_method != POWER_OF_TWO_ON_LEFT){
        cudaErr = cudaMalloc((void**) &d_aux_offsets_sort, (max_num_segments + 1) * sizeof(int));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaErr = cudaMalloc((void**) &A, (max_num_segments + 1)*sizeof(int));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaErr = cudaMalloc((void**) &B, n*sizeof(int));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaErr = cudaMalloc((void**) &d_bin_search_output, n*sizeof(int));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
        cudaErr = cudaMalloc((void**) &d_input_search, n*sizeof(int));
        if ( cudaErr != cudaSuccess ){ printf("CUDA Error: %s\n", cudaGetErrorString(cudaErr)); }
    }

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (n+numThreadsPerBlock-1)/numThreadsPerBlock;

    if(div_method != POWER_OF_TWO_ON_LEFT){
        initializeArrays<<<numBlocks, numThreadsPerBlock>>>(n, dim, d_values_in, d_curr_dim, d_offsets_sort, d_offsets_reduce, d_input_search, max_num_segments);
    } else {
        initializeArrays<<<numBlocks, numThreadsPerBlock>>>(n, d_values_in, d_curr_dim, max_num_segments);
    }
    cudaDeviceSynchronize();

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    unsigned int iteration = 0;

    #if 0
    if(div_method == DIVIDE_IN_HALF){
        while(!workDone)
    } else if (div_method == POWER_OF_TWO_ON_LEFT) {
        while(segment_size > bucket_size)
    } else {
        while(largest_segment_size > bucket_size)
    }
    #endif

    while(segment_size > bucket_size)
    {
        numThreadsPerBlock = 1024;
        numBlocks = (num_segments+1+numThreadsPerBlock-1)/numThreadsPerBlock;
        if(div_method == POWER_OF_TWO_ON_LEFT){
            fillOffsets<<<numBlocks, numThreadsPerBlock>>>(n, dim, num_segments, segment_size, d_offsets_sort, d_offsets_reduce);
        }
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

        if(div_method == DIVIDE_IN_HALF){
            fillCurrDim<<<numBlocks, numThreadsPerBlock>>> (n, num_segments, d_curr_dim, d_span_reduce_out, d_bit_vector);
        } else {
            fillCurrDim<<<numBlocks, numThreadsPerBlock>>> (n, num_segments, d_curr_dim, d_span_reduce_out);
        }
        cudaDeviceSynchronize();

        // fill keys_in array
        numThreadsPerBlock = 1024;
        numBlocks = (n+numThreadsPerBlock-1)/numThreadsPerBlock;

        if(div_method != POWER_OF_TWO_ON_LEFT){
            thrust::device_ptr<int> A = thrust::device_pointer_cast((int *)d_offsets_sort), B = thrust::device_pointer_cast((int *)d_input_search);
            thrust::device_vector<int> d_bin_search_output(n);
            thrust::upper_bound(A, A + num_segments + 1, B, B + n, d_bin_search_output.begin(), thrust::less<int>());
            d_thrust_v_bin_search_output = thrust::raw_pointer_cast(&d_bin_search_output[0]);
            fillKeysIn<<<numBlocks, numThreadsPerBlock>>> (n, d_keys_in, d_curr_dim, d_values_in, d_dataset, d_offsets_sort, d_thrust_v_bin_search_output);
        } else {
            fillKeysIn<<<numBlocks, numThreadsPerBlock>>> (n, segment_size, d_keys_in, d_curr_dim, d_values_in, d_dataset);
        }
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

        // TODO: fix cuda-memcheck bug in divide_in_half
        if(div_method == DIVIDE_IN_HALF){
            numThreadsPerBlock = 1024;
            numBlocks = (num_segments+numThreadsPerBlock-1)/numThreadsPerBlock;
            fillBitVector<<<numBlocks, numThreadsPerBlock>>>(num_segments, d_bit_vector, d_offsets_sort, bucket_size);
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
            fillOffsetsSort<<<numBlocks, numThreadsPerBlock>>>(n, dim, num_segments, d_offsets_sort, d_aux_offsets_sort, d_bit_vector, d_popc_scan, d_new_num_segments, d_workDone, bucket_size);
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

        } else if(div_method == POWER_OF_TWO_ON_LEFT){
            segment_size /= 2;
            num_segments = (n+segment_size-1)/segment_size;

        } else if(div_method == FULL_TREE){
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
        }

        num_segments_reduce = num_segments*dim;
    }

    cudaDeviceSynchronize();

    #if 0
    int *index_map = (int*)malloc(n*sizeof(int));
    cudaMemcpy(index_map, d_values_in, n*sizeof(int), cudaMemcpyDeviceToHost);
    printf("index max\n");
    for(int i=0; i<n; ++i){
        printf("%d ", index_map[i]);
    }
    printf("\n");
    free(index_map);
    #endif

    if(div_method == POWER_OF_TWO_ON_LEFT){
        printf("num segments :%d\n", num_segments);
        printf("segment size :%d\n", segment_size);
        fillOffsets<<<numBlocks, numThreadsPerBlock>>>(n, dim, num_segments, segment_size, d_offsets_sort, d_offsets_reduce);
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
    if(div_method != POWER_OF_TWO_ON_LEFT){
        cudaFree(d_aux_offsets_sort);
        cudaFree(A);
        cudaFree(B);
        cudaFree(d_bin_search_output);
        cudaFree(d_input_search);
    }
    if(div_method == DIVIDE_IN_HALF){
        free(new_num_segments);
        cudaFree(d_bit_vector);
        cudaFree(d_popc_bit_vector);
        cudaFree(d_popc_scan);
        cudaFree(d_new_num_segments);
        cudaFree(d_workDone);
    }
}