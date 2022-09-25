
#ifndef KDTREECONSTRUCTION_CUH
#define KDTREECONSTRUCTION_CUH

#include <assert.h>
#include <ctype.h>
#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

// TODO: clean this file
static void createKDTree(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, unsigned int bucketSize, uint64_t *numSegments, DIVISION_METHOD divMethod, int* &d_valuesIn, int* &d_offsetsSort, H2Opus_Real* d_dataset, uint64_t maxNumSegments) {

    *numSegments = 1;
    uint64_t numSegmentsReduce = *numSegments*dimensionOfInputPoints;
    uint64_t segmentSize = upperPowerOfTwo(numberOfInputPoints);

    int *d_offsetsReduce;
    H2Opus_Real *d_keysIn;
    H2Opus_Real *d_keysOut;
    int  *d_valuesOut;
    int *d_currentDim;
    H2Opus_Real *d_reduceIn;
    H2Opus_Real *d_reduceMinOut;
    H2Opus_Real *d_reduceMaxOut;
    int *d_temp;
    H2Opus_Real *d_span;
    int* d_spanOffsets;
    cub::KeyValuePair<int, H2Opus_Real> *d_spanReduceOut;

    #if 0
    unsigned int largestSegmentSize = numberOfInputPoints;
    #endif

    cudaMalloc((void**) &d_offsetsReduce, (maxNumSegments*dimensionOfInputPoints + 1)*sizeof(int));
    cudaMalloc((void**) &d_keysIn, numberOfInputPoints*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_keysOut, numberOfInputPoints*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_valuesOut, numberOfInputPoints*sizeof(int));
    cudaMalloc((void**) &d_currentDim, (maxNumSegments + 1)*sizeof(int));
    cudaMalloc((void**) &d_reduceIn, numberOfInputPoints*dimensionOfInputPoints*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_reduceMinOut, (maxNumSegments + 1)*dimensionOfInputPoints*sizeof(int));
    cudaMalloc((void**) &d_reduceMaxOut, (maxNumSegments + 1)*dimensionOfInputPoints*sizeof(int));
    cudaMalloc((void**) &d_span, (maxNumSegments + 1)*dimensionOfInputPoints*sizeof(int));
    cudaMalloc((void**) &d_spanOffsets, (maxNumSegments + 1)*sizeof(int));
    cudaMalloc((void**) &d_spanReduceOut, (maxNumSegments + 1)*sizeof(cub::KeyValuePair<int, H2Opus_Real>));

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (numberOfInputPoints + numThreadsPerBlock - 1)/numThreadsPerBlock;
    initializeArrays<<<numBlocks, numThreadsPerBlock>>>(numberOfInputPoints, d_valuesIn, d_currentDim, maxNumSegments);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    unsigned int iteration = 0;

    while(segmentSize > bucketSize) {
        numThreadsPerBlock = 1024;
        numBlocks = (*numSegments + 1 + numThreadsPerBlock - 1)/numThreadsPerBlock;
        fillOffsets<<<numBlocks, numThreadsPerBlock>>>(numberOfInputPoints, dimensionOfInputPoints, *numSegments, segmentSize, d_offsetsSort, d_offsetsReduce);

        numThreadsPerBlock = 1024;
        numBlocks = (long long)((long long)numberOfInputPoints*(long long)dimensionOfInputPoints + numThreadsPerBlock-1)/numThreadsPerBlock;

        fillReductionArray<<<numBlocks, numThreadsPerBlock>>> (numberOfInputPoints, dimensionOfInputPoints, d_dataset, d_valuesIn, d_reduceIn);
        cudaDeviceSynchronize();

        d_temp_storage = NULL;
        temp_storage_bytes = 0;

        cub::DeviceSegmentedReduce::Min(d_temp_storage, temp_storage_bytes, d_reduceIn, d_reduceMinOut,
            numSegmentsReduce, d_offsetsReduce, d_offsetsReduce + 1);

        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceSegmentedReduce::Min(d_temp_storage, temp_storage_bytes, d_reduceIn, d_reduceMinOut,
            numSegmentsReduce, d_offsetsReduce, d_offsetsReduce + 1);
        cudaFree(d_temp_storage);

        d_temp_storage = NULL;
        temp_storage_bytes = 0;

        cub::DeviceSegmentedReduce::Max(d_temp_storage, temp_storage_bytes, d_reduceIn, d_reduceMaxOut,
            numSegmentsReduce, d_offsetsReduce, d_offsetsReduce + 1);

        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceSegmentedReduce::Max(d_temp_storage, temp_storage_bytes, d_reduceIn, d_reduceMaxOut,
            numSegmentsReduce, d_offsetsReduce, d_offsetsReduce + 1);
        cudaFree(d_temp_storage);

        numThreadsPerBlock = 1024;
        numBlocks = (*numSegments + numThreadsPerBlock - 1)/numThreadsPerBlock;

        findSpan<<<numBlocks, numThreadsPerBlock>>> (numberOfInputPoints, dimensionOfInputPoints, *numSegments, d_reduceMinOut, d_reduceMaxOut, d_span, d_spanOffsets);
        cudaDeviceSynchronize();

        d_temp_storage = NULL;
        temp_storage_bytes = 0;

        cub::DeviceSegmentedReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_span, d_spanReduceOut,
            *numSegments, d_spanOffsets, d_spanOffsets + 1);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run argmax-reduction
        cub::DeviceSegmentedReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_span, d_spanReduceOut,
            *numSegments, d_spanOffsets, d_spanOffsets + 1);
        cudaFree(d_temp_storage);

        numThreadsPerBlock = 1024;
        numBlocks = (*numSegments + numThreadsPerBlock - 1)/numThreadsPerBlock;
        fillCurrDim<<<numBlocks, numThreadsPerBlock>>> (numberOfInputPoints, *numSegments, d_currentDim, d_spanReduceOut);

        // fill keys_in array
        numThreadsPerBlock = 1024;
        numBlocks = (numberOfInputPoints + numThreadsPerBlock - 1)/numThreadsPerBlock;
        fillKeysIn<<<numBlocks, numThreadsPerBlock>>> (numberOfInputPoints, segmentSize, d_keysIn, d_currentDim, d_valuesIn, d_dataset);
        cudaDeviceSynchronize();

        d_temp_storage = NULL;
        temp_storage_bytes = 0;

        cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keysIn, d_keysOut, d_valuesIn, d_valuesOut,
        numberOfInputPoints, *numSegments, d_offsetsSort, d_offsetsSort + 1);

        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keysIn, d_keysOut, d_valuesIn, d_valuesOut,
        numberOfInputPoints, *numSegments, d_offsetsSort, d_offsetsSort + 1);
        cudaFree(d_temp_storage);

        d_temp = d_valuesIn;
        d_valuesIn = d_valuesOut;
        d_valuesOut = d_temp;
        ++iteration;

        segmentSize /= 2;
        *numSegments = (numberOfInputPoints + segmentSize - 1)/segmentSize;
        numSegmentsReduce = *numSegments*dimensionOfInputPoints;
    }


    fillOffsets<<<numBlocks, numThreadsPerBlock>>>(numberOfInputPoints, dimensionOfInputPoints, *numSegments, segmentSize, d_offsetsSort, d_offsetsReduce);

    cudaFree(d_offsetsReduce);
    cudaFree(d_keysIn);
    cudaFree(d_keysOut);
    cudaFree(d_valuesOut);
    cudaFree(d_currentDim);
    cudaFree(d_reduceIn);
    cudaFree(d_reduceMinOut);
    cudaFree(d_reduceMaxOut);
    cudaFree(d_spanReduceOut);
    cudaFree(d_span);
    cudaFree(d_spanOffsets);
}

#endif
