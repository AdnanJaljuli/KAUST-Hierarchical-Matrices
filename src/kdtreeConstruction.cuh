
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

static void createKDTree(unsigned int numberOfInputPoints, unsigned int dimensionOfInputPoints, uint64_t numSegments, unsigned int bucketSize, DIVISION_METHOD divMethod, int* &d_valuesIn, int* &d_offsetsSort, H2Opus_Real* d_pointCloud) {

    uint64_t currentNumSegments = 1;
    uint64_t numSegmentsReduce = currentNumSegments*dimensionOfInputPoints;
    uint64_t segmentSize = upperPowerOfTwo(numberOfInputPoints);

    int *d_offsetsReduce;
    H2Opus_Real *d_keysIn;
    H2Opus_Real *d_keysOut;
    int  *d_valuesOut;
    int *d_currentDim;
    H2Opus_Real *d_reduceIn;
    H2Opus_Real *d_reduceMinOut;
    H2Opus_Real *d_reduceMaxOut;
    H2Opus_Real *d_span;
    int* d_spanOffsets;
    cub::KeyValuePair<int, H2Opus_Real> *d_spanReduceOut;

    cudaMalloc((void**) &d_offsetsReduce, (numSegments*dimensionOfInputPoints + 1)*sizeof(int));
    cudaMalloc((void**) &d_keysIn, numberOfInputPoints*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_keysOut, numberOfInputPoints*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_valuesOut, numberOfInputPoints*sizeof(int));
    cudaMalloc((void**) &d_currentDim, (numSegments + 1)*sizeof(int));
    cudaMalloc((void**) &d_reduceIn, numberOfInputPoints*dimensionOfInputPoints*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_reduceMinOut, (numSegments + 1)*dimensionOfInputPoints*sizeof(int));
    cudaMalloc((void**) &d_reduceMaxOut, (numSegments + 1)*dimensionOfInputPoints*sizeof(int));
    cudaMalloc((void**) &d_span, (numSegments + 1)*dimensionOfInputPoints*sizeof(int));
    cudaMalloc((void**) &d_spanOffsets, (numSegments + 1)*sizeof(int));
    cudaMalloc((void**) &d_spanReduceOut, (numSegments + 1)*sizeof(cub::KeyValuePair<int, H2Opus_Real>));

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (numberOfInputPoints + numThreadsPerBlock - 1)/numThreadsPerBlock;
    initializeArrays<<<numBlocks, numThreadsPerBlock>>>(numberOfInputPoints, d_valuesIn, d_currentDim, numSegments);

    void *d_tempStorage = NULL;
    size_t tempStorageBytes = 0;
    int *d_temp;

    while (segmentSize > bucketSize) {
        numThreadsPerBlock = 1024;
        numBlocks = (currentNumSegments + 1 + numThreadsPerBlock - 1)/numThreadsPerBlock;
        fillOffsets<<<numBlocks, numThreadsPerBlock>>>(numberOfInputPoints, dimensionOfInputPoints, currentNumSegments, segmentSize, d_offsetsSort, d_offsetsReduce);

        numThreadsPerBlock = 1024;
        numBlocks = (numberOfInputPoints*dimensionOfInputPoints + numThreadsPerBlock - 1)/numThreadsPerBlock;

        fillReductionArray<<<numBlocks, numThreadsPerBlock>>> (numberOfInputPoints, dimensionOfInputPoints, d_pointCloud, d_valuesIn, d_reduceIn);
        cudaDeviceSynchronize();

        d_tempStorage = NULL;
        tempStorageBytes = 0;
        cub::DeviceSegmentedReduce::Min(d_tempStorage, tempStorageBytes, d_reduceIn, d_reduceMinOut,
            numSegmentsReduce, d_offsetsReduce, d_offsetsReduce + 1);
        cudaMalloc(&d_tempStorage, tempStorageBytes);
        cub::DeviceSegmentedReduce::Min(d_tempStorage, tempStorageBytes, d_reduceIn, d_reduceMinOut,
            numSegmentsReduce, d_offsetsReduce, d_offsetsReduce + 1);
        cudaFree(d_tempStorage);

        d_tempStorage = NULL;
        tempStorageBytes = 0;
        cub::DeviceSegmentedReduce::Max(d_tempStorage, tempStorageBytes, d_reduceIn, d_reduceMaxOut,
            numSegmentsReduce, d_offsetsReduce, d_offsetsReduce + 1);
        cudaMalloc(&d_tempStorage, tempStorageBytes);
        cub::DeviceSegmentedReduce::Max(d_tempStorage, tempStorageBytes, d_reduceIn, d_reduceMaxOut,
            numSegmentsReduce, d_offsetsReduce, d_offsetsReduce + 1);
        cudaFree(d_tempStorage);

        numThreadsPerBlock = 1024;
        numBlocks = (currentNumSegments + numThreadsPerBlock - 1)/numThreadsPerBlock;
        findSpan<<<numBlocks, numThreadsPerBlock>>> (numberOfInputPoints, dimensionOfInputPoints, currentNumSegments, d_reduceMinOut, d_reduceMaxOut, d_span, d_spanOffsets);
        cudaDeviceSynchronize();

        d_tempStorage = NULL;
        tempStorageBytes = 0;
        cub::DeviceSegmentedReduce::ArgMax(d_tempStorage, tempStorageBytes, d_span, d_spanReduceOut,
            currentNumSegments, d_spanOffsets, d_spanOffsets + 1);
        cudaMalloc(&d_tempStorage, tempStorageBytes);
        cub::DeviceSegmentedReduce::ArgMax(d_tempStorage, tempStorageBytes, d_span, d_spanReduceOut,
            currentNumSegments, d_spanOffsets, d_spanOffsets + 1);
        cudaFree(d_tempStorage);

        numThreadsPerBlock = 1024;
        numBlocks = (currentNumSegments + numThreadsPerBlock - 1)/numThreadsPerBlock;
        fillCurrDim<<<numBlocks, numThreadsPerBlock>>> (numberOfInputPoints, currentNumSegments, d_currentDim, d_spanReduceOut);

        numThreadsPerBlock = 1024;
        numBlocks = (numberOfInputPoints + numThreadsPerBlock - 1)/numThreadsPerBlock;
        fillKeysIn<<<numBlocks, numThreadsPerBlock>>> (numberOfInputPoints, segmentSize, d_keysIn, d_currentDim, d_valuesIn, d_pointCloud);
        cudaDeviceSynchronize();

        d_tempStorage = NULL;
        tempStorageBytes = 0;

        cub::DeviceSegmentedRadixSort::SortPairs(d_tempStorage, tempStorageBytes,
        d_keysIn, d_keysOut, d_valuesIn, d_valuesOut,
        numberOfInputPoints, currentNumSegments, d_offsetsSort, d_offsetsSort + 1);
        cudaMalloc(&d_tempStorage, tempStorageBytes);
        cub::DeviceSegmentedRadixSort::SortPairs(d_tempStorage, tempStorageBytes,
        d_keysIn, d_keysOut, d_valuesIn, d_valuesOut,
        numberOfInputPoints, currentNumSegments, d_offsetsSort, d_offsetsSort + 1);
        cudaFree(d_tempStorage);

        d_temp = d_valuesIn;
        d_valuesIn = d_valuesOut;
        d_valuesOut = d_temp;
        segmentSize /= 2;
        currentNumSegments = (numberOfInputPoints + segmentSize - 1)/segmentSize;
        numSegmentsReduce = currentNumSegments*dimensionOfInputPoints;
    }

    fillOffsets<<<numBlocks, numThreadsPerBlock>>>(numberOfInputPoints, dimensionOfInputPoints, currentNumSegments, segmentSize, d_offsetsSort, d_offsetsReduce);

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