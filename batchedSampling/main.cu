#include "batchedSampling.cuh"

#include <algorithm>
#include <assert.h>
#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <inttypes.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <typeinfo>
#include <utility>
#include <bits/stdc++.h>
using namespace std;

__global__ void generateSamplingVectors(double *samplingVectors, int size) {
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < size) {
        unsigned int seed = i;
        curandState s;
        curand_init(seed, 0, 0, &s);
        samplingVectors[i] = curand_uniform(&s);
    }
}

__global__ void fillBatchedPtrs(double** d_UBatchPtrs, double** d_VBatchPtrs, double* d_U, double* d_V, int* d_scanRanks, int batchSize, int segmentSize, int unitSize) {
    for(int i = 0; i < batchSize; ++i) {
        d_UBatchPtrs[i] = &d_U[d_scanRanks[i*unitSize*unitSize]*segmentSize];
        d_VBatchPtrs[i] = &d_U[d_scanRanks[i*unitSize*unitSize]*segmentSize];
    }
}

int main() {
    // read a batch of n*n TLR matrices from a file
    fstream myFile("batchedMatrix.txt", ios_base::in);
    int unitSize, segmentSize, batchSize;
    myFile >> unitSize >> segmentSize >> batchSize;
    printf("%d %d %d\n", unitSize, segmentSize, batchSize);
    int *ranks = (int*)malloc(batchSize*unitSize*unitSize*sizeof(int));
    int rankSum = 0;
    double *U, *V;
    V = (double*)malloc(0);
    U = (double*)malloc(0);

    for(int i = 0; i < batchSize; ++i) {
        for(int j = 0; j < unitSize*unitSize; ++j) {
            int index = i*unitSize*unitSize + j;
            myFile >> ranks[index];
            rankSum += ranks[index];
            U = (double*)realloc(U, rankSum*segmentSize*sizeof(double));
            V = (double*)realloc(V, rankSum*segmentSize*sizeof(double));

            for(int k = 0; k < ranks[index]*segmentSize; ++k) {
                myFile >> U[rankSum - ranks[index] + k];
            }
            for(int k = 0; k < ranks[index]*segmentSize; ++k) {
                myFile >> V[rankSum -ranks[index] + k];
            }
        }
    }

    // TODO: make scanRanks a double pointer. make the scan of each batch unit independent of the other batches
    int *d_ranks, *d_scanRanks;
    double *d_U, *d_V;
    cudaMalloc((void**) &d_ranks, batchSize*unitSize*unitSize*sizeof(int));
    cudaMalloc((void**) &d_scanRanks, batchSize*unitSize*unitSize*sizeof(int));
    cudaMalloc((void**) &d_U, rankSum*segmentSize*sizeof(double));
    cudaMalloc((void**) &d_V, rankSum*segmentSize*sizeof(double));
    cudaMemcpy(d_ranks, ranks, batchSize*unitSize*unitSize*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, U, rankSum*segmentSize*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, rankSum*segmentSize*sizeof(double), cudaMemcpyHostToDevice);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_ranks, d_scanRanks, batchSize*unitSize*unitSize);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_ranks, d_scanRanks, batchSize*unitSize*unitSize);

    double **d_UBatchPtrs, **d_VBatchPtrs;
    cudaMalloc((void**) &d_UBatchPtrs, batchSize*sizeof(double*));
    cudaMalloc((void**) &d_VBatchPtrs, batchSize*sizeof(double*));

    unsigned int numBlocks = 1;
    unsigned int numThreadsPerBlock = 1;
    fillBatchedPtrs <<< numBlocks, numThreadsPerBlock >>> (d_UBatchPtrs, d_VBatchPtrs, d_U, d_V, d_scanRanks, batchSize, segmentSize, unitSize);

    // generate random sampling vectors
    unsigned int samplingVectorsWidth = 16;
    double *d_output;
    double *d_samplingVectors;
    cudaMalloc((void**) &d_output, samplingVectorsWidth*batchSize*segmentSize*unitSize*sizeof(double));
    cudaMalloc((void**) &d_samplingVectors, samplingVectorsWidth*batchSize*segmentSize*unitSize*sizeof(double));
    numThreadsPerBlock = 1024;
    numBlocks = (samplingVectorsWidth*batchSize*segmentSize*unitSize + numThreadsPerBlock - 1)/numThreadsPerBlock;
    generateSamplingVectors <<< numBlocks, numThreadsPerBlock >>> (d_samplingVectors, samplingVectorsWidth*batchSize*segmentSize*unitSize);

    // launch a kernel that takes as input the TLR matrices, sampling function and multiplies them and stores them in a matrix
    numThreadsPerBlock = 1024;
    numBlocks = batchSize*unitSize;
    batchedSampling <<< numBlocks, numThreadsPerBlock >>> (segmentSize, batchSize, unitSize, d_UBatchPtrs, d_VBatchPtrs, d_scanRanks, d_samplingVectors, samplingVectorsWidth);
    cudaDeviceSynchronize();

    // TODO: launch a kernel that checks the correctness of the multiplication
}