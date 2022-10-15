#include <algorithm>
#include <assert.h>
#include <inttypes.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <curand_kernel.h>
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

int main() {
    // TODO: read a batch of n*n TLR matrices from a file
    fstream myFile("batchedMatrix.txt", ios_base::in);

    int unitSize, segmentSize, batchSize, rank;
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
    int *d_ranks;
    double *d_U, *d_V;
    cudaMalloc((void**) &d_ranks, batchSize*unitSize*unitSize*sizeof(int));
    cudaMalloc((void**) &d_U, rankSum*segmentSize*sizeof(double));
    cudaMalloc((void**) &d_V, rankSum*segmentSize*sizeof(double));
    cudaMemcpy(d_ranks, ranks, batchSize*unitSize*unitSize*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, U, rankSum*segmentSize*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, rankSum*segmentSize*sizeof(double), cudaMemcpyHostToDevice);

    // TODO: generate random sampling vectors
    unsigned int samplingVectorsWidth = 16;
    double *d_samplingVectors;
    cudaMalloc((void**) &d_samplingVectors, samplingVectorsWidth*batchSize*segmentSize*unitSize*sizeof(double));
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (samplingVectorsWidth*batchSize*segmentSize*unitSize + numThreadsPerBlock - 1)/numThreadsPerBlock;
    generateSamplingVectors <<< numBlocks, numThreadsPerBlock >>> (d_samplingVectors, samplingVectorsWidth*batchSize*segmentSize*unitSize);

    // TODO: launch a kernel that takes as input the TLR matrices, sampling function and multiplies them and stores them in a matrix
    // TODO: launch a kernel that checks the correctness of the multiplication
}