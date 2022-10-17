
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
        // samplingVectors[i] = curand_uniform(&s);
        samplingVectors[i] = 1;
    }
}

__global__ void fillBatchedPtrs(double** d_UBatchPtrs, double** d_VBatchPtrs, double* d_U, double* d_V, int* d_scanRanks, int batchSize, int segmentSize, int unitSize) {
    int sumRanks = 0;
    for(int i = 0; i < batchSize; ++i) {
        d_UBatchPtrs[i] = &d_U[sumRanks*segmentSize];
        d_VBatchPtrs[i] = &d_V[sumRanks*segmentSize];
        sumRanks = d_scanRanks[(i + 1)*unitSize*unitSize - 1];
    }
}

__global__ void fillBatchSegments(int *batchSegments, int unitSize, int batchSize) {
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < batchSize*unitSize*unitSize) {
        batchSegments[i] = i/(unitSize*unitSize);
    }
}

__global__ void printArray(int* output, int size) {
    for(int i = 0; i < size; ++i) {
        printf("%d\n", output[i]);
    }
}

__global__ void printOutput(double* output, int size) {
    for(int i = 0; i < size; ++i) {
        printf("%lf\n", output[i]);
    }
}

__global__ void denseMatrixSampling(int batchSize, int matrixDim, double* denseMatrix, double* denseMatrixOutput, double* samplingVectors, int samplingVectorDim) {
    // unsigned int batch = blockIdx.y/(matrixDim/32);
    // unsigned int blockInBatch = blockIdx.y%(matrixDim/32);
    if(threadIdx.x < samplingVectorDim) {
        double sum = 0;
        for(unsigned int i = 0; i < matrixDim; ++i) {
            sum += denseMatrix[i*matrixDim + blockIdx.y*32 + threadIdx.y]*samplingVectors[threadIdx.x*matrixDim + i];
        }
        denseMatrixOutput[threadIdx.x*matrixDim + blockIdx.y*32 + threadIdx.y] = sum;
    }
}

__global__ void compareResults(double* denseMatrixOutput, double* output, int size, double* error, double* tmp) {
    for(unsigned int i = 0; i < size; ++ i) {
        double x = denseMatrixOutput[i];
        double y = output[i];
        atomicAdd(tmp, x*x);
        atomicAdd(error, (x - y)*(x - y));
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
                myFile >> U[(rankSum - ranks[index])*segmentSize + k];
            }
            for(int k = 0; k < ranks[index]*segmentSize; ++k) {
                myFile >> V[(rankSum - ranks[index])*segmentSize + k];
            }
        }
    }

    int *d_ranks, *d_scanRanks;
    double *d_U, *d_V;
    cudaMalloc((void**) &d_ranks, batchSize*unitSize*unitSize*sizeof(int));
    cudaMalloc((void**) &d_scanRanks, batchSize*unitSize*unitSize*sizeof(int));
    cudaMalloc((void**) &d_U, rankSum*segmentSize*sizeof(double));
    cudaMalloc((void**) &d_V, rankSum*segmentSize*sizeof(double));
    cudaMemcpy(d_ranks, ranks, batchSize*unitSize*unitSize*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, U, rankSum*segmentSize*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, rankSum*segmentSize*sizeof(double), cudaMemcpyHostToDevice);

    int *d_batchSegments;
    cudaMalloc((void**) &d_batchSegments, batchSize*unitSize*unitSize*sizeof(int));
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (batchSize*unitSize*unitSize + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillBatchSegments <<< numBlocks, numThreadsPerBlock >>> (d_batchSegments, unitSize, batchSize);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSumByKey(d_temp_storage, temp_storage_bytes, d_batchSegments, d_ranks, d_scanRanks, batchSize*unitSize*unitSize);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSumByKey(d_temp_storage, temp_storage_bytes, d_batchSegments, d_ranks, d_scanRanks, batchSize*unitSize*unitSize);

    double **d_UBatchPtrs, **d_VBatchPtrs;
    cudaMalloc((void**) &d_UBatchPtrs, batchSize*sizeof(double*));
    cudaMalloc((void**) &d_VBatchPtrs, batchSize*sizeof(double*));

    numBlocks = 1;
    numThreadsPerBlock = 1;
    fillBatchedPtrs <<< numBlocks, numThreadsPerBlock >>> (d_UBatchPtrs, d_VBatchPtrs, d_U, d_V, d_scanRanks, batchSize, segmentSize, unitSize);

    // generate random sampling vectors
    unsigned int samplingVectorsWidth = 32;
    double *d_output;
    double *d_bufferMemory;
    double *d_samplingVectors;
    cudaMalloc((void**) &d_output, samplingVectorsWidth*batchSize*segmentSize*unitSize*sizeof(double));
    cudaMalloc((void**) &d_bufferMemory, samplingVectorsWidth*batchSize*segmentSize*unitSize*sizeof(double));
    cudaMalloc((void**) &d_samplingVectors, samplingVectorsWidth*batchSize*segmentSize*unitSize*sizeof(double));
    
    numThreadsPerBlock = 1024;
    numBlocks = (samplingVectorsWidth*batchSize*segmentSize*unitSize + numThreadsPerBlock - 1)/numThreadsPerBlock;
    generateSamplingVectors <<< numBlocks, numThreadsPerBlock >>> (d_samplingVectors, samplingVectorsWidth*batchSize*segmentSize*unitSize);

    // launch a kernel that takes as input the TLR matrices, sampling function and multiplies them and stores them in a matrix
    dim3 m_numThreadsPerBlock(32, 32);
    dim3 m_numBlocks(batchSize*unitSize, 1);
    batchedSampling <<< m_numBlocks, m_numThreadsPerBlock >>> (segmentSize, batchSize, unitSize, d_UBatchPtrs, d_VBatchPtrs, d_scanRanks, d_samplingVectors, samplingVectorsWidth, d_output, d_bufferMemory);

    // read the batched dense tiles form the txt file
    fstream denseMatrixFile("denseMatrix.txt", ios_base::in);
    double* denseMatrix = (double*)malloc(batchSize*unitSize*segmentSize*unitSize*segmentSize*sizeof(double));
    for(unsigned int batch = 0; batch < batchSize; ++batch) {
        for(unsigned int row = 0; row < unitSize*segmentSize; ++row) {
            for(unsigned int col = 0; col < unitSize*segmentSize; ++col) {
                denseMatrixFile >> denseMatrix[batch*unitSize*segmentSize*unitSize*segmentSize + col*unitSize*segmentSize + row];
            }
        }
    }

    double* d_denseMatrix;
    cudaMalloc((void**) &d_denseMatrix, batchSize*unitSize*segmentSize*unitSize*segmentSize*sizeof(double));    
    cudaMemcpy(d_denseMatrix, denseMatrix, batchSize*unitSize*segmentSize*unitSize*segmentSize*sizeof(double), cudaMemcpyHostToDevice);
    double *d_denseMatrixOutput;
    cudaMalloc((void**) &d_denseMatrixOutput, samplingVectorsWidth*batchSize*segmentSize*unitSize*sizeof(double));

    // multiply the dense matrix by the sampling vectors
    dim3 dm_numThreadsPerBlock(samplingVectorsWidth, 32);
    dim3 dm_numBlocks(1, (unitSize*segmentSize)/32);
    denseMatrixSampling <<< dm_numBlocks, dm_numThreadsPerBlock >>> (batchSize, unitSize*segmentSize, d_denseMatrix, d_denseMatrixOutput, d_samplingVectors, samplingVectorsWidth);

    // compare the results
    double *d_error, *d_tmp;
    cudaMalloc((void**) &d_error, sizeof(double));
    cudaMalloc((void**) &d_tmp, sizeof(double));
    cudaMemset(d_error, 0, sizeof(double));
    cudaMemset(d_tmp, 0, sizeof(double));
    compareResults <<< 1, 1 >>> (d_denseMatrixOutput, d_output, samplingVectorsWidth*batchSize*segmentSize*unitSize, d_error, d_tmp);
    double h_error;
    double h_tmp;
    cudaMemcpy(&h_error, d_error, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_tmp, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);
    printf("error in matrix: %lf\n", sqrt(h_error)/sqrt(h_tmp));
    cudaFree(d_tmp);
    cudaFree(d_error);
    cudaDeviceSynchronize();
}