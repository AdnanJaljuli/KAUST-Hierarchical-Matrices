
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
        // samplingVectors[i] = 1;
    }
}

__global__ void fillBatchedPtrs(double** d_UBatchPtrs, double** d_VBatchPtrs, double* d_U, double* d_V, int* d_scanRanks, int batchSize, int segmentSize, int batchUnitSize) {
    int sumRanks = 0;
    for(int i = 0; i < batchSize; ++i) {
        d_UBatchPtrs[i] = &d_U[sumRanks*segmentSize];
        d_VBatchPtrs[i] = &d_V[sumRanks*segmentSize];
        sumRanks += d_scanRanks[(i + 1)*batchUnitSize*batchUnitSize - 1];
    }
}

__global__ void fillBatchSegments(int *batchSegments, int batchUnitSize, int batchSize) {
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < batchSize*batchUnitSize*batchUnitSize) {
        batchSegments[i] = i/(batchUnitSize*batchUnitSize);
    }
}

__global__ void fillSamplesBatch(int* samplesBatch, int batchSize) {
    samplesBatch[0] = 57;
    samplesBatch[1] = 24;
}

__global__ void fillScanRanksPtrs(int **d_scanRanksPtrs, int *d_scanRanks, int batchSize, int batchUnitSize) {
    for(int i = 0; i < batchSize; ++i) {
        d_scanRanksPtrs[i] = &d_scanRanks[i*batchUnitSize*batchUnitSize];
    }
}

__global__ void fillSamplingVectorsPtrs(double **d_samplingVectorsPtrs, double *d_samplingVectors, double **d_outputPtrs, double *d_output, int batchSize, int batchUnitSize, int segmentSize, int maxSamples) {
    for(int i = 0; i < batchSize; ++i) {
        d_samplingVectorsPtrs[i] = &d_samplingVectors[i*batchUnitSize*segmentSize*maxSamples];
        d_outputPtrs[i] = &d_output[i*batchUnitSize*segmentSize*maxSamples];
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

__global__ void denseMatrixSampling(int batchSize, int matrixDim, double* denseMatrix, double* denseMatrixOutput, double* samplingVectors, int samplingVectorDim, int *samplesBatch, int transpose) {
    double sum = 0;

    if(transpose == 0) {
        unsigned int batch = blockIdx.y;
        unsigned int blockInBatch = blockIdx.x;
        if(threadIdx.x < samplesBatch[batch]) {
            for(unsigned int i = 0; i < matrixDim; ++i) {
                sum += denseMatrix[batch*matrixDim*matrixDim + i*matrixDim + blockInBatch*32 + threadIdx.y]*samplingVectors[batch*matrixDim*samplingVectorDim + threadIdx.x*matrixDim + i];
            }
        }
        if(threadIdx.x < samplesBatch[batch]) {
            denseMatrixOutput[batch*matrixDim*samplingVectorDim + threadIdx.x*matrixDim + blockInBatch*32 + threadIdx.y] = sum;
        }
    }
    else {
        unsigned int batch = blockIdx.z;
        unsigned int blockInBatch = blockIdx.y;
        unsigned int colInBatch = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int rowInBatch = blockIdx.y*blockDim.y + threadIdx.y;
        if(colInBatch < samplesBatch[batch]) {
            for(unsigned int i = 0; i < matrixDim; ++i) {
                sum += denseMatrix[batch*matrixDim*matrixDim + rowInBatch*matrixDim + i]*samplingVectors[batch*matrixDim*samplingVectorDim + colInBatch*matrixDim + i];
            }
        }
        if(colInBatch < samplesBatch[batch]) {
            denseMatrixOutput[batch*matrixDim*samplingVectorDim + colInBatch*matrixDim + rowInBatch] = sum;
        }
    }
}

__global__ void compareResults(double* denseMatrixOutput, double* output, int batchSize, int batchUnitSize, int segmentSize, int maxSamples, int* samplesBatch, double* error, double* tmp, int transpose) {
    if(transpose == 0) {
        unsigned int batch = blockIdx.y;
        unsigned int blockInBatch = blockIdx.x;
        if(threadIdx.x < samplesBatch[batch]) {
            double x = denseMatrixOutput[batch*batchUnitSize*segmentSize*maxSamples + threadIdx.x*batchUnitSize*segmentSize + blockInBatch*segmentSize + threadIdx.y];
            double y = output[batch*batchUnitSize*segmentSize*maxSamples + threadIdx.x*batchUnitSize*segmentSize + blockInBatch*segmentSize + threadIdx.y];
            printf("%lf      %lf\n", x, y);
            atomicAdd(tmp, x*x);
            atomicAdd(error, (x - y)*(x - y));
        }
    }
    else {
        unsigned int batch = blockIdx.z;
        unsigned int blockInBatch = blockIdx.y;
        unsigned int colInBatch = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int rowInBatch = blockIdx.y*blockDim.y + threadIdx.y;
        if(colInBatch < samplesBatch[batch]) {
            double x = denseMatrixOutput[batch*batchUnitSize*segmentSize*maxSamples + colInBatch*batchUnitSize*segmentSize + rowInBatch];
            double y = output[batch*batchUnitSize*segmentSize*maxSamples + colInBatch*batchUnitSize*segmentSize + rowInBatch];
            printf("%lf      %lf\n", x, y);
            atomicAdd(tmp, x*x);
            atomicAdd(error, (x - y)*(x - y));
        }
    }
}

int main() {
    int transpose = 1;
    // read a batch of n*n TLR matrices from a file
    fstream myFile("batchedMatrix.txt", ios_base::in);
    int batchUnitSize, segmentSize, batchSize;
    myFile >> batchUnitSize >> segmentSize >> batchSize;
    printf("%d %d %d\n", batchUnitSize, segmentSize, batchSize);
    int *ranks = (int*)malloc(batchSize*batchUnitSize*batchUnitSize*sizeof(int));
    int rankSum = 0;
    double *U, *V;
    V = (double*)malloc(0);
    U = (double*)malloc(0);

    for(unsigned int i = 0; i < batchSize; ++i) {
        for(unsigned int j = 0; j < batchUnitSize*batchUnitSize; ++j) {
            int index = i*batchUnitSize*batchUnitSize + j;
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
    cudaMalloc((void**) &d_ranks, batchSize*batchUnitSize*batchUnitSize*sizeof(int));
    cudaMalloc((void**) &d_scanRanks, batchSize*batchUnitSize*batchUnitSize*sizeof(int));
    cudaMalloc((void**) &d_U, rankSum*segmentSize*sizeof(double));
    cudaMalloc((void**) &d_V, rankSum*segmentSize*sizeof(double));
    cudaMemcpy(d_ranks, ranks, batchSize*batchUnitSize*batchUnitSize*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, U, rankSum*segmentSize*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, rankSum*segmentSize*sizeof(double), cudaMemcpyHostToDevice);

    int *d_batchSegments;
    cudaMalloc((void**) &d_batchSegments, batchSize*batchUnitSize*batchUnitSize*sizeof(int));
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (batchSize*batchUnitSize*batchUnitSize + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillBatchSegments <<< numBlocks, numThreadsPerBlock >>> (d_batchSegments, batchUnitSize, batchSize);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSumByKey(d_temp_storage, temp_storage_bytes, d_batchSegments, d_ranks, d_scanRanks, batchSize*batchUnitSize*batchUnitSize);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSumByKey(d_temp_storage, temp_storage_bytes, d_batchSegments, d_ranks, d_scanRanks, batchSize*batchUnitSize*batchUnitSize);

    double **d_UBatchPtrs, **d_VBatchPtrs;
    cudaMalloc((void**) &d_UBatchPtrs, batchSize*sizeof(double*));
    cudaMalloc((void**) &d_VBatchPtrs, batchSize*sizeof(double*));

    numBlocks = 1;
    numThreadsPerBlock = 1;
    fillBatchedPtrs <<< numBlocks, numThreadsPerBlock >>> (d_UBatchPtrs, d_VBatchPtrs, d_U, d_V, d_scanRanks, batchSize, segmentSize, batchUnitSize);

    // generate random sampling vectors
    int* d_samplesBatch;
    cudaMalloc((void**) &d_samplesBatch, batchSize*sizeof(int));
    fillSamplesBatch <<< 1, 1 >>> (d_samplesBatch, batchSize);

    unsigned int maxSamples = 64;
    int maxRows = batchUnitSize*segmentSize/2, maxCols = batchUnitSize*segmentSize/2;
    double *d_output;
    double *d_samplingVectors;
    cudaMalloc((void**) &d_output, maxSamples*batchSize*segmentSize*batchUnitSize*sizeof(double));
    cudaMalloc((void**) &d_samplingVectors, maxSamples*batchSize*segmentSize*batchUnitSize*sizeof(double));

    numThreadsPerBlock = 1024;
    numBlocks = (maxSamples*batchSize*segmentSize*batchUnitSize + numThreadsPerBlock - 1)/numThreadsPerBlock;
    generateSamplingVectors <<< numBlocks, numThreadsPerBlock >>> (d_samplingVectors, maxSamples*batchSize*segmentSize*batchUnitSize);

    int **d_scanRanksPtrs;
    cudaMalloc((void**) &d_scanRanksPtrs, batchSize*sizeof(int*));
    fillScanRanksPtrs <<< 1, 1>>> (d_scanRanksPtrs, d_scanRanks, batchSize, batchUnitSize);

    double **d_samplingVectorsPtrs;
    cudaMalloc((void**) &d_samplingVectorsPtrs, batchSize*sizeof(double*));
    double **d_outputPtrs;
    cudaMalloc((void**) &d_outputPtrs, batchSize*sizeof(double*));
    fillSamplingVectorsPtrs <<< 1, 1 >>> (d_samplingVectorsPtrs, d_samplingVectors, d_outputPtrs, d_output, batchSize, batchUnitSize, segmentSize, maxSamples);

    // launch a kernel that takes as input the TLR matrices, sampling function and multiplies them and stores them in a matrix
    if(transpose == 0) {
        dim3 m_numThreadsPerBlock(maxSamples, 32);
        dim3 m_numBlocks(batchUnitSize, batchSize);
        batchedSampling <double> <<< m_numBlocks, m_numThreadsPerBlock >>> (segmentSize, batchUnitSize, d_UBatchPtrs, d_VBatchPtrs, d_scanRanksPtrs, d_samplingVectorsPtrs, d_outputPtrs, d_samplesBatch, maxRows, maxCols, maxSamples, transpose);
    }
    else {
        dim3 m_numThreadsPerBlock(min(32, maxSamples), 32);
        dim3 m_numBlocks((maxSamples + 31)/32, batchUnitSize, batchSize);
        batchedSampling <double> <<< m_numBlocks, m_numThreadsPerBlock >>> (segmentSize, batchUnitSize, d_UBatchPtrs, d_VBatchPtrs, d_scanRanksPtrs, d_samplingVectorsPtrs, d_outputPtrs, d_samplesBatch, maxRows, maxCols, maxSamples, transpose);
    }
    cudaDeviceSynchronize();

    // read the batched dense tiles form the txt file
    fstream denseMatrixFile("denseMatrix.txt", ios_base::in);
    double* denseMatrix = (double*)malloc(batchSize*batchUnitSize*segmentSize*batchUnitSize*segmentSize*sizeof(double));
    for(unsigned int batch = 0; batch < batchSize; ++batch) {
        for(unsigned int row = 0; row < batchUnitSize*segmentSize; ++row) {
            for(unsigned int col = 0; col < batchUnitSize*segmentSize; ++col) {
                denseMatrixFile >> denseMatrix[batch*batchUnitSize*segmentSize*batchUnitSize*segmentSize + col*batchUnitSize*segmentSize + row];
            }
        }
    }

    double* d_denseMatrix;
    cudaMalloc((void**) &d_denseMatrix, batchSize*batchUnitSize*segmentSize*batchUnitSize*segmentSize*sizeof(double));    
    cudaMemcpy(d_denseMatrix, denseMatrix, batchSize*batchUnitSize*segmentSize*batchUnitSize*segmentSize*sizeof(double), cudaMemcpyHostToDevice);
    double *d_denseMatrixOutput;
    cudaMalloc((void**) &d_denseMatrixOutput, maxSamples*batchSize*segmentSize*batchUnitSize*sizeof(double));

    // multiply the dense matrix by the sampling vectors
    if(transpose == 0) {
        dim3 dm_numThreadsPerBlock(maxSamples, 32);
        dim3 dm_numBlocks(batchUnitSize, batchSize);
        denseMatrixSampling <<< dm_numBlocks, dm_numThreadsPerBlock >>> (batchSize, batchUnitSize*segmentSize, d_denseMatrix, d_denseMatrixOutput, d_samplingVectors, maxSamples, d_samplesBatch, transpose);
    }
    else {
        dim3 dm_numThreadsPerBlock(min(32, maxSamples), 32);
        dim3 dm_numBlocks((maxSamples + 31)/32, batchUnitSize, batchSize);
        denseMatrixSampling <<< dm_numBlocks, dm_numThreadsPerBlock >>> (batchSize, batchUnitSize*segmentSize, d_denseMatrix, d_denseMatrixOutput, d_samplingVectors, maxSamples, d_samplesBatch, transpose);
    }

    // compare the results
    double *d_error, *d_tmp;
    cudaMalloc((void**) &d_error, sizeof(double));
    cudaMalloc((void**) &d_tmp, sizeof(double));
    cudaMemset(d_error, 0, sizeof(double));
    cudaMemset(d_tmp, 0, sizeof(double));
    if(transpose == 0) {
        dim3 dm_numThreadsPerBlock(maxSamples, 32);
        dim3 dm_numBlocks(batchUnitSize, batchSize);
        compareResults <<< dm_numBlocks, dm_numThreadsPerBlock >>> (d_denseMatrixOutput, d_output, batchSize, batchUnitSize, segmentSize, maxSamples, d_samplesBatch, d_error, d_tmp, transpose);
    }
    else {
        dim3 dm_numThreadsPerBlock(min(32, maxSamples), 32);
        dim3 dm_numBlocks((maxSamples + 31)/32, batchUnitSize, batchSize);
        compareResults <<< dm_numBlocks, dm_numThreadsPerBlock >>> (d_denseMatrixOutput, d_output, batchSize, batchUnitSize, segmentSize, maxSamples, d_samplesBatch, d_error, d_tmp, transpose);
    }
    double h_error;
    double h_tmp;
    cudaMemcpy(&h_error, d_error, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_tmp, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);
    printf("error in matrix: %lf\n", sqrt(h_error)/sqrt(h_tmp));
    cudaFree(d_tmp);
    cudaFree(d_error);
    cudaDeviceSynchronize();
}