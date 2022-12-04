
#include "HMatrixVectorMultiplication.cuh"
#include "cublas_v2.h"
#include "cutlassDiagonalXVector.cuh"
#include "cutlassHMatrixXVector.cuh"
#include "HMatrix.cuh"
#include <assert.h>
#include <stdio.h>

__global__ void checkErrorInDiagXVec(unsigned int size, H2Opus_Real *result, H2Opus_Real *originalResult, H2Opus_Real* error, H2Opus_Real* tmp) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < size) {
        H2Opus_Real x = originalResult[i];
        H2Opus_Real y = result[i];
        printf("x: %lf   y: %lf\n", x, y);
        atomicAdd(tmp, x*x);
        atomicAdd(error, (x - y)*(x - y));
    }
}

__global__ void d_gemmBatched(unsigned int numberOfInputPoints, unsigned int bucketSize, unsigned int vectorWidth, H2Opus_Real *diagonalBlocks, H2Opus_Real *inpuVectors, H2Opus_Real *tempResultsVectors) {
    H2Opus_Real sum = 0;
    for(unsigned int i=0; i<32; ++i) {
        sum += diagonalBlocks[blockIdx.x*bucketSize*bucketSize + i*bucketSize + threadIdx.y]*inpuVectors[threadIdx.x*numberOfInputPoints + blockIdx.x*bucketSize + i];
    }
    printf("sum: %lf\n", sum);
    tempResultsVectors[threadIdx.x*numberOfInputPoints + blockIdx.x*bucketSize + threadIdx.y] = sum;
}

void printDiagMatrix( unsigned int bucketSize, unsigned int numSegments, H2Opus_Real* diagonalBlocks) {
    H2Opus_Real *h_diagonalBlocks = (H2Opus_Real*)malloc(bucketSize*bucketSize*numSegments*sizeof(H2Opus_Real));
    cudaMemcpy(h_diagonalBlocks, diagonalBlocks, bucketSize*bucketSize*numSegments*sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);

    char filename[100] = "results/diagonalMatrix.txt";
    FILE *output_file = fopen(filename, "a");

    for(unsigned int i = 0; i < numSegments; ++i) {
        for(unsigned int j = 0; j < bucketSize; ++j) {
            for(unsigned int k = 0; k < bucketSize; ++k) {
                fprintf(output_file, "%lf ", h_diagonalBlocks[i*bucketSize*bucketSize + j*bucketSize + k]);
            }
            fprintf(output_file, "\n");
        }
        fprintf(output_file, "\n");
    }
    fclose(output_file);
}

void printVectors(unsigned int vectorWidth, unsigned int vectorHeight, H2Opus_Real* vector) {
    H2Opus_Real *h_vector = (H2Opus_Real*)malloc(vectorWidth*vectorHeight*sizeof(H2Opus_Real));
    cudaMemcpy(h_vector, vector, vectorWidth*vectorHeight*sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);

    char filename[100] = "results/vectors.txt";
    FILE *output_file = fopen(filename, "a");

    for(unsigned int i = 0; i < vectorHeight; ++i) {
        for(unsigned int j = 0; j < vectorWidth; ++j) {
            fprintf(output_file, "%lf ", h_vector[j*vectorHeight + i]);
        }
        fprintf(output_file, "\n");
    }
    fclose(output_file);
}

cudaError_t HMatrixVecMult(unsigned int numberOfInputPoints, unsigned int bucketSize, unsigned int numSegments, unsigned int vectorWidth, HMatrix hierarchicalMatrix, H2Opus_Real* inpuVectors, H2Opus_Real* resultVectors) {

    assert((bucketSize & (bucketSize - 1)) == 0);
    assert((vectorWidth & (vectorWidth - 1)) == 0);
    cudaError_t result;
    // multiply diagonal blocks first
    // TODO: replace this with cuBLAS batched gemm
    result = cutlassDiagonalXVec(numberOfInputPoints, bucketSize, numSegments, vectorWidth, hierarchicalMatrix.diagonalBlocks, inpuVectors, resultVectors);
    // TODO: add error checking and return error message if result != success

    #if 0
    H2Opus_Real *d_tempResultsVectors;
    cudaMalloc((void**) &d_tempResultsVectors, numberOfInputPoints*vectorWidth*sizeof(H2Opus_Real));
    dim3 d_numThreadsPerBlock(16, 32);
    dim3 d_numBlocks(numSegments);
    d_gemmBatched <<< d_numBlocks, d_numThreadsPerBlock >>> (numberOfInputPoints, bucketSize, vectorWidth, hierarchicalMatrix.diagonalBlocks, inpuVectors, d_tempResultsVectors);
    H2Opus_Real* d_error;
    H2Opus_Real* d_tmp;
    cudaMalloc((void**) &d_error, sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_tmp, sizeof(H2Opus_Real));
    cudaMemset(d_error, 0, sizeof(H2Opus_Real));
    cudaMemset(d_tmp, 0, sizeof(H2Opus_Real));

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (numberOfInputPoints*vectorWidth + numThreadsPerBlock - 1)/numThreadsPerBlock;
    checkErrorInDiagXVec <<< numBlocks, numThreadsPerBlock >>> (numberOfInputPoints*vectorWidth, resultVectors, d_tempResultsVectors, d_error, d_tmp);

    H2Opus_Real h_error;
    H2Opus_Real h_tmp;
    cudaMemcpy(&h_error, d_error, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_tmp, d_tmp, sizeof(H2Opus_Real), cudaMemcpyDeviceToHost);
    printf("error in diagXVec: %lf\n", sqrt(h_error)/sqrt(h_tmp));
    cudaFree(d_tmp);
    cudaFree(d_error);
    cudaFree(d_tempResultsVectors);
    #endif

    // multiply rest of hierarchical matrix by the vector
    H2Opus_Real *d_bufferVectors;
    cudaMalloc((void**) &d_bufferVectors, numberOfInputPoints*vectorWidth*sizeof(H2Opus_Real));
    result = cutlassHierarchicalXVec(numberOfInputPoints, bucketSize, numSegments, vectorWidth, hierarchicalMatrix, inpuVectors, d_bufferVectors, resultVectors);
    cudaFree(d_bufferVectors);

    return result;
}