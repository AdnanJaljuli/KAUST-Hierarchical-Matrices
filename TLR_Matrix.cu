#include "TLR_Matrix.h"
#include "Random_Matrix_Gen_kernel.cuh"

#include <stdio.h>
#include <stdint.h>
#include <cstdlib>
#include <iostream>
#include <assert.h>
#include <time.h>
using namespace std;

void TLR_Matrix::createRandomMatrix(unsigned int xn, unsigned int xblockSize){
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (n+numThreadsPerBlock-1)/numThreadsPerBlock;

    blockSize = xblockSize;
    n = xn;
    nBlocks = (n + blockSize - 1)/blockSize;
    blockRanks = (unsigned int *) malloc(sizeof(unsigned int)*((nBlocks * nBlocks)-nBlocks));
    float *blockRanks_d;
    cudaMalloc((void**)&blockRanks_d, sizeof(unsigned int)*((nBlocks * nBlocks)-nBlocks));
    RandomMatrixGen_kernel <<<numThreadsPerBlock, numBlocks>>> (blockRanks_d, nBlocks*nBlocks - nBlocks);
    cudaMemcpy(blockRanks, blockRanks_d, sizeof(unsigned int)*((nBlocks * nBlocks)-nBlocks), cudaMemcpyDeviceToHost);

    // srand(time(NULL));
    // for (unsigned int i = 0; i < ((nBlocks * nBlocks)-nBlocks); ++i){
    //     unsigned int randomNumber = rand() % 10 + 10;
    //     blockRanks[i] = randomNumber;
    // }

    unsigned int numberOfElements = 0;
    for (unsigned int i = 0; i < ((nBlocks * nBlocks)-nBlocks); ++i){
        numberOfElements += (blockSize * blockRanks[i]);
    }

    diagonal = (float *) malloc(sizeof(float)*nBlocks*(blockSize*blockSize));
    u = (float *) malloc(sizeof(float)*numberOfElements);
    v = (float *) malloc(sizeof(float)*numberOfElements);

    float *u_d, *v_d;
    cudaMalloc((void**)&u_d, sizeof(float)*numberOfElements);
    cudaMalloc((void**)&v_d, sizeof(float)*numberOfElements);
    RandomMatrixGen_kernel <<<numThreadsPerBlock, numBlocks>>> (u_d, numberOfElements);
    cudaMemcpy(u, u_d, sizeof(float)*numberOfElements, cudaMemcpyHostToDevice);
    RandomMatrixGen_kernel <<<numThreadsPerBlock, numBlocks>>> (v_d, numberOfElements);
    cudaMemcpy(v, v_d, sizeof(float)*numberOfElements, cudaMemcpyDeviceToHost);
    // TODO: replace with a kernel call
    // for (unsigned int i = 0; i < numberOfElements; ++i){
    //     unsigned int randomNumber1 = rand() % 10 + 10;
    //     unsigned int randomNumber2 = rand() % 10 + 10;
    //     u[i] = randomNumber1;
    //     v[i] = randomNumber2;
    // }

    float *diagonal_d;
    cudaMalloc((void**)&diagonal_d, sizeof(float)*nBlocks*(blockSize*blockSize));
    RandomMatrixGen_kernel <<<numThreadsPerBlock, numBlocks>>> (diagonal_d, nBlocks*(blockSize*blockSize));
    cudaMemcpy(diagonal, diagonal_d, sizeof(float)*nBlocks*(blockSize*blockSize), cudaMemcpyDeviceToHost);
    // TODO: replace with a kernel call
    // for (unsigned int i = 0; i < nBlocks*(blockSize*blockSize); ++i){
    //     unsigned int randomNumber = rand() % 10 + 10;
    //     diagonal[i] = randomNumber;
    // }

    lowRankBlockPointers = (unsigned int *) malloc(sizeof(unsigned int)*((nBlocks*nBlocks)-nBlocks));

    lowRankBlockPointers[0] = 0;
    for (unsigned int i = 1; i < ((nBlocks * nBlocks)-nBlocks); ++i){
        lowRankBlockPointers[i] = (blockSize*blockRanks[i-1]) + lowRankBlockPointers[i-1] ;
    }
}

void TLR_Matrix::createOutputMatrix(unsigned int xn, unsigned int xblockSize){
    blockSize = xblockSize;
    n = xn;
    nBlocks = (n +  blockSize -1)/ blockSize;
}

void TLR_Matrix::del(){
    free(blockRanks);
    free(lowRankBlockPointers);
    free(u);
    free(v);
    free(diagonal);
}

