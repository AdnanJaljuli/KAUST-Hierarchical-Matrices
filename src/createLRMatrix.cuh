
#ifndef __CREATE_LR_MATRIX_CUH__
#define __CREATE_LR_MATRIX_CUH__

#include <assert.h>
#include <ctype.h>
#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

// TODO: ask Izzat about the order of the below 3 header files. If their order is different, it throus a compilation error.
#include "cublas_v2.h"
#include "kblas.h"
#include "kDTree.h"
#include "batch_rand.h"
#include "helperFunctions.cuh"
#include "helperKernels.cuh"
#include "magma_auxiliary.h"

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

uint64_t createColumnMajorLRMatrix(unsigned int numberOfInputPoints, unsigned int maxSegmentSize, unsigned int bucketSize, unsigned int dimensionOfInputPoints, TLR_Matrix &matrix, KDTree kDTree, H2Opus_Real* &d_dataset, float tolerance, int ARA_R) {

    int *d_rowsBatch, *d_colsBatch, *d_ranks;
    int *d_LDMBatch, *d_LDABatch, *d_LDBBatch;
    H2Opus_Real *d_A, *d_B;
    H2Opus_Real **d_MPtrs, **d_APtrs, **d_BPtrs;
    gpuErrchk(cudaMalloc((void**) &d_rowsBatch, (kDTree.numSegments - 1)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_colsBatch, (kDTree.numSegments - 1)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_ranks, (kDTree.numSegments - 1)*kDTree.numSegments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_LDMBatch, (kDTree.numSegments - 1)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_LDABatch, (kDTree.numSegments - 1)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_LDBBatch, (kDTree.numSegments - 1)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_A, (kDTree.numSegments - 1)*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &d_B, (kDTree.numSegments - 1)*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &d_MPtrs, (kDTree.numSegments - 1)*sizeof(H2Opus_Real*)));
    gpuErrchk(cudaMalloc((void**) &d_APtrs, (kDTree.numSegments - 1)*sizeof(H2Opus_Real*)));
    gpuErrchk(cudaMalloc((void**) &d_BPtrs, (kDTree.numSegments - 1)*sizeof(H2Opus_Real*)));

    int numThreadsPerBlock = 1024;
    int numBlocks = (kDTree.numSegments - 1 + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillARAArrays <<< numBlocks, numThreadsPerBlock >>> (kDTree.numSegments - 1, maxSegmentSize, d_rowsBatch, d_colsBatch, d_LDMBatch, d_LDABatch, d_LDBBatch);
    gpuErrchk(cudaPeekAtLastError());

    magma_init();
    kblasHandle_t kblasHandle;
    kblasRandState_t randState;
    kblasCreate(&kblasHandle);
    gpuErrchk(cudaPeekAtLastError());

    kblasInitRandState(kblasHandle, &randState, 1<<15, 0);
    gpuErrchk(cudaPeekAtLastError());

    kblasEnableMagma(kblasHandle);
    kblas_gesvj_batch_wsquery<H2Opus_Real>(kblasHandle, maxSegmentSize, maxSegmentSize, kDTree.numSegments - 1);
    kblas_ara_batch_wsquery<H2Opus_Real>(kblasHandle, bucketSize, kDTree.numSegments - 1);
    gpuErrchk(cudaPeekAtLastError());
    kblasAllocateWorkspace(kblasHandle);

    float ARATotalTime = 0;
    uint64_t rankSum = 0;
    uint64_t* totalMem = (uint64_t*)malloc(sizeof(uint64_t));
    uint64_t* d_totalMem;
    cudaMalloc((void**) &d_totalMem, sizeof(uint64_t));

    H2Opus_Real* d_inputMatrixSegmented;
    int* d_scanRanksSegmented;
    gpuErrchk(cudaMalloc((void**) &d_inputMatrixSegmented, maxSegmentSize*maxSegmentSize*kDTree.numSegments*(uint64_t)sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &d_scanRanksSegmented, (kDTree.numSegments - 1)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &matrix.blockRanks, kDTree.numSegments*kDTree.numSegments*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &matrix.diagonal, kDTree.numSegments*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real)));
    H2Opus_Real** d_UTiledTemp = (H2Opus_Real**)malloc(kDTree.numSegments*sizeof(H2Opus_Real*));
    H2Opus_Real** d_VTiledTemp = (H2Opus_Real**)malloc(kDTree.numSegments*sizeof(H2Opus_Real*));

    dim3 m_numThreadsPerBlock(min(32, (int)maxSegmentSize), min(32, (int)maxSegmentSize));
    dim3 m_numBlocks(1, kDTree.numSegments);

    for(unsigned int segment = 0; segment < kDTree.numSegments; ++segment) {

        generateDenseBlockColumn <<< m_numBlocks, m_numThreadsPerBlock >>> (numberOfInputPoints, maxSegmentSize, dimensionOfInputPoints, d_inputMatrixSegmented, d_dataset, kDTree, segment, matrix.diagonal);
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        generateArrayOfPointersT<H2Opus_Real>(d_inputMatrixSegmented, d_MPtrs, maxSegmentSize*maxSegmentSize, kDTree.numSegments - 1, 0);
        generateArrayOfPointersT<H2Opus_Real>(d_A, d_APtrs, maxSegmentSize*maxSegmentSize, kDTree.numSegments - 1, 0);
        generateArrayOfPointersT<H2Opus_Real>(d_B, d_BPtrs, maxSegmentSize*maxSegmentSize, kDTree.numSegments - 1, 0);
        gpuErrchk(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());

        cudaEvent_t startARA, stopARA;
        cudaEventCreate(&startARA);
        cudaEventCreate(&stopARA);
        cudaEventRecord(startARA);

        int kblas_ara_return = kblas_ara_batch(
            kblasHandle, d_rowsBatch, d_colsBatch, d_MPtrs, d_LDMBatch, 
            d_APtrs, d_LDABatch, d_BPtrs, d_LDBBatch, d_ranks + segment*(kDTree.numSegments - 1),
            tolerance, maxSegmentSize, maxSegmentSize, maxSegmentSize, 32, ARA_R, randState, 0, kDTree.numSegments - 1
        );
        assert(kblas_ara_return == 1);

        cudaEventRecord(stopARA);
        cudaEventSynchronize(stopARA);
        float ARA_time = 0;
        cudaEventElapsedTime(&ARA_time, startARA, stopARA);
        ARATotalTime += ARA_time;
        cudaEventDestroy(startARA);
        cudaEventDestroy(stopARA);
        cudaDeviceSynchronize();

        void* d_tempStorage = NULL;
        size_t tempStorageBytes = 0;
        cub::DeviceScan::ExclusiveSum(d_tempStorage, tempStorageBytes, d_ranks + segment*(kDTree.numSegments - 1), d_scanRanksSegmented, kDTree.numSegments - 1);
        cudaMalloc(&d_tempStorage, tempStorageBytes);
        cub::DeviceScan::ExclusiveSum(d_tempStorage, tempStorageBytes, d_ranks + segment*(kDTree.numSegments - 1), d_scanRanksSegmented, kDTree.numSegments - 1);
        cudaDeviceSynchronize();
        cudaFree(d_tempStorage);

        printK<<<1, 1>>>(d_ranks + segment*(kDTree.numSegments - 1), kDTree.numSegments - 1);
        getTotalMem<<<1, 1>>> (d_totalMem, d_ranks + segment*(kDTree.numSegments - 1), d_scanRanksSegmented, kDTree.numSegments - 1);
        cudaMemcpy(totalMem, d_totalMem, sizeof(uint64_t), cudaMemcpyDeviceToHost);

        gpuErrchk(cudaMalloc((void**) &d_UTiledTemp[segment], maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real)));
        gpuErrchk(cudaMalloc((void**) &d_VTiledTemp[segment], maxSegmentSize*(*totalMem)*sizeof(H2Opus_Real)));

        // TODO: optimize thread allocation here
        int numThreadsPerBlock = maxSegmentSize;
        int numBlocks = kDTree.numSegments - 1;
        copyTiles<<<numBlocks, numThreadsPerBlock>>>(kDTree.numSegments - 1, maxSegmentSize, d_ranks + segment*(kDTree.numSegments - 1), d_scanRanksSegmented, d_UTiledTemp[segment], d_A, d_VTiledTemp[segment], d_B);

        rankSum += (*totalMem);
    }

    free(totalMem);
    cudaFree(d_totalMem);
    cudaFree(d_inputMatrixSegmented);
    cudaFree(d_scanRanksSegmented);
    cudaFree(d_rowsBatch);
    cudaFree(d_colsBatch);
    cudaFree(d_LDMBatch);
    cudaFree(d_LDABatch);
    cudaFree(d_LDBBatch);
    cudaFree(d_MPtrs);
    cudaFree(d_APtrs);
    cudaFree(d_BPtrs);
    cudaFree(d_A);
    cudaFree(d_B);

    gpuErrchk(cudaMalloc((void**) &matrix.U, rankSum*maxSegmentSize*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &matrix.V, rankSum*maxSegmentSize*sizeof(H2Opus_Real)));
    gpuErrchk(cudaMalloc((void**) &matrix.blockOffsets, kDTree.numSegments*kDTree.numSegments*sizeof(int)));

    numThreadsPerBlock = 1024;
    numBlocks = ((kDTree.numSegments - 1)*kDTree.numSegments + numThreadsPerBlock - 1)/numThreadsPerBlock;
    copyRanks <<< numBlocks, numThreadsPerBlock >>> (kDTree.numSegments, maxSegmentSize, d_ranks, matrix.blockRanks);
    cudaDeviceSynchronize();
    cudaFree(d_ranks);

    void *d_tempStorage = NULL;
    size_t tempStorageBytes = 0;
    cub::DeviceScan::ExclusiveSum(d_tempStorage, tempStorageBytes, matrix.blockRanks, matrix.blockOffsets, kDTree.numSegments*kDTree.numSegments);
    cudaMalloc(&d_tempStorage, tempStorageBytes);
    cub::DeviceScan::ExclusiveSum(d_tempStorage, tempStorageBytes, matrix.blockRanks, matrix.blockOffsets, kDTree.numSegments*kDTree.numSegments);
    cudaDeviceSynchronize();
    cudaFree(d_tempStorage);
    gpuErrchk(cudaPeekAtLastError());

    int* h_scanRanks = (int*)malloc(kDTree.numSegments*kDTree.numSegments*sizeof(int));
    gpuErrchk(cudaMemcpy(h_scanRanks, matrix.blockOffsets, kDTree.numSegments*kDTree.numSegments*sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaPeekAtLastError());

    for(unsigned int segment = 0; segment < kDTree.numSegments - 1; ++segment) {
        gpuErrchk(cudaMemcpy(&matrix.U[h_scanRanks[kDTree.numSegments*segment]*maxSegmentSize], d_UTiledTemp[segment], (h_scanRanks[kDTree.numSegments*(segment + 1)] - h_scanRanks[kDTree.numSegments*segment])*maxSegmentSize*(uint64_t)sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy(&matrix.V[h_scanRanks[kDTree.numSegments*segment]*maxSegmentSize], d_VTiledTemp[segment], (h_scanRanks[kDTree.numSegments*(segment + 1)] - h_scanRanks[kDTree.numSegments*segment])*maxSegmentSize*(uint64_t)sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice));
    }
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpy(&matrix.U[h_scanRanks[kDTree.numSegments*(kDTree.numSegments - 1)]*maxSegmentSize], d_UTiledTemp[kDTree.numSegments - 1], (rankSum - h_scanRanks[kDTree.numSegments*(kDTree.numSegments - 1)])*maxSegmentSize*(uint64_t)sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(&matrix.V[h_scanRanks[kDTree.numSegments*(kDTree.numSegments - 1)]*maxSegmentSize], d_VTiledTemp[kDTree.numSegments - 1], (rankSum - h_scanRanks[kDTree.numSegments*(kDTree.numSegments - 1)])*maxSegmentSize*(uint64_t)sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice));
    free(h_scanRanks);
    gpuErrchk(cudaPeekAtLastError());

    for(unsigned int segment = 0; segment < kDTree.numSegments; ++segment) {
        cudaFree(d_UTiledTemp[segment]);
        cudaFree(d_VTiledTemp[segment]);
    }
    free(d_UTiledTemp);
    free(d_VTiledTemp);
    gpuErrchk(cudaPeekAtLastError());
    return rankSum;
}

#endif
