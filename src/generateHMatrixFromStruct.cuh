
#ifndef __GENERATE_HIERARCHICALMATRIX_H__
#define __GENERATE_HIERARCHICALMATRIX_H__

#include "config.h"
#include "counters.h"
#include "generateHMatrixHelpers.cuh"
#include "hierarchicalMatrix.h"
#include "kDTree.h"
#include "TLRMatrix.h"

void generateHMatrixFromStruct(unsigned int numberOfInputPoints, unsigned int bucketSize, unsigned int numSegments, unsigned int segmentSize, TLR_Matrix mortonOrderedMatrix, int ARA_R, float tolerance, HMatrix hierarchicalMatrix, WeakAdmissibility WAStruct) {
    // n = 512, level = 1, tileSize = 32, batchUnitSize = 8
    int level = 1, batchUnitSize = 8, batchSize = 2;

    // fill UPtrs and VPtrs
    H2Opus_Real **d_UPtrs, **d_VPtrs;
    cudaMalloc((void**) &d_UPtrs, batchSize*sizeof(H2Opus_Real*));
    cudaMalloc((void**) &d_VPtrs, batchSize*sizeof(H2Opus_Real*));
    fillBatchedPtrs <<< 1, 1 >>> (d_UPtrs, d_VPtrs, mortonOrderedMatrix, batchSize, segmentSize, batchUnitSize);
    gpuErrchk(cudaPeekAtLastError());

    // fill scan ranks
    int *d_scanRanks;
    cudaMalloc((void**) &d_scanRanks, batchSize*batchUnitSize*batchUnitSize*sizeof(H2Opus_Real));
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, mortonOrderedMatrix.blockRanks + 64, d_scanRanks, batchUnitSize*batchUnitSize);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, mortonOrderedMatrix.blockRanks + 64, d_scanRanks, batchUnitSize*batchUnitSize);
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, mortonOrderedMatrix.blockRanks + 128, d_scanRanks + 64, batchUnitSize*batchUnitSize);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, mortonOrderedMatrix.blockRanks + 128, d_scanRanks + 64, batchUnitSize*batchUnitSize);
    gpuErrchk(cudaPeekAtLastError());

    printK <<< 1, 1 >>> (d_scanRanks, batchSize*batchUnitSize*batchUnitSize);

    int **d_scanRanksPtrs;
    cudaMalloc((void**) &d_scanRanksPtrs, batchSize*sizeof(H2Opus_Real*));
    fillScanRankPtrs <<< 1, 1 >>> (d_scanRanksPtrs, d_scanRanks, batchUnitSize);
    gpuErrchk(cudaPeekAtLastError());

    int maxRows = 256;
    int maxCols = 256;
    int maxRank = maxRows/2;
    int *d_rowsBatch, *d_colsBatch, *d_ranks;
    int *d_LDABatch, *d_LDBBatch;
    H2Opus_Real *d_A, *d_B;
    H2Opus_Real **d_APtrs, **d_BPtrs;

    cudaMalloc((void**) &d_ranks, batchSize*sizeof(int));
    cudaMalloc((void**) &d_rowsBatch, batchSize*sizeof(int));
    cudaMalloc((void**) &d_colsBatch, batchSize*sizeof(int));
    cudaMalloc((void**) &d_LDABatch, batchSize*sizeof(int));
    cudaMalloc((void**) &d_LDBBatch, batchSize*sizeof(int));
    cudaMalloc((void**) &d_APtrs, batchSize*sizeof(H2Opus_Real*));
    cudaMalloc((void**) &d_BPtrs, batchSize*sizeof(H2Opus_Real*));
    cudaMalloc((void**) &d_A, batchSize*maxRows*maxRank*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_B, batchSize*maxRows*maxRank*sizeof(H2Opus_Real));

    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (batchSize + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillLRARAArrays <<< numBlocks, numThreadsPerBlock >>> (batchSize, maxRows, maxCols, d_rowsBatch, d_colsBatch, d_LDABatch, d_LDBBatch);
    gpuErrchk(cudaPeekAtLastError());

    generateArrayOfPointersT<H2Opus_Real>(d_A, d_APtrs, maxRows*maxCols, batchSize, 0);
    generateArrayOfPointersT<H2Opus_Real>(d_B, d_BPtrs, maxRows*maxCols, batchSize, 0);
    kblasHandle_t kblasHandle;
    kblasRandState_t randState;
    kblasCreate(&kblasHandle);
    kblasInitRandState(kblasHandle, &randState, 1<<15, 0);
    kblasEnableMagma(kblasHandle);
    kblas_gesvj_batch_wsquery<H2Opus_Real>(kblasHandle, maxRows, maxCols, batchSize);
    kblas_ara_batch_wsquery<H2Opus_Real>(kblasHandle, maxRows, batchSize);
    kblasAllocateWorkspace(kblasHandle);
    gpuErrchk(cudaPeekAtLastError());
    
    int lr_ARA_return = lr_kblas_ara_batch(kblasHandle, segmentSize, batchUnitSize, d_rowsBatch, d_colsBatch, d_UPtrs, d_VPtrs, d_scanRanksPtrs,
            d_APtrs, d_LDABatch, d_BPtrs, d_LDBBatch, d_ranks,
            tolerance, maxRows, maxCols, maxRank, 16, ARA_R, randState, 0, batchSize
    );
    assert(lr_ARA_return == 1);
    printK <<< 1, 1 >>> (d_ranks, batchSize);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    // TODO: error checking
    
}

#endif