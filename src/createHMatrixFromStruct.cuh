
#ifndef __GENERATE_HIERARCHICALMATRIX_H__
#define __GENERATE_HIERARCHICALMATRIX_H__

#include "config.h"
#include "counters.h"
#include "HMatrixHelpers.cuh"
#include "HMatrix.h"
#include "kDTree.h"
#include "TLRMatrix.h"

// TODO: break this code into smaller pieces and make it more readable
void generateHMatrixFromStruct(unsigned int numberOfInputPoints, unsigned int bucketSize, unsigned int numSegments, unsigned int segmentSize, TLR_Matrix mortonOrderedMatrix, int ARA_R, float tolerance, HMatrix hierarchicalMatrix, H2Opus_Real* d_denseMatrix) {
    
    WeakAdmissibility WAStruct;
    allocateWeakAdmissibilityStruct(WAStruct, numberOfInputPoints, bucketSize);

    magma_init();
    kblasHandle_t kblasHandle;
    kblasRandState_t randState;
    kblasCreate(&kblasHandle);
    kblasInitRandState(kblasHandle, &randState, 1<<15, 0);
    kblasEnableMagma(kblasHandle);
    // TODO: allocate memory outside the loop
    // TODO: use multistreaming

    for(unsigned int level = WAStruct.numLevels - 2; level > 0; --level) {
        int batchSize = WAStruct.numTiles[level - 1];
        if(batchSize == 0) {
            continue;
        }
        int batchUnitSize = 1 << (WAStruct.numLevels - (level + 1));
        tolerance *= 2;

        // preprocessing
        H2Opus_Real **d_UPtrs, **d_VPtrs;
        cudaMalloc((void**) &d_UPtrs, batchSize*sizeof(H2Opus_Real*));
        cudaMalloc((void**) &d_VPtrs, batchSize*sizeof(H2Opus_Real*));

        int* d_tileIndices;
        cudaMalloc((void**) &d_tileIndices, batchSize*sizeof(int));
        cudaMemcpy(d_tileIndices, WAStruct.tileIndices[level], batchSize*sizeof(int), cudaMemcpyHostToDevice);

        dim3 numThreadsPerBlock(1024);
        dim3 numBlocks((batchSize + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x, 2);
        fillBatchedPtrs <<< numBlocks, numThreadsPerBlock >>> (d_UPtrs, d_VPtrs, mortonOrderedMatrix, batchSize, segmentSize, batchUnitSize, d_tileIndices, level);

        // TODO: replace for loop with inclusiveSumByKey when using cuda/11
        int *d_scanRanks;
        cudaMalloc((void**) &d_scanRanks, batchSize*batchUnitSize*batchUnitSize*sizeof(int));
        for(unsigned int batch = 0; batch < batchSize; ++batch) {
            void *d_temp_storage = NULL;
            size_t temp_storage_bytes = 0;
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, mortonOrderedMatrix.blockRanks + WAStruct.tileIndices[level - 1][batch]*batchUnitSize*batchUnitSize, d_scanRanks + batch*batchUnitSize*batchUnitSize, batchUnitSize*batchUnitSize);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, mortonOrderedMatrix.blockRanks + WAStruct.tileIndices[level - 1][batch]*batchUnitSize*batchUnitSize, d_scanRanks + batch*batchUnitSize*batchUnitSize, batchUnitSize*batchUnitSize);
            cudaFree(d_temp_storage);
        }

        int **d_scanRanksPtrs;
        cudaMalloc((void**) &d_scanRanksPtrs, batchSize*sizeof(int*));
        numThreadsPerBlock.x = 1024;
        numBlocks.x = (batchSize + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x;
        numBlocks.y = 1;
        fillScanRankPtrs <<< numBlocks, numThreadsPerBlock >>> (d_scanRanksPtrs, d_scanRanks, batchUnitSize, batchSize);

        int maxRows = batchUnitSize*bucketSize;
        int maxCols = maxRows;
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

        numThreadsPerBlock.x = 1024;
        numBlocks.x = (batchSize + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x;
        fillLRARAArrays <<< numBlocks, numThreadsPerBlock >>> (batchSize, maxRows, d_rowsBatch, d_colsBatch, d_LDABatch, d_LDBBatch);
        gpuErrchk(cudaPeekAtLastError());

        generateArrayOfPointersT<H2Opus_Real>(d_A, d_APtrs, maxRows*maxRank, batchSize, 0);
        generateArrayOfPointersT<H2Opus_Real>(d_B, d_BPtrs, maxRows*maxRank, batchSize, 0);
        kblas_gesvj_batch_wsquery<H2Opus_Real>(kblasHandle, maxRows, maxCols, batchSize);
        kblas_ara_batch_wsquery<H2Opus_Real>(kblasHandle, maxRows, batchSize);
        kblasAllocateWorkspace(kblasHandle);

        int lr_ARA_return = lr_kblas_ara_batch(kblasHandle, segmentSize, batchUnitSize, d_rowsBatch, d_colsBatch, d_UPtrs, d_VPtrs, d_scanRanksPtrs,
            d_APtrs, d_LDABatch, d_BPtrs, d_LDBBatch, d_ranks,
            tolerance, maxRows, maxCols, maxRank, 16, ARA_R, randState, 0, batchSize
        );
        assert(lr_ARA_return == 1);
        gpuErrchk(cudaPeekAtLastError());
        printK <<< 1, 1 >>> (d_ranks, batchSize);

        // allocate HMatrix level
        allocateHMatrixLevel(hierarchicalMatrix.levels[level - 1], d_ranks, WAStruct, level, d_A, d_B, maxRows, maxRank);
        gpuErrchk(cudaPeekAtLastError());

        #if EXPAND_MATRIX
        // expand H matrix level
        H2Opus_Real *d_expandedMatrix;
        cudaMalloc((void**) &d_expandedMatrix, batchSize*batchUnitSize*bucketSize*batchUnitSize*bucketSize*sizeof(H2Opus_Real));
        dim3 m_numBlocks(batchUnitSize, batchUnitSize, batchSize);
        dim3 m_numThreadsPerBlock(32, 32);
        expandMatrix <<< m_numBlocks, m_numThreadsPerBlock >>> (d_APtrs, d_BPtrs, batchUnitSize*bucketSize, d_expandedMatrix, d_ranks);
        cudaDeviceSynchronize();
        // compare expanded H matrix level with dense matrix
        double *d_error, *d_tmp;
        cudaMalloc((void**) &d_error, sizeof(double));
        cudaMalloc((void**) &d_tmp, sizeof(double));
        cudaMemset(d_error, 0, sizeof(double));
        cudaMemset(d_tmp, 0, sizeof(double));
        compareResults <<< m_numBlocks, m_numThreadsPerBlock >>> (numberOfInputPoints, d_denseMatrix, d_expandedMatrix, WAStruct.tileIndices[level - 1], batchSize, batchUnitSize, d_error, d_tmp);
        double h_error;
        double h_tmp;
        cudaMemcpy(&h_error, d_error, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_tmp, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);
        printf("error in matrix: %lf\n", sqrt(h_error)/sqrt(h_tmp));
        cudaFree(d_tmp);
        cudaFree(d_error);
        cudaDeviceSynchronize();
        #endif
        gpuErrchk(cudaPeekAtLastError());

        // free memory
        cudaFree(d_UPtrs);
        cudaFree(d_VPtrs);
        cudaFree(d_tileIndices);
        cudaFree(d_scanRanks);
        cudaFree(d_scanRanksPtrs);
        cudaFree(d_ranks);
        cudaFree(d_rowsBatch);
        cudaFree(d_colsBatch);
        cudaFree(d_LDABatch);
        cudaFree(d_LDBBatch);
        cudaFree(d_APtrs);
        cudaFree(d_BPtrs);
        cudaFree(d_A);
        cudaFree(d_B);
        gpuErrchk(cudaPeekAtLastError());

    }
    // TODO: free WAStruct
}

#endif