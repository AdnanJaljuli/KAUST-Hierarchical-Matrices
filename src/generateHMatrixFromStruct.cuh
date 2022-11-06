
#ifndef __GENERATE_HIERARCHICALMATRIX_H__
#define __GENERATE_HIERARCHICALMATRIX_H__

#include "config.h"
#include "counters.h"
#include "generateHMatrixHelpers.cuh"
#include "hierarchicalMatrix.h"
#include "kDTree.h"
#include "TLRMatrix.h"

__global__ void expandMatrix(H2Opus_Real **A, H2Opus_Real **B, int size, H2Opus_Real* output, int* ranks) {
    unsigned int batch = blockIdx.z;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    if(col < size && row < size) {
        H2Opus_Real sum = 0;
        for(unsigned int i = 0; i < ranks[0]; ++i) {
            sum += A[batch][i*size + row]*B[batch][i*size + col];
        }
        output[batch*size*size + col*size + row] = sum;
    }
}

__global__ void compareResults(double* denseMatrix, double* output, int size, double* error, double* tmp) {
    unsigned int batch = blockIdx.z;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    if(batch == 0) {
        double x = denseMatrix[(col+64)*128 + row];
        double y = output[col*64 + row];
        atomicAdd(tmp, x*x);
        atomicAdd(error, (x - y)*(x - y));
    }
    else {
        double x = denseMatrix[col*128 + 64 + row];
        double y = output[batch*64*64 + col*64 + row];
        atomicAdd(tmp, x*x);
        atomicAdd(error, (x - y)*(x - y));
    }
}

void generateHMatrixFromStruct(unsigned int numberOfInputPoints, unsigned int bucketSize, unsigned int numSegments, unsigned int segmentSize, TLR_Matrix mortonOrderedMatrix, int ARA_R, float tolerance, HMatrix hierarchicalMatrix, WeakAdmissibility WAStruct, H2Opus_Real* d_denseMatrix) {
    for(unsigned int level = WAStruct.numLevels - 2; level > 0; --level) {
        tolerance *= 2;
        // TODO: allocate outside the loop
        int batchUnitSize = 1 << (WAStruct.numLevels - (level + 1));
        int batchSize = WAStruct.numTiles[level - 1];
        printf("level: %d   batchSize: %d   bathcUnitSize: %d\n", level, batchSize, batchUnitSize);

        H2Opus_Real **d_UPtrs, **d_VPtrs;
        cudaMalloc((void**) &d_UPtrs, batchSize*sizeof(H2Opus_Real*));
        cudaMalloc((void**) &d_VPtrs, batchSize*sizeof(H2Opus_Real*));
        
        int* d_tileIndices;
        cudaMalloc((void**) &d_tileIndices, WAStruct.numTiles[level - 1]*sizeof(int));
        cudaMemcpy(d_tileIndices, WAStruct.tileIndices[level], WAStruct.numTiles[level - 1]*sizeof(int), cudaMemcpyHostToDevice);
        
        dim3 numThreadsPerBlock(1024);
        dim3 numBlocks((batchSize + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x, 2);
        fillBatchedPtrs <<< numBlocks, numThreadsPerBlock >>> (d_UPtrs, d_VPtrs, mortonOrderedMatrix, batchSize, segmentSize, batchUnitSize, d_tileIndices, level);
        gpuErrchk(cudaPeekAtLastError());

        int *d_scanRanks;
        cudaMalloc((void**) &d_scanRanks, batchSize*batchUnitSize*batchUnitSize*sizeof(int));

        // TODO: replace for loop with inclusiveSumByKey
        for(unsigned int batch = 0; batch < batchSize; ++batch) {
            void *d_temp_storage = NULL;
            size_t temp_storage_bytes = 0;
            printf("batch index: %d\n", WAStruct.tileIndices[level - 1][batch]*4);
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, mortonOrderedMatrix.blockRanks + WAStruct.tileIndices[level - 1][batch]*batchUnitSize*batchUnitSize, d_scanRanks + batch*batchUnitSize*batchUnitSize, batchUnitSize*batchUnitSize);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, mortonOrderedMatrix.blockRanks + WAStruct.tileIndices[level - 1][batch]*batchUnitSize*batchUnitSize, d_scanRanks + batch*batchUnitSize*batchUnitSize, batchUnitSize*batchUnitSize);
        }
        gpuErrchk(cudaPeekAtLastError());

        int **d_scanRanksPtrs;
        cudaMalloc((void**) &d_scanRanksPtrs, batchSize*sizeof(int*));
        numThreadsPerBlock.x = 1024;
        numBlocks.x = (batchSize + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x;
        numBlocks.y = 1;
        fillScanRankPtrs <<< numBlocks, numThreadsPerBlock >>> (d_scanRanksPtrs, d_scanRanks, batchUnitSize, batchSize);
        gpuErrchk(cudaPeekAtLastError());

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
        magma_init();
        kblasHandle_t kblasHandle;
        kblasRandState_t randState;
        kblasCreate(&kblasHandle);
        kblasInitRandState(kblasHandle, &randState, 1<<15, 0);
        kblasEnableMagma(kblasHandle);
        kblas_gesvj_batch_wsquery<H2Opus_Real>(kblasHandle, maxRows, maxCols, batchSize);
        kblas_ara_batch_wsquery<H2Opus_Real>(kblasHandle, 32, batchSize);
        kblasAllocateWorkspace(kblasHandle);
        gpuErrchk(cudaPeekAtLastError());

        int lr_ARA_return = lr_kblas_ara_batch(kblasHandle, segmentSize, batchUnitSize, d_rowsBatch, d_colsBatch, d_UPtrs, d_VPtrs, d_scanRanksPtrs,
            d_APtrs, d_LDABatch, d_BPtrs, d_LDBBatch, d_ranks,
            tolerance, maxRows, maxCols, maxRank, 16, ARA_R, randState, 0, batchSize
        );
        assert(lr_ARA_return == 1);
        printK <<< 1, 1 >>> (d_ranks, batchSize);
        gpuErrchk(cudaPeekAtLastError());
        cudaDeviceSynchronize();

        // TODO: copy back tiles to H matrix
    }

    #if 0
    // TODO: error checking
    #if EXPAND_MATRIX
    // launch a kernel that multiplies U by V
    H2Opus_Real *d_expandedMatrix;
    cudaMalloc((void**) &d_expandedMatrix, 2*64*64*sizeof(H2Opus_Real));
    dim3 m_numBlocks(2, 2, 2);
    dim3 m_numThreadsPerBlock(32, 32);
    expandMatrix <<< m_numBlocks, m_numThreadsPerBlock >>> (d_APtrs, d_BPtrs, 64, d_expandedMatrix, d_ranks);
    cudaDeviceSynchronize();
    // lanch a kernel that checks the error
    double *d_error, *d_tmp;
    cudaMalloc((void**) &d_error, sizeof(double));
    cudaMalloc((void**) &d_tmp, sizeof(double));
    cudaMemset(d_error, 0, sizeof(double));
    cudaMemset(d_tmp, 0, sizeof(double));
    compareResults <<< m_numBlocks, m_numThreadsPerBlock >>> (d_denseMatrix, d_expandedMatrix, 128, d_error, d_tmp);
    double h_error;
    double h_tmp;
    cudaMemcpy(&h_error, d_error, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_tmp, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);
    printf("error in matrix: %lf\n", sqrt(h_error)/sqrt(h_tmp));
    cudaFree(d_tmp);
    cudaFree(d_error);
    cudaDeviceSynchronize();
    #endif
    #endif
}

#endif