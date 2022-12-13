
#include "constructTLRMatrix.cuh"
#include "TLRMatrix.cuh"
#include "TLRMatrixHelpers.cuh"
#include "cublas_v2.h"
#include "helperFunctions.cuh"
#include "helperKernels.cuh"
#include "kblas.h"
#include "batch_rand.h"

#include <assert.h>

uint64_t createColumnMajorLRMatrix(unsigned int numberOfInputPoints, unsigned int leafSize, unsigned int dimensionOfInputPoints, TLR_Matrix &matrix, KDTree kDTree, H2Opus_Real* &d_dataset, float tolerance, int ARA_R) {

    int maxRank = kDTree.leafSize/2;
    int *d_rowsBatch, *d_colsBatch, *d_ranks;
    int *d_LDMBatch, *d_LDABatch, *d_LDBBatch;
    H2Opus_Real *d_A, *d_B;
    H2Opus_Real **d_MPtrs, **d_APtrs, **d_BPtrs;
    cudaMalloc((void**) &d_rowsBatch, (kDTree.numSegments - 1)*sizeof(int));
    cudaMalloc((void**) &d_colsBatch, (kDTree.numSegments - 1)*sizeof(int));
    cudaMalloc((void**) &d_ranks, (kDTree.numSegments - 1)*kDTree.numSegments*sizeof(int));
    cudaMalloc((void**) &d_LDMBatch, (kDTree.numSegments - 1)*sizeof(int));
    cudaMalloc((void**) &d_LDABatch, (kDTree.numSegments - 1)*sizeof(int));
    cudaMalloc((void**) &d_LDBBatch, (kDTree.numSegments - 1)*sizeof(int));
    cudaMalloc((void**) &d_A, (kDTree.numSegments - 1)*kDTree.leafSize*maxRank*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_B, (kDTree.numSegments - 1)*kDTree.leafSize*maxRank*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_MPtrs, (kDTree.numSegments - 1)*sizeof(H2Opus_Real*));
    cudaMalloc((void**) &d_APtrs, (kDTree.numSegments - 1)*sizeof(H2Opus_Real*));
    cudaMalloc((void**) &d_BPtrs, (kDTree.numSegments - 1)*sizeof(H2Opus_Real*));

    int numThreadsPerBlock = 1024;
    int numBlocks = (kDTree.numSegments - 1 + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillARAArrays <<< numBlocks, numThreadsPerBlock >>> (kDTree.numSegments - 1, kDTree.leafSize, d_rowsBatch, d_colsBatch, d_LDMBatch, d_LDABatch, d_LDBBatch);

    kblasHandle_t kblasHandle;
    kblasRandState_t randState;
    kblasCreate(&kblasHandle);
    kblasInitRandState(kblasHandle, &randState, 1 << 15, 0);
    kblasEnableMagma(kblasHandle);
    kblas_ara_batch_wsquery<H2Opus_Real>(kblasHandle, leafSize, kDTree.numSegments - 1);
    kblasAllocateWorkspace(kblasHandle);

    float ARATotalTime = 0;
    uint64_t rankSum = 0;
    int totalMem;
    int* d_totalMem;
    cudaMalloc((void**) &d_totalMem, sizeof(int));

    H2Opus_Real* d_inputMatrixSegmented;
    int* d_scanRanksSegmented;
    cudaMalloc((void**) &d_inputMatrixSegmented, kDTree.leafSize*kDTree.leafSize*kDTree.numSegments*sizeof(H2Opus_Real));
    cudaMalloc((void**) &d_scanRanksSegmented, (kDTree.numSegments - 1)*sizeof(int));
    cudaMalloc((void**) &matrix.blockRanks, kDTree.numSegments*kDTree.numSegments*sizeof(int));
    cudaMalloc((void**) &matrix.diagonal, kDTree.numSegments*kDTree.leafSize*kDTree.leafSize*sizeof(H2Opus_Real));
    H2Opus_Real **d_UTiledTemp = (H2Opus_Real**)malloc(kDTree.numSegments*sizeof(H2Opus_Real*));
    H2Opus_Real **d_VTiledTemp = (H2Opus_Real**)malloc(kDTree.numSegments*sizeof(H2Opus_Real*));

    dim3 m_numThreadsPerBlock(min(32, (int)kDTree.leafSize), min(32, (int)kDTree.leafSize));
    dim3 m_numBlocks(1, kDTree.numSegments);

    for(unsigned int segment = 0; segment < kDTree.numSegments; ++segment) {
        generateDenseBlockColumn <<< m_numBlocks, m_numThreadsPerBlock >>> (numberOfInputPoints, kDTree.leafSize, dimensionOfInputPoints, d_inputMatrixSegmented, d_dataset, kDTree, segment, matrix.diagonal);

        generateArrayOfPointersT<H2Opus_Real>(d_inputMatrixSegmented, d_MPtrs, kDTree.leafSize*kDTree.leafSize, kDTree.numSegments - 1, 0);
        generateArrayOfPointersT<H2Opus_Real>(d_A, d_APtrs, kDTree.leafSize*maxRank, kDTree.numSegments - 1, 0);
        generateArrayOfPointersT<H2Opus_Real>(d_B, d_BPtrs, kDTree.leafSize*maxRank, kDTree.numSegments - 1, 0);
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

        int kblas_ara_return = kblas_ara_batch(
            kblasHandle, d_rowsBatch, d_colsBatch, d_MPtrs, d_LDMBatch, 
            d_APtrs, d_LDABatch, d_BPtrs, d_LDBBatch, d_ranks + segment*(kDTree.numSegments - 1),
            tolerance, kDTree.leafSize, kDTree.leafSize, maxRank, 16, ARA_R, randState, 0, kDTree.numSegments - 1
        );
        assert(kblas_ara_return == 1);

        void* d_tempStorage = NULL;
        size_t tempStorageBytes = 0;
        cub::DeviceScan::InclusiveSum(d_tempStorage, tempStorageBytes, d_ranks + segment*(kDTree.numSegments - 1), d_scanRanksSegmented, kDTree.numSegments - 1);
        cudaMalloc(&d_tempStorage, tempStorageBytes);
        cub::DeviceScan::InclusiveSum(d_tempStorage, tempStorageBytes, d_ranks + segment*(kDTree.numSegments - 1), d_scanRanksSegmented, kDTree.numSegments - 1);
        cudaFree(d_tempStorage);

        cudaMemcpy(&totalMem, d_scanRanksSegmented + kDTree.numSegments - 2, sizeof(int), cudaMemcpyDeviceToHost);

        cudaMalloc((void**) &d_UTiledTemp[segment], kDTree.leafSize*totalMem*sizeof(H2Opus_Real));
        cudaMalloc((void**) &d_VTiledTemp[segment], kDTree.leafSize*totalMem*sizeof(H2Opus_Real));
        gpuErrchk(cudaPeekAtLastError());

        // TODO: optimize thread allocation here OR replace with cudaMemcpys
        int numThreadsPerBlock = kDTree.leafSize;
        int numBlocks = kDTree.numSegments - 1;
        copyTiles <<< numBlocks, numThreadsPerBlock >>> (kDTree.numSegments - 1, kDTree.leafSize, d_ranks + segment*(kDTree.numSegments - 1), d_scanRanksSegmented, d_UTiledTemp[segment], d_A, d_VTiledTemp[segment], d_B, maxRank);
        rankSum += static_cast<uint64_t>(totalMem);
    }

    kblasDestroy(&kblasHandle);
    kblasDestroyRandState(randState);

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

    cudaMalloc((void**) &matrix.U, rankSum*kDTree.leafSize*sizeof(H2Opus_Real));
    cudaMalloc((void**) &matrix.V, rankSum*kDTree.leafSize*sizeof(H2Opus_Real));
    cudaMalloc((void**) &matrix.blockOffsets, kDTree.numSegments*kDTree.numSegments*sizeof(int));

    numThreadsPerBlock = 1024;
    numBlocks = ((kDTree.numSegments - 1)*kDTree.numSegments + numThreadsPerBlock - 1)/numThreadsPerBlock;
    // TODO: no need for this. Instead, replace d_ranks with matrix.blockRanks
    copyRanks <<< numBlocks, numThreadsPerBlock >>> (kDTree.numSegments, kDTree.leafSize, d_ranks, matrix.blockRanks);
    cudaFree(d_ranks);

    void *d_tempStorage = NULL;
    size_t tempStorageBytes = 0;
    cub::DeviceScan::ExclusiveSum(d_tempStorage, tempStorageBytes, matrix.blockRanks, matrix.blockOffsets, kDTree.numSegments*kDTree.numSegments);
    cudaMalloc(&d_tempStorage, tempStorageBytes);
    cub::DeviceScan::ExclusiveSum(d_tempStorage, tempStorageBytes, matrix.blockRanks, matrix.blockOffsets, kDTree.numSegments*kDTree.numSegments);
    cudaFree(d_tempStorage);

    int* h_scanRanks = (int*)malloc(kDTree.numSegments*kDTree.numSegments*sizeof(int));
    cudaMemcpy(h_scanRanks, matrix.blockOffsets, kDTree.numSegments*kDTree.numSegments*sizeof(int), cudaMemcpyDeviceToHost);

    for(unsigned int segment = 0; segment < kDTree.numSegments - 1; ++segment) {
        cudaMemcpy(&matrix.U[static_cast<uint64_t>(h_scanRanks[kDTree.numSegments*segment])*kDTree.leafSize], d_UTiledTemp[segment], static_cast<uint64_t>(h_scanRanks[kDTree.numSegments*(segment + 1)] - h_scanRanks[kDTree.numSegments*segment])*kDTree.leafSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&matrix.V[static_cast<uint64_t>(h_scanRanks[kDTree.numSegments*segment])*kDTree.leafSize], d_VTiledTemp[segment], static_cast<uint64_t>(h_scanRanks[kDTree.numSegments*(segment + 1)] - h_scanRanks[kDTree.numSegments*segment])*kDTree.leafSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(&matrix.U[static_cast<uint64_t>(h_scanRanks[kDTree.numSegments*(kDTree.numSegments - 1)])*kDTree.leafSize], d_UTiledTemp[kDTree.numSegments - 1], static_cast<uint64_t>(rankSum - h_scanRanks[kDTree.numSegments*(kDTree.numSegments - 1)])*kDTree.leafSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
    cudaMemcpy(&matrix.V[static_cast<uint64_t>(h_scanRanks[kDTree.numSegments*(kDTree.numSegments - 1)])*kDTree.leafSize], d_VTiledTemp[kDTree.numSegments - 1], static_cast<uint64_t>(rankSum - h_scanRanks[kDTree.numSegments*(kDTree.numSegments - 1)])*kDTree.leafSize*sizeof(H2Opus_Real), cudaMemcpyDeviceToDevice);
    free(h_scanRanks);

    for(unsigned int segment = 0; segment < kDTree.numSegments; ++segment) {
        cudaFree(d_UTiledTemp[segment]);
        cudaFree(d_VTiledTemp[segment]);
    }
    free(d_UTiledTemp);
    free(d_VTiledTemp);
    
    return rankSum;
}
