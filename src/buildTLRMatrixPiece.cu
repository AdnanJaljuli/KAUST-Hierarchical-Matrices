
#include "buildTLRMatrixPiece.cuh"
#include "buildTLRMatrixPiece_helpers.cuh"
#include "helperKernels.cuh"
#include "kDTree.cuh"
#include "cublas_v2.h"
#include "kblas.h"
#include "batch_rand.h"
#include "magma_auxiliary.h"

#include <assert.h>
#include <cub/cub.cuh>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

template <class T>
void buildTLRMatrixPiece(
    TLR_Matrix *matrix,
    KDTree kdtree,
    T* d_pointCloud,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    T tol) {

        magma_init();

        uint64_t totalRankSum = 0;
        int ARA_R = 10;
        matrix->tileSize = kdtree.maxLeafSize;
        assert(upperPowerOfTwo(kdtree.N)/numPiecesInAxis >= matrix->tileSize);

        matrix->numTilesInAxis = (upperPowerOfTwo(kdtree.N)/numPiecesInAxis)/matrix->tileSize;
        bool isDiagonal = isPieceDiagonal(pieceMortonIndex);
        unsigned int batchCount = isDiagonal ? matrix->numTilesInAxis - 1: matrix->numTilesInAxis;
        unsigned int maxRank = matrix->tileSize/2;
        printf("maxrank : %d\n", maxRank);
        printf("batch count: %d  isDiagonal: %d  tile size: %d  numTilesInAxis: %d\n", batchCount, isDiagonal, matrix->tileSize, matrix->numTilesInAxis);

        int *d_rowsBatch, *d_colsBatch, *d_ranksOutput;
        int *d_LDMBatch, *d_LDABatch, *d_LDBBatch;
        T *d_UOutput, *d_VOutput;
        T **d_MPtrs, **d_UOutputPtrs, **d_VOutputPtrs;
        cudaMalloc((void**) &d_rowsBatch, batchCount*sizeof(int));
        cudaMalloc((void**) &d_colsBatch, batchCount*sizeof(int));
        cudaMalloc((void**) &d_ranksOutput, batchCount*matrix->numTilesInAxis*sizeof(int));
        cudaMalloc((void**) &d_LDMBatch, batchCount*sizeof(int));
        cudaMalloc((void**) &d_LDABatch, batchCount*sizeof(int));
        cudaMalloc((void**) &d_LDBBatch, batchCount*sizeof(int));
        cudaMalloc((void**) &d_UOutput, batchCount*matrix->tileSize*maxRank*sizeof(T));
        cudaMalloc((void**) &d_VOutput, batchCount*matrix->tileSize*maxRank*sizeof(T));
        cudaMalloc((void**) &d_MPtrs, batchCount*sizeof(T*));
        cudaMalloc((void**) &d_UOutputPtrs, batchCount*sizeof(T*));
        cudaMalloc((void**) &d_VOutputPtrs, batchCount*sizeof(T*));

        fillARAHelpers(
            batchCount, matrix->tileSize, 
            d_rowsBatch, d_colsBatch, 
            d_LDMBatch, d_LDABatch, d_LDBBatch);

        kblasHandle_t kblasHandle;
        kblasRandState_t randState;
        kblasCreate(&kblasHandle);
        kblasInitRandState(kblasHandle, &randState, 1 << 15, 0);
        kblasEnableMagma(kblasHandle);
        kblas_ara_batch_wsquery<T>(kblasHandle, matrix->tileSize, batchCount);
        kblasAllocateWorkspace(kblasHandle);

        T *d_denseTileCol;
        int* d_scannedRanks;
        cudaMalloc((void**) &d_denseTileCol, batchCount*matrix->tileSize*matrix->tileSize*sizeof(T));
        cudaMalloc((void**) &d_scannedRanks, batchCount*sizeof(int));

        for(unsigned int tileColIdx = 0; tileColIdx < matrix->numTilesInAxis; ++tileColIdx) {
            generateDenseTileCol <T> (
                d_denseTileCol,
                d_pointCloud,
                kdtree,
                tileColIdx,
                matrix->tileSize,
                batchCount,
                pieceMortonIndex, numPiecesInAxis,
                matrix->numTilesInAxis,
                isDiagonal);

            generateArrayOfPointersT<T>(d_denseTileCol, d_MPtrs, matrix->tileSize*matrix->tileSize, batchCount, 0);
            generateArrayOfPointersT<T>(d_UOutput, d_UOutputPtrs, matrix->tileSize*maxRank, batchCount, 0);
            generateArrayOfPointersT<T>(d_VOutput, d_VOutputPtrs, matrix->tileSize*maxRank, batchCount, 0);
            cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

            int kblas_ara_return = kblas_ara_batch(
                kblasHandle, d_rowsBatch, d_colsBatch, d_MPtrs, d_LDMBatch, 
                d_UOutputPtrs, d_LDABatch, d_VOutputPtrs, d_LDBBatch, d_ranksOutput + tileColIdx*batchCount,
                tol, matrix->tileSize, matrix->tileSize, maxRank, 16, ARA_R, randState, 0, batchCount);
            assert(kblas_ara_return == 1);

            void *d_temp_storage = NULL;
            size_t temp_storage_bytes = 0;
            cub::DeviceScan::InclusiveSum(
                d_temp_storage, temp_storage_bytes, d_ranksOutput + tileColIdx*batchCount, d_scannedRanks, batchCount);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceScan::InclusiveSum(
                d_temp_storage, temp_storage_bytes, d_ranksOutput + tileColIdx*batchCount, d_scannedRanks, batchCount);

            int colRankSum;
            cudaMemcpy(&colRankSum, &d_scannedRanks[batchCount - 1], sizeof(int), cudaMemcpyDeviceToHost);

            matrix->d_U.resize((totalRankSum + colRankSum)*matrix->tileSize);
            matrix->d_V.resize((totalRankSum + colRankSum)*matrix->tileSize);

            assert(matrix->tileSize <= 512);

            copyTiles <T> (matrix, d_UOutput, d_VOutput, d_scannedRanks, maxRank, batchCount);

            totalRankSum += colRankSum;
        }
        printf("rank sum: %d\n", totalRankSum);

        cudaFree(d_denseTileCol);
        cudaFree(d_scannedRanks);
        cudaFree(d_rowsBatch);
        cudaFree(d_colsBatch);
        cudaFree(d_ranksOutput);
        cudaFree(d_LDMBatch);
        cudaFree(d_LDABatch);
        cudaFree(d_LDBBatch);
        cudaFree(d_MPtrs);
        cudaFree(d_UOutputPtrs);
        cudaFree(d_VOutputPtrs);
        cudaFree(d_UOutput);
        cudaFree(d_VOutput);

        kblasFreeWorkspace(kblasHandle);
        kblasDestroy(&kblasHandle);
        kblasHandle = NULL;
        kblasDestroyRandState(randState);
        magma_finalize();
}

template void buildTLRMatrixPiece <H2Opus_Real> (
    TLR_Matrix *matrix,
    KDTree kdtree,
    H2Opus_Real* d_dataset,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    H2Opus_Real tol);


template <class T>
void checkErrorInTLRPiece(
    TLR_Matrix matrix,
    KDTree kdtree,
    T* d_pointCloud,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis) {

        bool isDiagonal = isPieceDiagonal(pieceMortonIndex);
        unsigned int numTilesInCol = isDiagonal ? matrix.numTilesInAxis - 1: matrix.numTilesInAxis;

        T *d_densePiece;
        cudaMalloc(
            (void**) &d_densePiece,matrix.numTilesInAxis*numTilesInCol*matrix.tileSize*matrix.tileSize*sizeof(T));
        generateDensePiece <T> (
            d_densePiece,
            kdtree,
            d_pointCloud,
            pieceMortonIndex, numPiecesInAxis,
            matrix.numTilesInAxis, numTilesInCol,
            isDiagonal);

        H2Opus_Real* d_expandedTLRPiece;
        cudaMalloc(
            (void**) &d_expandedTLRPiece,
            matrix.numTilesInAxis*numTilesInCol*matrix.tileSize*matrix.tileSize*sizeof(T));

        dim3 numBlocks(matrix.numTilesInAxis, numTilesInCol);
        dim3 numThreadsPerBlock(32, 32);
        expandTLRPiece <<< numBlocks, numThreadsPerBlock >>> (matrix.numTilesInAxis, numTilesInCol, d_expandedTLRPiece, matrix);
}

template void checkErrorInTLRPiece <H2Opus_Real> (
    TLR_Matrix matrix,
    KDTree kdtree,
    H2Opus_Real* d_pointCloud,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis);
