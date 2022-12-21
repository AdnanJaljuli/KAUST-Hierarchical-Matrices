
#include "buildTLRPiece.cuh"
#include "buildTLRPiece_helpers.cuh"
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

__global__ void printScannedRanks(int *array, int size) {
    printf("scanned ranks:\n");
    for(unsigned int i = 0; i < size; ++i) {
        for(unsigned int j = 0; j < size; ++j) {
            printf("%d ", array[i*size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

template <class T>
void buildTLRPiece(
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
        cudaMalloc((void**) &matrix->d_tileOffsets, matrix->numTilesInAxis*matrix->numTilesInAxis*sizeof(int));

        printf("maxrank : %d\n", maxRank);
        printf("batch count: %d  isDiagonal: %d  tile size: %d  numTilesInAxis: %d\n", batchCount, isDiagonal, matrix->tileSize, matrix->numTilesInAxis);

        int *d_rowsBatch, *d_colsBatch, *d_colRanks;
        int *d_LDMBatch, *d_LDABatch, *d_LDBBatch;
        T *d_UOutput, *d_VOutput;
        T **d_MPtrs, **d_UOutputPtrs, **d_VOutputPtrs;
        cudaMalloc((void**) &d_rowsBatch, batchCount*sizeof(int));
        cudaMalloc((void**) &d_colsBatch, batchCount*sizeof(int));
        cudaMalloc((void**) &d_colRanks, batchCount*sizeof(int));
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
        int *d_pieceTileRanks;
        cudaMalloc((void**) &d_denseTileCol, batchCount*matrix->tileSize*matrix->tileSize*sizeof(T));
        cudaMalloc((void**) &d_scannedRanks, batchCount*sizeof(int));
        cudaMalloc((void**) &d_pieceTileRanks, matrix->numTilesInAxis*matrix->numTilesInAxis*sizeof(int));

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
                d_UOutputPtrs, d_LDABatch, d_VOutputPtrs, d_LDBBatch, d_colRanks,
                tol, matrix->tileSize, matrix->tileSize, maxRank, 16, ARA_R, randState, 0, batchCount);
            assert(kblas_ara_return == 1);

            void *d_temp_storage = NULL;
            size_t temp_storage_bytes = 0;
            cub::DeviceScan::InclusiveSum(
                d_temp_storage, temp_storage_bytes, d_colRanks, d_scannedRanks, batchCount);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceScan::InclusiveSum(
                d_temp_storage, temp_storage_bytes, d_colRanks, d_scannedRanks, batchCount);

            int colRankSum;
            cudaMemcpy(&colRankSum, &d_scannedRanks[batchCount - 1], sizeof(int), cudaMemcpyDeviceToHost);

            matrix->d_U.resize((totalRankSum + colRankSum)*matrix->tileSize);
            matrix->d_V.resize((totalRankSum + colRankSum)*matrix->tileSize);

            assert(matrix->tileSize <= 512);

            copyTiles <T> (matrix, d_UOutput, d_VOutput, d_scannedRanks, maxRank, totalRankSum, batchCount);

            unsigned int numThreadsPerBlock = 1024;
            unsigned int numBlocks = (batchCount + numThreadsPerBlock - 1)/numThreadsPerBlock;
            copyRanks <<< numBlocks, numThreadsPerBlock >>> (
                batchCount,
                d_colRanks,
                &d_pieceTileRanks[tileColIdx*matrix->numTilesInAxis],
                tileColIdx,
                isDiagonal);

            totalRankSum += colRankSum;
        }

        matrix->rankSum = totalRankSum;

        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(
            d_temp_storage, temp_storage_bytes, d_pieceTileRanks, matrix->d_tileOffsets, matrix->numTilesInAxis*matrix->numTilesInAxis);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::InclusiveSum(
            d_temp_storage, temp_storage_bytes, d_pieceTileRanks, matrix->d_tileOffsets, matrix->numTilesInAxis*matrix->numTilesInAxis);


        cudaFree(d_denseTileCol);
        cudaFree(d_scannedRanks);
        cudaFree(d_rowsBatch);
        cudaFree(d_colsBatch);
        cudaFree(d_colRanks);
        cudaFree(d_LDMBatch);
        cudaFree(d_LDABatch);
        cudaFree(d_LDBBatch);
        cudaFree(d_MPtrs);
        cudaFree(d_UOutputPtrs);
        cudaFree(d_VOutputPtrs);
        cudaFree(d_UOutput);
        cudaFree(d_VOutput);
        cudaFree(d_pieceTileRanks);

        printf("rank sum: %d\n", totalRankSum);

        kblasFreeWorkspace(kblasHandle);
        kblasDestroy(&kblasHandle);
        kblasHandle = NULL;
        kblasDestroyRandState(randState);
        magma_finalize();

        // printScannedRanks <<< 1, 1 >>> (matrix->d_tileOffsets, matrix->numTilesInAxis);
}

template void buildTLRPiece <H2Opus_Real> (
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
            (void**) &d_densePiece, matrix.numTilesInAxis*numTilesInCol*matrix.tileSize*matrix.tileSize*sizeof(T));
        generateDensePiece <T> (
            d_densePiece,
            kdtree,
            d_pointCloud,
            pieceMortonIndex, numPiecesInAxis,
            matrix.numTilesInAxis, numTilesInCol,
            isDiagonal);

        T *d_expandedTLRPiece;
        cudaMalloc(
            (void**) &d_expandedTLRPiece,
            matrix.numTilesInAxis*numTilesInCol*matrix.tileSize*matrix.tileSize*sizeof(T));

        T *d_UPtr = thrust::raw_pointer_cast(matrix.d_U.data());
        T *d_VPtr = thrust::raw_pointer_cast(matrix.d_V.data());

        dim3 numBlocks(matrix.numTilesInAxis, numTilesInCol);
        dim3 numThreadsPerBlock(32, 32);
        expandTLRPiece <T> <<< numBlocks, numThreadsPerBlock >>>
            (d_UPtr, d_VPtr, matrix.d_tileOffsets, d_expandedTLRPiece, matrix.numTilesInAxis, numTilesInCol, matrix.tileSize, isDiagonal);

        T *d_error, *d_tmp;
        cudaMalloc((void**) &d_error, sizeof(T));
        cudaMalloc((void**) &d_tmp, sizeof(T));
        cudaMemset(d_error, 0, sizeof(T));
        cudaMemset(d_tmp, 0, sizeof(T));
        calcErrorInPiece <T> <<< numBlocks, numThreadsPerBlock >>> (d_expandedTLRPiece, d_densePiece, numTilesInCol, matrix.tileSize, d_error, d_tmp);
        T h_error;
        T h_tmp;
        cudaMemcpy(&h_error, d_error, sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_tmp, d_tmp, sizeof(T), cudaMemcpyDeviceToHost);
        printf("error in LR matrix: %lf\n", sqrt(h_error)/sqrt(h_tmp));
        cudaFree(d_tmp);
        cudaFree(d_error);
        cudaFree(d_densePiece);
        cudaFree(d_expandedTLRPiece);
}

template void checkErrorInTLRPiece <H2Opus_Real> (
    TLR_Matrix matrix,
    KDTree kdtree,
    H2Opus_Real* d_pointCloud,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis);
