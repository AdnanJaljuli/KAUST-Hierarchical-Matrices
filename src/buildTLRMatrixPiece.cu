
#include "buildTLRMatrixPiece.cuh"
#include "buildTLRMatrixPiece_helpers.cuh"
#include "helperKernels.cuh"
#include "kDTree.cuh"
#include "cublas_v2.h"
#include "kblas.h"
#include "batch_rand.h"
#include "magma_auxiliary.h"

#include <assert.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

__global__ void printK(int *ranksOutput, int batchCount) {
    for(unsigned int i = 0; i < batchCount; ++i) {
        printf("%d ", ranksOutput[i]);
    }
    printf("\n");
}

template <class T>
void buildTLRMatrixPiece(
    TLR_Matrix *matrix,
    KDTree kdtree,
    T* d_pointCloud,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    T tol) {

        magma_init();

        uint64_t rankSum = 0;
        int totalMem;
        int ARA_R = 10;
        matrix->tileSize = kdtree.maxLeafSize;
        matrix->numTilesInAxis = (upperPowerOfTwo(kdtree.N)/numPiecesInAxis)/matrix->tileSize;
        bool isDiagonal = isPieceDiagonal(pieceMortonIndex);
        unsigned int batchCount = isDiagonal ? matrix->numTilesInAxis - 1: matrix->numTilesInAxis;
        unsigned int maxRank = matrix->tileSize>>1;
        assert(numPiecesInAxis <= matrix->numTilesInAxis);

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
        int* d_colScanRanks;
        cudaMalloc((void**) &d_denseTileCol, batchCount*matrix->tileSize*matrix->tileSize*sizeof(T));
        cudaMalloc((void**) &d_colScanRanks, batchCount*sizeof(int));

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

            printK <<< 1, 1 >>> (d_ranksOutput + tileColIdx*batchCount, batchCount);
        }

        
        cudaFree(d_denseTileCol);
        cudaFree(d_rowsBatch);
        cudaFree(d_colsBatch);
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

template void buildTLRMatrixPiece<H2Opus_Real>(
    TLR_Matrix *matrix,
    KDTree kdtree,
    H2Opus_Real* d_dataset,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    H2Opus_Real tol);
