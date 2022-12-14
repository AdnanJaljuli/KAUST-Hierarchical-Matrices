
#include "buildTLRMatrixPiece.cuh"
#include "helperKernels.cuh"
#include "kDTree.cuh"
#include "cublas_v2.h"
#include "kblas.h"
#include "batch_rand.h"
#include "magma_auxiliary.h"

#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

bool isPieceDiagonal(unsigned int pieceMortonIndex);
void fillARAHelpers(
    unsigned int batchCount, int tileSize,
    int *d_rowsBatch, int *d_colsBatch,
    int *d_LDMBatch, int *d_LDABatch, int *d_LDBBatch);


template <class T>
void buildTLRMatrixPiece(
    TLR_Matrix *matrix,
    KDTree kdtree,
    T* d_pointCloud,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    T tolerance) {

        magma_init();

        uint64_t rankSum = 0;
        int totalMem;
        matrix->tileSize = kdtree.leafSize;
        matrix->numTilesInAxis = (upperPowerOfTwo(kdtree.N)/numPiecesInAxis)/matrix->tileSize;
        unsigned int numTilesInPieceCol = 
            isPieceDiagonal(pieceMortonIndex) ? matrix->numTilesInAxis : matrix->numTilesInAxis - 1;
        unsigned int maxRank = matrix->tileSize>>1;

        int *d_rowsBatch, *d_colsBatch, *d_ranksOutput;
        int *d_LDMBatch, *d_LDABatch, *d_LDBBatch;
        T *d_UOutput, *d_VOutput;
        T **d_MPtrs, **d_UOutputPtrs, **d_VOutputPtrs;
        cudaMalloc((void**) &d_rowsBatch, numTilesInPieceCol*sizeof(int));
        cudaMalloc((void**) &d_colsBatch, numTilesInPieceCol*sizeof(int));
        cudaMalloc((void**) &d_ranksOutput, numTilesInPieceCol*kdtree.numLeaves*sizeof(int));
        cudaMalloc((void**) &d_LDMBatch, numTilesInPieceCol*sizeof(int));
        cudaMalloc((void**) &d_LDABatch, numTilesInPieceCol*sizeof(int));
        cudaMalloc((void**) &d_LDBBatch, numTilesInPieceCol*sizeof(int));
        cudaMalloc((void**) &d_UOutput, numTilesInPieceCol*matrix->tileSize*maxRank*sizeof(T));
        cudaMalloc((void**) &d_VOutput, numTilesInPieceCol*matrix->tileSize*maxRank*sizeof(T));
        cudaMalloc((void**) &d_MPtrs, numTilesInPieceCol*sizeof(T*));
        cudaMalloc((void**) &d_UOutputPtrs, numTilesInPieceCol*sizeof(T*));
        cudaMalloc((void**) &d_VOutputPtrs, numTilesInPieceCol*sizeof(T*));
        fillARAHelpers(
            numTilesInPieceCol, matrix->tileSize, 
            d_rowsBatch, d_colsBatch, 
            d_LDMBatch, d_LDABatch, d_LDBBatch);

        kblasHandle_t kblasHandle;
        kblasRandState_t randState;
        kblasCreate(&kblasHandle);
        kblasInitRandState(kblasHandle, &randState, 1 << 15, 0);
        kblasEnableMagma(kblasHandle);
        kblas_ara_batch_wsquery<T>(kblasHandle, matrix->tileSize, numTilesInPieceCol);
        kblasAllocateWorkspace(kblasHandle);

        T *d_denseTileCol;
        int* d_colScanRanks;
        cudaMalloc((void**) &d_denseTileCol, matrix->tileSize*matrix->tileSize*numTilesInPieceCol*sizeof(T));
        cudaMalloc((void**) &d_colScanRanks, numTilesInPieceCol*sizeof(int));

        for(unsigned int colSegment = 0; colSegment < matrix->numTilesInAxis; ++colSegment) {
            // generateDenseTileCol();
            
        }

        magma_finalize();
}

bool isPieceDiagonal(unsigned int pieceMortonIndex) {
    uint32_t pieceRow, pieceCol;
    morton2columnMajor((uint32_t)pieceMortonIndex, pieceCol, pieceRow);
    return (pieceCol == pieceRow);
}

void fillARAHelpers(
    unsigned int batchCount, int tileSize,
    int *d_rowsBatch, int *d_colsBatch,
    int *d_LDMBatch, int *d_LDABatch, int *d_LDBBatch) {

        thrust::device_vector<int> devicePtr;

        // devicePtr = thrust::device_pointer_cast(d_rowsBatch);
        // thrust::fill(devicePtr, devicePtr + batchCount, tileSize);

        // devicePtr = thrust::device_pointer_cast(d_colsBatch);
        // thrust::fill(devicePtr, devicePtr + batchCount, tileSize);

        // devicePtr = thrust::device_pointer_cast(d_LDMBatch);
        // thrust::fill(devicePtr, devicePtr + batchCount, tileSize);

        // devicePtr = thrust::device_pointer_cast(d_LDABatch);
        // thrust::fill(devicePtr, devicePtr + batchCount, tileSize);

        // devicePtr = thrust::device_pointer_cast(d_LDBBatch);
        // thrust::fill(devicePtr, devicePtr + batchCount, tileSize);
}

#if 0
generateDenseTileCol() {
    dim3 m_numThreadsPerBlock(min(32, matrix->tileSize), min(32, matrix->tileSize));
    dim3 m_numBlocks(1, numTilesInPieceCol);
    generateDenseTileColumn_kernel <<< m_numBlocks, m_numThreadsPerBlock >>> (
        numberOfInputPoints, kDTree.segmentSize, dimensionOfInputPoints, d_inputMatrixSegmented, d_dataset, kDTree, segment, matrix.diagonal);
}

template <class T>
__global__ void generateDenseTileColumn_kernel(T *denseTileCol, T *pointCloud, KDTree kdtree, unsigned int colIndex) {

    for(unsigned int i = 0; i < (kdtree.leafSize/blockDim.x); ++i) {
        for(unsigned int j = 0; j < (kdtree.leafSize/blockDim.x); ++j) {

            unsigned int diff = (blockIdx.y >= colIndex) ? 1 : 0;
            unsigned int rowTileIdx = blockIdx.y + diff;

            unsigned int row = rowTileIdx*kdtree.leafSize + i*blockDim.x + threadIdx.y;
            unsigned int col = colIndex*kdtree.leafSize + j*blockDim.x + threadIdx.x;
            int xDim = kDTree.segmentOffsets[colIndex + 1] - kDTree.segmentOffsets[colIndex];
            int yDim = kDTree.segmentOffsets[rowTileIdx + 1] - kDTree.segmentOffsets[rowTileIdx];

            uint64_t matrixIndex = static_cast<uint64_t>(rowTileIdx)*kdtree.leafSize*kdtree.leafSize - diff*kdtree.leafSize*kdtree.leafSize + j*blockDim.x*kdtree.leafSize + threadIdx.x*kdtree.leafSize + i*blockDim.y + threadIdx.y;

            if(threadIdx.x + j*blockDim.x >= xDim || threadIdx.y + i*blockDim.x >= yDim) {
                if(col == row) {
                    matrix[matrixIndex] = 1;
                }
                else {
                    matrix[matrixIndex] = 0;
                }
            }
            else {
                matrix[matrixIndex] = interaction(numberOfInputPoints, dimensionOfInputPoints, kDTree.segmentIndices[kDTree.segmentOffsets[colIndex] + blockDim.x*j + threadIdx.x], kDTree.segmentIndices[kDTree.segmentOffsets[blockIdx.y] + i*blockDim.x + threadIdx.y], pointCloud);
            }
        }
    }
}
#endif

template void buildTLRMatrixPiece<H2Opus_Real>(
    TLR_Matrix *matrix,
    KDTree kdtree,
    H2Opus_Real* d_dataset,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    H2Opus_Real tolerance);
