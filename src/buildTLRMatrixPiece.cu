
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
        bool isDiagonal = isPieceDiagonal(pieceMortonIndex);
        unsigned int batchCount = isDiagonal ? matrix->numTilesInAxis : matrix->numTilesInAxis - 1;
        unsigned int maxRank = matrix->tileSize>>1;

        int *d_rowsBatch, *d_colsBatch, *d_ranksOutput;
        int *d_LDMBatch, *d_LDABatch, *d_LDBBatch;
        T *d_UOutput, *d_VOutput;
        T **d_MPtrs, **d_UOutputPtrs, **d_VOutputPtrs;
        cudaMalloc((void**) &d_rowsBatch, batchCount*sizeof(int));
        cudaMalloc((void**) &d_colsBatch, batchCount*sizeof(int));
        cudaMalloc((void**) &d_ranksOutput, batchCount*kdtree.numLeaves*sizeof(int));
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
        cudaMalloc((void**) &d_denseTileCol, matrix->tileSize*matrix->tileSize*batchCount*sizeof(T));
        cudaMalloc((void**) &d_colScanRanks, batchCount*sizeof(int));

        #if 0
        for(unsigned int colSegment = 0; colSegment < matrix->numTilesInAxis; ++colSegment) {
            generateDenseTileCol(
                d_denseTileCol, 
                d_pointCloud, 
                kdtree, 
                colSegment, 
                matrix->tileSize, 
                batchCount,
                pieceMortonIndex, numPiecesInAxis,
                isDiagonal);
        }
        #endif

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

        thrust::device_ptr<int> devicePtr;

        devicePtr = thrust::device_pointer_cast(d_rowsBatch);
        thrust::fill(devicePtr, devicePtr + batchCount, tileSize);

        devicePtr = thrust::device_pointer_cast(d_colsBatch);
        thrust::fill(devicePtr, devicePtr + batchCount, tileSize);

        devicePtr = thrust::device_pointer_cast(d_LDMBatch);
        thrust::fill(devicePtr, devicePtr + batchCount, tileSize);

        devicePtr = thrust::device_pointer_cast(d_LDABatch);
        thrust::fill(devicePtr, devicePtr + batchCount, tileSize);

        devicePtr = thrust::device_pointer_cast(d_LDBBatch);
        thrust::fill(devicePtr, devicePtr + batchCount, tileSize);
}

#if 0
template <class T>
void generateDenseTileCol(
    T *d_denseTileCol, 
    T *d_pointCloud, 
    KDTree kdtree, 
    unsigned int tileColIdx, 
    unsigned int tileSize, 
    unsigned int batchCount,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    bool isDiagonal) {

        dim3 m_numThreadsPerBlock(min(32, tileSize), min(32, tileSize));
        dim3 m_numBlocks(1, batchCount);
        generateDenseTileColumn_kernel <<< m_numBlocks, m_numThreadsPerBlock >>> 
            (d_denseTileCol, d_pointCloud, kdtree, tileColIdx, pieceMortonIndex, numPiecesInAxis);
}

__device__ H2Opus_Real getCorrelationLength(int dim){
    return dim == 3 ? 0.2 : 0.1;
}

__device__ H2Opus_Real interaction(int n, int dim, int col, int row, H2Opus_Real* pointCloud){
    assert(col<n);
    assert(row<n);

    H2Opus_Real diff = 0;
    H2Opus_Real x, y;
    for (int d = 0; d < dim; ++d){
        x = pointCloud[d*n + col];
        y = pointCloud[d*n + row];
        diff += (x - y)*(x - y);
    }

    H2Opus_Real dist = sqrt(diff);
    return exp(-dist/getCorrelationLength(dim));
}

template <class T>
__global__ void generateDenseTileColumn_kernel(
    T *denseTileCol, 
    T *pointCloud, 
    KDTree kdtree, 
    unsigned int tileColIdxInPiece,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    bool isDiagonal) {

        for(unsigned int i = 0; i < (kdtree.leafSize/blockDim.x); ++i) {
            for(unsigned int j = 0; j < (kdtree.leafSize/blockDim.x); ++j) {

                unsigned int tileRowIdxInPiece;
                if(isDiagonal) {
                    unsigned int diff = (blockIdx.y >= tileColIdxInPiece) ? 1 : 0;
                    tileRowIdxInPiece = blockIdx.y + diff;
                }
                else {
                    tileRowIdxInPiece = blockIdx.y;
                }

                // unsigned int col = tileColIdxInPiece*kdtree.leafSize + j*blockDim.x + threadIdx.x;
                // unsigned int row = tileRowIdxInPiece*kdtree.leafSize + i*blockDim.x + threadIdx.y;
                uint32_t pieceCol, pieceRow;
                morton2columnMajor(pieceMortonIndex, pieceCol, pieceRow);
                
                int xDim = kdtree.leafOffsets[tileColIdxInPiece + 1] - kdtree.leafOffsets[tileColIdxInPiece];
                int yDim = kdtree.leafOffsets[tileRowIdxInPiece + 1] - kdtree.leafOffsets[tileRowIdxInPiece];

                uint64_t writeToIdx = 
                    static_cast<uint64_t>(blockIdx.y)*kdtree.leafSize*kdtree.leafSize + 
                    j*blockDim.x*kdtree.leafSize + 
                    threadIdx.x*kdtree.leafSize + 
                    i*blockDim.y + 
                    threadIdx.y;

                if(threadIdx.x + j*blockDim.x >= xDim || threadIdx.y + i*blockDim.x >= yDim) {
                    denseTileCol[writeToIdx] = 0;
                }
                else {
                    denseTileCol[writeToIdx] = interaction(
                                            kdtree.N, kdtree.nDim, 
                                            kdtree.leafIndices[kdtree.leafOffsets[tileColIdxInPiece] + blockDim.x*j + threadIdx.x],
                                            kdtree.leafIndices[kdtree.leafOffsets[tileRowIdxInPiece] + i*blockDim.y + threadIdx.y],
                                            pointCloud);
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
