
#include "buildTLRMatrixPiece_helpers.cuh"
#include "helperKernels.cuh"
#include "kDTree.cuh"
#include "TLRMatrix.cuh"

#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

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
__global__ void printDenseTileColumn(H2Opus_Real *denseTileCol, unsigned int batchCount, unsigned int tileSize) {
    for(unsigned int i = 0; i < batchCount; ++i) {
        for(unsigned int j = 0; j < tileSize; ++j) {
            for(unsigned int k = 0; k < tileSize; ++k) {
                printf("%lf ", denseTileCol[i*tileSize*tileSize + k*tileSize + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
#endif

__device__ float getCorrelationLength(int nDim) {
    return nDim == 3 ? 0.2 : 0.1;
}

template <class T>
__device__ T interaction(int N, int nDim, int col, int row, T* pointCloud) {
    assert(col < N);
    assert(row < N);

    T diff = 0;
    T x, y;
    for (int d = 0; d < nDim; ++d){
        x = pointCloud[d*N + col];
        y = pointCloud[d*N + row];
        diff += (x - y)*(x - y);
    }
    T dist = sqrt(diff);
    return exp(-dist/getCorrelationLength(nDim));
}

template <class T>
__global__ void generateDenseTileColumn_kernel(
    T *denseTileCol, 
    T *pointCloud, 
    KDTree kdtree, 
    unsigned int tileColIdxInPiece,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    unsigned int numTilesInAxis,
    bool isDiagonal) {

        for(unsigned int i = 0; i < (kdtree.maxLeafSize/blockDim.x); ++i) {
            for(unsigned int j = 0; j < (kdtree.maxLeafSize/blockDim.x); ++j) {

                unsigned int tileRowIdxInPiece;
                if(isDiagonal) {
                    unsigned int diff = (blockIdx.y >= tileColIdxInPiece) ? 1 : 0;
                    tileRowIdxInPiece = blockIdx.y + diff;
                }
                else {
                    tileRowIdxInPiece = blockIdx.y;
                }

                uint32_t pieceCol, pieceRow;
                morton2columnMajor(pieceMortonIndex, pieceCol, pieceRow);

                int xDim = 
                    kdtree.leafOffsets[pieceCol*numTilesInAxis + tileColIdxInPiece + 1] - 
                    kdtree.leafOffsets[pieceCol*numTilesInAxis + tileColIdxInPiece];
                int yDim = 
                    kdtree.leafOffsets[pieceRow*numTilesInAxis + tileRowIdxInPiece + 1] - 
                    kdtree.leafOffsets[pieceRow*numTilesInAxis + tileRowIdxInPiece];

                uint64_t writeToIdx = 
                    static_cast<uint64_t>(blockIdx.y)*kdtree.maxLeafSize*kdtree.maxLeafSize +
                    j*blockDim.x*kdtree.maxLeafSize +
                    threadIdx.x*kdtree.maxLeafSize +
                    i*blockDim.y +
                    threadIdx.y;

                if(threadIdx.x + j*blockDim.x >= xDim || threadIdx.y + i*blockDim.x >= yDim) {
                    denseTileCol[writeToIdx] = 0;
                }
                else {
                    denseTileCol[writeToIdx] = 
                        interaction <T> (
                            kdtree.N, kdtree.nDim,
                            kdtree.leafIndices[kdtree.leafOffsets[pieceCol*numTilesInAxis + tileColIdxInPiece] + j*blockDim.x + threadIdx.x],
                            kdtree.leafIndices[kdtree.leafOffsets[pieceRow*numTilesInAxis + tileRowIdxInPiece] + i*blockDim.y + threadIdx.y],
                            pointCloud);
                }
            }
        }
}

template <class T>
void generateDenseTileCol(
    T *d_denseTileCol, 
    T *d_pointCloud, 
    KDTree kdtree, 
    unsigned int tileColIdx, 
    unsigned int tileSize, 
    unsigned int batchCount,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    unsigned int numTilesInAxis,
    bool isDiagonal) {

        dim3 m_numThreadsPerBlock(min(32, tileSize), min(32, tileSize));
        dim3 m_numBlocks(1, batchCount);
        generateDenseTileColumn_kernel<T><<< m_numBlocks, m_numThreadsPerBlock >>> (
            d_denseTileCol, 
            d_pointCloud, 
            kdtree, 
            tileColIdx, 
            pieceMortonIndex, numPiecesInAxis, 
            numTilesInAxis, 
            isDiagonal);
}


// _______________________________________________Explicit Instantiations_______________________________________________


template void generateDenseTileCol <H2Opus_Real> (
    H2Opus_Real *d_denseTileCol, 
    H2Opus_Real *d_pointCloud, 
    KDTree kdtree, 
    unsigned int tileColIdx, 
    unsigned int tileSize, 
    unsigned int batchCount,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    unsigned int numTilesInAxis,
    bool isDiagonal);