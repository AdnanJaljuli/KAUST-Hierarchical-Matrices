
#include "buildTLRPiece_helpers.cuh"
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


template <class T>
__global__ void copyTiles(
    T *UPtr, T *UOutput,
    T *VPtr, T *VOutput,
    int *scannedRanks,
    unsigned int tileSize,
    unsigned int maxRank) {

        unsigned int previousBlockScanRank = (blockIdx.x == 0) ? 0 : scannedRanks[blockIdx.x - 1];
        unsigned int blockRank = scannedRanks[blockIdx.x] - previousBlockScanRank;
        unsigned int numElementsInTile = blockRank*tileSize;
 
        for(unsigned int i = threadIdx.x; i < numElementsInTile; i += blockDim.x) {
            UPtr[previousBlockScanRank*tileSize + i] = UOutput[blockIdx.x*tileSize*maxRank + i];
            VPtr[previousBlockScanRank*tileSize + i] = VOutput[blockIdx.x*tileSize*maxRank + i];
        }
}

template <class T>
void copyTiles(
    TLR_Matrix *matrix, T *d_UOutput, T *d_VOutput, int *d_scannedRanks, int maxRank, unsigned int totalRankSum, int batchCount) {

        T *d_UPtr = thrust::raw_pointer_cast(matrix->d_U.data());
        T *d_VPtr = thrust::raw_pointer_cast(matrix->d_V.data());

        unsigned int numThreadsPerBlock = matrix->tileSize*2;
        unsigned int numBlocks = batchCount;
        copyTiles <T> <<< numBlocks, numThreadsPerBlock >>> (
            &d_UPtr[totalRankSum*matrix->tileSize], d_UOutput,
            &d_VPtr[totalRankSum*matrix->tileSize], d_VOutput,
            d_scannedRanks,
            matrix->tileSize,
            maxRank);
}

template void copyTiles <H2Opus_Real> (
    TLR_Matrix *matrix,
    H2Opus_Real *d_UOutput, H2Opus_Real *d_VOutput,
    int *d_scannedRanks,
    int maxRank,
    unsigned int totalRankSum,
    int batchCount);


__global__ void copyRanks(
    unsigned int numElements,
    int* fromRanks,
    int* toRanks,
    unsigned int tileColIdx,
    bool isDiagonal) {

        int i = blockIdx.x*blockDim.x + threadIdx.x;

        if(i < numElements) {
            if(!isDiagonal) {
                toRanks[i] = fromRanks[i];
            }
            else {
                int diff = (i >= tileColIdx) ? 1 : 0;
                toRanks[i + diff] = fromRanks[i];
                if(i == 0) {
                    toRanks[tileColIdx] = 0;
                }
            }
        }
}

template <class T>
void generateDensePiece(
    T *d_densePiece, 
    KDTree kdtree, 
    T *d_pointCloud, 
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    unsigned int numTilesInAxis, unsigned int numTilesInCol,
    bool isDiagonal) {

        for(unsigned int tileColIdx = 0; tileColIdx < numTilesInAxis; ++tileColIdx) {
            generateDenseTileCol <T> (
                &d_densePiece[tileColIdx*numTilesInCol*kdtree.maxLeafSize*kdtree.maxLeafSize],
                d_pointCloud,
                kdtree,
                tileColIdx,
                kdtree.maxLeafSize,
                numTilesInCol,
                pieceMortonIndex, numPiecesInAxis,
                numTilesInAxis,
                isDiagonal);
        }
}

template void generateDensePiece <H2Opus_Real> (
    H2Opus_Real *d_densePiece, 
    KDTree kdtree, 
    H2Opus_Real *d_pointCloud, 
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis,
    unsigned int numTilesInAxis, unsigned int numTilesInCol,
    bool isDiagonal);


template <class T>
__global__ void expandTLRPiece(
    T *UPtr, T *VPtr,
    int *tileOffsets,
    T* expandedPiece,
    unsigned int numTilesInAxis,
    unsigned int numTilesInCol,
    unsigned int tileSize,
    bool isDiagonal) {

        unsigned int col = blockIdx.x;
        unsigned int row;
        if(isDiagonal) {
            if(blockIdx.y < blockIdx.x) {
                row = blockIdx.y;
            }
            else {
                row = blockIdx.y + 1;
            }
        }
        else {
            row = blockIdx.y;
        }

        unsigned int index = col*numTilesInAxis + row;

        unsigned int previousBlockScanRank = (index == 0) ? 0 : tileOffsets[index - 1];
        unsigned int tileRank = tileOffsets[index] - previousBlockScanRank;

        T sum = 0;
        for(unsigned int i = 0; i < tileRank; ++i) {
            T u = UPtr[previousBlockScanRank*tileSize + i*tileSize + threadIdx.y];
            T v = VPtr[previousBlockScanRank*tileSize + i*tileSize + threadIdx.x];
            sum += u*v;
        }
        expandedPiece[col*numTilesInCol*tileSize*tileSize + 
                      blockIdx.y*tileSize*tileSize + 
                      threadIdx.x*tileSize + 
                      threadIdx.y] = sum;
}

template __global__ void expandTLRPiece <H2Opus_Real> (
    H2Opus_Real *UPtr, H2Opus_Real *VPtr,
    int *tileOffsets,
    H2Opus_Real* expandedPiece,
    unsigned int numTilesInAxis,
    unsigned int numTilesInCol,
    unsigned int tileSize,
    bool isDiagonal);


template <class T>
__global__ void calcErrorInPiece (T *expandedTLRPiece, T *densePiece, unsigned int numTilesInCol, unsigned int tileSize, T *error, T *tmp) {

    T x = densePiece[blockIdx.x*numTilesInCol*tileSize*tileSize +
                      blockIdx.y*tileSize*tileSize +
                      threadIdx.x*tileSize +
                      threadIdx.y];

    T y = expandedTLRPiece[blockIdx.x*numTilesInCol*tileSize*tileSize +
                      blockIdx.y*tileSize*tileSize +
                      threadIdx.x*tileSize +
                      threadIdx.y];

    // printf("%lf: x          %lf: y          %d %d %d %d\n", x, y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
    atomicAdd(tmp, x*x);
    atomicAdd(error, (x - y)*(x - y));
}

template __global__ void calcErrorInPiece <H2Opus_Real> (H2Opus_Real *expandedTLRPiece, H2Opus_Real *densePiece, unsigned int numTilesInCol, unsigned int tileSize, H2Opus_Real *error, H2Opus_Real *tmp);