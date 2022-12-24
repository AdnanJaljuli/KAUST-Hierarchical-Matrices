
#include "buildHMatrix_helpers.cuh"
#include "helperKernels.cuh"

#include <algorithm>
#include <vector>
#include <utility>

void generateHMatMaxRanks(unsigned int numLevels, unsigned int tileSize, std::vector<unsigned int> maxRanks) {
    for(unsigned int i = 0; i < numLevels - 2; ++i) {
        maxRanks.push_back(tileSize*(1 << i));
    }
}

std::pair<int, int> getTilesInPiece(
    std::vector<int> tileIndices,
    unsigned int tileLevel,
    unsigned int pieceMortonIndex, unsigned int pieceLevel) {

        // unsigned int levelDiff = tileLevel/pieceLevel;
        // unsigned int numBLocksInPieceAxis = 1<<(levelDiff - 1);
        unsigned int numBLocksInPieceAxis = (1<<tileLevel)/(1<<pieceLevel);
        printf("numBLocksInPieceAxis: %d\n", numBLocksInPieceAxis);
        unsigned int left = pieceMortonIndex*numBLocksInPieceAxis*numBLocksInPieceAxis;
        unsigned int right = (pieceMortonIndex + 1)*numBLocksInPieceAxis*numBLocksInPieceAxis - 1;

        // binary search
        std::vector<int>::iterator lower = lower_bound(tileIndices.begin(), tileIndices.end(), left);
        std::vector<int>::iterator upper = upper_bound(tileIndices.begin(), tileIndices.end(), right);

        std::pair<int, int> ans;
        ans.first = upper - lower;
        ans.second = lower - tileIndices.begin();

        return ans;
}

template <class T>
__global__ void fillBatchPtrs(
    LevelTilePtrs <T> *tilePtrs,
    T *UPtr,
    T *VPtr,
    int *tileOffsets,
    int batchSize,
    int tileSize,
    int batchUnitSize,
    int *tileIndices,
    int tileLevel) {

        unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
        if(i < batchSize) {
            unsigned int numTilesInAxis = 1<<tileLevel;
            // unsigned int tileCol, tileRow;
            // morton2columnMajor(tileIndices[i], tileCol, tileRow);

            if(blockIdx.y == 0) {
                // tilePtrs->d_U[i] = &UPtr[
                //     static_cast<uint64_t>(
                //         tileOffsets[tileCol*numTilesInAxis*batchUnitSize*batchUnitSize + 
                //         tileRow*batchUnitSize*batchUnitSize])*
                //     tileSize];
                tilePtrs->d_U[i] = &UPtr[
                    static_cast<uint64_t>(tileOffsets[tileIndices[i]*batchUnitSize*batchUnitSize])
                    *tileSize
                ];
            }
            else {
                // tilePtrs->d_V[i] = &VPtr[
                //     static_cast<uint64_t>(
                //         tileOffsets[tileCol*numTilesInAxis*batchUnitSize*batchUnitSize + 
                //         tileRow*batchUnitSize*batchUnitSize])*
                //     tileSize];
                tilePtrs->d_V[i] = &VPtr[
                    static_cast<uint64_t>(
                        tileOffsets[tileIndices[i]*batchUnitSize*batchUnitSize]
                    )*tileSize
                ];
            }
        }
}

template <class T>
void allocateTilePtrs(
    int batchSize,
    int batchUnitSize,
    int tileSize,
    int tileLevel,
    int *d_tileIndices,
    LevelTilePtrs <T> *tilePtrs,
    TLR_Matrix TLRPiece) {

        cudaMalloc((void**) &tilePtrs->d_U, batchSize*sizeof(T*));
        cudaMalloc((void**) &tilePtrs->d_V, batchSize*sizeof(T*));

        T *d_UPtr = thrust::raw_pointer_cast(TLRPiece.d_U.data());
        T *d_VPtr = thrust::raw_pointer_cast(TLRPiece.d_V.data());

        dim3 numThreadsPerBlock(1024);
        dim3 numBlocks((batchSize + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x, 2);
        fillBatchPtrs <<< numBlocks, numThreadsPerBlock >>> (tilePtrs, d_UPtr, d_VPtr, TLRPiece.d_tileOffsets, batchSize, tileSize, batchUnitSize, d_tileIndices, tileLevel);
}

template void allocateTilePtrs <H2Opus_Real> (
    int batchSize,
    int batchUnitSize,
    int tileSize,
    int tileLevel,
    int *d_tileIndices,
    LevelTilePtrs <H2Opus_Real> *tilePtrs,
    TLR_Matrix TLRPiece);

template <class T>
void freeLevelTilePtrs(LevelTilePtrs <T> tilePtrs) {
    cudaFree(tilePtrs.d_U);
    cudaFree(tilePtrs.d_V);
}

__global__ void getRanks_kernel(int *blockRanks, int *blockScanRanks, int size) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < size) {
        int prevScanRanks = (i == 0) ? 0 : blockScanRanks[i - 1];
        blockRanks[i] = blockScanRanks[i] - prevScanRanks;
    }
}

void getRanks(int *d_blockRanks, int *d_blockScanRanks, int size) {
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (size + numThreadsPerBlock - 1)/numThreadsPerBlock;
    getRanks_kernel <<< numBlocks, numThreadsPerBlock >>> (d_blockRanks, d_blockScanRanks, size);
}

__global__ void fillScanRankPtrs(int **d_scanRanksPtrs, int *d_scanRanks, int batchUnitSize, int batchSize) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < batchSize) {
        d_scanRanksPtrs[i] = &d_scanRanks[i*batchUnitSize*batchUnitSize];
    }
}

void generateScanRanks(int batchSize, int batchUnitSize, int *ranks, int *scanRanks, int **scanRanksPtrs, int *levelTileIndices) {
    // TODO: we already have a scanRanks array of all the ranks in the MOMatrix. Use that one instead of this
    for(unsigned int batch = 0; batch < batchSize; ++batch) {
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, ranks + levelTileIndices[batch]*batchUnitSize*batchUnitSize, scanRanks + batch*batchUnitSize*batchUnitSize, batchUnitSize*batchUnitSize);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, ranks + levelTileIndices[batch]*batchUnitSize*batchUnitSize, scanRanks + batch*batchUnitSize*batchUnitSize, batchUnitSize*batchUnitSize);
        cudaFree(d_temp_storage);
    }

    // fillScanRanksPtrs
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (batchSize + numThreadsPerBlock - 1)/numThreadsPerBlock;
    fillScanRankPtrs <<< numBlocks, numThreadsPerBlock >>> (scanRanksPtrs, scanRanks, batchUnitSize, batchSize);
}