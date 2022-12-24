
#include "buildHMatrixPiece.cuh"
#include "buildHMatrix_helpers.cuh"
#include "cublas_v2.h"
#include "HMatrix.cuh"
#include "TLRMatrix.cuh"
#include "precision.h"
#include "kblas.h"
#include "batch_rand.h"

#include <algorithm> 
#include <cub/cub.cuh>
#include <vector>
#include <utility>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

// TODO: add TLRtolerance as a parameter. if eps(TLR) == eps(HMatrix) skip finest level, else: do kblas_ara on finest level to produce a finer tile
template <class T>
void buildHMatrixPiece (
    HMatrix <T> HMat,
    TLR_Matrix TLRMatrix,
    std::vector<unsigned int> maxRanks,
    float lowestLevelTolerance,
    unsigned int pieceMortonIndex, unsigned int pieceLevel) {

        unsigned int numPiecesInAxis = 1<<pieceLevel;

        kblasHandle_t kblasHandle;
        kblasRandState_t randState;
        kblasCreate(&kblasHandle);
        kblasInitRandState(kblasHandle, &randState, 1<<15, 0);
        kblasEnableMagma(kblasHandle);

        // TODO: get rid of maxCols
        int maxRows;
        int maxCols;
        int *d_rowsBatch, *d_colsBatch, *d_ranks;
        int *d_LDABatch, *d_LDBBatch;
        H2Opus_Real *d_A, *d_B;
        H2Opus_Real **d_APtrs, **d_BPtrs;

        printf("here 1\n");

        int *d_blockRanks;
        cudaMalloc((void**) &d_blockRanks, TLRMatrix.numTilesInAxis*TLRMatrix.numTilesInAxis*sizeof(int));
        getRanks(d_blockRanks, TLRMatrix.d_tileOffsets, TLRMatrix.numTilesInAxis*TLRMatrix.numTilesInAxis);

        printf("here 2\n");

        // TODO: allocate memory outside the loop
        for(unsigned int tileLevel = HMat.structure.numLevels - 2; tileLevel > 0; --tileLevel) {
            printf("here 3\n");
            assert(pieceLevel <= tileLevel);
            printf("4\n");
            std::pair<int, int> tilesInPiece = getTilesInPiece(
                HMat.structure.tileIndices[tileLevel],
                tileLevel,
                pieceMortonIndex, pieceLevel);

            printf("here 5\n");

            if(tilesInPiece.first != 0) {
                printf("here\n");

                unsigned int batchUnitSize = 1 << (HMat.structure.numLevels - (tileLevel + 1));
                int* d_tileIndices;
                cudaMalloc((void**) &d_tileIndices, tilesInPiece.first*sizeof(int));
                cudaMemcpy(
                    d_tileIndices,
                    &HMat.structure.tileIndices[tileLevel - 1][tilesInPiece.second],
                    tilesInPiece.first*sizeof(int),
                    cudaMemcpyHostToDevice);


                LevelTilePtrs <T> tilePtrs;
                allocateTilePtrs <T> (
                    tilesInPiece.first,
                    batchUnitSize,
                    TLRMatrix.tileSize,
                    tileLevel,
                    d_tileIndices,
                    &tilePtrs,
                    TLRMatrix);


                // scan the ranks array
                int *d_scanRanks;
                int **d_scanRanksPtrs;
                cudaMalloc((void**) &d_scanRanks, tilesInPiece.first*batchUnitSize*batchUnitSize*sizeof(int));
                cudaMalloc((void**) &d_scanRanksPtrs, tilesInPiece.first*sizeof(int*));
                generateScanRanks(
                    tilesInPiece.first,
                    batchUnitSize,
                    d_blockRanks,
                    d_scanRanks,
                    d_scanRanksPtrs,
                    &HMat.structure.tileIndices[tileLevel - 1][tilesInPiece.second]);
            }

        }
        cudaFree(d_blockRanks);
}

template void buildHMatrixPiece <H2Opus_Real> (
    HMatrix <H2Opus_Real> HMat,
    TLR_Matrix TLRMatrix,
    std::vector<unsigned int> maxRanks,
    float lowestLevelTolerance,
    unsigned int pieceMortonIndex, unsigned int pieceLevel);

// void generateScanRanks(int batchSize, int batchUnitSize, int *scanRanks, int **scanRanksPtrs, int *levelTileIndices) {
//     // fillScanRanksPtrs
//     unsigned int numThreadsPerBlock = 1024;
//     unsigned int numBlocks = (batchSize + numThreadsPerBlock - 1)/numThreadsPerBlock;
//     fillScanRankPtrs <<< numBlocks, numThreadsPerBlock >>> (scanRanksPtrs, scanRanks, batchUnitSize, batchSize);
// }