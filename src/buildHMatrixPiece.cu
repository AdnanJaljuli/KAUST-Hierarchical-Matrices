
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
    HMatrix <T> *HMat,
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
        T *d_A, *d_B;
        T **d_APtrs, **d_BPtrs;

        int *d_blockRanks;
        cudaMalloc((void**) &d_blockRanks, TLRMatrix.numTilesInAxis*TLRMatrix.numTilesInAxis*sizeof(int));
        getRanks(d_blockRanks, TLRMatrix.d_tileOffsets, TLRMatrix.numTilesInAxis*TLRMatrix.numTilesInAxis);

        // TODO: allocate memory outside the loop
        for(unsigned int tileLevel = HMat->structure.numLevels - 2; tileLevel > 0; --tileLevel) {
            assert(pieceLevel <= tileLevel);
            printf("beginning of for loop\n");
            std::pair<int, int> tilesInPiece = getTilesInPiece(
                HMat->structure.tileIndices[tileLevel - 1],
                tileLevel,
                pieceMortonIndex, pieceLevel);

            printf("batchSize: %d    %d\n", tilesInPiece.first, tilesInPiece.second);

            if(tilesInPiece.first != 0) {
                printf("in if statement\n");

                unsigned int batchUnitSize = 1 << (HMat->structure.numLevels - (tileLevel + 1));
                int* d_tileIndices;
                cudaMalloc((void**) &d_tileIndices, tilesInPiece.first*sizeof(int));
                cudaMemcpy(
                    d_tileIndices,
                    &HMat->structure.tileIndices[tileLevel - 1][tilesInPiece.second],
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
                    &HMat->structure.tileIndices[tileLevel - 1][tilesInPiece.second]);

                lowestLevelTolerance *= 2;
                maxRows = batchUnitSize*TLRMatrix.tileSize;
                maxCols = maxRows;
                cudaMalloc((void**) &d_ranks, tilesInPiece.first*sizeof(int)); // TODO: allocate and free outside loop and reuse memory pools across levels
                cudaMalloc((void**) &d_rowsBatch, tilesInPiece.first*sizeof(int));
                cudaMalloc((void**) &d_colsBatch, tilesInPiece.first*sizeof(int));
                cudaMalloc((void**) &d_LDABatch, tilesInPiece.first*sizeof(int));
                cudaMalloc((void**) &d_LDBBatch, tilesInPiece.first*sizeof(int));
                cudaMalloc((void**) &d_APtrs, tilesInPiece.first*sizeof(T*));
                cudaMalloc((void**) &d_BPtrs, tilesInPiece.first*sizeof(T*));
                cudaMalloc((void**) &d_A, static_cast<uint64_t>(tilesInPiece.first)*maxRows*maxRanks[HMat->structure.numLevels - (tileLevel + 2)]*sizeof(T));
                cudaMalloc((void**) &d_B, static_cast<uint64_t>(tilesInPiece.first)*maxRows*maxRanks[HMat->structure.numLevels - (tileLevel + 2)]*sizeof(T));

                unsigned int numThreadsPerBlock = 1024;
                unsigned int numBlocks = (tilesInPiece.first + numThreadsPerBlock - 1)/numThreadsPerBlock;
                fillLRARAArrays <<< numBlocks, numThreadsPerBlock >>> (tilesInPiece.first, maxRows, d_rowsBatch, d_colsBatch, d_LDABatch, d_LDBBatch);

                generateArrayOfPointersT<T>(d_A, d_APtrs, maxRows*maxRanks[HMat->structure.numLevels - (tileLevel + 2)], tilesInPiece.first, 0);
                generateArrayOfPointersT<T>(d_B, d_BPtrs, maxRows*maxRanks[HMat->structure.numLevels - (tileLevel + 2)], tilesInPiece.first, 0);
                kblas_ara_batch_wsquery<T>(kblasHandle, maxRows, tilesInPiece.first);
                kblasAllocateWorkspace(kblasHandle);
            }

        }
        cudaFree(d_blockRanks);
}

template void buildHMatrixPiece <H2Opus_Real> (
    HMatrix <H2Opus_Real> *HMat,
    TLR_Matrix TLRMatrix,
    std::vector<unsigned int> maxRanks,
    float lowestLevelTolerance,
    unsigned int pieceMortonIndex, unsigned int pieceLevel);
