
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

// TODO: add TLRtolerance as a parameter. if eps(TLR) == eps(HMatrix) skip finest level, else: do kblas_ara on finest level to produce a finer tile
template <class T>
void buildHMatrixPiece (
    HMatrix <T> hierarchicalMatrix,
    TLR_Matrix TLRMatrix,
    std::vector<unsigned int> maxRanks,
    float lowestLevelTolerance,
    unsigned int pieceMortonIndex, unsigned int pieceLevel) {

        unsigned int  numPiecesInAxis = 1<<pieceLevel;

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
        // TODO: allocate memory outside the loop
        // TODO: use multiple streams

        for(unsigned int tileLevel = hierarchicalMatrix.structure.numLevels - 2; tileLevel > 0; --tileLevel) {
            #if 1
            assert(pieceLevel <= tileLevel);
            std::pair<int, int> tilesInPiece = getTilesInPiece(
                hierarchicalMatrix.structure.tileIndices[tileLevel],
                tileLevel,
                pieceMortonIndex, pieceLevel);

            if(tilesInPiece.first != 0) {

                unsigned int batchUnitSize = 1 << (hierarchicalMatrix.structure.numLevels - (tileLevel + 1));
                int* d_tileIndices;
                cudaMalloc((void**) &d_tileIndices, tilesInPiece.first*sizeof(int));
                cudaMemcpy(
                    d_tileIndices,
                    &hierarchicalMatrix.structure.tileIndices[tileLevel - 1][tilesInPiece.second],
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

                //         // scan the ranks array
                // int *d_scanRanks;
                // int **d_scanRanksPtrs;
                // cudaMalloc((void**) &d_scanRanks, batchSize*batchUnitSize*batchUnitSize*sizeof(int));
                // cudaMalloc((void**) &d_scanRanksPtrs, batchSize*sizeof(int*));
                // generateScanRanks(batchSize, batchUnitSize, mortonOrderedMatrix.blockRanks, d_scanRanks, d_scanRanksPtrs, hierarchicalMatrix.matrixStructure.tileIndices[level - 1]);
            }
            #endif
        }
}

template void buildHMatrixPiece <H2Opus_Real> (
    HMatrix <H2Opus_Real> hierarchicalMatrix,
    TLR_Matrix TLRMatrix,
    std::vector<unsigned int> maxRanks,
    float lowestLevelTolerance,
    unsigned int pieceMortonIndex, unsigned int pieceLevel);

