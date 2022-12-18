
#include "buildHMatrix.cuh"
#include "HMatrix.cuh"
#include "TLRMatrix.cuh"
#include "precision.h"
#include "kblas.h"
#include "batch_rand.h"

#include <algorithm> 
#include <cub/cub.cuh>
#include <vector>

// TODO: add TLRtolerance as a parameter. if eps(TLR) == eps(HMatrix) skip finest level, else: do kblas_ara on finest level to produce a finer tile
template <class T>
void buildHMatrixPiece (
    HMatrix <T> hierarchicalMatrix,
    TLR_Matrix TLRMatrix,
    std::vector<unsigned int> maxRanks,
    float lowestLevelTolerance,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis) {

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
            assert(pieceLevel <= tileLevel);
            pair<int, int> tilesInPiece = getTilesInPiece(
                hierarchicalMatrix.structure.tileIndices[tileLevel],
                tileLevel,
                pieceMortonIndex, pieceLevel);

            if(tilesInPiece.first != 0) {

                unsigned int batchUnitSize = 1 << (hierarchicalMatrix.structure.numLevels - (tileLevel + 1));
                int* d_tileIndices;
                cudaMalloc((void**) &d_tileIndices, tilesInPiece.first*sizeof(int));
                cudaMemcpy(d_tileIndices, hierarchicalMatrix.structure.tileIndices[tileLevel - 1][tilesInPiece.second], tilesInPiece.first*sizeof(int), cudaMemcpyHostToDevice);
            }
        }
}

template void buildHMatrixPiece <H2Opus_Real> (
    HMatrix <H2Opus_Real> hierarchicalMatrix,
    TLR_Matrix TLRMatrix,
    std::vector<unsigned int> maxRanks,
    float lowestLevelTolerance,
    unsigned int pieceMortonIndex, unsigned int numPiecesInAxis);

void generateHMatMaxRanks(unsigned int numLevels, unsigned int tileSize, std::vector<unsigned int> maxRanks) {
    for(unsigned int i = 0; i < numLevels - 2; ++i) {
        maxRanks.push_back(tileSize*(1 << i));
    }
}

std::pair<int, int> getTilesInPiece(
    std::vector<int> tileIndices,
    unsigned int tileLevel,
    unsigned int pieceMortonIndex, unsigned int pieceLevel) {

        unsigned int levelDiff = tileLevel/pieceLevel;
        unsigned int numBLocksInPieceAxis = 1<<(levelDiff - 1);
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