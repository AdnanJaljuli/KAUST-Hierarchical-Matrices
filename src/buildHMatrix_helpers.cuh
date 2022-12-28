
#ifndef BUILD_HMATRIX_HELPERS_H
#define BUILD_HMATRIX_HELPERS_H

#include "TLRMatrix.cuh"

#include <vector>
#include <utility>

void generateHMatMaxRanks(unsigned int numLevels, unsigned int tileSize, std::vector<unsigned int> *maxRanks);

std::pair<int, int> getTilesInPiece(
    std::vector<int> tileIndices,
    unsigned int tileLevel,
    unsigned int pieceMortonIndex, unsigned int pieceLevel);

struct LevelTilePtrs {
    H2Opus_Real **d_U;
    H2Opus_Real **d_V;
};

template <class T>
void allocateTilePtrs(
    int batchSize,
    int batchUnitSize,
    int tileSize,
    int tileLevel,
    int *d_tileIndices,
    LevelTilePtrs &tilePtrs,
    TLR_Matrix TLRPiece);

template <class T>
void freeLevelTilePtrs(LevelTilePtrs tilePtrs);

void getRanks(int *d_blockRanks, int *d_blockScanRanks, int size);

void generateScanRanks(
    int batchSize,
    int batchUnitSize,
    int *ranks,
    int *scanRanks,
    int **scanRanksPtrs,
    int *levelTileIndices);

__global__ void fillLRARAArrays(
    int batchSize,
    int maxRows,
    int* d_rowsBatch, int* d_colsBatch,
    int* d_LDABatch, int* d_LDBBatch);

#endif