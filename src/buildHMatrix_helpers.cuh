
#ifndef BUILD_HMATRIX_HELPERS_H
#define BUILD_HMATRIX_HELPERS_H

#include "TLRMatrix.cuh"

#include <vector>

void generateHMatMaxRanks(unsigned int numLevels, unsigned int tileSize, std::vector<unsigned int> maxRanks);

std::pair<int, int> getTilesInPiece(
    std::vector<int> tileIndices,
    unsigned int tileLevel,
    unsigned int pieceMortonIndex, unsigned int pieceLevel);

template <class T>
struct LevelTilePtrs {
    T **d_U;
    T **d_V;
};

template <class T>
void allocateTilePtrs(
    int batchSize,
    int batchUnitSize,
    int tileSize,
    int tileLevel,
    int *d_tileIndices,
    LevelTilePtrs<T> *tilePtrs,
    TLR_Matrix TLRPiece);

template <class T>
void freeLevelTilePtrs(LevelTilePtrs <T> tilePtrs);

#endif