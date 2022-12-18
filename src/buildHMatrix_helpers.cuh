
#ifndef BUILD_HMATRIX_HELPERS_H
#define BUILD_HMATRIX_HELPERS_H

#include <vector>

void generateHMatMaxRanks(unsigned int numLevels, unsigned int tileSize, std::vector<unsigned int> maxRanks);

std::pair<int, int> getTilesInPiece(
    std::vector<int> tileIndices,
    unsigned int tileLevel,
    unsigned int pieceMortonIndex, unsigned int pieceLevel);

#endif