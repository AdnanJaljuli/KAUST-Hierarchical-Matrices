
#ifndef BUILD_H_MATRIX_h
#define BUILD_H_MATRIX_h

#include "HMatrix.cuh"
#include "TLRMatrix.cuh"

#include <vector>

template <class T>
void buildHMatrixPiece (
    HMatrix <T> hierarchicalMatrix,
    TLR_Matrix TLRMatrix,
    std::vector<unsigned int> maxRanks,
    float lowestLevelTolerance,
    unsigned int pieceMortonIndex, unsigned int pieceLevel);

void generateHMatMaxRanks(unsigned int numLevels, unsigned int tileSize, std::vector<unsigned int> maxRanks);

#endif