
#include "buildHMatrix_helpers.cuh"

#include <algorithm>
#include <vector>

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