
// #include <cub/cub.cuh>
#include <algorithm> 
#include <stdio.h>
#include <vector>
#include <iostream>

std::pair<int, int> getTilesInPiece(std::vector<int> tileIndices, unsigned int tileLevel, unsigned int pieceMortonIndex, unsigned int pieceLevel) {
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

int main() {
    std::vector<int> v = {0, 2, 3};
    std::pair<int, int> ans = getTilesInPiece(v, 2, 1, 1);

    printf("%d   %d\n", ans.first, ans.second);
}