
#include "buildHMatrix.cuh"

#include <vector>

void generateHMatMaxRanks(unsigned int numLevels, unsigned int tileSize, std::vector<unsigned int> maxRanks) {
    for(unsigned int i = 0; i < numLevels - 2; ++i) {
        maxRanks.push_back(tileSize*(1 << i));
    }
}
