
#include "admissibilityFunctions.cuh"
#include "buildTLRPiece.cuh"
#include "buildHMatrixPiece.cuh"
#include "config.h"
#include "counters.h"
#include "generateDataset.cuh"
#include "helperFunctions.cuh"
#include "HMatrix.cuh"
#include "HMatrixStructure.cuh"
#include "kDTree.cuh"
#include "kDTreeHelpers.cuh"
#include "kDTreeConstruction.cuh"
#include "TLRMatrix.cuh"

#include <algorithm>
#include <assert.h>
#include <inttypes.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <typeinfo>
#include <utility>
using namespace std;

// TODO: template all functions that deal with H2Opus_Real
int main(int argc, char *argv[]) {

    cudaDeviceSynchronize();

    Config config = parseArgs(argc, argv);
    printArgs(config);

    #if USE_COUNTERS
    Counters counters;
    initCounters(&counters);
    startTime(TOTAL_TIME, &counters);
    #endif

    // Generate the points
    #if USE_COUNTERS
    startTime(GENERATE_DATASET, &counters);
    #endif
    H2Opus_Real* d_pointCloud;
    cudaMalloc((void**) &d_pointCloud, config.N*config.nDim*sizeof(H2Opus_Real));
    generateDataset(config.N, config.nDim, d_pointCloud);
    #if USE_COUNTERS
    endTime(GENERATE_DATASET, &counters);
    #endif

    #if EXPAND_MATRIX
    printPointCloud(config.N, config.nDim, d_pointCloud);
    #endif

    // Build the KD-tree
    #if USE_COUNTERS
    startTime(KDTREE, &counters);
    #endif
    KDTree kDTree;
    allocateKDTree(
        kDTree,
        config.N,
        config.nDim,
        config.leafSize,
        config.divMethod);

    constructKDTree(
        kDTree,
        d_pointCloud,
        config.divMethod); // TODO: pass a reference to kdtree
    printf("segment size: %lu\n", kDTree.maxLeafSize);
    printf("num segments: %lu\n", kDTree.numLeaves);
    printf("num levels: %d\n", kDTree.numLevels);
    #if USE_COUNTERS
    endTime(KDTREE, &counters);
    #endif

    // create HMatrixStructure
    #if USE_COUNTERS
    startTime(HMATRIX_STRUCTURE, &counters);
    #endif
    HMatrix <H2Opus_Real> hierarchicalMatrix;
    allocateHMatrixStructure(&hierarchicalMatrix.structure, kDTree.numLevels);

    if(config.admissibilityCondition == BOX_CENTER_ADMISSIBILITY) {
        H2Opus_Real eta = 1;
        BBoxCenterAdmissibility <H2Opus_Real> admissibility(eta, kDTree.nDim);
        constructHMatrixStructure <H2Opus_Real>(
            &hierarchicalMatrix.structure,
            admissibility,
            kDTree,
            kDTree);
    }
    else if(config.admissibilityCondition == WEAK_ADMISSIBILITY) {
        WeakAdmissibility <H2Opus_Real> admissibility;
        constructHMatrixStructure <H2Opus_Real>(
            &hierarchicalMatrix.structure,
            admissibility,
            kDTree,
            kDTree);
    }
    #if USE_COUNTERS
    endTime(HMATRIX_STRUCTURE, &counters);
    #endif

    #if EXPAND_MATRIX

    for (int i = 0; i < hierarchicalMatrix.structure.tileIndices.size(); i++) {
        printf("[ ");
        for (int j = 0; j < hierarchicalMatrix.structure.tileIndices[i].size(); j++) {
            printf("%d ", hierarchicalMatrix.structure.tileIndices[i][j]);
        }
        printf(" ]");
        printf("\n");
    }
    printf("\n");


    printKDTree(config.N, config.nDim, config.divMethod, config.leafSize, kDTree, d_pointCloud);
    printMatrixStructure(hierarchicalMatrix.structure);
    #endif

    // build TLR and Hmatrix piece
    allocateHMatrix <H2Opus_Real> (hierarchicalMatrix, kDTree.maxLeafSize, kDTree.numLeaves);
    vector<unsigned int> maxRanks;
    generateHMatMaxRanks(hierarchicalMatrix.structure.numLevels, kDTree.maxLeafSize, maxRanks);

    for(unsigned int piece = 0; piece < config.numPiecesInAxis*config.numPiecesInAxis; ++piece) {
        TLR_Matrix TLRMatrix;
        TLRMatrix.ordering = COLUMN_MAJOR;

        // TODO: pass level instead of numPiecesInAxis
        buildTLRPiece <H2Opus_Real> (
            &TLRMatrix,
            kDTree,
            d_pointCloud,
            piece, config.numPiecesInAxis,
            config.lowestLevelTolerance);

        #if EXPAND_MATRIX
        checkErrorInTLRPiece <H2Opus_Real> (
            TLRMatrix,
            kDTree,
            d_pointCloud,
            piece, config.numPiecesInAxis);
        #endif

        TLR_Matrix mortonOrderedMatrix;
        mortonOrderedMatrix.ordering = MORTON;
        convertColumnMajorToMorton <H2Opus_Real> (TLRMatrix, &mortonOrderedMatrix);

        #if EXPAND_MATRIX
        checkErrorInTLRPiece <H2Opus_Real> (
            mortonOrderedMatrix,
            kDTree,
            d_pointCloud,
            piece, config.numPiecesInAxis);
        #endif

        unsigned int pieceLevel = __builtin_ctz(config.numPiecesInAxis);
        printf("piece level: %d\n", pieceLevel);
        buildHMatrixPiece <H2Opus_Real> (
            &hierarchicalMatrix,
            mortonOrderedMatrix,
            maxRanks,
            config.lowestLevelTolerance,
            piece, pieceLevel
        );

        freeTLRPiece(&TLRMatrix);
        freeTLRPiece(&mortonOrderedMatrix);
    }

    maxRanks.clear();
    freeKDTree(kDTree);

    #if USE_COUNTERS
    endTime(TOTAL_TIME, &counters);
    printCountersInFile(config, &counters);
    #endif

    printf("done :)\n");

    return 0;
}