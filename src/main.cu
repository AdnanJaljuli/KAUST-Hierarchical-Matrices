
#include "admissibilityFunctions.cuh"
#include "config.h"
#include "counters.h"
#include "constructTLRMatrix.cuh"
#include "expandMatrix.cuh"
#include "generateDataset.cuh"
#include "helperFunctions.cuh"
#include "HMatrix.cuh"
#include "HMatrixStructure.cuh"
#include "HMatrixVectorMultiplication.cuh"
#include "constructHMatrixFromStruct.cuh"
#include "kDTree.cuh"
#include "kDTreeHelpers.cuh"
#include "kDTreeConstruction.cuh"
#include "magma_auxiliary.h"
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
    printf("segment size: %lu\n", kDTree.leafSize);
    printf("num segments: %lu\n", kDTree.numSegments);
    printf("num levels: %d\n", kDTree.numLevels);
    #if USE_COUNTERS
    endTime(KDTREE, &counters);
    #endif

    // create HMatrixStructure
    HMatrix hierarchicalMatrix;
    allocateHMatrixStructure(&hierarchicalMatrix.matrixStructure, kDTree.numLevels);
    if(config.admissibilityCondition == BOX_CENTER_ADMISSIBILITY) {
        H2Opus_Real eta = 1;
        BBoxCenterAdmissibility admissibility(eta, kDTree.nDim);
        constructHMatrixStructure(
            &hierarchicalMatrix.matrixStructure,
            admissibility,
            kDTree,
            kDTree);
    }
    else if(config.admissibilityCondition == WEAK_ADMISSIBILITY) {
        WeakAdmissibility admissibility;
        constructHMatrixStructure(
            &hierarchicalMatrix.matrixStructure,
            admissibility,
            kDTree,
            kDTree);
    }

    #if EXPAND_MATRIX
    printKDTree(config.N, config.nDim, config.divMethod, config.leafSize, kDTree, d_pointCloud);
    printMatrixStructure(hierarchicalMatrix.matrixStructure);
    #endif

    freeKDTree(kDTree);

    #if USE_COUNTERS
    endTime(TOTAL_TIME, &counters);
    printCountersInFile(config, &counters);
    #endif

    printf("done :)\n");

    return 0;

}