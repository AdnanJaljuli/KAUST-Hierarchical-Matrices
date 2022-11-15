
#include "config.h"
#include "counters.h"
#include "createLRMatrix.cuh"
#include "expandMatrix.cuh"
#include "helperFunctions.cuh"
#include "createHMatrixFromStruct.cuh"
#include "kblas.h"
#include "kDTree.cuh"
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

// TODO: debug dealing with non-powers of two
// TODO: add counters for functions in main
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
    H2Opus_Real* d_pointCloud;
    cudaMalloc((void**) &d_pointCloud, config.numberOfInputPoints*config.dimensionOfInputPoints*sizeof(H2Opus_Real));
    generateDataset(config.numberOfInputPoints, config.dimensionOfInputPoints, d_pointCloud);

    // Build the KD-tree
    magma_init();
    KDTree kDTree;
    allocateKDTree(kDTree, config.numberOfInputPoints, config.bucketSize);
    constructKDTree(config.numberOfInputPoints, config.dimensionOfInputPoints, config.bucketSize, kDTree, d_pointCloud);

    printf("segment size: %lu\n", kDTree.segmentSize);
    printf("num segments: %lu\n", kDTree.numSegments);

    // Build the TLR matrix
    const int ARA_R = 10;
    TLR_Matrix matrix;
    matrix.ordering = COLUMN_MAJOR;
    uint64_t rankSum = createColumnMajorLRMatrix(config.numberOfInputPoints, config.bucketSize, config.dimensionOfInputPoints, matrix, kDTree, d_pointCloud, config.lowestLevelTolerance, ARA_R);

    #if EXPAND_MATRIX
    H2Opus_Real* d_denseMatrix;
    // TODO: assert that dense matrix doesn't exceed memory limit
    cudaMalloc((void**) &d_denseMatrix, kDTree.numSegments*kDTree.numSegments*kDTree.segmentSize*kDTree.segmentSize*sizeof(H2Opus_Real));
    generateDenseMatrix(config.numberOfInputPoints, kDTree.numSegments, kDTree.segmentSize, config.dimensionOfInputPoints, d_denseMatrix, kDTree.segmentIndices, kDTree.segmentOffsets, d_pointCloud);
    #endif

    cudaFree(d_pointCloud);

    #if EXPAND_MATRIX
    checkErrorInLRMatrix(kDTree.numSegments, kDTree.segmentSize, matrix, d_denseMatrix);
    #endif

    // Convert TLR matrix to morton order
    TLR_Matrix mortonOrderedMatrix;
    mortonOrderedMatrix.ordering = MORTON;
    convertColumnMajorToMorton(kDTree.numSegments, kDTree.segmentSize, rankSum, matrix, mortonOrderedMatrix);
    freeMatrix(matrix);

    #if EXPAND_MATRIX
    checkErrorInLRMatrix(kDTree.numSegments, kDTree.segmentSize, mortonOrderedMatrix, d_denseMatrix);
    #endif

    // Build hierarchical matrix
    WeakAdmissibility WAStruct;
    allocateWeakAdmissibilityStruct(WAStruct, config.numberOfInputPoints, config.bucketSize);
    HMatrix hierarchicalMatrix;
    allocateHMatrix(hierarchicalMatrix, mortonOrderedMatrix, kDTree.segmentSize, kDTree.numSegments, config.numberOfInputPoints, config.bucketSize);
    generateHMatrixFromStruct(config.numberOfInputPoints, config.bucketSize, kDTree.numSegments, kDTree.segmentSize, mortonOrderedMatrix, ARA_R, config.lowestLevelTolerance, hierarchicalMatrix, WAStruct);

    #if EXPAND_MATRIX
    checkErrorInHMatrix(config.numberOfInputPoints, config.bucketSize, hierarchicalMatrix, d_denseMatrix);
    #endif

    freeWeakAdmissbilityStruct(WAStruct);
    freeKDTree(kDTree);
    freeMatrix(mortonOrderedMatrix);
    freeHMatrix(hierarchicalMatrix);
    #if EXPAND_MATRIX
    cudaFree(d_denseMatrix);
    #endif

    magma_finalize();

    #if USE_COUNTERS
    endTime(TOTAL_TIME, &counters);
    printCountersInFile(config, &counters);
    #endif

    return 0;

}

