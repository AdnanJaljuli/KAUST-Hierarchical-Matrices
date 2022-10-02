
#include "config.h"
#include "counters.h"
#include "createLRMatrix.cuh"
#include "expandMatrixHelpers.cuh"
#include "helperFunctions.cuh"
#include "hierarchicalMatrixFunctions.cuh"
#include "kblas.h"
#include "kDTree.h"
#include "kDTreeConstruction.cuh"
#include "TLRMatrix.h"

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
    gpuErrchk(cudaMalloc((void**) &d_pointCloud, config.numberOfInputPoints*config.dimensionOfInputPoints*sizeof(H2Opus_Real)));
    generateDataset(config.numberOfInputPoints, config.dimensionOfInputPoints, d_pointCloud);

    // Build the KD-tree
    KDTree kDTree;
    allocateKDTree(kDTree, config.numberOfInputPoints, config.bucketSize);
    createKDTree(config.numberOfInputPoints, config.dimensionOfInputPoints, config.bucketSize, kDTree, d_pointCloud);

    // Build the TLR matrix
    uint64_t maxSegmentSize = config.bucketSize;
    printf("max segment size: %lu\n", maxSegmentSize);
    printf("num segments: %lu\n", kDTree.numSegments);

    const int ARA_R = 10;
    TLR_Matrix matrix;
    matrix.ordering = COLUMN_MAJOR;
    uint64_t rankSum = createColumnMajorLRMatrix(config.numberOfInputPoints, maxSegmentSize, config.bucketSize, config.dimensionOfInputPoints, matrix, kDTree, d_pointCloud, config.lowestLevelTolerance, ARA_R);

    #if EXPAND_MATRIX
    // TODO: assert that this doesn't exceed memory limit
    H2Opus_Real* d_denseMatrix;
    cudaMalloc((void**) &d_denseMatrix, kDTree.numSegments*kDTree.numSegments*maxSegmentSize*maxSegmentSize*sizeof(H2Opus_Real));
    generateDenseMatrix(config.numberOfInputPoints, kDTree.numSegments, maxSegmentSize, config.dimensionOfInputPoints, d_denseMatrix, kDTree.segmentIndices, kDTree.segmentOffsets, d_pointCloud);
    #endif

    cudaFree(d_pointCloud);
    cudaFreeKDTree(kDTree);
    gpuErrchk(cudaPeekAtLastError());

    #if EXPAND_MATRIX
    checkErrorInLRMatrix(kDTree.numSegments, maxSegmentSize, matrix, d_denseMatrix);
    #endif

    // Convert TLR matrix to morton order
    TLR_Matrix mortonMatrix;
    mortonMatrix.ordering = MORTON;
    convertColumnMajorToMorton(kDTree.numSegments, maxSegmentSize, rankSum, matrix, mortonMatrix);
    cudaFreeMatrix(matrix);

    #if EXPAND_MATRIX
    checkErrorInLRMatrix(kDTree.numSegments, maxSegmentSize, mortonMatrix, d_denseMatrix);
    #endif

    // Build hierarchical matrix
    // TODO: move declarations not used later inside the function
    #if 0
    int max_rows = maxSegmentSize;
    int max_cols = maxSegmentSize;
    int max_rank = max_cols;
    const int numLevels = __builtin_ctz(config.numberOfInputPoints/config.bucketSize) + 1;
    printf("numLevels: %d\n", numLevels);
    int** HMatrixExistingRanks = (int**)malloc((numLevels - 1)*sizeof(int*));
    int** HMatrixExistingTiles = (int**)malloc((numLevels - 1)*sizeof(int*));
    genereateHierarchicalMatrix(config.numberOfInputPoints, config.bucketSize, kDTree.numSegments, maxSegmentSize, numLevels, mortonMatrix, HMatrixExistingRanks, HMatrixExistingTiles);
    #endif

    cudaFreeMatrix(mortonMatrix);
    gpuErrchk(cudaPeekAtLastError());
    #if EXPAND_MATRIX
    cudaFree(d_denseMatrix);
    #endif

    #if USE_COUNTERS
    endTime(TOTAL_TIME, &counters);
    printCountersInFile(config, &counters);
    #endif

    return 0;
}