
#include "config.h"
#include "counters.h"
#include "constructTLRMatrix.cuh"
#include "expandMatrix.cuh"
#include "generateDataset.cuh"
#include "helperFunctions.cuh"
#include "HMatrix.cuh"
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
    #if USE_COUNTERS
    startTime(GENERATE_DATASET, &counters);
    #endif
    H2Opus_Real* d_pointCloud;
    cudaMalloc((void**) &d_pointCloud, config.numberOfInputPoints*config.dimensionOfInputPoints*sizeof(H2Opus_Real));
    generateDataset(config.numberOfInputPoints, config.dimensionOfInputPoints, d_pointCloud);
    #if USE_COUNTERS
    endTime(GENERATE_DATASET, &counters);
    #endif


    // Build the KD-tree
    magma_init();
    #if USE_COUNTERS
    startTime(KDTREE, &counters);
    #endif
    KDTree kDTree;
    allocateKDTree(
        kDTree,
        config.numberOfInputPoints,
        config.dimensionOfInputPoints,
        config.bucketSize,
        config.divMethod);

    constructKDTree(
        config.numberOfInputPoints,
        config.dimensionOfInputPoints,
        config.bucketSize,
        kDTree,
        d_pointCloud,
        config.divMethod); // TODO: pass a reference to kdtree
    printf("segment size: %lu\n", kDTree.segmentSize);
    printf("num segments: %lu\n", kDTree.numSegments);
    #if USE_COUNTERS
    endTime(KDTREE, &counters);
    #endif


    // create HMatrixStructure
    HMatrix hierarchicalMatrix;
    constructHMatrixStructure(
        &hierarchicalMatrix.matrixStructure,
        upperPowerOfTwo(config.numberOfInputPoints),
        config.dimensionOfInputPoints,
        config.bucketSize,
        config.admissibilityCondition,
        kDTree.boundingBoxes,
        kDTree.boundingBoxes);

    // Build the TLR matrix
    #if USE_COUNTERS
    startTime(TLR_MATRIX, &counters);
    #endif
    const int ARA_R = 10;
    TLR_Matrix matrix;
    matrix.ordering = COLUMN_MAJOR;
    // TODO: dont return rankSum. 
    // TODO: generate Morton ordering directly
    // TODO: print how much memory the matrix is consuming
    // TODO: print the rank distribution of the tiles and more statistics
    uint64_t rankSum = createColumnMajorLRMatrix(config.numberOfInputPoints, config.bucketSize, config.dimensionOfInputPoints, matrix, kDTree, d_pointCloud, config.lowestLevelTolerance, ARA_R);
    #if USE_COUNTERS
    endTime(TLR_MATRIX, &counters);
    #endif


    #if EXPAND_MATRIX
    H2Opus_Real* d_denseMatrix;
    // TODO: assert that dense matrix doesn't exceed memory limit
    cudaMalloc((void**) &d_denseMatrix, kDTree.numSegments*kDTree.numSegments*kDTree.segmentSize*kDTree.segmentSize*sizeof(H2Opus_Real));
    generateDenseMatrix(config.numberOfInputPoints, kDTree.numSegments, kDTree.segmentSize, config.dimensionOfInputPoints, d_denseMatrix, kDTree.segmentIndices, kDTree.segmentOffsets, d_pointCloud);
    checkErrorInLRMatrix(kDTree.numSegments, kDTree.segmentSize, matrix, d_denseMatrix);
    #endif

    cudaFree(d_pointCloud);

    // Convert TLR matrix to morton order
    #if USE_COUNTERS
    startTime(CMTOMO, &counters);
    #endif
    TLR_Matrix mortonOrderedMatrix;
    mortonOrderedMatrix.ordering = MORTON;
    convertColumnMajorToMorton(kDTree.numSegments, kDTree.segmentSize, rankSum, matrix, mortonOrderedMatrix);
    freeMatrix(matrix);
    #if USE_COUNTERS
    endTime(CMTOMO, &counters);
    #endif

    #if EXPAND_MATRIX
    checkErrorInLRMatrix(kDTree.numSegments, kDTree.segmentSize, mortonOrderedMatrix, d_denseMatrix);
    #endif

    // Build hierarchical matrix
    #if USE_COUNTERS
    startTime(HMATRIX, &counters);
    #endif
    allocateHMatrix(hierarchicalMatrix, mortonOrderedMatrix, kDTree.segmentSize, kDTree.numSegments, config.numberOfInputPoints, config.bucketSize, hierarchicalMatrix.matrixStructure);
    unsigned int *maxRanks = (unsigned int*)malloc((hierarchicalMatrix.matrixStructure.numLevels - 2)*sizeof(unsigned int));
    generateMaxRanks(hierarchicalMatrix.matrixStructure.numLevels, config.bucketSize, maxRanks);
    generateHMatrixFromStruct(config.numberOfInputPoints, config.bucketSize, kDTree.numSegments, kDTree.segmentSize, mortonOrderedMatrix, ARA_R, config.lowestLevelTolerance, hierarchicalMatrix, hierarchicalMatrix.matrixStructure, maxRanks);
    #if USE_COUNTERS
    endTime(HMATRIX, &counters);
    #endif

    #if EXPAND_MATRIX
    checkErrorInHMatrix(config.numberOfInputPoints, config.bucketSize, hierarchicalMatrix, d_denseMatrix);
    #endif
    free(maxRanks);
    freeHMatrixStructure(hierarchicalMatrix.matrixStructure);
    freeMatrix(mortonOrderedMatrix);

    #if 0

    // TODO: generate random vector
    H2Opus_Real *d_inputVectors, *d_resultVectors;
    cudaMalloc((void**) &d_inputVectors, config.vectorWidth*config.numberOfInputPoints*sizeof(H2Opus_Real));
    generateRandomVector(config.vectorWidth, config.numberOfInputPoints, d_inputVectors);
    cudaMalloc((void**) &d_resultVectors, config.vectorWidth*config.numberOfInputPoints*sizeof(H2Opus_Real));
    cudaMemset(d_resultVectors, 0, config.vectorWidth*config.numberOfInputPoints*sizeof(H2Opus_Real));

    // hierarchical matrix - vector multiplication
    HMatrixVecMult(config.numberOfInputPoints, config.bucketSize, kDTree.numSegments, config.vectorWidth, hierarchicalMatrix, d_inputVectors, d_resultVectors);

    #if EXPAND_MATRIX
    checkErrorInHmatrixVecMult(config.numberOfInputPoints, config.vectorWidth, kDTree.numSegments, d_denseMatrix, d_inputVectors, d_resultVectors);
    #endif

    cudaFree(d_inputVectors);
    cudaFree(d_resultVectors);
    freeKDTree(kDTree);
    freeHMatrix(hierarchicalMatrix);
    magma_finalize();
    #if EXPAND_MATRIX
    cudaFree(d_denseMatrix);
    #endif

    #endif

    #if USE_COUNTERS
    endTime(TOTAL_TIME, &counters);
    printCountersInFile(config, &counters);
    #endif

    printf("done :)\n");

    return 0;

}

